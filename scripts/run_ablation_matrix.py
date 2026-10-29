#!/usr/bin/env python
"""四个组件消融实验的运行矩阵（隔离产物 + 配对统计）。

消融臂（component-removal only，其余数据/训练/阈值全部沿用主线）：

    full              主线（model_mode=causal_graph_crossmodal，加载预训练）
    no_pre_train      预训练消融：架构不变，load_pretrained=false 从随机初始化训练
    no_history_context 历史上下文消融：邻居张量置零、mask 全 False（只剩 null token）
    no_retrieval      检索消融：保留邻居与 mask，把可学习 attention 换成 mask-aware 均值
    no_gated_residual 门控残差消融：去门控，改无门控残差直连
    no_modal_attn     模态级自注意力消融：去掉 3-token self-attention

设计要点：
- 每个 (arm, class, seed) 用派生配置驱动正式入口 ``scripts/run_graph_tfusion.py``，
  校验、防泄露守卫、落盘路径完全复用主线；消融只通过 ``--ablation`` 注入模型工厂；
- 产物写到 ``{output_root}/{arm}/{class}/seed_{seed}``，互不覆盖，也不动
  ``save/detection_model`` 里的既有产物；
- 可续跑：某格已有 metrics.json 就跳过（除非 --force）；
- 跑完输出 summary.json / summary.csv / report.txt，含配对均值差、bootstrap 95% CI。

用法：
    python scripts/run_ablation_matrix.py --dataset DDOS_19 --seeds 777 888 999 --ratio 0.02
    python scripts/run_ablation_matrix.py --only DrDoS_NTP --seeds 777 --workers 1   # pilot
    python scripts/run_ablation_matrix.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PYTHON = "/liuluping/miniconda3/envs/shenzhiju/bin/python"
ENTRYPOINT = os.path.join("scripts", "run_graph_tfusion.py")

# (arm 名, --ablation 取值, 是否禁用预训练)
ARMS = (
    ("full", "none", False),
    ("no_pre_train", "none", True),
    ("no_history_context", "no_history_context", False),
    ("no_retrieval", "no_retrieval", False),
    ("no_gated_residual", "no_gated_residual", False),
    ("no_modal_attn", "no_modal_attn", False),
)

METRIC_KEYS = (
    "auc", "attack_f1", "f1", "macro_f1", "accuracy", "auprc",
    "attack_precision", "attack_recall", "fpr", "fnr",
    "support_benign", "support_attack", "threshold",
)

# 报告主指标：默认 Macro-F1，可用 --primary-metric 覆盖。
METRIC_NAMES = {"auc": "AUC", "attack_f1": "Attack-F1", "f1": "F1", "macro_f1": "Macro-F1"}
# 饱和类（无下降空间）/ 病态类（指标近随机）的判定阈值，随主指标取值。
SATURATED_THRESHOLD = {"auc": 0.999, "attack_f1": 0.99, "f1": 0.99, "macro_f1": 0.99}
DEGENERATE_THRESHOLD = {"auc": 0.70, "attack_f1": 0.70, "f1": 0.70, "macro_f1": 0.70}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="DDOS_19",
                        help="config/train-test 下的数据集子目录名（默认 DDOS_19=CICDDoS2019）")
    parser.add_argument("--config-root", default="config/train-test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[777, 888, 999])
    parser.add_argument("--ratio", type=float, default=0.02,
                        help="data_construct_param.train_sample_ratio（默认 0.02=逐类配置的少样本档；"
                             "1.0 会把任务推到 near-saturated，消融差异被天花板掩盖）")
    parser.add_argument("--context-isolation", default="none",
                        choices=["none", "train_only"],
                        help="上下文取用口径：none=全量流构建（默认，含转导通道）；"
                             "train_only=训练侧流建上下文（可辩护口径，AUC 下降以创造下降空间）")
    parser.add_argument("--context-stage", default=None,
                        choices=["before_context", "after_context"],
                        help="训练集抽样与上下文构建的先后顺序；after_context=先建完整 "
                             "train 上下文再抽训练行，用于消除训练/测试邻居密度差")
    # R2/R3 配方：梯度步预算、参数组学习率、门控零初始化、上下文密度
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--finetune-batch-size", type=int, default=None)
    parser.add_argument("--finetune-lr", type=float, default=None,
                        help="backbone 的基础微调学习率")
    parser.add_argument("--finetune-graph-lr", type=float, default=None,
                        help="图分支独立学习率；不给则沿用 finetune_lr")
    parser.add_argument("--finetune-backbone-lr-scale", type=float, default=None)
    parser.add_argument("--finetune-max-steps", type=int, default=None,
                        help="梯度步预算；达到后停止（优先于 epoch 数）")
    parser.add_argument("--finetune-patience", type=int, default=None,
                        help="验证指标连续多少轮不提升就早停")
    parser.add_argument("--finetune-scope", default=None,
                        choices=["full", "graph", "head", "layerwise"])
    parser.add_argument("--finetune-eval-metric", default=None,
                        choices=["auc", "loss"],
                        help="早停指标；默认 auc（阈值无关）")
    parser.add_argument("--zero-init-gate", action="store_true",
                        help="把三个门控残差初始化为接近恒等（gate≈sigmoid(bias)）")
    parser.add_argument("--gate-init-bias", type=float, default=-6.0)
    parser.add_argument("--graph-window", type=float, default=None,
                        help="图上下文时间窗（秒）")
    parser.add_argument("--neighbor-count", type=int, default=None,
                        help="每个流最多检索的历史邻居数")
    parser.add_argument("--external-val", action="store_true",
                        help="启用 data_split.val 作为早停/阈值验证集（默认关闭）")
    parser.add_argument("--optimize-threshold", choices=["true", "false"], default=None,
                        help="若提供，覆盖 train_param.optimize_threshold。"
                             "true = 按验证集选阈值（此时 F1 才是'最优 F1'）")
    parser.add_argument("--threshold-objective", choices=["macro_f1", "attack_f1"],
                        default=None,
                        help="阈值的寻优目标（train_param.threshold_objective），"
                             "默认沿用基础配置")
    parser.add_argument("--arms", nargs="+", default=[arm for arm, _, _ in ARMS],
                        choices=[arm for arm, _, _ in ARMS])
    parser.add_argument("--only", nargs="+", default=None,
                        help="仅跑这些类别（= 配置文件名去扩展名），用于 pilot")
    parser.add_argument("--output-root", default="save/ablations")
    parser.add_argument("--log-root", default="log/ablations")
    parser.add_argument("--fig-root", default="figure/ablations")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpu", type=int, default=None,
                        help="写入派生配置 gpu_enable 的设备号；不传则沿用旧的 CPU 行为")
    parser.add_argument("--force", action="store_true",
                        help="已有 metrics.json 也重跑")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-process", default=None, metavar="SUBSTR",
                        help="开跑前先等待：直到没有任何进程的命令行包含 SUBSTR 为止"
                             "（用于避开占用 CPU 额度的并发 sweep）")
    parser.add_argument("--wait-poll", type=int, default=60,
                        help="等待轮询间隔秒数（默认 60）")
    parser.add_argument("--no-summary", action="store_true",
                        help="只跑不汇总（用于分批跑）")
    parser.add_argument("--primary-metric", choices=["auc", "attack_f1", "f1", "macro_f1"],
                        default="macro_f1",
                        help="报告主指标（默认 macro_f1=Macro-F1）")
    parser.add_argument("--resummarize", action="store_true",
                        help="不跑任务，直接读取已有 summary.csv 重新生成报告")
    return parser.parse_args()


def _process_matching(substring: str):
    """返回第一个"python 进程"中命令行包含 substring 的 pid（不含自身）。

    只认 python 解释器拉起的进程：调用方的终端 shell 会把整条命令（含
    substring）留在自己的 cmdline 里，若把 shell 也算进去，等待条件永远
    不会满足。
    """
    me = str(os.getpid())
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or entry == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        argv = [part.decode("utf-8", "ignore") for part in raw.split(b"\x00") if part]
        if not argv or "python" not in os.path.basename(argv[0]):
            continue
        if substring in " ".join(argv):
            return entry
    return None


def wait_for_process(substring: str, poll_seconds: int) -> None:
    pid = _process_matching(substring)
    if pid is None:
        print(f"[ablation] 未发现匹配 {substring!r} 的进程，直接开始。", flush=True)
        return
    print(f"[ablation] 等待并发任务结束：pid={pid} 匹配 {substring!r}；"
          f"每 {poll_seconds}s 轮询。", flush=True)
    while (pid := _process_matching(substring)) is not None:
        time.sleep(poll_seconds)
    print(f"[ablation] 匹配 {substring!r} 的并发任务已结束，开始消融矩阵。", flush=True)


def iter_classes(config_root: str, dataset: str):
    dataset_dir = os.path.join(ROOT, config_root, dataset)
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"配置目录不存在: {dataset_dir}")
    for name in sorted(os.listdir(dataset_dir)):
        if name.endswith(".json") and not name.startswith("."):
            yield os.path.splitext(name)[0], os.path.join(dataset_dir, name)


def build_derived_config(base, arm, scratch, ratio, seed, dataset, stem, args):
    config = copy.deepcopy(base)
    config.setdefault("data_construct_param", {})["train_sample_ratio"] = ratio
    if args.context_isolation != "none":
        # train_only 让 train/pre-train 行只用训练侧流建上下文，堵掉转导通道；
        # test 行仍用其时间点之前的全部流（部署中合法可得的观测历史）。
        config["data_construct_param"]["context_isolation"] = args.context_isolation
    config.setdefault("train_param", {})["seed"] = int(seed)
    if args.optimize_threshold is not None:
        optimize = args.optimize_threshold == "true"
        config["train_param"]["optimize_threshold"] = optimize
        # 新字段优先，避免派生配置里的旧 false 覆盖 CLI 选择。
        config["train_param"]["threshold_strategy"] = (
            "validation" if optimize else "fixed_0.5"
        )
    if args.threshold_objective is not None:
        config["train_param"]["threshold_objective"] = args.threshold_objective
    if args.context_stage is not None:
        config.setdefault("data_construct_param", {})["train_sample_stage"] = (
            args.context_stage
        )
    data_params = config.setdefault("data_construct_param", {})
    train_params = config.setdefault("train_param", {})
    if args.graph_window is not None:
        data_params["graph_window"] = float(args.graph_window)
    if args.neighbor_count is not None:
        data_params["neighbor_count"] = int(args.neighbor_count)
    for key, value in (
        ("finetune_epochs", args.finetune_epochs),
        ("finetune_batch_size", args.finetune_batch_size),
        ("finetune_lr", args.finetune_lr),
        ("finetune_graph_lr", args.finetune_graph_lr),
        ("finetune_backbone_lr_scale", args.finetune_backbone_lr_scale),
        ("finetune_max_steps", args.finetune_max_steps),
        ("finetune_patience", args.finetune_patience),
        ("finetune_scope", args.finetune_scope),
        ("finetune_eval_metric", args.finetune_eval_metric),
    ):
        if value is not None:
            train_params[key] = value
    if args.zero_init_gate:
        train_params["zero_init_gate"] = True
        train_params["gate_init_bias"] = float(args.gate_init_bias)
    if args.external_val:
        config.setdefault("data_split", {}).setdefault("val", {})["enable"] = True
    # 沙箱内 CUDA 不可用；显式置 -1 让下游直接走 CPU，避免无意义的 CUDA 初始化告警。
    config["gpu_enable"] = [args.gpu] if args.gpu is not None else [-1]
    run_root = os.path.join(args.output_root, arm, dataset, stem, f"seed_{seed}")
    config["detection_model_path"] = run_root
    config["log_path"] = os.path.join(args.log_root, arm, dataset, stem, f"seed_{seed}")
    config["fig_path"] = os.path.join(args.fig_root, arm, dataset, stem, f"seed_{seed}")
    # 预训练消融：架构与主线逐位相同，只把加载开关关掉（provenance 会记 disabled_by_config）
    config["load_pretrained"] = not scratch
    config["ablation_arm"] = arm
    return config


def run_one(arm, ablation_flag, scratch, dataset, stem, base_path, seed, args):
    with open(base_path, encoding="utf-8") as handle:
        base = json.load(handle)
    config = build_derived_config(base, arm, scratch, args.ratio, seed, dataset, stem, args)
    metrics_path = os.path.join(config["detection_model_path"], "metrics.json")
    derived_path = os.path.join(
        args.log_root, "derived_configs", arm, dataset, f"{stem}_seed{seed}.json")
    os.makedirs(os.path.dirname(derived_path), exist_ok=True)
    with open(derived_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    label = f"{arm}/{dataset}/{stem}/seed_{seed}"
    if os.path.isfile(metrics_path) and not args.force:
        return label, "cached", 0.0, _read_metrics(metrics_path)
    if args.dry_run:
        return label, "dry-run", 0.0, None

    command = [args.python, os.path.join(ROOT, ENTRYPOINT), "-c", derived_path,
               "--ablation", ablation_flag]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    begin = time.time()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
    elapsed = time.time() - begin
    if result.returncode == 0 and os.path.isfile(metrics_path):
        return label, "ok", elapsed, _read_metrics(metrics_path)
    tail = (result.stderr or result.stdout or "").strip().splitlines()[-4:]
    return label, "FAIL(exit=%d) | %s" % (result.returncode, " / ".join(tail)), elapsed, None


def _read_metrics(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: payload.get(key) for key in METRIC_KEYS}


def bootstrap_ci(deltas, iterations=10000, seed=777):
    import random

    if not deltas:
        return None, None
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iterations):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def summarize(record, args) -> None:
    arms = [arm for arm in args.arms]
    classes = sorted({entry["class"] for entries in record.values() for entry in entries})
    seeds = sorted({entry["seed"] for entries in record.values() for entry in entries})
    metric_key = args.primary_metric
    metric_name = METRIC_NAMES[metric_key]
    sat_threshold = SATURATED_THRESHOLD[metric_key]
    deg_threshold = DEGENERATE_THRESHOLD[metric_key]

    # 逐 (arm, class, seed) 的指标表
    table = {}
    for arm, entries in record.items():
        for entry in entries:
            table[(arm, entry["class"], entry["seed"])] = entry["metrics"]

    report = []
    report.append(f"数据集: {args.dataset}  比例 train_sample_ratio={args.ratio}  "
                  f"种子={seeds}  类别数={len(classes)}")
    report.append(f"报告主指标: {metric_name}（{metric_key}）")
    report.append("")

    # 覆盖度
    for arm in arms:
        have = sum(1 for c in classes for s in seeds if (arm, c, s) in table)
        report.append(f"[覆盖] {arm:<20s} {have}/{len(classes) * len(seeds)}")

    summary = {"dataset": args.dataset, "ratio": args.ratio, "seeds": seeds,
               "classes": classes, "primary_metric": metric_key,
               "coverage": {}, "per_arm": {}, "paired_vs_full": {}}

    for arm in arms:
        vals = [table[(arm, c, s)][metric_key] for c in classes for s in seeds
                if (arm, c, s) in table and table[(arm, c, s)][metric_key] is not None]
        summary["coverage"][arm] = len(vals)
        if not vals:
            continue
        mean_v = sum(vals) / len(vals)
        std_v = (sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        summary["per_arm"][arm] = {"n": len(vals), "mean": mean_v, "std": std_v}

    report.append("")
    report.append(f"{'臂':<20s} {'n':>4s} {metric_name + '(mean±sd)':>22s}")
    for arm in arms:
        stats = summary["per_arm"].get(arm)
        if not stats:
            continue
        report.append(f"{arm:<20s} {stats['n']:>4d} "
                      f"{stats['mean']:.4f} ± {stats['std']:.4f}")

    report.append("")
    report.append(f"配对分析（同一 class×seed 下，与 full 臂逐对相减；负 = 该消融使 {metric_name} 下降）")
    report.append(f"  [{metric_name}]")
    for arm in arms:
        if arm == "full":
            continue
        deltas, improved = [], 0
        for c in classes:
            for s in seeds:
                full = table.get(("full", c, s))
                other = table.get((arm, c, s))
                if not full or not other:
                    continue
                if full.get(metric_key) is None or other.get(metric_key) is None:
                    continue
                d = other[metric_key] - full[metric_key]
                deltas.append(d)
                if d > 0:
                    improved += 1
        if not deltas:
            report.append(f"    {arm:<20s} 无可用配对")
            continue
        lo, hi = bootstrap_ci(deltas, seed=args.seeds[0])
        mean_d = sum(deltas) / len(deltas)
        summary["paired_vs_full"].setdefault(metric_key, {})[arm] = {
            "pairs": len(deltas), "mean_delta": mean_d,
            "ci95": [lo, hi], "improved_pairs": improved,
        }
        report.append(
            f"    {arm:<20s} pairs={len(deltas):>3d}  "
            f"Δ{metric_name}={mean_d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
            f"上升对={improved}/{len(deltas)}")

    per_class_all = {}
    report.append("")
    report.append(f"逐类 Δ{metric_name}（消融 − full，按 class 平均种子）")
    report.append(f"{'class':<22s}"
                  + "".join(f"{arm[:16]:>18s}" for arm in arms if arm != "full"))
    per_class = {}
    for c in classes:
        row = [f"{c:<22s}"]
        per_class[c] = {}
        for arm in arms:
            if arm == "full":
                continue
            ds = [table[(arm, c, s)][metric_key] - table[("full", c, s)][metric_key]
                  for s in seeds
                  if (arm, c, s) in table and ("full", c, s) in table
                  and table[(arm, c, s)][metric_key] is not None
                  and table[("full", c, s)][metric_key] is not None]
            if ds:
                value = sum(ds) / len(ds)
                per_class[c][arm] = value
                row.append(f"{value:>+18.4f}")
            else:
                row.append(f"{'n/a':>18s}")
        report.append("".join(row))
    per_class_all[metric_key] = per_class
    summary["per_class_delta"] = per_class_all

    # 饱和/病态类识别：饱和类（full 主指标≈1）没有下降空间，消融对它们恒为 0；
    # 病态类（full 主指标接近随机）的"差异"是噪声却会主导聚合，故两者都要单列，
    # 并给出"剔除病态类与饱和类"后的稳健配对聚合。
    full_mean = {}
    for c in classes:
        vals = [table[("full", c, s)][metric_key] for s in seeds
                if ("full", c, s) in table and table[("full", c, s)][metric_key] is not None]
        if vals:
            full_mean[c] = sum(vals) / len(vals)
    saturated = sorted(c for c, v in full_mean.items() if v >= sat_threshold)
    degenerate = sorted(c for c, v in full_mean.items() if v < deg_threshold)
    # 稳健聚合必须同时剔除两类：病态类（差值即噪声）与饱和类（恒为 0 的差值
    # 会把均值稀释到无意义量级）。只剔除病态类会让结果被天花板效应主导。
    excluded = set(degenerate) | set(saturated)
    robust = [c for c in classes if c not in excluded]
    report.append("")
    report.append(f"饱和类(full {metric_name}≥{sat_threshold:g}, 无下降空间) n={len(saturated)}: {saturated}")
    report.append(f"病态类(full {metric_name}<{deg_threshold:g}, 指标近随机) n={len(degenerate)}: {degenerate}")
    report.append("")
    report.append(f"剔除病态类与饱和类后的稳健配对 Δ{metric_name}（{len(robust)} 类 × {len(seeds)} seed）")
    summary["saturated_classes"] = saturated
    summary["degenerate_classes"] = degenerate
    summary["paired_vs_full_robust"] = {}
    for arm in arms:
        if arm == "full":
            continue
        ds = [table[(arm, c, s)][metric_key] - table[("full", c, s)][metric_key]
              for c in robust for s in seeds
              if (arm, c, s) in table and ("full", c, s) in table
              and table[(arm, c, s)][metric_key] is not None
              and table[("full", c, s)][metric_key] is not None]
        if not ds:
            report.append(f"  {arm:<20s} 无可用配对")
            continue
        lo, hi = bootstrap_ci(ds, seed=args.seeds[0])
        mean_d = sum(ds) / len(ds)
        summary["paired_vs_full_robust"][arm] = {
            "pairs": len(ds), "mean_delta": mean_d, "ci95": [lo, hi],
        }
        report.append(f"  {arm:<20s} pairs={len(ds):>3d}  Δ{metric_name}={mean_d:+.5f} "
                      f"[{lo:+.5f}, {hi:+.5f}]")

    summary_dir = os.path.join(ROOT, args.log_root)
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(summary_dir, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    with open(os.path.join(summary_dir, "summary.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "class", "seed"] + list(METRIC_KEYS))
        for (arm, c, s), metrics in sorted(table.items()):
            writer.writerow([arm, c, s] + [metrics.get(k) for k in METRIC_KEYS])
    print("\n".join(report))


def load_record_from_csv(log_root: str):
    """从已有 summary.csv 重建 record 结构（--resummarize 用）。

    CSV 指标列是字符串：空串/缺失记 None，其余转 float。
    """
    csv_path = os.path.join(ROOT, log_root, "summary.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"找不到 summary.csv: {csv_path}")
    record = {}
    with open(csv_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics = {}
            for key in METRIC_KEYS:
                raw = (row.get(key) or "").strip()
                metrics[key] = float(raw) if raw else None
            record.setdefault(row["arm"], []).append({
                "class": row["class"],
                "seed": int(row["seed"]),
                "metrics": metrics,
            })
    return record


def main() -> int:
    args = parse_args()
    if args.resummarize:
        # 不跑任务：从既有 summary.csv 重建报告（如更换主指标、重算统计口径）。
        # 注意 ratio/dataset 不在 CSV 里，需在命令行传对（如 --ratio 1.0）。
        record = load_record_from_csv(args.log_root)
        if not record:
            raise SystemExit("summary.csv 为空，无法重建报告。")
        summarize(record, args)
        return 0
    if args.wait_for_process and not args.dry_run:
        wait_for_process(args.wait_for_process, args.wait_poll)
    classes = list(iter_classes(args.config_root, args.dataset))
    if args.only:
        wanted = set(args.only)
        classes = [(stem, path) for stem, path in classes if stem in wanted]
    if not classes:
        raise SystemExit("没有匹配的类别配置。")

    jobs = []
    for arm, ablation_flag, scratch in ARMS:
        if arm not in args.arms:
            continue
        for stem, base_path in classes:
            for seed in args.seeds:
                jobs.append((arm, ablation_flag, scratch, args.dataset, stem,
                             base_path, seed))

    total = len(jobs)
    print(f"[ablation] dataset={args.dataset} ratio={args.ratio} arms={args.arms} "
          f"classes={len(classes)} seeds={args.seeds} -> {total} 个任务, "
          f"workers={args.workers}")
    if args.dry_run:
        for job in jobs:
            print(f"  {job[0]:<20s} {job[4]:<22s} seed={job[6]}")
        return 0

    record = {}
    results = []
    started = time.time()
    done = 0
    if args.workers <= 1:
        for job in jobs:
            results.append(run_one(*job, args))
            done += 1
            _log_result(results[-1], done, total)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, *job, args): job for job in jobs}
            for future in as_completed(futures):
                res = future.result()
                done += 1
                results.append(res)
                _log_result(res, done, total)

    for label, status, elapsed, metrics in results:
        if status not in ("ok", "cached") or metrics is None:
            continue
        arm, dataset, stem, seed_part = label.split("/")
        record.setdefault(arm, []).append({
            "class": stem,
            "seed": int(seed_part.replace("seed_", "")),
            "metrics": metrics,
        })

    failures = [r for r in results if r[1].startswith("FAIL")]
    print(f"[ablation] 完成：成功 {len(results) - len(failures)} / 失败 {len(failures)}，"
          f"耗时 {int((time.time() - started) / 60)} 分钟")
    for label, status, _, _ in failures:
        print(f"  FAIL {label}: {status}")

    if not args.no_summary and record:
        summarize(record, args)
    return 0


def _log_result(result, done, total):
    label, status, elapsed, _ = result
    print(f"[{done:>3d}/{total}] {label} {status} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
