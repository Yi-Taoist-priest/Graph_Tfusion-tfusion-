#!/usr/bin/env python
"""汇总 Graph_Tfusion/log 下各消融集合的结果（主指标 Macro-F1）。

- 自动发现 ``log/ablations*`` 中已有 ``summary.csv`` 且含 full 臂的集合；
- 剔除单样本类（默认 ``IDS2017/Infiltration`` 与
  ``IoT-23/C_C-HeartBeat-FileDownload``；后者在主仓库消融中不存在，仅在清单里说明）；
- 每个集合输出：各臂主指标均值、配对 Δ（vs full，同 class×seed）、
  以及剔除"饱和类 + 病态类"后的稳健配对 Δ；
- 额外给出跨集合池化的总体 Δ（每对 class×seed 等权各计一次）。

产物：``log/ablations_summary.txt`` 与 ``log/ablations_summary.csv``。

用法：
    python scripts/tools/summarize_ablations.py
    python scripts/tools/summarize_ablations.py --primary-metric auc
    python scripts/tools/summarize_ablations.py --exclude IDS2017/Infiltration
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_ablation_matrix import (  # noqa: E402
    DEGENERATE_THRESHOLD,
    METRIC_NAMES,
    SATURATED_THRESHOLD,
    bootstrap_ci,
    load_record_from_csv,
)

# 单样本类（support_attack<=1）默认剔除；按 "数据集/类别" 指定，兼容两套命名。
DEFAULT_EXCLUDE = (
    "IDS2017/Infiltration",
    "IDS-17/Infiltration",
    "IoT-23/C_C-HeartBeat-FileDownload",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", default="log",
                        help="消融集合所在目录（默认 Graph_Tfusion/log）")
    parser.add_argument("--primary-metric", default="macro_f1",
                        choices=["macro_f1", "auc", "attack_f1", "f1"],
                        help="汇总主指标（默认 macro_f1=Macro-F1）")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                        help="要剔除的 数据集/类别（默认两个单样本类）")
    parser.add_argument("--out", default="log/ablations_summary.txt",
                        help="汇总报告输出路径（相对 Graph_Tfusion）")
    return parser.parse_args()


def discover_sets(log_dir: str) -> list[str]:
    """返回 log 下含 summary.csv 的 ablations* 集合名（排序）。"""
    names = []
    for name in sorted(os.listdir(log_dir)):
        if not name.startswith("ablations"):
            continue
        if os.path.isfile(os.path.join(log_dir, name, "summary.csv")):
            names.append(name)
    return names


def table_of(record: dict) -> dict:
    """键为 (arm, class, seed)，值为该格的指标 dict。"""
    return {(arm, e["class"], e["seed"]): e["metrics"]
            for arm, entries in record.items() for e in entries}


def analyze(record: dict, dataset: str, metric_key: str, exclude: set) -> dict:
    classes = sorted({e["class"] for es in record.values() for e in es})
    seeds = sorted({e["seed"] for es in record.values() for e in es})
    table = table_of(record)
    arms = sorted(record)

    excluded = {c for c in classes if f"{dataset}/{c}" in exclude}

    # 单样本类自动识别（供报告透明化，不自动剔除，仅按 --exclude 生效）
    single = []
    for c in classes:
        m = next((table[("full", c, s)] for s in seeds
                  if ("full", c, s) in table), None)
        if m and ((m.get("support_attack") is not None and m["support_attack"] <= 1)
                  or (m.get("support_benign") is not None and m["support_benign"] <= 1)):
            single.append(c)

    kept = [c for c in classes if c not in excluded]

    sat_thr = SATURATED_THRESHOLD[metric_key]
    deg_thr = DEGENERATE_THRESHOLD[metric_key]
    full_mean = {}
    for c in kept:
        vals = [table[("full", c, s)][metric_key] for s in seeds
                if ("full", c, s) in table
                and table[("full", c, s)][metric_key] is not None]
        if vals:
            full_mean[c] = sum(vals) / len(vals)
    saturated = sorted(c for c, v in full_mean.items() if v >= sat_thr)
    degenerate = sorted(c for c, v in full_mean.items() if v < deg_thr)
    robust = [c for c in kept if c not in set(saturated) | set(degenerate)]

    result = {"dataset": dataset, "classes_all": classes, "excluded": sorted(excluded),
              "kept": kept, "single_sample": single, "saturated": saturated,
              "degenerate": degenerate, "robust": robust, "seeds": seeds,
              "arms": {}, "paired": {}, "paired_robust": {}}

    for arm in arms:
        vals = [table[(arm, c, s)][metric_key] for c in kept for s in seeds
                if (arm, c, s) in table and table[(arm, c, s)][metric_key] is not None]
        if vals:
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            result["arms"][arm] = {"n": len(vals), "mean": mean, "sd": sd}

    for arm in arms:
        if arm == "full":
            continue
        deltas, improved = [], 0
        for c in kept:
            for s in seeds:
                f = table.get(("full", c, s))
                o = table.get((arm, c, s))
                if not f or not o:
                    continue
                if f.get(metric_key) is None or o.get(metric_key) is None:
                    continue
                d = o[metric_key] - f[metric_key]
                deltas.append(d)
                improved += d > 0
        if deltas:
            lo, hi = bootstrap_ci(deltas, seed=seeds[0] if seeds else 777)
            result["paired"][arm] = {
                "pairs": len(deltas), "mean": sum(deltas) / len(deltas),
                "lo": lo, "hi": hi, "improved": improved, "deltas": deltas}
        # 稳健：仅保留非饱和非病态类
        rdeltas = []
        for c in robust:
            for s in seeds:
                f = table.get(("full", c, s))
                o = table.get((arm, c, s))
                if not f or not o:
                    continue
                if f.get(metric_key) is None or o.get(metric_key) is None:
                    continue
                rdeltas.append(o[metric_key] - f[metric_key])
        if rdeltas:
            lo, hi = bootstrap_ci(rdeltas, seed=seeds[0] if seeds else 777)
            result["paired_robust"][arm] = {
                "pairs": len(rdeltas), "mean": sum(rdeltas) / len(rdeltas),
                "lo": lo, "hi": hi}
    return result


def main() -> int:
    args = parse_args()
    log_dir = os.path.join(ROOT, args.log_dir)
    metric_key = args.primary_metric
    metric_name = METRIC_NAMES[metric_key]
    exclude = set(args.exclude)

    sets = discover_sets(log_dir)
    reports = []
    for name in sets:
        summary_path = os.path.join(log_dir, name, "summary.json")
        with open(summary_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        dataset = meta.get("dataset", name)
        record = load_record_from_csv(os.path.join(args.log_dir, name))
        if "full" not in record:
            reports.append({"name": name, "dataset": dataset, "skip": "无 full 臂"})
            continue
        res = analyze(record, dataset, metric_key, exclude)
        res["name"] = name
        res["ratio"] = meta.get("ratio")
        reports.append(res)

    lines = []
    lines.append("消融实验汇总（Graph_Tfusion/log）")
    lines.append(f"报告主指标: {metric_name}（{metric_key}）")
    lines.append(f"剔除的单样本类: {', '.join(sorted(exclude))}")
    lines.append("说明: 单样本类指测试集攻击样本数=1（macro-F1 恒≈0.5，无统计意义）。")
    hit = sorted({f"{r['dataset']}/{c}" for r in reports
                  if not r.get("skip") for c in r["excluded"]})
    lines.append(f"本仓库实际命中剔除: {hit if hit else '无'}")
    lines.append("")

    lines.append("=" * 72)
    lines.append("一、集合清单")
    lines.append(f"{'集合':<32s} {'数据集':<10s} {'比例':>5s} {'纳入类':>6s} {'剔除':>5s}")
    for r in reports:
        if r.get("skip"):
            lines.append(f"{r['name']:<32s} {r['dataset']:<10s} {'-':>5s} {'-':>6s} 跳过({r['skip']})")
            continue
        lines.append(f"{r['name']:<32s} {r['dataset']:<10s} {r.get('ratio', '-'):>5} "
                     f"{len(r['kept']):>6d} {len(r['excluded']):>5d}")
    lines.append("")
    lines.append("二、逐集合明细（配对 Δ 与稳健 Δ，均为 消融 − full）")

    pooled = {}
    csv_rows = []
    for r in reports:
        if r.get("skip"):
            continue
        lines.append("-" * 72)
        lines.append(f"{r['name']}（{r['dataset']} @ ratio={r.get('ratio')}，"
                     f"类数 {len(r['kept'])}，seed={r['seeds']}）")
        if r["excluded"]:
            lines.append(f"  已剔除单样本类: {r['excluded']}")
        other_single = [c for c in r["single_sample"] if c not in r["excluded"]]
        if other_single:
            lines.append(f"  其它单样本类(未剔除): {other_single}")
        lines.append(f"  {'臂':<20s} {'n':>4s} {metric_name + '(mean±sd)':>22s}")
        for arm, st in r["arms"].items():
            lines.append(f"  {arm:<20s} {st['n']:>4d} {st['mean']:.4f} ± {st['sd']:.4f}")
        lines.append(f"  配对 Δ{metric_name}（消融 − full；正 = 消融反而更好）")
        for arm in sorted(r["paired"]):
            p = r["paired"][arm]
            lines.append(f"    {arm:<20s} pairs={p['pairs']:>3d}  "
                         f"Δ={p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]  "
                         f"上升对={p['improved']}/{p['pairs']}")
            pooled.setdefault(arm, []).extend(p["deltas"])
            csv_rows.append([r["name"], r["dataset"], r.get("ratio"), arm,
                             len(r["kept"]), len(r["seeds"]), round(r["arms"][arm]["mean"], 6),
                             p["pairs"], round(p["mean"], 6), p["improved"],
                             round(p["lo"], 6), round(p["hi"], 6)])
        lines.append(f"  稳健配对 Δ{metric_name}（剔除饱和类 {len(r['saturated'])} 个"
                     f"与病态类 {len(r['degenerate'])} 个，剩 {len(r['robust'])} 类）")
        if r["degenerate"]:
            lines.append(f"    病态类: {r['degenerate']}")
        if r["saturated"]:
            lines.append(f"    饱和类: {r['saturated']}")
        for arm in sorted(r["paired_robust"]):
            p = r["paired_robust"][arm]
            lines.append(f"    {arm:<20s} pairs={p['pairs']:>3d}  "
                         f"Δ={p['mean']:+.5f} [{p['lo']:+.5f}, {p['hi']:+.5f}]")
        lines.append("")

    lines.append("=" * 72)
    lines.append("三、跨集合池化总体配对 Δ（所有集合的 class×seed 对等权合并）")
    for arm in sorted(pooled):
        deltas = pooled[arm]
        lo, hi = bootstrap_ci(deltas, seed=777)
        improved = sum(d > 0 for d in deltas)
        lines.append(f"  {arm:<20s} pairs={len(deltas):>4d}  "
                     f"Δ{metric_name}={sum(deltas) / len(deltas):+.4f} "
                     f"[{lo:+.4f}, {hi:+.4f}]  上升对={improved}/{len(deltas)}")
        csv_rows.append(["__POOLED__", "ALL", "-", arm, "-", "-", "-",
                         len(deltas), round(sum(deltas) / len(deltas), 6), improved,
                         round(lo, 6), round(hi, 6)])

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    csv_path = os.path.splitext(out_path)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["set", "dataset", "ratio", "arm", "n_classes", "n_seeds",
                         f"mean_{metric_key}", "pairs", f"delta_{metric_key}",
                         "improved_pairs", "ci_lo", "ci_hi"])
        writer.writerows(csv_rows)

    print("\n".join(lines))
    print(f"\n[写出] {os.path.relpath(out_path, ROOT)}")
    print(f"[写出] {os.path.relpath(csv_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
