#!/usr/bin/env python
"""Step 0：数据集「上下文信息量」screening（不训练深度模型）。

回答一个问题：在跑图模块消融之前，这个数据集的上下文里到底有没有
可用的判别信息。四项指标：

- S1 邻居可得性：10s 窗口内至少 1 个有效邻居的流占比 / 平均邻居数；
- S2 邻居标签同质性：邻居与当前流同标签占比 vs 随机基线 p^2+(1-p)^2；
- S3 单流可分性：RF(当前流特征) 的 test AUC / Macro-F1（天花板）；
- S4 上下文增量信息：RF(当前流 + 邻居聚合) 相对 RF(当前流) 的 ΔMacro-F1 / ΔAUC。

S4 是判定门槛：若 RF 都无法从邻居聚合里拿到增量信息，深度图模块更拿不到。

实现复用主线的 ``build_causal_context_table``，只读预切分 JSON，不训练
深度模型、不写 save/ 下任何产物。

用法：
    python scripts/tools/dataset_screening.py --datasets DDOS_19 --max-classes 1
    python scripts/tools/dataset_screening.py --datasets DDOS_19 IoT-23 UNSW_NB15
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
import ujson

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_pipeline.context import build_causal_context_table  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

CONFIG_ROOT = os.path.join(ROOT, "config", "train-test")
DEFAULT_OUTPUT = os.path.join(ROOT, "log", "dataset_screening_20261025")


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(ROOT, path))


def _read_split(path: str):
    with open(path, "r") as handle:
        payload = ujson.load(handle)
    return list(payload.get("benign") or []), list(payload.get("attack") or [])


def _subsample(items, limit: int, seed: int):
    if limit <= 0 or len(items) <= limit:
        return list(items)
    order = np.random.RandomState(seed).permutation(len(items))[:limit]
    return [items[index] for index in sorted(order.tolist())]


def _flow_vector(flow: dict) -> list:
    nonseq = flow.get("nonseq_feature") or [0.0] * 6
    length = flow.get("len")
    length = 0.0 if length is None else float(length)
    return [float(value) for value in nonseq] + [math.log1p(max(0.0, length))]


def _build_features(flows, table):
    current = np.asarray([_flow_vector(flow) for flow in flows], dtype=np.float64)
    neighbor_idx = table.neighbor_indices
    valid = neighbor_idx >= 0
    row_count = len(flows)
    safe_idx = np.clip(neighbor_idx, 0, max(row_count - 1, 0))
    neighbor_values = np.where(valid[..., None], current[safe_idx], 0.0)
    counts = valid.sum(axis=1)
    denom = np.maximum(counts, 1).astype(np.float64)[:, None]
    neighbor_mean = neighbor_values.sum(axis=1) / denom
    delta = np.where(valid, table.neighbor_delta_t, 0.0).sum(axis=1) / denom[:, 0]
    coverage = counts > 0
    context = np.concatenate(
        [current, neighbor_mean, counts[:, None].astype(np.float64), delta[:, None]],
        axis=1,
    )
    stats = {
        "rows": row_count,
        "coverage": float(coverage.mean()) if row_count else 0.0,
        "mean_neighbors": float(counts.mean()) if row_count else 0.0,
    }
    return current, context, neighbor_idx, valid, stats


def _neighbor_label_stats(labels: np.ndarray, neighbor_idx: np.ndarray, valid: np.ndarray):
    if not valid.any():
        return {"same_rate": float("nan"), "chance": float("nan"),
                "same_attack": float("nan"), "same_benign": float("nan")}
    safe_idx = np.clip(neighbor_idx, 0, len(labels) - 1)
    neighbor_labels = labels[safe_idx]
    same = (neighbor_labels == labels[:, None]) & valid
    same_rate = float(same.sum() / valid.sum())
    attack_rate = float(labels.mean())
    chance = attack_rate ** 2 + (1.0 - attack_rate) ** 2
    attack_rows = labels == 1
    benign_rows = labels == 0
    same_attack = (
        float(same[attack_rows].sum() / valid[attack_rows].sum())
        if attack_rows.any() and valid[attack_rows].any() else float("nan")
    )
    same_benign = (
        float(same[benign_rows].sum() / valid[benign_rows].sum())
        if benign_rows.any() and valid[benign_rows].any() else float("nan")
    )
    return {"same_rate": same_rate, "chance": chance,
            "same_attack": same_attack, "same_benign": same_benign}


def _fit_probe(x_train, y_train, x_test, y_test):
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auc": float("nan"), "macro_f1": float("nan"), "attack_f1": float("nan")}
    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=8,
        random_state=0,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "attack_f1": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
    }


def screen_class(config_path: str, max_rows: int, graph_window: float, neighbors: int,
                 isolate_context: bool = False):
    with open(config_path, "r") as handle:
        config = ujson.load(handle)
    split = config.get("data_split") or {}
    split_sets = {}
    for name in ("train", "test", "val"):
        entry = split.get(name) or {}
        if not entry.get("enable"):
            continue
        benign, attack = _read_split(_resolve(entry["path"]))
        split_sets[name] = (
            _subsample(benign, max_rows, 11),
            _subsample(attack, max_rows, 13),
        )

    train_benign, train_attack = split_sets["train"]
    test_benign, test_attack = split_sets["test"]
    val_benign, val_attack = split_sets.get("val", ([], []))

    train_flows = train_benign + train_attack
    val_flows = val_benign + val_attack
    test_flows = test_benign + test_attack
    train_labels = [0] * len(train_benign) + [1] * len(train_attack)
    test_labels = [0] * len(test_benign) + [1] * len(test_attack)

    context_flows = train_flows + val_flows + test_flows
    if isolate_context:
        segments = [segment for segment in (train_flows, val_flows, test_flows) if segment]
    else:
        segments = [context_flows]

    currents, contexts, neighbor_blocks, valid_blocks = [], [], [], []
    offset = 0
    for segment in segments:
        table = build_causal_context_table(
            segment, graph_window=graph_window, neighbor_count=neighbors
        )
        current_block, context_block, neighbor_block, valid_block, _ = _build_features(
            segment, table
        )
        neighbor_blocks.append(np.where(valid_block, neighbor_block + offset, -1))
        currents.append(current_block)
        contexts.append(context_block)
        valid_blocks.append(valid_block)
        offset += len(segment)

    current = np.concatenate(currents, axis=0)
    context = np.concatenate(contexts, axis=0)
    neighbor_idx = np.concatenate(neighbor_blocks, axis=0)
    valid = np.concatenate(valid_blocks, axis=0)
    stats = {
        "rows": int(len(valid)),
        "coverage": float((valid.sum(axis=1) > 0).mean()) if len(valid) else 0.0,
        "mean_neighbors": float(valid.sum(axis=1).mean()) if len(valid) else 0.0,
    }
    labels = np.asarray(
        train_labels + [0] * len(val_benign) + [1] * len(val_attack) + test_labels,
        dtype=np.int64,
    )

    train_end = len(train_flows)
    val_end = train_end + len(val_flows)
    x_train_cur, x_train_ctx = current[:train_end], context[:train_end]
    y_train = labels[:train_end]
    x_test_cur, x_test_ctx = current[val_end:], context[val_end:]
    y_test = labels[val_end:]

    current_metrics = _fit_probe(x_train_cur, y_train, x_test_cur, y_test)
    context_metrics = _fit_probe(x_train_ctx, y_train, x_test_ctx, y_test)
    context_only = _fit_probe(
        x_train_ctx[:, current.shape[1]:], y_train,
        x_test_ctx[:, current.shape[1]:], y_test,
    )
    label_stats = _neighbor_label_stats(labels, neighbor_idx, valid)

    row = {
        "class": os.path.splitext(os.path.basename(config_path))[0],
        "n_train": int(len(train_flows)),
        "n_test": int(len(test_flows)),
        "attack_train": int(sum(train_labels)),
        "attack_test": int(sum(test_labels)),
        "rows": stats["rows"],
        "coverage": stats["coverage"],
        "mean_neighbors": stats["mean_neighbors"],
        **label_stats,
        "cur_auc": current_metrics["auc"],
        "cur_macro_f1": current_metrics["macro_f1"],
        "ctx_auc": context_metrics["auc"],
        "ctx_macro_f1": context_metrics["macro_f1"],
        "ctx_only_auc": context_only["auc"],
        "ctx_only_macro_f1": context_only["macro_f1"],
        "d_auc": context_metrics["auc"] - current_metrics["auc"],
        "d_macro_f1": context_metrics["macro_f1"] - current_metrics["macro_f1"],
    }
    return row


def _mean(values):
    finite = [value for value in values if value is not None and not math.isnan(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _write_csv(path: str, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="数据集上下文信息量 screening")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--config-root", default=CONFIG_ROOT, help="配置根目录")
    parser.add_argument("--max-classes", type=int, default=0)
    parser.add_argument("--classes", nargs="*", default=None, help="只跑指定类别名")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--graph-window", type=float, default=10.0)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument(
        "--isolate-context", action="store_true",
        help="train/val/test 各自建上下文，禁止跨划分邻居（检验爆发式泄漏）",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    datasets = args.datasets or sorted(
        name for name in os.listdir(args.config_root)
        if os.path.isdir(os.path.join(args.config_root, name))
    )
    os.makedirs(args.output, exist_ok=True)

    all_rows = []
    summary_rows = []
    for dataset in datasets:
        dataset_dir = os.path.join(args.config_root, dataset)
        configs = sorted(
            os.path.join(dataset_dir, name)
            for name in os.listdir(dataset_dir)
            if name.endswith(".json")
        )
        if args.max_classes > 0:
            configs = configs[: args.max_classes]
        if args.classes:
            wanted = set(args.classes)
            configs = [
                path for path in configs
                if os.path.splitext(os.path.basename(path))[0] in wanted
            ]
        rows = []
        for config_path in configs:
            row = screen_class(
                config_path, args.max_rows, args.graph_window, args.neighbors,
                isolate_context=args.isolate_context,
            )
            row["dataset"] = dataset
            rows.append(row)
            print(
                f"[{dataset}/{row['class']}] cover={row['coverage']:.2f} "
                f"same={row['same_rate']:.3f} (chance {row['chance']:.3f}) "
                f"cur_auc={row['cur_auc']:.4f} ctx_auc={row['ctx_auc']:.4f} "
                f"d_macro_f1={row['d_macro_f1']:+.4f}",
                flush=True,
            )
        all_rows.extend(rows)
        summary_rows.append({
            "dataset": dataset,
            "n_classes": len(rows),
            "attack_test_mean": _mean([row["attack_test"] for row in rows]),
            "coverage": _mean([row["coverage"] for row in rows]),
            "mean_neighbors": _mean([row["mean_neighbors"] for row in rows]),
            "same_rate": _mean([row["same_rate"] for row in rows]),
            "chance": _mean([row["chance"] for row in rows]),
            "same_attack": _mean([row["same_attack"] for row in rows]),
            "cur_auc": _mean([row["cur_auc"] for row in rows]),
            "cur_macro_f1": _mean([row["cur_macro_f1"] for row in rows]),
            "ctx_auc": _mean([row["ctx_auc"] for row in rows]),
            "ctx_macro_f1": _mean([row["ctx_macro_f1"] for row in rows]),
            "ctx_only_auc": _mean([row["ctx_only_auc"] for row in rows]),
            "ctx_only_macro_f1": _mean([row["ctx_only_macro_f1"] for row in rows]),
            "d_auc": _mean([row["d_auc"] for row in rows]),
            "d_macro_f1": _mean([row["d_macro_f1"] for row in rows]),
        })
        print(
            f"[{dataset}] classes={len(rows)} cover={summary_rows[-1]['coverage']:.2f} "
            f"cur_auc={summary_rows[-1]['cur_auc']:.4f} "
            f"d_macro_f1={summary_rows[-1]['d_macro_f1']:+.4f}",
            flush=True,
        )

    _write_csv(os.path.join(args.output, "per_class.csv"), all_rows)
    _write_csv(os.path.join(args.output, "dataset_summary.csv"), summary_rows)
    print(f"written: {os.path.join(args.output, 'per_class.csv')}")
    print(f"written: {os.path.join(args.output, 'dataset_summary.csv')}")


if __name__ == "__main__":
    main()
