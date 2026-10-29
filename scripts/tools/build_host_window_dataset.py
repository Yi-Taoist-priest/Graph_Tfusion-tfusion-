#!/usr/bin/env python
"""构建「主机 × 时间窗」标签数据集（hostwin）。

与逐流标签的区别：
- 单元是 ``(capture, src_host, window_id)``；
- 单元标签 = 该窗口内是否出现恶意流（来自 Zeek conn.log.labeled）；
- 窗口内所有流继承单元标签（包括原本 benign 的流）；
- 正窗口按上限抽样，负窗口全保留；
- 每个 capture 内部按窗口时间前向 6:2:2 切分，边界留 embargo 窗口。

输出仍使用 ``{"benign": [...], "attack": [...]}`` 结构，因此
``src/data_pipeline/loader.py`` 不需要改动；额外字段仅供审计与窗口级评估。

用法：
    python scripts/tools/build_host_window_dataset.py
    python scripts/tools/build_host_window_dataset.py --cap 5 --window 60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict

import ujson

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(ROOT)

DEFAULT_SOURCES = (
    # tag, extracted feature dir, Zeek conn.log.labeled
    ("c1", "data/hostwin_iot23/raw/c1",
     "../datasets/IoT-23/PCAPs/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled"),
    ("c3", "data/hostwin_iot23/raw/c3",
     "../datasets/IoT-23/PCAPs/CTU-IoT-Malware-Capture-3-1/bro/conn.log.labeled"),
    ("c34", "data/hostwin_iot23/raw/c34",
     "../datasets/IoT-23/PCAPs/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled"),
)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(ROOT, path))


def read_cell_labels(conn_log: str, window: int):
    """读取 Zeek conn.log.labeled，返回 {(host, window_id): 0/1}。"""
    cells = {}
    with open(_resolve(conn_log), "r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 21:
                continue
            tail = parts[-1].split()
            if len(tail) < 2:
                continue
            try:
                window_id = int(float(parts[0]) // window)
            except ValueError:
                continue
            host = parts[2]
            key = (host, window_id)
            cells[key] = max(cells.get(key, 0), 1 if tail[-2] == "Malicious" else 0)
    return cells


def read_flows(raw_dir: str):
    flows = []
    for name in sorted(os.listdir(_resolve(raw_dir))):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(_resolve(raw_dir), name), "r") as handle:
            payload = ujson.load(handle)
        flows.extend(payload.get("benign") or [])
        flows.extend(payload.get("attack") or [])
    return flows


def attach_cells(flows, cell_labels, capture: str, window: int):
    """把流按 (host, window) 分组，丢弃没有 conn.log 覆盖的流。"""
    grouped = defaultdict(list)
    unlabeled = 0
    for flow in flows:
        try:
            host = flow["five_tuple"][1]
            start_ts = float(flow["start_ts"])
        except (KeyError, IndexError, TypeError, ValueError):
            unlabeled += 1
            continue
        key = (host, int(start_ts // window))
        if key not in cell_labels:
            unlabeled += 1
            continue
        grouped[key].append(flow)
    return grouped, unlabeled


def cap_positive_cells(grouped, cell_labels, cap: int, seed: int, capture: str):
    kept = {}
    dropped = 0
    for key, items in grouped.items():
        if cell_labels[key] == 1 and cap > 0 and len(items) > cap:
            rng = random.Random(f"{seed}:{capture}:{key[0]}:{key[1]}")
            items = rng.sample(items, cap)
            dropped += 1
        kept[key] = items
    return kept, dropped


def _split_cells_sorted(cells, split, embargo):
    total = len(cells)
    train_end = int(total * split[0])
    val_end = int(total * (split[0] + split[1]))
    train = set(cells[: max(0, train_end - embargo)])
    val = set(cells[train_end: max(train_end, val_end - embargo)])
    test = set(cells[val_end:])
    return train, val, test


def split_windows(cell_keys, cell_labels, split, embargo):
    """以 (host, window) 单元做时间前向切分，正负单元各自分层。

    分层避免负样本被时段挤到单一 split；单元整体进入一个 split，
    因此不会出现同一 host 的同一窗口被切开。
    """
    positives = sorted(
        (key for key in cell_keys if cell_labels[key] == 1),
        key=lambda key: (key[1], key[0]),
    )
    negatives = sorted(
        (key for key in cell_keys if cell_labels[key] == 0),
        key=lambda key: (key[1], key[0]),
    )
    pos_train, pos_val, pos_test = _split_cells_sorted(positives, split, embargo)
    neg_train, neg_val, neg_test = _split_cells_sorted(negatives, split, embargo)
    windows = {
        "train": pos_train | neg_train,
        "val": pos_val | neg_val,
        "test": pos_test | neg_test,
    }
    return windows, positives + negatives


def enrich(flow, capture, key, label, index):
    record = dict(flow)
    record["capture"] = capture
    record["host_id"] = key[0]
    record["window_id"] = int(key[1])
    record["window_label"] = int(label)
    record["flow_uid"] = f"{capture}:{key[0]}:{key[1]}:{index}"
    return record


def build(args):
    os.makedirs(_resolve(args.output_dir), exist_ok=True)
    split_flows = {"train": {"benign": [], "attack": []},
                   "val": {"benign": [], "attack": []},
                   "test": {"benign": [], "attack": []}}
    audit = {"window": args.window, "cap": args.cap, "embargo": args.embargo,
             "sources": {}, "splits": {}}
    cell_rows = []

    for tag, raw_dir, conn_log in args.sources:
        cell_labels = read_cell_labels(conn_log, args.window)
        flows = read_flows(raw_dir)
        grouped, unlabeled = attach_cells(flows, cell_labels, tag, args.window)
        kept, capped_cells = cap_positive_cells(
            grouped, cell_labels, args.cap, args.seed, tag
        )
        windows, all_windows = split_windows(
            kept.keys(), cell_labels, args.split, args.embargo
        )

        source_stats = {
            "flows_extracted": len(flows),
            "flows_unlabeled": unlabeled,
            "cells": len(kept),
            "positive_cells": sum(1 for key in kept if cell_labels[key] == 1),
            "negative_cells": sum(1 for key in kept if cell_labels[key] == 0),
            "capped_positive_cells": capped_cells,
            "positive_flows_kept": sum(len(kept[k]) for k in kept if cell_labels[k] == 1),
            "negative_flows_kept": sum(len(kept[k]) for k in kept if cell_labels[k] == 0),
            "windows_total": len(all_windows),
        }
        audit["sources"][tag] = source_stats
        print(f"[{tag}] cells={source_stats['cells']} pos_cells={source_stats['positive_cells']} "
              f"flow_kept={source_stats['positive_flows_kept'] + source_stats['negative_flows_kept']} "
              f"unlabeled={unlabeled}", flush=True)

        for key, items in kept.items():
            label = int(cell_labels[key])
            if key in windows["train"]:
                name = "train"
            elif key in windows["val"]:
                name = "val"
            elif key in windows["test"]:
                name = "test"
            else:
                continue
            bucket = split_flows[name]["attack" if label else "benign"]
            for index, flow in enumerate(items):
                bucket.append(enrich(flow, tag, key, label, index))
            cell_rows.append({
                "capture": tag, "host_id": key[0], "window_id": key[1],
                "window_label": label, "split": name, "flows": len(items),
            })

    for name, payload in split_flows.items():
        attack = len(payload["attack"])
        benign = len(payload["benign"])
        total = attack + benign
        path = os.path.join(_resolve(args.output_dir), f"{name}.json")
        payload["meta"] = {
            "dataset": "hostwin_iot23",
            "split": name,
            "window_seconds": args.window,
            "cap_positive_flows": args.cap,
            "attack_flows": attack,
            "benign_flows": benign,
        }
        with open(path, "w") as handle:
            ujson.dump(payload, handle)
        audit["splits"][name] = {
            "attack_flows": attack, "benign_flows": benign,
            "attack_rate": (attack / total) if total else 0.0,
            "windows": len({row["window_id"] for row in cell_rows if row["split"] == name}),
            "hosts": len({row["host_id"] for row in cell_rows if row["split"] == name}),
        }
        print(f"[{name}] attack={attack} benign={benign} rate={audit['splits'][name]['attack_rate']:.3f}")

    summary_path = os.path.join(_resolve(args.output_dir), "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    cells_path = os.path.join(_resolve(args.output_dir), "cells.csv")
    with open(cells_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cell_rows)
    print(f"written {summary_path}")
    print(f"written {cells_path}")
    return audit


def parse_source(value: str):
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("source must be tag:raw_dir:conn_log")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="构建 host×window 标签数据集")
    parser.add_argument("--source", action="append", type=parse_source, default=None,
                        help="tag:raw_dir:conn_log，可重复；默认 c1/c3/c34")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--cap", type=int, default=5, help="每个正窗口保留的最大流数")
    parser.add_argument("--embargo", type=int, default=1, help="切分边界丢弃的窗口数")
    parser.add_argument("--split", nargs=3, type=float, default=[0.6, 0.2, 0.2])
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output-dir", default="data/hostwin_iot23/hostwin")
    args = parser.parse_args()
    args.sources = args.source or list(DEFAULT_SOURCES)
    build(args)


if __name__ == "__main__":
    main()
