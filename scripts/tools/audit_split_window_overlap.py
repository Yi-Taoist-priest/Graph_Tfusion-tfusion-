#!/usr/bin/env python3
"""预切分数据划分的上下文窗口重叠审计（只读）。

目的
----
预切分模式下 loader 仍把 ``train + val + test`` 拼成"全量总体"来构建因果
上下文（见 ``data_pipeline/loader.load_graph_tfusion_data``），因此训练侧样本
仍可能检索到测试侧邻居（转导通道）。``data_construct_param.context_isolation
="train_only"`` 是关闭该通道的运行时开关，但候选邻居只在同一个
``graph_window`` 窗口内取（见 ``data_pipeline/context.build_causal_context_table``），
所以当 ``train/val`` 与 ``test`` 没有任何窗口重叠时，``train_only`` 是**恒等
操作**（对 pre-train/train/val 行的上下文逐位相同），跑对照纯属浪费。

本脚本按类报告 ``train×test`` 与 ``(train∪val)×test`` 的窗口重叠数，用于在
批量跑 ``train_only`` 对照前判断哪些类真的会变。

口径
----
- ``window_id = floor(start_ts / graph_window)``，``graph_window`` 取配置
  ``data_construct_param.graph_window``（缺省 10.0），与 loader 完全一致。
- 窗口重叠是 ``train_only`` 生效的**必要条件**（同窗口才可能成为邻居）。
- 恒等判定额外要求 ``pre_train_ratio == 0``：>0 时 pre-train 前缀在 ``none``
  口径下不在上下文总体里、在 ``train_only`` 口径下在内，即使窗口不重叠也会有
  小幅差异。

只读：只打开数据 JSON 读取流记录，不修改、不移动、不删除任何数据。

用法
----
    python scripts/tools/audit_split_window_overlap.py
    python scripts/tools/audit_split_window_overlap.py \
        --config-glob 'config/train-test/DDOS_19/*.json' \
        --out audit/split_window_overlap_DDOS_19.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import math
import os
from typing import Dict, List, Sequence, Set

try:
    import ujson as json  # 与生产加载器 data_pipeline/loader.py 保持一致
except ImportError:  # pragma: no cover - 环境兜底
    import json


AUDIT_SCHEMA_VERSION = 1
DEFAULT_GRAPH_WINDOW = 10.0
SPLIT_FILES = ("train", "test", "val")


def window_id_set(flows: Sequence[dict], graph_window: float) -> Set[int]:
    """流集合占据的窗口 id 集合；缺 start_ts 或非有限值的流被忽略。"""
    window = max(float(graph_window), 1e-9)
    ids: Set[int] = set()
    for flow in flows:
        if not isinstance(flow, dict):
            continue
        ts = flow.get("start_ts")
        if ts is None:
            continue
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ts):
            continue
        ids.add(int(math.floor(ts / window)))
    return ids


def classify(
    train_flows: Sequence[dict],
    val_flows: Sequence[dict],
    test_flows: Sequence[dict],
    graph_window: float,
    pre_train_ratio: float,
) -> Dict:
    """报告窗口重叠与 train_only 是否为恒等操作。"""
    train_windows = window_id_set(train_flows, graph_window)
    val_windows = window_id_set(val_flows, graph_window)
    test_windows = window_id_set(test_flows, graph_window)
    overlap_train_test = len(train_windows & test_windows)
    overlap_trainval_test = len((train_windows | val_windows) & test_windows)
    is_noop = overlap_trainval_test == 0 and float(pre_train_ratio) == 0.0
    return {
        "windows_train": len(train_windows),
        "windows_test": len(test_windows),
        "overlap_train_test": overlap_train_test,
        "overlap_trainval_test": overlap_trainval_test,
        "pre_train_ratio": float(pre_train_ratio),
        "train_only_is_noop": is_noop,
    }


def _read_split(path: str) -> List[dict]:
    with open(path, "r") as source:
        payload = json.load(source)
    return list(payload.get("benign") or []) + list(payload.get("attack") or [])


def _resolve(repo_root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    rel = path[2:] if path.startswith("./") else path
    return os.path.join(repo_root, rel)


def audit_config(repo_root: str, config_path: str) -> Dict:
    """审计单个配置；未启用预切分或缺文件时返回 status 说明。"""
    with open(config_path, "r") as source:
        config = json.load(source)
    record: Dict = {
        "config": os.path.relpath(config_path, repo_root),
        "status": None,
        "graph_window": DEFAULT_GRAPH_WINDOW,
        "missing_files": [],
    }
    section = config.get("data_split") or {}
    if not section.get("use_pre_split"):
        record["status"] = "not_pre_split"
        return record

    params = config.get("data_construct_param") or {}
    graph_window = float(params.get("graph_window", DEFAULT_GRAPH_WINDOW))
    pre_train_ratio = float(params.get("pre_train_ratio", 0.0) or 0.0)
    record["graph_window"] = graph_window

    flows = {}
    for name in SPLIT_FILES:
        entry = section.get(name) or {}
        if not entry.get("enable"):
            flows[name] = []
            continue
        path = _resolve(repo_root, entry.get("path") or "")
        if not path or not os.path.isfile(path):
            record["missing_files"].append(entry.get("path"))
            flows[name] = []
            continue
        flows[name] = _read_split(path)

    record["status"] = "ok" if not record["missing_files"] else "missing_files"
    record.update(classify(
        flows["train"], flows["val"], flows["test"], graph_window, pre_train_ratio,
    ))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--repo-root", default=default_root)
    parser.add_argument("--config-glob", action="append", default=None,
                        help="可重复；相对 repo-root 的配置 glob，"
                             "默认 config/train-test/**/*.json")
    parser.add_argument("--out", default="audit/split_window_overlap.json")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    patterns = args.config_glob or ["config/train-test/**/*.json"]
    config_paths: List[str] = []
    for pattern in patterns:
        config_paths.extend(glob.glob(os.path.join(repo_root, pattern), recursive=True))
    config_paths = sorted(set(config_paths))
    if not config_paths:
        raise SystemExit(f"未找到任何配置：{patterns}")

    records: List[Dict] = []
    for index, path in enumerate(config_paths, start=1):
        print(f"[{index}/{len(config_paths)}] auditing {os.path.relpath(path, repo_root)} ...",
              flush=True)
        records.append(audit_config(repo_root, path))

    checked = [r for r in records if r["status"] == "ok"]
    noop = [r for r in checked if r.get("train_only_is_noop")]
    overlap = [r for r in checked if not r.get("train_only_is_noop")]

    document = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "repo_root": repo_root,
        "frozen_criteria": {
            "window_id_rule": "floor(start_ts / graph_window)",
            "train_only_is_noop": "(train ∪ val) 与 test 无窗口重叠 且 pre_train_ratio == 0",
            "note": "窗口重叠是 train_only 生效的必要条件",
        },
        "summary": {
            "configs_total": len(records),
            "checked": len(checked),
            "train_only_noop": len(noop),
            "train_only_active": len(overlap),
            "not_pre_split": len([r for r in records if r["status"] == "not_pre_split"]),
            "missing_files": len([r for r in records if r["status"] == "missing_files"]),
        },
        "classes": records,
    }
    out_path = os.path.join(repo_root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    temporary = out_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as target:
        json.dump(document, target, indent=2, ensure_ascii=False, sort_keys=True)
        target.write("\n")
    os.replace(temporary, out_path)

    ordered = sorted(
        checked, key=lambda r: (-r["overlap_trainval_test"], r["config"]))
    print()
    print("=" * 100)
    print(f"{'class':<52}{'win_tr':>7}{'win_te':>7}{'ov_tr':>7}{'ov_trval':>9}"
          f"{'pre':>6}{'verdict':>10}")
    print("-" * 100)
    for record in ordered:
        name = record["config"]
        if len(name) > 51:
            name = "..." + name[-48:]
        verdict = "noop" if record["train_only_is_noop"] else "active"
        print(f"{name:<52}{record['windows_train']:>7}{record['windows_test']:>7}"
              f"{record['overlap_train_test']:>7}{record['overlap_trainval_test']:>9}"
              f"{record['pre_train_ratio']:>6.2f}{verdict:>10}")
    print("-" * 100)
    print(f"检查 {len(checked)} 个预切分配置：train_only 恒等 {len(noop)} 个，"
          f"有效 {len(overlap)} 个")
    for record in records:
        if record["status"] == "missing_files":
            print(f"  缺文件跳过 {record['config']}: {record['missing_files']}")
    print("full report:", os.path.relpath(out_path, repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
