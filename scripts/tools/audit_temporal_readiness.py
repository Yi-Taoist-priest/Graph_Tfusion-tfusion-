#!/usr/bin/env python3
"""严格全局时间切分协议 —— 数据就绪度只读审计。

本脚本是只读的：只打开数据 JSON 读取流记录，不修改、不移动、不删除任何数据。
输出口径冻结（AUDIT_SCHEMA_VERSION=1），所有判定阈值都在下方常量区显式定义，
禁止在看过结果后通过调整阈值改变 verdict；调整阈值必须升 schema 版本。

审计内容（对应专家评审第 7、8 节）：
1. 时间戳质量：缺失/不可解析/非有限值数量、时间单位推断、同时间戳组、存储序逆序；
2. 严格全局时间切分模拟：合并 benign/attack 后按 start_ts 全局排序，
   相同 start_ts 作为不可拆分组，按预声明的组分位点 0.60/0.80 切 fit/validation/test；
3. 协议适用性判定：supervised_ready / anomaly_only_ready / unsupported / hard_fail。

用法：
    python scripts/tools/audit_temporal_readiness.py \
        --data-root data/train-test \
        --config-glob 'config/train-*/*.json' \
        --out audit/temporal_readiness.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import ujson as json  # 与生产加载器 data_pipeline/loader.py 保持一致
except ImportError:  # pragma: no cover - 环境兜底
    import json


AUDIT_SCHEMA_VERSION = 1

# ── 冻结口径（修改即升 schema 版本） ──────────────────────────────────────────
FIT_GROUP_QUANTILE = 0.60       # fit 段占不同时间戳组的比例（预声明，非事后搜索）
VAL_GROUP_QUANTILE = 0.80       # validation 截止分位点；其后为 test
MIN_SUPERVISED_PER_CLASS = 10   # 与 MIN_ATTACK_TRAIN_SIZE 对齐；fit/val 每类下限
MIN_TEST_PER_CLASS = 1          # test 段两类都必须出现
SUPPORT_CURVE_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# 与 data_pipeline.context.TimestampDiagnostics.degenerate 同口径：
# distinct_timestamps <= max(1, flow_count // 10) 即视为退化
DEGENERATE_DIVISOR = 10

BENIGN = "benign"
ATTACK = "attack"
CLASSES = (BENIGN, ATTACK)


# ── 基础工具 ─────────────────────────────────────────────────────────────────

def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def infer_time_unit(median_abs_ts: float) -> str:
    """仅做量级标注（白名单值），不参与 hard_fail 判定。

    epoch 秒   ≈ 1e9..1e12 （约 2001-2028）
    epoch 毫秒 ≈ 1e12..1e15
    相对秒     < 1e5 且非负
    其余标 unknown，需人工确认，不猜单位。
    """
    value = abs(median_abs_ts)
    if 1e9 <= value < 1e12:
        return "epoch_seconds"
    if 1e12 <= value < 1e15:
        return "epoch_milliseconds"
    if 0.0 <= value < 1e5:
        return "relative_seconds"
    return "unknown"


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# ── 单文件审计 ───────────────────────────────────────────────────────────────

def audit_dataset_file(path: str, compute_sha256: bool) -> Dict:
    payload: Dict = {
        "path": path,
        "exists": os.path.isfile(path),
        "size_bytes": None,
        "sha256": None,
        "load_error": None,
        "counts": {BENIGN: 0, ATTACK: 0, "total": 0},
        "timestamp_quality": {},
        "simulated_strict_temporal": None,
        "verdict": None,
        "hard_fail_reasons": [],
    }
    if not payload["exists"]:
        payload["load_error"] = "file_not_found"
        payload["verdict"] = "hard_fail"
        payload["hard_fail_reasons"].append("data file does not exist")
        return payload

    payload["size_bytes"] = os.path.getsize(path)
    if compute_sha256:
        payload["sha256"] = sha256_file(path)

    try:
        with open(path, "r") as source:
            data = json.load(source)
    except Exception as exc:  # 解析失败本身就是硬门槛失败
        payload["load_error"] = f"{type(exc).__name__}: {exc}"
        payload["verdict"] = "hard_fail"
        payload["hard_fail_reasons"].append("dataset JSON cannot be parsed")
        return payload

    # (start_ts, class_label)；非法时间戳不进入排序，单独计数
    valid_records: List[Tuple[float, str]] = []
    quality = {
        cls: {
            "flows": 0,
            "missing_start_ts": 0,
            "unparseable_start_ts": 0,
            "non_finite_start_ts": 0,
            "min_ts": None,
            "max_ts": None,
            "storage_adjacent_inversions": 0,
        }
        for cls in CLASSES
    }

    for cls in CLASSES:
        flows = data.get(cls) or []
        if not isinstance(flows, list):
            payload["hard_fail_reasons"].append(f"{cls} block is not a JSON list")
            flows = []
        stats = quality[cls]
        stats["flows"] = len(flows)
        previous: Optional[float] = None
        cls_values: List[float] = []
        for flow in flows:
            if not isinstance(flow, dict) or "start_ts" not in flow or flow["start_ts"] is None:
                stats["missing_start_ts"] += 1
                continue
            try:
                ts = float(flow["start_ts"])
            except (TypeError, ValueError):
                stats["unparseable_start_ts"] += 1
                continue
            if not math.isfinite(ts):
                stats["non_finite_start_ts"] += 1
                continue
            cls_values.append(ts)
            valid_records.append((ts, cls))
            if previous is not None and ts < previous:
                stats["storage_adjacent_inversions"] += 1
            previous = ts
        if cls_values:
            stats["min_ts"] = min(cls_values)
            stats["max_ts"] = max(cls_values)

    total = sum(quality[c]["flows"] for c in CLASSES)
    payload["counts"] = {
        BENIGN: quality[BENIGN]["flows"],
        ATTACK: quality[ATTACK]["flows"],
        "total": total,
    }

    valid_total = len(valid_records)
    invalid_total = sum(
        quality[c]["missing_start_ts"] + quality[c]["unparseable_start_ts"]
        + quality[c]["non_finite_start_ts"]
        for c in CLASSES
    )
    if invalid_total:
        payload["hard_fail_reasons"].append(
            f"{invalid_total} flow(s) have missing/unparseable/non-finite start_ts"
        )
    if valid_total == 0:
        payload["hard_fail_reasons"].append("no flows with a valid start_ts")

    # 全局时间戳分布（不区分类）
    distinct_ts = sorted({ts for ts, _ in valid_records})
    largest_tie_group = 0
    if valid_records:
        ordered_ts = sorted(ts for ts, _ in valid_records)
        run = 1
        for index in range(1, len(ordered_ts)):
            if ordered_ts[index] == ordered_ts[index - 1]:
                run += 1
            else:
                largest_tie_group = max(largest_tie_group, run)
                run = 1
        largest_tie_group = max(largest_tie_group, run)

    all_values = [ts for ts, _ in valid_records]
    median_ts = _median([abs(v) for v in all_values])
    global_stats = {
        "valid_flows": valid_total,
        "invalid_flows": invalid_total,
        "min_ts": min(all_values) if all_values else None,
        "max_ts": max(all_values) if all_values else None,
        "span_seconds": (max(all_values) - min(all_values)) if all_values else None,
        "distinct_timestamps": len(distinct_ts),
        "largest_tie_group": largest_tie_group,
        "tie_group_ratio": (largest_tie_group / valid_total) if valid_total else 0.0,
        "inferred_time_unit": infer_time_unit(median_ts) if median_ts is not None else "unknown",
        "degenerate": (
            valid_total > 0
            and len(distinct_ts) <= max(1, valid_total // DEGENERATE_DIVISOR)
        ),
    }
    if global_stats["degenerate"]:
        payload["hard_fail_reasons"].append(
            "timestamp anchors degenerate: distinct_timestamps <= max(1, total//10)"
        )
    quality["global"] = global_stats
    payload["timestamp_quality"] = quality

    if valid_records and not global_stats["degenerate"] and invalid_total == 0:
        payload["simulated_strict_temporal"] = _simulate_strict_temporal(valid_records, distinct_ts)

    payload["verdict"] = _decide_verdict(payload)
    return payload


def _simulate_strict_temporal(
    records: Sequence[Tuple[float, str]], distinct_ts: Sequence[float]
) -> Dict:
    """按冻结的 0.60/0.80 组分位点模拟 fit/validation/test。

    相同 start_ts 永远落在同一分区（先在 distinct 时间戳上切，再把流映射到组）。
    """
    group_count = len(distinct_ts)

    def cut_index(quantile: float) -> int:
        return int(math.floor(quantile * group_count))

    fit_end = cut_index(FIT_GROUP_QUANTILE)
    val_end = cut_index(VAL_GROUP_QUANTILE)
    # 边界保护：三个分区都至少要有一个时间戳组
    if fit_end == 0:
        fit_end = 1
    if val_end <= fit_end:
        val_end = min(group_count, fit_end + 1)

    fit_border = distinct_ts[fit_end - 1]
    val_border = distinct_ts[val_end - 1]

    partitions = {
        "fit": {BENIGN: 0, ATTACK: 0, "min_ts": None, "max_ts": None},
        "validation": {BENIGN: 0, ATTACK: 0, "min_ts": None, "max_ts": None},
        "test": {BENIGN: 0, ATTACK: 0, "min_ts": None, "max_ts": None},
    }
    support_curve = []
    for ts, cls in records:
        if ts <= fit_border:
            part = "fit"
        elif ts <= val_border:
            part = "validation"
        else:
            part = "test"
        bucket = partitions[part]
        bucket[cls] += 1
        bucket["min_ts"] = ts if bucket["min_ts"] is None else min(bucket["min_ts"], ts)
        bucket["max_ts"] = ts if bucket["max_ts"] is None else max(bucket["max_ts"], ts)

    # 描述性（非搜索用）：固定十分位的累计类别支持，用于判断攻击在时间轴上的爆发位置
    for quantile in SUPPORT_CURVE_QUANTILES:
        end_index = cut_index(quantile)
        if end_index == 0:
            end_index = 1
        border = distinct_ts[end_index - 1]
        benign_before = sum(1 for ts, cls in records if ts <= border and cls == BENIGN)
        attack_before = sum(1 for ts, cls in records if ts <= border and cls == ATTACK)
        support_curve.append({
            "cumulative_quantile": quantile,
            "border_ts": border,
            "benign_at_or_before": benign_before,
            "attack_at_or_before": attack_before,
        })

    return {
        "rule": "global_timestamp_group_quantiles",
        "fit_group_quantile": FIT_GROUP_QUANTILE,
        "validation_group_quantile": VAL_GROUP_QUANTILE,
        "group_count": group_count,
        "fit_end_ts": fit_border,
        "validation_end_ts": val_border,
        "partitions": partitions,
        "fixed_decile_support_curve": support_curve,
    }


def _decide_verdict(payload: Dict) -> str:
    if payload["hard_fail_reasons"]:
        return "hard_fail"
    simulated = payload["simulated_strict_temporal"]
    if simulated is None:
        return "hard_fail"
    parts = simulated["partitions"]
    fit, val, test = parts["fit"], parts["validation"], parts["test"]
    supervised_ok = (
        fit[BENIGN] >= MIN_SUPERVISED_PER_CLASS
        and fit[ATTACK] >= MIN_SUPERVISED_PER_CLASS
        and val[BENIGN] >= MIN_SUPERVISED_PER_CLASS
        and val[ATTACK] >= MIN_SUPERVISED_PER_CLASS
        and test[BENIGN] >= MIN_TEST_PER_CLASS
        and test[ATTACK] >= MIN_TEST_PER_CLASS
    )
    if supervised_ok:
        return "supervised_ready"
    anomaly_ok = (
        fit[BENIGN] >= MIN_SUPERVISED_PER_CLASS
        and test[BENIGN] >= MIN_TEST_PER_CLASS
        and test[ATTACK] >= MIN_TEST_PER_CLASS
    )
    if anomaly_ok:
        return "anomaly_only_ready"
    return "unsupported"


# ── 配置交叉核对 ─────────────────────────────────────────────────────────────

def audit_configured_targets(repo_root: str, config_patterns: Sequence[str]) -> List[Dict]:
    """列出配置声明的数据文件及存在性（HyperVision 等未落地数据由此暴露）。"""
    targets: List[Dict] = []
    seen = set()
    for pattern in config_patterns:
        for config_path in sorted(glob.glob(os.path.join(repo_root, pattern))):
            try:
                with open(config_path, "r") as source:
                    config = json.load(source)
            except Exception:
                continue
            data_path = config.get("data_path")
            tags = config.get("data_tag") or {}
            if not data_path or not isinstance(tags, dict):
                continue
            for tag in tags:
                rel_dir = data_path[2:] if data_path.startswith("./") else data_path
                rel_file = os.path.join(rel_dir, f"{tag}.json")
                abs_file = os.path.join(repo_root, rel_file)
                key = os.path.abspath(abs_file)
                if key in seen:
                    continue
                seen.add(key)
                targets.append({
                    "config": os.path.relpath(config_path, repo_root),
                    "data_file": rel_file,
                    "exists": os.path.isfile(abs_file),
                })
    return targets


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--repo-root", default=default_root)
    parser.add_argument("--data-root", default="data/train-test")
    parser.add_argument(
        "--config-glob", action="append", default=None,
        help="可重复；相对 repo-root 的配置 glob，默认核对 config/train-*/*.json",
    )
    parser.add_argument("--out", default="audit/temporal_readiness.json")
    parser.add_argument("--no-sha256", action="store_true", help="跳过数据文件 SHA-256")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    data_root = os.path.join(repo_root, args.data_root)
    if not os.path.isdir(data_root):
        raise SystemExit(f"data root not found: {data_root}")

    # 数据目录形如 data/train-test/<数据集>_v2/<数据集>/<小类>.json，深度不固定，
    # 因此递归收集；转换器写的 conversion_manifest.json 不是流量数据集，跳过。
    data_files = sorted(
        os.path.join(current, name)
        for current, _, names in os.walk(data_root)
        for name in names
        if name.endswith(".json") and name != "conversion_manifest.json"
    )

    reports = []
    for index, path in enumerate(sorted(data_files), start=1):
        rel = os.path.relpath(path, repo_root)
        print(f"[{index}/{len(data_files)}] auditing {rel} ...", flush=True)
        report = audit_dataset_file(path, compute_sha256=not args.no_sha256)
        report["path"] = rel
        reports.append(report)

    config_patterns = args.config_glob or ["config/train-*/*.json"]
    configured = audit_configured_targets(repo_root, config_patterns)

    summary = {
        "total_files": len(reports),
        "by_verdict": {},
        "missing_configured_files": [
            {"config": item["config"], "data_file": item["data_file"]}
            for item in configured
            if not item["exists"]
        ],
    }
    for report in reports:
        summary["by_verdict"][report["verdict"]] = (
            summary["by_verdict"].get(report["verdict"], 0) + 1
        )

    document = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "repo_root": repo_root,
        "frozen_criteria": {
            "fit_group_quantile": FIT_GROUP_QUANTILE,
            "validation_group_quantile": VAL_GROUP_QUANTILE,
            "min_supervised_per_class_fit_and_validation": MIN_SUPERVISED_PER_CLASS,
            "min_test_per_class": MIN_TEST_PER_CLASS,
            "degenerate_rule": "distinct_timestamps <= max(1, valid_total // 10)",
            "tie_group_rule": "equal float start_ts are one indivisible group",
            "verdict_values": [
                "supervised_ready", "anomaly_only_ready", "unsupported", "hard_fail",
            ],
        },
        "summary": summary,
        "configured_targets": configured,
        "datasets": reports,
    }

    out_path = os.path.join(repo_root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    temporary = out_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as target:
        json.dump(document, target, indent=2, sort_keys=True)
        target.write("\n")
    os.replace(temporary, out_path)

    print()
    print("=" * 104)
    print(f"{'dataset':<45}{'total':>8}{'attack':>8}{'distinct_ts':>12}{'tie_max':>9}{'verdict':>22}")
    print("-" * 104)
    for report in reports:
        quality = report.get("timestamp_quality") or {}
        globq = quality.get("global") or {}
        name = report["path"]
        if len(name) > 44:
            name = "..." + name[-41:]
        print(
            f"{name:<45}"
            f"{report['counts']['total']:>8}"
            f"{report['counts'][ATTACK]:>8}"
            f"{globq.get('distinct_timestamps', '-'):>12}"
            f"{globq.get('largest_tie_group', '-'):>9}"
            f"{report['verdict']:>22}"
        )
    print("-" * 104)
    print("verdict summary:", summary["by_verdict"])
    if summary["missing_configured_files"]:
        print("configured but absent data files:")
        for item in summary["missing_configured_files"]:
            print(f"  - {item['config']} -> {item['data_file']}")
    print("full report:", os.path.relpath(out_path, repo_root))


if __name__ == "__main__":
    main()
