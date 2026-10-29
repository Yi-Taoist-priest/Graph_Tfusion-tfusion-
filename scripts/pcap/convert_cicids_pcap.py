#!/usr/bin/env python3
"""
CICIDS2017 PCAP + CSV labels -> tFusion JSON converter.

Features are extracted from real PCAP packets. CICIDS CSV files are used only
as five-tuple label sources.

mini_datasets 模式（--mini-root）：把 split_datasets.py 的切分产物
    {mini_root}/{Dataset}/{Class}/{Class}.pcap|.csv|manifest.json
转换为 tFusion JSON（每类一个 {output}/{Dataset}/{Class}.json，含
benign+attack）。支持 CICIDS2017 / CICDDoS2019 / CIRACICDoHBrw2020 /
CTU-13 / Kitsune / UNSW_NB15 / CSECICIDS2018 / IoT-23：
  - CICFlowMeter 系（CICIDS2017/CICDDoS2019）：Label 列 + 5 元组 + 时间；
  - CTU-13：Botnet/Normal 标签前缀 + 协议名映射；
  - DoH：DoH 布尔列（True=攻击），无协议列，TCP/UDP 双键；
  - Kitsune：包级标签 CSV，按 5 元组+空闲重建流，包多数表决定流标签；
  - UNSW_NB15：proto 字符串 + Stime epoch 秒 + Label(0/1) + attack_cat；
  - IoT-23：Zeek conn.log.labeled 的列（ts/5 元组/duration/label_tail），
    5 元组不可直接作索引（同一 5 元组在 120s 内被大量短连接复用），改按
    切分端同规则“行时间窗回放”把包归属到行；
  - CSECICIDS2018：官方 CSV 无源 IP/源端口，按切分端同规则用指纹
    (协议, 目的端口, 首包时间±1s, 时长µs, 前/后向包数与 payload 字节)
    配对流，偏移取切分 manifest 的指纹推断结果（12 小时制双候选）。
时间偏移优先取切分 manifest 中已推断的值（注意符号约定：切分端为
PCAP-CSV，本转换端为 CSV-PCAP，需取反）。
"""

import pandas as pd
import numpy as np
import os
import glob
import time
import sys
import re
import bisect
import statistics
import hashlib
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import ujson as json
except ImportError:
    import json

# ─── 常量 ───
SEGMENT_LEN = 100
LABEL_IDENTITY_COLUMNS = {
    'Flow ID', 'Protocol', 'Source IP', 'Destination IP', 'Source Port',
    'Destination Port', 'Timestamp', 'Label',
}

_WORKER_HOST_WINDOWS = None
_WORKER_BASE_TS = 0.0
_WORKER_SEGMENT_LEN = SEGMENT_LEN
_WORKER_MAX_PACKET_LEN = 1500
_WORKER_GAP_THRESHOLD = 1.0
_WORKER_HOST_WINDOW = 10.0

# ─── CICIDS2017 标签 → 任务类别映射 ───
# 每个类别对应一个输出 JSON 文件，含 benign + 该类攻击流
ATTACK_CATEGORY_MAP = {
    # DoS/DDoS 类
    'DDoS':               'dos',
    'DoS Hulk':           'dos',
    'DoS GoldenEye':      'dos',
    'DoS slowloris':      'dos',
    'DoS Slowhttptest':   'dos',
    # Bot 类
    'Bot':                'bot',
    # 暴力破解类
    'FTP-Patator':        'brute_force',
    'SSH-Patator':        'brute_force',
    # 渗透类
    'Infiltration':       'infiltration',
    # 端口扫描类
    'PortScan':           'portscan',
}


@dataclass(frozen=True)
class SourceLabelRecord:
    source_id: str
    key: str
    label: str
    category: str
    timestamp_epoch: float


@dataclass(frozen=True)
class LabelResolution:
    label: Optional[str]
    category: Optional[str]
    matched: bool
    reason: str
    time_delta_seconds: Optional[float]


def classify_attack(label_str):
    """把 CICIDS2017 标签映射到任务类别，非攻击返回 None"""
    s = str(label_str).strip()
    # 直接匹配
    if s in ATTACK_CATEGORY_MAP:
        return ATTACK_CATEGORY_MAP[s]
    # Web Attack 系列（编码可能含特殊字符，用 ascii fallback 匹配）
    s_ascii = s.encode('ascii', errors='replace').decode('ascii')
    if 'Web Attack' in s_ascii:
        return 'webattack'
    return None


def load_csvs(csv_dir):
    """读取目录下所有 CSV，清洗后合并"""
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    if not csv_files:
        print(f'错误: {csv_dir} 中没有找到 CSV 文件')
        sys.exit(1)

    print(f'找到 {len(csv_files)} 个 CSV 文件')
    dfs = []
    for f in csv_files:
        t0 = time.time()
        usecols = lambda column: str(column).strip() in LABEL_IDENTITY_COLUMNS
        try:
            df = pd.read_csv(f, encoding='utf-8', usecols=usecols)
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding='cp1252', usecols=usecols)
        df.columns = [c.strip() for c in df.columns]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        df["Source File"] = os.path.basename(f)
        dfs.append(df)
        print(f'  读取 {os.path.basename(f)[:50]}: {len(df)} 行 ({time.time() - t0:.1f}s)')

    df_all = pd.concat(dfs, ignore_index=True)
    print(f'总行数: {len(df_all)}, 列数: {len(df_all.columns)}')
    return df_all


def _safe_int(value, default=0):
    try:
        if value is None or value == '':
            return default
        if isinstance(value, str):
            value = value.strip()
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_str(value):
    return str(value).strip()


def source_id_from_name(value) -> str:
    """Return the CICIDS capture-day identity shared by PCAP and label CSV names."""
    match = re.search(r"(monday|tuesday|wednesday|thursday|friday)", Path(str(value)).name, re.I)
    if match is None:
        raise ValueError(f"Cannot infer CICIDS source day from {value!r}.")
    return match.group(1).capitalize()


def describe_input_file(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return {
        'path': str(path),
        'size_bytes': path.stat().st_size,
        'sha256': digest.hexdigest(),
    }


def parse_cic_timestamp(value) -> Optional[float]:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed) or parsed.year != 2017:
        return None
    # Pandas keeps the timestamp as an integer epoch internally.  Converting a
    # malformed/pre-1970 value through datetime.timestamp() raises OSError on
    # Windows, so retain only CICIDS2017 dates and avoid the platform API.
    return float(parsed.value / 1_000_000_000)


def _pcap_label_key(proto, src, dst, sport, dport):
    return "|".join([
        str(_safe_int(proto)),
        _safe_str(src),
        _safe_str(dst),
        str(_safe_int(sport)),
        str(_safe_int(dport)),
    ])


def _reverse_pcap_label_key(key):
    proto, src, dst, sport, dport = key.split('|')
    return "|".join([proto, dst, src, dport, sport])


def _pcap_label_key_from_flow_id(flow_id):
    parts = _safe_str(flow_id).split('-')
    if len(parts) != 5:
        return None
    src, dst, sport, dport, proto = parts
    return _pcap_label_key(proto, src, dst, sport, dport)


def _prefer_label(old_label, new_label):
    if old_label is None:
        return new_label
    old_is_benign = _safe_str(old_label).upper() == 'BENIGN'
    new_is_benign = _safe_str(new_label).upper() == 'BENIGN'
    if old_is_benign and not new_is_benign:
        return new_label
    return old_label


def build_pcap_label_maps_from_df(df):
    """
    Build five-tuple label maps from CICIDS/CICFlowMeter CSV rows.

    Returned keys match extract_pcap_features.FlowRecord:
      proto|src|dst|sport|dport
    """
    required = ['Protocol', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Label']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'CICIDS label CSV missing required columns: {missing}')

    label_map = {}
    category_map = {}

    def add_label_key(key, label):
        if key is None:
            return
        chosen = _prefer_label(label_map.get(key), label)
        label_map[key] = chosen
        category_map[key] = 'benign' if chosen.upper() == 'BENIGN' else classify_attack(chosen)

    flow_ids = df['Flow ID'].values if 'Flow ID' in df.columns else [None] * len(df)
    rows = zip(
        df['Protocol'].values,
        df['Source IP'].values,
        df['Destination IP'].values,
        df['Source Port'].values,
        df['Destination Port'].values,
        df['Label'].values,
        flow_ids,
    )
    for proto, src, dst, sport, dport, label_raw, flow_id in rows:
        col_key = _pcap_label_key(
            proto,
            src,
            dst,
            sport,
            dport,
        )
        label = _safe_str(label_raw or 'BENIGN')
        flow_id_key = _pcap_label_key_from_flow_id(flow_id)
        for key in [col_key, flow_id_key]:
            add_label_key(key, label)
            if key is not None:
                add_label_key(_reverse_pcap_label_key(key), label)
    return label_map, category_map


def build_source_label_index(
    df,
    source_id_resolver=source_id_from_name,
    category_resolver=None,
) -> Dict[str, Dict[str, List[SourceLabelRecord]]]:
    """Keep labels separate by capture source and timestamp.

    Defaults retain the CICIDS day/category conventions.  Other CIC datasets
    can supply explicit resolvers while reusing the same safe tuple-and-time
    matching contract.
    """
    required = [
        "Source File", "Timestamp", "Protocol", "Source IP", "Destination IP",
        "Source Port", "Destination Port", "Label",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"CICIDS label CSV missing required columns: {missing}")

    parsed_timestamps = pd.to_datetime(
        df["Timestamp"], dayfirst=True, format="mixed", errors="coerce"
    )
    valid_rows = df.loc[
        parsed_timestamps.notna() & (parsed_timestamps.dt.year == 2017),
        required,
    ].copy()
    valid_rows["Timestamp"] = (
        parsed_timestamps.loc[valid_rows.index].astype("int64") / 1_000_000_000
    )

    index: Dict[str, Dict[str, List[SourceLabelRecord]]] = {}
    for row in valid_rows.itertuples(index=False, name=None):
        source_file, timestamp_epoch, proto, src, dst, sport, dport, label_raw = row
        source_id = source_id_resolver(source_file)
        label = _safe_str(label_raw or "BENIGN")
        category = (
            category_resolver(label)
            if category_resolver is not None
            else ("benign" if label.upper() == "BENIGN" else classify_attack(label))
        )
        if category is None:
            continue
        primary_key = _pcap_label_key(proto, src, dst, sport, dport)
        for key in (primary_key, _reverse_pcap_label_key(primary_key)):
            record = SourceLabelRecord(
                source_id=source_id,
                key=key,
                label=label,
                category=category,
                timestamp_epoch=float(timestamp_epoch),
            )
            index.setdefault(source_id, {}).setdefault(key, []).append(record)

    for source_records in index.values():
        for records in source_records.values():
            records.sort(key=lambda record: record.timestamp_epoch)
    return index


def infer_source_time_offset(
    source_id: str,
    flows,
    label_index: Dict[str, Dict[str, List[SourceLabelRecord]]],
) -> Optional[float]:
    """Estimate the offset from source-local tuples that are unique in CSV."""
    offsets = []
    source_records = label_index.get(source_id, {})
    for flow in flows:
        key = _pcap_label_key(flow.proto, flow.src, flow.dst, flow.sport, flow.dport)
        candidates = source_records.get(key, [])
        if len(candidates) != 1:
            continue
        offsets.append(candidates[0].timestamp_epoch - float(flow.start_ts))
    return float(statistics.median(offsets)) if offsets else None


def resolve_source_flow_label(
    source_id: str,
    flow,
    label_index: Dict[str, Dict[str, List[SourceLabelRecord]]],
    offset_seconds: float,
    tolerance_seconds: float,
) -> LabelResolution:
    """Resolve one PCAP flow only when a source-local timestamp match is unique."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative.")
    key = _pcap_label_key(flow.proto, flow.src, flow.dst, flow.sport, flow.dport)
    candidates = label_index.get(source_id, {}).get(key, [])
    if not candidates:
        return LabelResolution(None, None, False, "no_candidate", None)

    expected_timestamp = float(flow.start_ts) + float(offset_seconds)
    ranked = sorted(
        ((abs(record.timestamp_epoch - expected_timestamp), record) for record in candidates),
        key=lambda item: item[0],
    )
    best_delta, best_record = ranked[0]
    if best_delta > tolerance_seconds:
        return LabelResolution(None, None, False, "outside_tolerance", best_delta)
    if len(ranked) > 1 and abs(ranked[1][0] - best_delta) <= 1e-9:
        return LabelResolution(None, None, False, "ambiguous_timestamp", best_delta)
    return LabelResolution(
        best_record.label,
        best_record.category,
        True,
        "matched",
        best_delta,
    )


def build_conversion_manifest(
    source_reports,
    category_counts,
    flow_idle_timeout_seconds: float,
    label_time_tolerance_seconds: float,
    inputs=None,
):
    return {
        "contract_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "flow_idle_timeout_seconds": float(flow_idle_timeout_seconds),
        "label_time_tolerance_seconds": float(label_time_tolerance_seconds),
        "inputs": list(inputs or []),
        "sources": list(source_reports),
        "category_counts": dict(sorted(category_counts.items())),
    }


def _pcap_flow_key(flow):
    return _pcap_label_key(flow.proto, flow.src, flow.dst, flow.sport, flow.dport)


def _pcap_flow_to_item(flow, host_windows, base_ts, segment_len, max_packet_len, gap_threshold, host_window):
    from extract_pcap_features import (
        ip_to_int_for_project,
        make_flow_feature,
        make_host_feature,
        make_seq_feature,
    )

    item = {
        'seq_feature': make_seq_feature(flow, segment_len, max_packet_len, gap_threshold),
        'nonseq_feature': make_flow_feature(flow),
        'interact_feature': make_host_feature(flow, host_windows, base_ts, host_window),
        'len': int(flow.packet_count),
        'id': [ip_to_int_for_project(flow.src), ip_to_int_for_project(flow.dst)],
        'five_tuple': [flow.proto, flow.src, flow.dst, flow.sport, flow.dport],
        'start_ts': flow.start_ts,
        'end_ts': flow.end_ts,
    }
    group_id = getattr(flow, 'group', None)
    if group_id is not None:
        item['group'] = group_id
    return item


def _init_pcap_item_worker(host_windows, base_ts, segment_len, max_packet_len, gap_threshold, host_window):
    global _WORKER_HOST_WINDOWS
    global _WORKER_BASE_TS
    global _WORKER_SEGMENT_LEN
    global _WORKER_MAX_PACKET_LEN
    global _WORKER_GAP_THRESHOLD
    global _WORKER_HOST_WINDOW

    _WORKER_HOST_WINDOWS = host_windows
    _WORKER_BASE_TS = base_ts
    _WORKER_SEGMENT_LEN = segment_len
    _WORKER_MAX_PACKET_LEN = max_packet_len
    _WORKER_GAP_THRESHOLD = gap_threshold
    _WORKER_HOST_WINDOW = host_window


def _pcap_item_worker(task):
    idx, label, category, flow = task
    item = _pcap_flow_to_item(
        flow,
        _WORKER_HOST_WINDOWS,
        _WORKER_BASE_TS,
        _WORKER_SEGMENT_LEN,
        _WORKER_MAX_PACKET_LEN,
        _WORKER_GAP_THRESHOLD,
        _WORKER_HOST_WINDOW,
    )
    return idx, label, category, item


def _freeze_host_windows(host_windows):
    frozen = {}
    for wid, host_map in host_windows.items():
        frozen_hosts = {}
        for host, stats in host_map.items():
            frozen_hosts[host] = {
                'send': list(stats.get('send', [0.0, 0.0, 0.0])),
                'recv': list(stats.get('recv', [0.0, 0.0, 0.0])),
            }
        frozen[int(wid)] = frozen_hosts
    return frozen


def _materialize_labeled_tasks(
    tasks,
    host_windows,
    base_ts,
    segment_len,
    max_packet_len,
    gap_threshold,
    host_window,
    workers,
    chunksize,
):
    benign_flows = []
    attack_by_cat = defaultdict(list)
    chunksize = max(1, int(chunksize))
    workers = max(1, int(workers))
    worker_host_windows = _freeze_host_windows(host_windows)

    if workers == 1 or len(tasks) <= 1:
        _init_pcap_item_worker(worker_host_windows, base_ts, segment_len, max_packet_len, gap_threshold, host_window)
        results = [_pcap_item_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_pcap_item_worker,
            initargs=(worker_host_windows, base_ts, segment_len, max_packet_len, gap_threshold, host_window),
        ) as executor:
            results = list(executor.map(_pcap_item_worker, tasks, chunksize=chunksize))

    for _, label, category, item in results:
        if str(label).strip().upper() == 'BENIGN' or category == 'benign':
            benign_flows.append(item)
        else:
            attack_by_cat[category].append(item)
    return benign_flows, attack_by_cat


def convert_labeled_pcap_flows(
    flows,
    label_map,
    category_map,
    host_windows,
    base_ts,
    segment_len=SEGMENT_LEN,
    max_packet_len=1500,
    gap_threshold=1.0,
    host_window=10.0,
    max_benign=50000,
    max_attack_per_cat=50000,
    unmatched_policy='skip',
    workers=1,
    chunksize=1000,
):
    benign_tasks = 0
    attack_tasks_by_cat = defaultdict(int)
    tasks = []
    unmatched = 0
    unknown_attack = 0

    for flow in flows:
        key = _pcap_flow_key(flow)
        if key not in label_map:
            unmatched += 1
            if unmatched_policy == 'skip':
                continue
            if unmatched_policy == 'error':
                raise ValueError(f'Unmatched PCAP flow has no CICIDS label: {key}')
            label = 'BENIGN'
            category = 'benign'
        else:
            label = label_map[key]
            category = category_map.get(key, 'benign')

        if category is None:
            unknown_attack += 1
            continue

        is_benign = str(label).strip().upper() == 'BENIGN' or category == 'benign'
        if is_benign:
            if max_benign >= 0 and benign_tasks >= max_benign:
                continue
            benign_tasks += 1
        else:
            if max_attack_per_cat >= 0 and attack_tasks_by_cat[category] >= max_attack_per_cat:
                continue
            attack_tasks_by_cat[category] += 1

        tasks.append((len(tasks), label, category, flow))

    benign_flows, attack_by_cat = _materialize_labeled_tasks(
        tasks,
        host_windows,
        base_ts,
        segment_len,
        max_packet_len,
        gap_threshold,
        host_window,
        workers,
        chunksize,
    )

    return benign_flows, attack_by_cat, unmatched, unknown_attack


def convert_labeled_source_flows(
    flows,
    source_id,
    label_index,
    source_time_offset_seconds,
    host_windows,
    base_ts,
    segment_len=SEGMENT_LEN,
    max_packet_len=1500,
    gap_threshold=1.0,
    host_window=10.0,
    label_time_tolerance_seconds=60.0,
    max_benign=50000,
    max_attack_per_cat=50000,
    workers=1,
    chunksize=1000,
):
    """Convert only source-local, uniquely resolved labels; unmatched flows are skipped."""
    benign_tasks = 0
    attack_tasks_by_cat = defaultdict(int)
    tasks = []
    unmatched = 0
    matched = 0
    matched_by_category = defaultdict(int)
    rejection_reasons = defaultdict(int)

    for flow in flows:
        resolution = resolve_source_flow_label(
            source_id,
            flow,
            label_index,
            offset_seconds=source_time_offset_seconds,
            tolerance_seconds=label_time_tolerance_seconds,
        )
        if not resolution.matched:
            unmatched += 1
            rejection_reasons[resolution.reason] += 1
            continue
        matched += 1
        label = resolution.label
        category = resolution.category
        matched_by_category[category] += 1
        if category == 'benign':
            if max_benign >= 0 and benign_tasks >= max_benign:
                continue
            benign_tasks += 1
        else:
            if max_attack_per_cat >= 0 and attack_tasks_by_cat[category] >= max_attack_per_cat:
                continue
            attack_tasks_by_cat[category] += 1
        tasks.append((len(tasks), label, category, flow))

    benign_flows, attack_by_cat = _materialize_labeled_tasks(
        tasks,
        host_windows,
        base_ts,
        segment_len,
        max_packet_len,
        gap_threshold,
        host_window,
        workers,
        chunksize,
    )
    return benign_flows, attack_by_cat, {
        'source_id': source_id,
        'pcap_flows': len(flows),
        'matched': matched,
        'matched_by_category': dict(sorted(matched_by_category.items())),
        'unmatched': unmatched,
        'rejections': dict(sorted(rejection_reasons.items())),
        'time_offset_seconds': float(source_time_offset_seconds),
    }


def convert_pcaps_with_cicids_labels(
    pcaps,
    csv_dir,
    output_dir,
    max_benign=50000,
    max_attack_per_cat=50000,
    segment_len=SEGMENT_LEN,
    max_packet_len=1500,
    gap_threshold=1.0,
    host_window=10.0,
    min_packets=1,
    compact=False,
    no_progress=False,
    unmatched_policy='skip',
    workers=1,
    chunksize=1000,
    flow_idle_timeout_seconds=120.0,
    label_time_tolerance_seconds=60.0,
    skip_malformed_dns=False,
):
    """Convert source-isolated PCAP flows with source-local CICIDS labels."""
    from extract_pcap_features import build_host_windows, collect_flows

    if unmatched_policy != 'skip':
        raise ValueError("Source-aware conversion only supports unmatched_policy='skip'.")
    print('=' * 50)
    print('1. 读取 CICIDS 标签 CSV')
    label_df = load_csvs(csv_dir)
    label_index = build_source_label_index(label_df)
    print(f'来源标签集合数: {sum(len(records) for records in label_index.values())}')

    print('=' * 50)
    print('2. 读取 PCAP 并构造真实包级 flow')
    pcap_paths = [Path(p) for p in pcaps]
    missing = [str(p) for p in pcap_paths if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing input pcap file(s): ' + ', '.join(missing))
    input_descriptions = [describe_input_file(path) for path in pcap_paths]

    show_progress = not no_progress
    source_reports = []
    all_benign = []
    all_attack_by_cat = defaultdict(list)

    print('=' * 50)
    print('3. 按来源独立解析、校准并转换 PCAP flows')
    print(f'特征转换 workers={max(1, int(workers))}, chunksize={max(1, int(chunksize))}')
    for pcap_path in pcap_paths:
        source_id = source_id_from_name(pcap_path.name)
        flows = collect_flows(
            [pcap_path],
            min_packets=min_packets,
            show_progress=show_progress,
            flow_idle_timeout_seconds=flow_idle_timeout_seconds,
            skip_malformed_dns=skip_malformed_dns,
        )
        if not flows:
            source_reports.append({
                'source_id': source_id,
                'pcap_path': str(pcap_path),
                'pcap_flows': 0,
                'matched': 0,
                'unmatched': 0,
                'rejections': {},
                'time_offset_seconds': None,
            })
            continue

        source_offset = infer_source_time_offset(source_id, flows, label_index)
        if source_offset is None:
            source_reports.append({
                'source_id': source_id,
                'pcap_path': str(pcap_path),
                'pcap_flows': len(flows),
                'matched': 0,
                'unmatched': len(flows),
                'rejections': {'no_unique_offset': len(flows)},
                'time_offset_seconds': None,
            })
            continue

        host_windows = build_host_windows(flows, host_window, show_progress=show_progress)
        remaining_benign = -1 if max_benign < 0 else max(0, max_benign - len(all_benign))
        remaining_attack = {
            category: -1 if max_attack_per_cat < 0 else max(0, max_attack_per_cat - len(all_attack_by_cat.get(category, [])))
            for category in set(ATTACK_CATEGORY_MAP.values())
        }
        benign_flows, attack_by_cat, source_report = convert_labeled_source_flows(
            flows,
            source_id=source_id,
            label_index=label_index,
            source_time_offset_seconds=source_offset,
            host_windows=host_windows,
            base_ts=flows[0].start_ts,
            segment_len=segment_len,
            max_packet_len=max_packet_len,
            gap_threshold=gap_threshold,
            host_window=host_window,
            label_time_tolerance_seconds=label_time_tolerance_seconds,
            max_benign=remaining_benign,
            max_attack_per_cat=max_attack_per_cat,
            workers=workers,
            chunksize=chunksize,
        )
        source_report['pcap_path'] = str(pcap_path)
        source_reports.append(source_report)
        all_benign.extend(benign_flows)
        for category, items in attack_by_cat.items():
            limit = remaining_attack.get(category, max_attack_per_cat)
            if limit >= 0:
                items = items[:limit]
            all_attack_by_cat[category].extend(items)

    os.makedirs(output_dir, exist_ok=True)
    all_attack_flows = []
    category_counts = {}
    for cat_name in sorted(all_attack_by_cat.keys()):
        attack_flows = all_attack_by_cat[cat_name]
        category_counts[cat_name] = len(attack_flows)
        all_attack_flows.extend(attack_flows)
        save_json_with_compact({'benign': all_benign, 'attack': attack_flows}, os.path.join(output_dir, f'{cat_name}.json'), compact=compact)

    full_data = {'benign': all_benign, 'attack': all_attack_flows}
    save_json_with_compact(full_data, os.path.join(output_dir, 'cicids2017.json'), compact=compact)
    manifest = build_conversion_manifest(
        source_reports,
        category_counts,
        flow_idle_timeout_seconds=flow_idle_timeout_seconds,
        label_time_tolerance_seconds=label_time_tolerance_seconds,
        inputs=input_descriptions,
    )
    with open(os.path.join(output_dir, 'conversion_manifest.json'), 'w', encoding='utf-8') as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
    print('PCAP + CICIDS 标签转换完成!')
    return manifest


# ═════════════════════════════════════════════════════════════════════
# mini_datasets 模式：split_datasets.py 切分产物 → tFusion JSON
# ═════════════════════════════════════════════════════════════════════
# 输入布局（每类一个目录）：
#   {mini_root}/{Dataset}/{Class}/{Class}.pcap    选中流的原始包（PCAP 时钟）
#   {mini_root}/{Dataset}/{Class}/{Class}.csv     选中流的原始行（CSV 时钟）
#   {mini_root}/{Dataset}/{Class}/manifest.json   切分审计（含已推断偏移）
# CSV 行与 PCAP 包由切分脚本保证严格对应；本转换按 5 元组 + 时间偏移
# 重新匹配，为重建的 PCAP 流打标签。

MINI_DEFAULT_ROOT = '/liuluping/shenzhiju/TFusion/mini_datasets'
MINI_SUPPORTED_DATASETS = (
    'CICIDS2017', 'CICDDoS2019', 'CIRACICDoHBrw2020', 'CTU-13', 'Kitsune',
    'UNSW_NB15', 'CSECICIDS2018', 'IoT-23',
)
_UNSW_PROTO_MAP = {'tcp': 6, 'udp': 17, 'icmp': 1, 'icmp6': 58, 'arp': 0}
_CTU_PROTO_MAP = {'tcp': 6, 'udp': 17, 'icmp': 1, 'icmp6': 58, 'arp': 0}
# Zeek（IoT-23）与 CSE-CIC-IDS2018 的自定义路径常量
_IOT23_PROTO_MAP = {'tcp': 6, 'udp': 17, 'icmp': 1, 'icmp6': 58, 'sctp': 132}
_IOT23_BASE_TOL = 1.0          # 切分端 ZeekConnParser.t_tol
_IOT23_SUBSECOND_TOL = 0.01    # 切分端 _fractional_tol（Zeek ts 带亚秒小数）
_CSE2018_CSV_COLUMNS = (
    'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
    'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Label',
)
_CSE2018_TS_FORMAT = '%d/%m/%Y %H:%M:%S'
_CSE2018_TIME_TOLERANCE = 1.0  # 切分端 CSECICIDS2018Splitter.TIME_TOL
_TS_ISO_RE = re.compile(
    r'^(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?$')
_TS_SLASH_RE = re.compile(
    r'^(\d{1,2})/(\d{1,2})/(\d{4})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?'
    r'(?:\.(\d+))?\s*([AaPp][Mm])?$')
_TS_YEARFIRST_RE = re.compile(
    r'^(\d{4})[/-](\d{1,2})[/-](\d{1,2})[ T](\d{1,2}):(\d{2}):(\d{2})'
    r'(?:\.(\d+))?$')


def parse_mini_ts(value) -> Optional[float]:
    """CSV 时间戳 → epoch 秒。与 split_datasets.parse_ts 同语义：
    ISO 快速路径 + 斜杠日期 MDY 优先（月>12 时自动按 DMY）+ AM/PM。
    切分端的偏移就是按这套语义推断的，转换端必须保持一致，
    否则 7/4/2017 之类会被解析成不同日期、整批错位。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    t = str(value).strip()
    m = _TS_ISO_RE.match(t) or _TS_YEARFIRST_RE.match(t)
    if m:
        y, mo, d, hh, mi, se, frac = m.groups()
        try:
            dt = datetime(int(y), int(mo), int(d), int(hh), int(mi), int(se))
        except ValueError:
            return None
        if frac:
            dt += timedelta(seconds=float('0.' + frac))
        return dt.timestamp()
    m = _TS_SLASH_RE.match(t)
    if m:
        a, b, y, hh, mi, se, frac, ap = m.groups()
        mo, d = int(a), int(b)
        if mo > 12 and d <= 12:          # 13/03/2019 之类只能按 DMY 解释
            mo, d = d, mo
        hh = int(hh)
        if ap:
            ap = ap.lower()
            if ap == 'pm' and hh < 12:
                hh += 12
            elif ap == 'am' and hh == 12:
                hh = 0
        try:
            dt = datetime(int(y), mo, d, hh, int(mi), int(se or 0))
        except ValueError:
            return None
        if frac:
            dt += timedelta(seconds=float('0.' + frac))
        return dt.timestamp()
    return None


def mini_adapter(dataset_name: str) -> str:
    if dataset_name in ('CICIDS2017', 'CICDDoS2019'):
        return 'cicflowmeter'
    if dataset_name == 'CTU-13':
        return 'ctu'
    if dataset_name == 'CIRACICDoHBrw2020':
        return 'doh'
    if dataset_name == 'Kitsune':
        return 'kitsune'
    if dataset_name == 'UNSW_NB15':
        return 'unsw'
    if dataset_name == 'IoT-23':
        return 'iot23'
    if dataset_name == 'CSECICIDS2018':
        return 'cse2018'
    raise ValueError(f'不支持的 mini 数据集: {dataset_name}')


def _mini_row_to_record(dataset_name, row, class_name):
    """一行 mini CSV → (keys, ts, label, category)；无法解析返回 None。
    攻击行 category=类名，良性行='benign'。"""
    adapter = mini_adapter(dataset_name)
    if adapter == 'cicflowmeter':
        ts = parse_mini_ts(row.get('Timestamp'))
        if ts is None:
            return None
        try:
            proto = int(float(str(row['Protocol'])))
            sp = int(float(str(row['Source Port'])))
            dp = int(float(str(row['Destination Port'])))
        except (KeyError, TypeError, ValueError):
            return None
        src = _safe_str(row.get('Source IP'))
        dst = _safe_str(row.get('Destination IP'))
        label = _safe_str(row.get('Label') or 'BENIGN')
        category = 'benign' if label.upper() == 'BENIGN' else class_name
        return [_pcap_label_key(proto, src, dst, sp, dp)], ts, label, category
    if adapter == 'ctu':
        ts = parse_mini_ts(row.get('StartTime'))
        if ts is None:
            return None
        proto = _CTU_PROTO_MAP.get(_safe_str(row.get('Proto')).lower())
        if proto is None:
            return None
        try:
            sp = int(float(str(row['Sport']).split('/')[0]))
            dp = int(float(str(row['Dport']).split('/')[0]))
        except (TypeError, ValueError):
            return None
        src = _safe_str(row.get('SrcAddr'))
        dst = _safe_str(row.get('DstAddr'))
        label = _safe_str(row.get('Label'))
        low = label.lower()
        if 'botnet' in low:
            category = class_name
        elif 'normal' in low:
            category = 'benign'
        else:
            return None       # 背景行：无法标注，跳过（与 unmatched=skip 同口径）
        return [_pcap_label_key(proto, src, dst, sp, dp)], ts, label, category
    if adapter == 'doh':
        ts = parse_mini_ts(row.get('TimeStamp'))
        if ts is None:
            return None
        try:
            sp = int(float(str(row['SourcePort'])))
            dp = int(float(str(row['DestinationPort'])))
        except (TypeError, ValueError):
            return None
        src = _safe_str(row.get('SourceIP'))
        dst = _safe_str(row.get('DestinationIP'))
        raw = _safe_str(row.get('DoH')).lower()
        if raw in ('true', '1'):
            label, category = class_name, class_name
        elif raw in ('false', '0'):
            label, category = 'BENIGN', 'benign'
        else:
            return None
        # 无协议列：TCP/UDP 双键兜底（与切分端 DoHSessionParser 一致）
        return [_pcap_label_key(6, src, dst, sp, dp),
                _pcap_label_key(17, src, dst, sp, dp)], ts, label, category
    if adapter == 'unsw':
        # UNSW-NB15：proto 为字符串名，Stime 已是 epoch 秒，
        # Label=0 良性 / 1 攻击，attack_cat 为类别名（可能含前导空格）。
        proto = _UNSW_PROTO_MAP.get(_safe_str(row.get('proto')).lower())
        if proto is None:
            return None
        try:
            ts = float(str(row['Stime']))
            sp = int(float(str(row['sport'])))
            dp = int(float(str(row['dsport'])))
        except (KeyError, TypeError, ValueError):
            return None
        src = _safe_str(row.get('srcip'))
        dst = _safe_str(row.get('dstip'))
        label = _safe_str(row.get('Label') or '0').strip()
        if label == '0':
            category = 'benign'
            label = 'BENIGN'
        else:
            category = class_name
            cat = _safe_str(row.get('attack_cat')).strip()
            label = cat or class_name
        return [_pcap_label_key(proto, src, dst, sp, dp)], ts, label, category
    return None


def build_mini_label_index(dataset_name, class_name, csv_path):
    """mini 类 CSV → {class_name: {五元组键: [SourceLabelRecord]}}。
    每类目录是单一来源（source_id=类名）。返回 (index, 行统计)。"""
    adapter = mini_adapter(dataset_name)
    if adapter in ('iot23', 'cse2018'):
        raise ValueError(
            f'{dataset_name} 的标签表无法用五元组键索引，'
            '请走 load_labeled_mini_flows 专用路径')
    usecols = {
        'cicflowmeter': {'Source IP', 'Destination IP', 'Source Port',
                         'Destination Port', 'Protocol', 'Timestamp', 'Label'},
        'ctu': {'StartTime', 'Proto', 'SrcAddr', 'Sport', 'DstAddr',
                'Dport', 'Label'},
        'doh': {'SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort',
                'TimeStamp', 'DoH'},
        'unsw': {'srcip', 'sport', 'dstip', 'dsport', 'proto', 'Stime',
                 'attack_cat', 'Label'},
    }[adapter]
    df = pd.read_csv(csv_path, dtype=str,
                     usecols=lambda c: c.strip() in usecols)
    df.columns = [str(c).strip() for c in df.columns]
    index: Dict[str, Dict[str, List[SourceLabelRecord]]] = {}
    stats = {'rows': len(df), 'attack': 0, 'benign': 0, 'skipped': 0}
    for row in df.to_dict('records'):
        record = _mini_row_to_record(dataset_name, row, class_name)
        if record is None:
            stats['skipped'] += 1
            continue
        keys, ts, label, category = record
        stats['attack' if category != 'benign' else 'benign'] += 1
        for key in keys:
            for k in (key, _reverse_pcap_label_key(key)):
                index.setdefault(class_name, {}).setdefault(k, []).append(
                    SourceLabelRecord(
                        source_id=class_name, key=k, label=label,
                        category=category, timestamp_epoch=ts,
                    )
                )
    for source in index.values():
        for records in source.values():
            records.sort(key=lambda r: r.timestamp_epoch)
    return index, stats


def mini_manifest_offsets(manifest_path) -> List[float]:
    """读取切分 manifest 的 source_units 偏移集合。切分端约定
    offset_seconds = PCAP_ts - CSV_ts；本转换端约定为 CSV-PCAP，
    故返回值逐个取反。CICDDoS2019 部分类合并了不同捕获日的来源，
    各来源偏移不同（如 14400 与 10800），必须保留全部候选逐流
    匹配——取中位数会让两个真实来源都落在容差外（Syn/UDPLag
    曾因此全量 unmatched）。"""
    if not manifest_path.exists():
        return []
    with manifest_path.open('r', encoding='utf-8') as fh:
        manifest = json.load(fh)
    offsets = []
    for unit in manifest.get('source_units', []):
        value = unit.get('offset_seconds')
        if value is not None:
            candidate = -float(value)
            if candidate not in offsets:
                offsets.append(candidate)
    return offsets


def resolve_mini_flow_labels(flows, source_id, label_index, offsets,
                             tolerance):
    """流标签解析（规则与 convert_labeled_source_flows 相同）：对每个
    偏移候选取容差内最近且唯一的匹配，再跨候选取时间差最小者。
    候选间相差远大于容差（如 3600s vs 60s）时不会误配。"""
    matched, unmatched = 0, 0
    matched_by_category = defaultdict(int)
    matched_by_offset = defaultdict(int)
    rejections = defaultdict(int)
    resolutions: List[Optional[tuple]] = []
    for flow in flows:
        best = None          # (delta, label, category, offset)
        best_reason = None   # 无匹配时保留最小 delta 的原因
        best_reason_delta = None
        for offset in offsets:
            resolution = resolve_source_flow_label(
                source_id, flow, label_index,
                offset_seconds=offset, tolerance_seconds=tolerance,
            )
            if resolution.matched:
                if best is None or \
                        resolution.time_delta_seconds < best[0]:
                    best = (resolution.time_delta_seconds,
                            resolution.label, resolution.category, offset)
            else:
                if best_reason is None or (
                        resolution.time_delta_seconds is not None
                        and best_reason_delta is not None
                        and resolution.time_delta_seconds
                        < best_reason_delta):
                    best_reason = resolution.reason
                    best_reason_delta = resolution.time_delta_seconds
        if best is None:
            unmatched += 1
            rejections[best_reason or 'no_candidate'] += 1
            resolutions.append(None)
            continue
        matched += 1
        matched_by_category[best[2]] += 1
        matched_by_offset[best[3]] += 1
        resolutions.append((best[1], best[2]))
    report = {
        'matched': matched,
        'unmatched': unmatched,
        'matched_by_category': dict(sorted(matched_by_category.items())),
        'matched_by_offset': {
            str(round(off, 3)): n
            for off, n in sorted(matched_by_offset.items())
        },
        'rejections': dict(sorted(rejections.items())),
    }
    return resolutions, report


# ─────────────────────────────────────────────────────────────────────
# IoT-23 / CSECICIDS2018 专用取流路径
#
# 这两个数据集的 mini CSV 都不能用“五元组键 + 时间偏移”直接索引：
#   - IoT-23：同一五元组在 120s 内被大量短连接复用，按五元组重建流会把
#     上千条 Zeek 行并成一条巨流（实测 DDoS 类 11923 行 → 952 条），必须
#     按切分端“行定义时间窗、包归属最近窗口中点”的规则回放；
#   - CSECICIDS2018：官方 CSV 无源 IP / 源端口，只能按切分端的指纹规则
#     （canonical 双向流 + 时长/前向后向包数与 payload 字节）配对，偏移取
#     切分 manifest 的指纹推断值。
# ─────────────────────────────────────────────────────────────────────
def _iot23_row_tolerance(raw_ts) -> float:
    """与切分端 _fractional_tol 同口径：带亚秒小数的行用 0.01s 容差，
    整秒行才用基准 1.0s。"""
    text = str(raw_ts or '')
    for separator in ('.', ','):
        position = text.find(separator)
        if position >= 0 and position + 1 < len(text) \
                and text[position + 1].isdigit():
            return _IOT23_SUBSECOND_TOL
    return _IOT23_BASE_TOL


def _canonical_pcap_key(proto, src, dst, sport, dport):
    """规范化双向五元组（与切分端 canonical_key / extract_pcap_features 同口径）。

    直接复用 `extract_pcap_features.canonical_five_tuple`，避免两处各写一份
    排序规则而在边界（同一 IP、端口相等）上产生分歧。
    """
    from extract_pcap_features import canonical_five_tuple

    return canonical_five_tuple((proto, src, dst, sport, dport))


def _iter_pcap_l4_packets(pcap_path):
    """逐包产出 (proto, src, dst, sport, dport, ts, ip_len, payload, fin_rst)。

    ip_len 按 IP 总长还原（与 extract_pcap_features 同口径）；payload 为
    L4 载荷字节数、fin_rst 为 TCP FIN/RST 标志，供 CSECICIDS2018 指纹重建
    使用（对应切分端 dpkt 的 len(l4.data) 与 FIN|RST）。
    """
    from extract_pcap_features import _require_scapy

    IP, IPv6, PcapReader, TCP, UDP = _require_scapy()
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            ip_layer = packet.getlayer(IP) or packet.getlayer(IPv6)
            if ip_layer is None:
                continue
            proto = int(getattr(ip_layer, 'proto', getattr(ip_layer, 'nh', 0)))
            sport = dport = 0
            payload = 0
            fin_rst = False
            tcp = packet.getlayer(TCP)
            udp = packet.getlayer(UDP)
            if tcp is not None:
                sport, dport = int(tcp.sport), int(tcp.dport)
                payload = len(bytes(tcp.payload))
                fin_rst = bool(int(tcp.flags) & 0x05)      # FIN | RST
            elif udp is not None:
                sport, dport = int(udp.sport), int(udp.dport)
                payload = len(bytes(udp.payload))

            # 过滤损坏的数据包
            if isinstance(ip_layer, IPv6):
                ip_len = max(int(len(ip_layer)), 40 + int(ip_layer.plen))
            else:
                ip_len = max(int(len(ip_layer)), int(ip_layer.len))
            yield (proto, str(ip_layer.src), str(ip_layer.dst), sport, dport,
                   float(packet.time), ip_len, payload, fin_rst)


class _MiniRowWindow:
    """一行 mini CSV 定义的时间窗（start/end/tol）、标签与其归属的包。"""
    __slots__ = ('dkey', 'ckey', 'start', 'end', 'tol', 'label', 'category',
                 'packets')

    def __init__(self, dkey, start, end, tol, label, category):
        self.dkey = dkey
        self.ckey = _canonical_pcap_key(*dkey)
        self.start = start
        self.end = end
        self.tol = tol
        self.label = label
        self.category = category
        self.packets = []


def load_iot23_rows(csv_path, class_name):
    """Zeek conn.log 列 → 行时间窗列表（按 start 升序）。返回 (rows, 统计)。"""
    frame = pd.read_csv(
        csv_path, dtype=str, keep_default_na=False,
        usecols=['ts', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
                 'proto', 'duration', 'label_tail'])
    frame.columns = [str(column).strip() for column in frame.columns]
    rows = []
    stats = {'rows': len(frame), 'benign': 0, 'attack': 0, 'skipped': 0}
    for record in frame.to_dict('records'):
        proto = _IOT23_PROTO_MAP.get(_safe_str(record.get('proto')).lower())
        try:
            timestamp = float(_safe_str(record.get('ts')))
            sport = int(float(_safe_str(record.get('id.orig_p'))))
            dport = int(float(_safe_str(record.get('id.resp_p'))))
        except ValueError:
            stats['skipped'] += 1
            continue
        if proto is None:
            stats['skipped'] += 1
            continue
        try:
            duration = max(float(_safe_str(record.get('duration'))), 0.0)
        except ValueError:
            duration = 0.0              # Zeek 的 '-'（无时长）按 0 处理
        segments = _safe_str(record.get('label_tail')).split('   ')
        label = segments[1].strip() if len(segments) > 1 else ''
        detailed = segments[2].strip() if len(segments) > 2 else '-'
        lowered = label.lower()
        if lowered == 'benign':
            category = 'benign'
        elif lowered == 'malicious':
            category = class_name
            label = detailed if detailed not in ('', '-') else label
        else:
            stats['skipped'] += 1       # '-' 等未标注背景行：不猜良性
            continue
        stats['benign' if category == 'benign' else 'attack'] += 1
        dkey = (proto, _safe_str(record.get('id.orig_h')),
                _safe_str(record.get('id.resp_h')), sport, dport)
        rows.append(_MiniRowWindow(
            dkey, timestamp, timestamp + duration,
            _iot23_row_tolerance(record.get('ts')), label, category))
    rows.sort(key=lambda row: row.start)
    return rows, stats


def _locate_row_window(flows, timestamp):
    """时间窗 [start-tol, end+tol] 覆盖 timestamp 的行中取中点最近者；
    距离并列即歧义（与切分端 MatchEngine._locate_flow 同语义）。"""
    if not flows:
        return None, False
    index = 0
    while index < len(flows) and flows[index].end + flows[index].tol < timestamp:
        index += 1
    if index:
        del flows[:index]
    best = None
    best_distance = None
    tied = False
    for flow in flows:
        if flow.start - flow.tol <= timestamp <= flow.end + flow.tol:
            distance = abs((flow.start + flow.end) * 0.5 - timestamp)
            if best is None or distance < best_distance:
                best, best_distance, tied = flow, distance, False
            elif distance == best_distance:
                tied = True
        if flow.start - flow.tol > timestamp and best is not None:
            break
    return best, bool(best is not None and tied)


def assign_iot23_packets(pcap_path, rows):
    """按行时间窗回放 mini PCAP，把每个包归属到唯一行（切分端同规则）。"""
    from extract_pcap_features import PacketPoint

    directed = defaultdict(list)
    canonical = defaultdict(list)
    for row in rows:
        directed[row.dkey].append(row)
        canonical[row.ckey].append(row)
    stats = {'packets': 0, 'assigned': 0, 'rejected': 0}
    for proto, src, dst, sport, dport, timestamp, ip_len, _, _ in \
            _iter_pcap_l4_packets(pcap_path):
        stats['packets'] += 1
        window, ambiguous = _locate_row_window(
            directed.get((proto, src, dst, sport, dport)), timestamp)
        if window is None:
            window, ambiguous = _locate_row_window(
                canonical.get(
                    _canonical_pcap_key(proto, src, dst, sport, dport)),
                timestamp)
        if window is None or ambiguous:
            stats['rejected'] += 1
            continue
        window.packets.append(PacketPoint(ts=timestamp, length=ip_len))
        stats['assigned'] += 1
    return stats


def _cse2018_timestamp(value) -> Optional[float]:
    """官方 CSV 时间戳（DD/MM/YYYY HH:MM:SS，12 小时制）→ epoch 秒（UTC）。"""
    text = _safe_str(value)
    if not text:
        return None
    try:
        parsed = datetime.strptime(text, _CSE2018_TS_FORMAT)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def load_cse2018_fingerprints(csv_path):
    """官方 CSV → (rows, index, 统计)。

    rows  : (ts, 时长µs, 前向包数, 后向包数, 前向 payload, 后向 payload, 标签)
    index : (协议, 目的端口, int(ts)) -> [行号]
    """
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False,
                        usecols=list(_CSE2018_CSV_COLUMNS))
    frame.columns = [str(column).strip() for column in frame.columns]
    rows = []
    index = defaultdict(list)
    label_counts = defaultdict(int)
    stats = {'rows': len(frame), 'indexed': 0, 'skipped': 0}
    for record in frame.to_dict('records'):
        timestamp = _cse2018_timestamp(record.get('Timestamp'))
        numbers = []
        for column in ('Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
                       'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts'):
            value = _safe_int(record.get(column), None)
            if value is None:
                break
            numbers.append(value)
        if timestamp is None or len(numbers) < 7:
            stats['skipped'] += 1       # 取值无效行（含重复表头）
            continue
        dport, proto, duration, forward, backward, forward_payload, \
            backward_payload = numbers
        label = _safe_str(record.get('Label'))
        rows.append((timestamp, duration, forward, backward, forward_payload,
                     backward_payload, label))
        index[(proto, dport, int(timestamp))].append(len(rows) - 1)
        label_counts[label] += 1
        stats['indexed'] += 1
    stats['labels'] = dict(sorted(label_counts.items()))
    return rows, index, stats


def cse2018_manifest_offsets(manifest_path) -> List[float]:
    """读切分 manifest 的指纹偏移（小时网格）。切分端约定
    offset = PCAP - CSV，本转换端为 CSV - PCAP，故逐个取反。"""
    if not manifest_path.exists():
        return []
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    hours = manifest.get('pair_audit', {}).get('offset_inference', {}) \
        .get('chosen_offsets_hours') or []
    offsets: List[float] = []
    for value in hours:
        candidate = -float(value) * 3600.0
        if candidate not in offsets:
            offsets.append(candidate)
    return offsets


class _Cse2018Flow:
    """指纹配对用的双向流（前向 = 首包方向）。"""
    __slots__ = ('key', 'proto', 'sport', 'dport', 'start', 'last',
                 'fwd_pkts', 'bwd_pkts', 'fwd_payload', 'bwd_payload',
                 'fin_rst', 'packets')

    def __init__(self, key, proto, sport, dport, start):
        self.key = key
        self.proto = proto
        self.sport = sport
        self.dport = dport
        self.start = start
        self.last = start
        self.fwd_pkts = 0
        self.bwd_pkts = 0
        self.fwd_payload = 0
        self.bwd_payload = 0
        self.fin_rst = False
        self.packets = []

    @property
    def duration_us(self) -> int:
        return int(round((self.last - self.start) * 1e6))


def collect_cse2018_flows(pcap_path, idle_timeout_seconds):
    """按切分端同规则重建双向流：canonical 五元组 + 空闲超时 / TCP FIN|RST
    关闭；前向 = 首包方向，包长按 IP 总长。"""
    from extract_pcap_features import PacketPoint

    active: Dict[tuple, _Cse2018Flow] = {}
    closed: List[_Cse2018Flow] = []
    for proto, src, dst, sport, dport, timestamp, ip_len, payload, fin_rst in \
            _iter_pcap_l4_packets(pcap_path):
        directed_key = (proto, src, dst, sport, dport)
        key = _canonical_pcap_key(*directed_key)
        flow = active.get(key)
        if flow is not None and (timestamp - flow.last > idle_timeout_seconds
                                 or flow.fin_rst):
            closed.append(flow)
            del active[key]
            flow = None
        if flow is None:
            flow = _Cse2018Flow(directed_key, proto, sport, dport, timestamp)
            active[key] = flow
        if directed_key == flow.key:
            flow.fwd_pkts += 1
            flow.fwd_payload += payload
        else:
            flow.bwd_pkts += 1
            flow.bwd_payload += payload
        flow.packets.append(PacketPoint(ts=timestamp, length=ip_len))
        flow.last = timestamp
        if fin_rst:
            flow.fin_rst = True
    closed.extend(active.values())
    closed.sort(key=lambda flow: flow.start)
    return closed


def match_cse2018_fingerprints(flows, rows, index, offsets):
    """指纹配对：同一行至多被一条流占用，|Δt| 最小者优先（切分端 _settle
    同口径）；跨标签歧义直接丢弃。返回 (每条流的标签或 None, 统计)。"""
    used = set()
    resolutions: List[Optional[str]] = []
    rejections = defaultdict(int)
    matched_by_label = defaultdict(int)
    stats = {'matched': 0, 'same_label_multi': 0, 'cross_label_ambiguous': 0}
    for flow in flows:
        duration = flow.duration_us
        candidates = []
        for offset in offsets:
            expected = flow.start + offset
            for delta in (-1, 0, 1):
                second = int(expected) + delta
                for orientation, port in ((0, flow.dport), (1, flow.sport)):
                    for row_index in index.get((flow.proto, port, second), ()):
                        if row_index in used:
                            continue
                        entry = rows[row_index]
                        if entry[1] != duration or abs(entry[0] - expected) \
                                > _CSE2018_TIME_TOLERANCE:
                            continue
                        if orientation == 0:
                            hit = (entry[2], entry[3], entry[4], entry[5]) == (
                                flow.fwd_pkts, flow.bwd_pkts,
                                flow.fwd_payload, flow.bwd_payload)
                        else:
                            hit = (entry[2], entry[3], entry[4], entry[5]) == (
                                flow.bwd_pkts, flow.fwd_pkts,
                                flow.bwd_payload, flow.fwd_payload)
                        if hit:
                            candidates.append(
                                (abs(entry[0] - expected), row_index, entry[6]))
        if not candidates:
            rejections['no_candidate'] += 1
            resolutions.append(None)
            continue
        labels = {candidate[2] for candidate in candidates}
        if len(labels) > 1:
            stats['cross_label_ambiguous'] += 1
            rejections['cross_label_ambiguous'] += 1
            resolutions.append(None)
            continue
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        if len(candidates) > 1:
            stats['same_label_multi'] += 1
        used.add(candidates[0][1])
        label = candidates[0][2]
        stats['matched'] += 1
        matched_by_label[label] += 1
        resolutions.append(label)
    stats['matched_by_label'] = dict(sorted(matched_by_label.items()))
    stats['rejections'] = dict(sorted(rejections.items()))
    return resolutions, stats


def load_labeled_mini_flows(dataset_name, class_name, csv_path, pcap_path,
                            manifest_path, flow_idle_timeout_seconds,
                            min_packets):
    """IoT-23 / CSECICIDS2018 专用取流：返回 ([(flow, label, category)], 审计)。

    其他 mini 数据集走“五元组键 + 时间偏移”的通用路径，不经过这里。
    """
    from extract_pcap_features import FlowRecord

    matched_by_category = defaultdict(int)
    if mini_adapter(dataset_name) == 'iot23':
        rows, row_stats = load_iot23_rows(csv_path, class_name)
        packet_stats = assign_iot23_packets(pcap_path, rows)
        labeled = []
        for row in rows:
            if len(row.packets) < min_packets:
                continue
            row.packets.sort(key=lambda point: point.ts)
            labeled.append((FlowRecord(key=row.dkey, packets=row.packets),
                            row.label, row.category))
            matched_by_category[row.category] += 1
        audit = {
            'label_rule': 'iot23_row_window_replay',
            'csv_rows': row_stats,
            'packets': packet_stats,
            'label_resolution': {
                'matched': len(labeled),
                'unmatched': row_stats['rows'] - len(labeled),
                'matched_by_category': dict(
                    sorted(matched_by_category.items())),
                'rejections': {
                    'rows_without_packets':
                        sum(1 for row in rows if not row.packets),
                    'packets_unassigned': packet_stats['rejected'],
                },
            },
        }
        return labeled, audit

    rows, index, csv_stats = load_cse2018_fingerprints(csv_path)
    offsets = cse2018_manifest_offsets(manifest_path)
    offset_source = 'manifest' if offsets else 'zero'
    if not offsets:
        offsets = [0.0]
    flows = collect_cse2018_flows(pcap_path, flow_idle_timeout_seconds)
    resolutions, match_stats = match_cse2018_fingerprints(
        flows, rows, index, offsets)
    labeled = []
    for flow, label in zip(flows, resolutions):
        if label is None or len(flow.packets) < min_packets:
            continue
        category = 'benign' if label.upper() == 'BENIGN' else class_name
        labeled.append((FlowRecord(key=flow.key, packets=flow.packets),
                        label, category))
        matched_by_category[category] += 1
    audit = {
        'label_rule': 'cse2018_fingerprint_unique',
        'csv_rows': csv_stats,
        'pcap_flows': len(flows),
        'pcap_packets': sum(flow.fwd_pkts + flow.bwd_pkts for flow in flows),
        'time_offsets_seconds': offsets,
        'offset_source': offset_source,
        'label_resolution': {
            'matched': len(labeled),
            'unmatched': csv_stats['rows'] - len(labeled),
            'matched_by_category': dict(sorted(matched_by_category.items())),
            'rejections': match_stats['rejections'],
            'same_label_multi': match_stats['same_label_multi'],
        },
    }
    return labeled, audit


def split_points_into_packet_chunks(group, packet_labels, chunk_packets):
    """Split one idle-group's (ts, length, ordinal) points into contiguous
    fixed-size packet pieces and carry their per-packet labels along.

    Returns [(piece_points, piece_labels), ...] in arrival order. Pure helper
    so the chunk/label-purity logic is testable without scapy.
    """
    if chunk_packets and int(chunk_packets) > 0:
        step = int(chunk_packets)
        pieces = [
            (group[start:start + step], packet_labels[start:start + step])
            for start in range(0, len(group), step)
        ]
    else:
        pieces = [(group, packet_labels)]
    return pieces


def collect_kitsune_labeled_flows(pcap_path, csv_path, idle_timeout,
                                  min_packets, attack_ratio=0.5,
                                  chunk_packets=0):
    """Kitsune：CSV 是包级标签 (packet_index,label)，与输出 PCAP 的包
    序号一一对应（含非 IP 包——序号必须按全部包计数）。按 5 元组 +
    空闲超时重建流；流内攻击包比例 >= attack_ratio 判为攻击流
    （与切分端口径一致）。

    chunk_packets>0 时，每个空闲流再切成连续的定长包段；每段用段内包
    标签多数表决独立打标签，并写入同一个 group 标识，供加载器做同组分
    裂防护，避免相邻近重复段跨 train/test 造成泄漏。"""
    from extract_pcap_features import FlowRecord, PacketPoint, _require_scapy
    IP, IPv6, PcapReader, TCP, UDP = _require_scapy()

    labels = [int(v) for v in pd.read_csv(csv_path, usecols=['label'])['label']]
    per_tuple: Dict[tuple, list] = {}
    total = 0
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            ordinal = total
            total += 1
            ip_layer = pkt.getlayer(IP) or pkt.getlayer(IPv6)
            if ip_layer is None:
                continue
            proto = int(getattr(ip_layer, 'proto',
                                getattr(ip_layer, 'nh', 0)))
            sport = dport = 0
            if pkt.haslayer(TCP):
                sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
            elif pkt.haslayer(UDP):
                sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
            key = (proto, str(ip_layer.src), str(ip_layer.dst),
                   sport, dport)
            per_tuple.setdefault(key, []).append(
                (float(pkt.time), int(len(ip_layer)), ordinal))
    if len(labels) != total:
        raise ValueError(
            f'Kitsune 标签数({len(labels)}) != PCAP 包数({total}): {pcap_path}')

    chunk_packets = int(chunk_packets or 0)
    segmented = chunk_packets > 0
    benign_flows, attack_flows = [], []
    parent_flow_count = 0
    mixed_label_parents = 0
    group_counter = 0
    for key, points in per_tuple.items():
        points.sort(key=lambda p: p[0])
        groups, current = [], []
        for point in points:
            if current and point[0] - current[-1][0] > idle_timeout:
                groups.append(current)
                current = []
            current.append(point)
        if current:
            groups.append(current)
        for group in groups:
            if len(group) < min_packets:
                continue
            parent_flow_count += 1
            group_counter += 1
            group_id = f'k{group_counter:08d}' if segmented else None
            ordinals = [p[2] for p in group]
            group_packet_labels = [labels[idx] for idx in ordinals]
            piece_decisions = []
            for piece, piece_labels in split_points_into_packet_chunks(
                    group, group_packet_labels, chunk_packets):
                is_attack = sum(piece_labels) / len(piece_labels) >= attack_ratio
                flow = FlowRecord(
                    key=key,
                    packets=[PacketPoint(ts=p[0], length=p[1]) for p in piece],
                    group=group_id,
                )
                piece_decisions.append(is_attack)
                if is_attack:
                    attack_flows.append(flow)
                else:
                    benign_flows.append(flow)
            if segmented and len(set(piece_decisions)) > 1:
                mixed_label_parents += 1
    benign_flows.sort(key=lambda f: f.start_ts)
    attack_flows.sort(key=lambda f: f.start_ts)
    segment_count = len(benign_flows) + len(attack_flows)
    audit = {
        'packets': total,
        'label_rows': len(labels),
        'raw_five_tuples': len(per_tuple),
        'flows': segment_count,
        'attack_flows': len(attack_flows),
        'benign_flows': len(benign_flows),
        'attack_ratio_threshold': attack_ratio,
        'segmentation': {
            'method': 'fixed_packet_chunks' if segmented else 'idle_flow',
            'chunk_packets': chunk_packets if segmented else 0,
            'parent_idle_flows': parent_flow_count,
            'segments': segment_count,
            'mixed_label_parent_flows': mixed_label_parents,
        },
    }
    return benign_flows, attack_flows, audit


# ─────────────────────────────────────────────────────────────────────
# v4 折布局：{mini_root}/{Dataset}/{Class}/{train,val,test}/{split}.pcap|.csv
#
# 与旧的「平铺」布局（{Class}/{Class}.pcap + {Class}.csv）并存：v4 由项目共享
# cutter 直切 pcap，CSV 统一表头为
#     flow_id,label,first_seen,end_ts,<数据集原生列...>
# 且 first_seen/end_ts 是微秒精度的 epoch 秒。因为折内 pcap 恰好只包含这些流，
# 标签可以直接按 first_seen 精确匹配，无需旧的「五元组 + 时间偏移」校准，也无需
# IoT-23 / CSECICIDS2018 的专用取流路径（那些是为「pcap 与 CSV 时间轴不一致」
# 的旧数据准备的）。
# ─────────────────────────────────────────────────────────────────────
SPLIT_FOLDS = ('train', 'val', 'test')

# mini_root 下不是数据集目录，不参与转换也不报「不支持」警告。
MINI_IGNORED_DIRS = {'code', 'golden', '__pycache__'}


def is_split_layout(class_dir) -> bool:
    """判断类目录是否为 v4 折布局（至少一个折有 {split}.pcap）。"""
    class_dir = Path(class_dir)
    return any((class_dir / split / f'{split}.pcap').exists() for split in SPLIT_FOLDS)


def _parse_epoch(value):
    """把 CSV 里的 epoch 秒字段解析为 float；空值/NaN/非有限返回 None。"""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _proto_column_to_int(value):
    """折 CSV 的协议列 → IP 协议号；无法解析返回 None。"""
    text = _safe_str(value).strip().lower()
    named = {'tcp': 6, 'udp': 17, 'icmp': 1, 'icmp6': 58, 'icmpv6': 58, 'ipv6-icmp': 58}
    if text in named:
        return named[text]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


# 各数据集折 CSV 里的五元组列名；None 表示该 CSV 没有完整五元组，
# 只能退回「时间戳」匹配（实测 Kitsune / CSECICIDS2018 的 first_seen 唯一，够用）。
SPLIT_FLOW_KEY_COLUMNS = {
    'cicflowmeter': {'proto': 'Protocol', 'src': 'Source IP', 'sport': 'Source Port',
                     'dst': 'Destination IP', 'dport': 'Destination Port'},
    'ctu': None,
    'doh': None,
    'unsw': {'proto': 'proto', 'src': 'srcip', 'sport': 'sport',
             'dst': 'dstip', 'dport': 'dsport'},
    'iot23': {'proto': 'proto', 'src': 'id.orig_h', 'sport': 'id.orig_p',
              'dst': 'id.resp_h', 'dport': 'id.resp_p'},
    'kitsune': None,
    'cse2018': None,
}

# pcap 写入会把时间戳舍入到微秒，实测 first_seen 与重建流相差 ~1e-6；
# 容差窗口内只接受五元组完全相同（或该数据集无五元组）的候选，避免错配。
SPLIT_TIME_TOLERANCE = 1e-3


class SplitLabelIndex:
    """v4 折标签索引：主键 ``(first_seen, canonical 五元组)``，时间戳做兜底。

    CICIDS2017 实测一个时间戳下最多压了 292 条流（cutter 的 flow_id 也是按
    ``(first_seen, ckey)`` 分配的），因此仅按时间匹配会漏配/错配。
    """

    def __init__(self, entries, stats, has_five_tuple):
        self.stats = stats
        self.has_five_tuple = has_five_tuple
        self._by_key = {}
        self._by_time = defaultdict(list)
        for first_seen, end_ts, ckey, label, category in entries:
            self._by_key[(first_seen, ckey)] = (label, category)
            self._by_time[first_seen].append((end_ts, ckey, label, category))
        self._times = sorted(self._by_time)

    def __len__(self) -> int:
        return len(self._by_key)

    def _pick_same_key(self, candidates, ckey, flow):
        """在候选里挑五元组相同（或该数据集无五元组）的那些，再用 end_ts 消歧。"""
        same = [c for c in candidates if c[1] == ckey]
        if not same and not self.has_five_tuple:
            same = [c for c in candidates if c[1] is None]
        if not same:
            return None
        if len(same) == 1:
            return same[0][2], same[0][3]
        ranked = sorted(
            ((abs(c[0] - flow.end_ts) if c[0] is not None else float('inf'), c)
             for c in same),
            key=lambda item: item[0],
        )
        if not np.isfinite(ranked[0][0]):
            return None
        if abs(ranked[1][0] - ranked[0][0]) <= 1e-6:
            return None
        return ranked[0][1][2], ranked[0][1][3]

    def resolve(self, flow):
        """解析一条重建流的标签；无法唯一确定时返回 None（宁可不标）。"""
        from extract_pcap_features import canonical_five_tuple

        ckey = canonical_five_tuple(flow.key)
        first_seen = round(float(flow.start_ts), 6)

        exact = self._by_key.get((first_seen, ckey))
        if exact is not None:
            return exact

        candidates = self._by_time.get(first_seen)
        if candidates:
            hit = self._pick_same_key(candidates, ckey, flow)
            if hit is not None:
                return hit

        # 时间戳舍入兜底：只在容差窗口内找，且必须五元组完全一致。
        # 必须同时看 index-1/index/index+1：精确命中时间戳但五元组不同时
        # （实测同一条流在 CSV 里差 1µs），正确答案可能在下一个时间戳上。
        index = bisect.bisect_left(self._times, first_seen)
        for position in (index - 1, index, index + 1):
            if not 0 <= position < len(self._times):
                continue
            near = self._times[position]
            if near == first_seen or abs(near - first_seen) > SPLIT_TIME_TOLERANCE:
                continue
            if not self.has_five_tuple:
                hit = self._pick_same_key(self._by_time[near], ckey, flow)
            else:
                hit = self._by_key.get((near, ckey))
            if hit is not None:
                return hit
        return None


def build_split_label_index(csv_path, class_name, dataset=None):
    """折 CSV → ``SplitLabelIndex``（+ 行统计）。

    v4 契约的 label ∈ {atk, ben}：atk 归到本类，ben 归到 benign。
    """
    header = pd.read_csv(csv_path, nrows=0)
    columns = {str(column).strip() for column in header.columns}
    missing = [name for name in ('label', 'first_seen', 'end_ts')
               if name not in columns]
    if missing:
        raise ValueError(f'v4 折 CSV 缺少必需列 {missing}: {csv_path}')

    key_columns = SPLIT_FLOW_KEY_COLUMNS.get(
        mini_adapter(dataset)) if dataset else None
    if key_columns and not set(key_columns.values()) <= columns:
        print(f'[WARN] {csv_path.name} 缺少五元组列，退回按时间戳匹配')
        key_columns = None

    wanted = {'flow_id', 'label', 'first_seen', 'end_ts'}
    if key_columns:
        wanted |= set(key_columns.values())
    df = pd.read_csv(csv_path, dtype=str,
                     usecols=lambda column: str(column).strip() in wanted)
    df.columns = [str(column).strip() for column in df.columns]

    stats = {'rows': len(df), 'attack': 0, 'benign': 0, 'skipped': 0}
    entries = []
    for row in df.to_dict('records'):
        first_seen = _parse_epoch(row.get('first_seen'))
        if first_seen is None:
            stats['skipped'] += 1
            continue
        end_ts = _parse_epoch(row.get('end_ts'))
        raw = _safe_str(row.get('label')).strip().lower()
        if raw in ('ben', 'benign', '0', 'normal', 'background'):
            label, category = 'BENIGN', 'benign'
            stats['benign'] += 1
        else:
            label, category = class_name, class_name
            stats['attack'] += 1

        ckey = None
        if key_columns:
            proto = _proto_column_to_int(row.get(key_columns['proto']))
            if proto is not None:
                ckey = _canonical_pcap_key(
                    proto,
                    _safe_str(row.get(key_columns['src'])),
                    _safe_str(row.get(key_columns['dst'])),
                    _safe_int(row.get(key_columns['sport'])),
                    _safe_int(row.get(key_columns['dport'])),
                )
        entries.append((round(first_seen, 6), end_ts, ckey, label, category))

    return SplitLabelIndex(entries, stats, has_five_tuple=key_columns is not None), stats


def resolve_fold_label(flow, index):
    """解析折标签；``index`` 为 ``SplitLabelIndex``。"""
    return index.resolve(flow)


def convert_split_class(dataset_name, class_name, class_dir, output_dir,
                        splits=SPLIT_FOLDS, max_benign=50000,
                        max_attack_per_cat=50000, segment_len=SEGMENT_LEN,
                        max_packet_len=1500, gap_threshold=1.0, host_window=10.0,
                        min_packets=1, flow_idle_timeout_seconds=120.0,
                        compact=False, no_progress=False, workers=1,
                        chunksize=1000, skip_malformed_dns=False):
    """v4 折布局：逐折转换，输出 {Dataset}/{Class}/{split}.json（benign+attack）。

    每个折单独重建流与 host 统计（折内自成一套时间基准），标签按 first_seen 匹配。
    """
    from extract_pcap_features import (
        build_host_windows, collect_flows, write_tfusion_json,
    )

    class_dir = Path(class_dir)
    out_dir = Path(output_dir) / dataset_name / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = {
        'dataset': dataset_name, 'class': class_name, 'layout': 'v4_split',
        'folds': {}, 'outputs': {'benign': 0, 'attack': 0},
    }

    for split in splits:
        fold_dir = class_dir / split
        pcap_path = fold_dir / f'{split}.pcap'
        csv_path = fold_dir / f'{split}.csv'
        if not pcap_path.exists() or not csv_path.exists():
            audit['folds'][split] = {'skipped': 'missing pcap or csv'}
            continue

        print(f'[{dataset_name}/{class_name}/{split}] 解析 PCAP 流')
        index, csv_stats = build_split_label_index(
            csv_path, class_name, dataset=dataset_name)
        flow_stats: dict = {}
        flows = collect_flows(
            [pcap_path], min_packets=min_packets,
            show_progress=not no_progress,
            flow_idle_timeout_seconds=flow_idle_timeout_seconds,
            skip_malformed_dns=skip_malformed_dns,
            stats=flow_stats,
        )
        host_windows = build_host_windows(
            flows, host_window, show_progress=not no_progress)
        base_ts = flows[0].start_ts if flows else 0.0

        tasks, matched, unmatched = [], 0, 0
        for flow in flows:
            resolved = resolve_fold_label(flow, index)
            if resolved is None:
                unmatched += 1
                continue
            label, category = resolved
            matched += 1
            tasks.append((len(tasks), label, category, flow))

        benign_items, attack_by_cat = _materialize_labeled_tasks(
            tasks, host_windows, base_ts, segment_len, max_packet_len,
            gap_threshold, host_window, workers, chunksize)

        data = {'benign': list(benign_items), 'attack': []}
        for category in sorted(attack_by_cat):
            data['attack'].extend(attack_by_cat[category])

        out_path = out_dir / f'{split}.json'
        write_tfusion_json(data, out_path, compact=compact)

        # CSV 有行但没有对应流 = 该标签未被消费（通常是 cutter 与本端空闲切分
        # 边界约定差几条）。这属于 0.0x% 级别的边界差异，不猜、但要显式暴露。
        valid_rows = csv_stats['rows'] - csv_stats['skipped']
        labels_unused = max(0, valid_rows - matched)
        if labels_unused:
            print(f'[WARN] {dataset_name}/{class_name}/{split}: '
                  f'CSV 有效 {valid_rows} 行，仅匹配到 {matched} 条流，'
                  f'{labels_unused} 个标签未被消费（多来源/切分边界差异）')

        # min_packets 过滤掉的条数必须报出来，否则会表现为"样本莫名变少"。
        dropped_short = flow_stats.get('flows_dropped_short', 0)
        if dropped_short:
            print(f'[WARN] {dataset_name}/{class_name}/{split}: '
                  f'min_packets={flow_stats.get("min_packets")} 过滤掉 '
                  f'{dropped_short} 条过短流（占保留数的 '
                  f'{dropped_short / max(1, len(flows)):.1%}）')

        print(f'[{dataset_name}/{class_name}/{split}] 完成: '
              f'benign={len(data["benign"])}, attack={len(data["attack"])}, '
              f'matched={matched}, unmatched={unmatched}')

        audit['folds'][split] = {
            'pcap': str(pcap_path), 'csv': str(csv_path),
            'csv_rows': csv_stats, 'flow_stats': flow_stats,
            'matched': matched, 'unmatched': unmatched,
            'labels_unused': labels_unused,
            'flows_dropped_short': dropped_short,
            'benign': len(data['benign']), 'attack': len(data['attack']),
            'output': str(out_path),
        }
        audit['outputs']['benign'] += len(data['benign'])
        audit['outputs']['attack'] += len(data['attack'])

    return audit


def convert_mini_class(dataset_name, class_name, class_dir, output_dir,
                       max_benign=50000, max_attack_per_cat=50000,
                       segment_len=SEGMENT_LEN, max_packet_len=1500,
                       gap_threshold=1.0, host_window=10.0, min_packets=1,
                       flow_idle_timeout_seconds=120.0,
                       label_time_tolerance_seconds=60.0, compact=False,
                       no_progress=False, workers=1, chunksize=1000,
                       packet_chunk_size=0, skip_malformed_dns=False):
    """一个类目录 → {output_dir}/{Dataset}/{Class}.json（benign+attack）。
    返回该类的审计记录。"""
    from extract_pcap_features import build_host_windows, collect_flows

    adapter = mini_adapter(dataset_name)
    class_dir = Path(class_dir)
    csv_path = class_dir / f'{class_name}.csv'
    pcap_path = class_dir / f'{class_name}.pcap'
    manifest_path = class_dir / 'manifest.json'
    for path in (csv_path, pcap_path):
        if not path.exists():
            raise FileNotFoundError(str(path))

    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{class_name}.json'

    if packet_chunk_size and adapter != 'kitsune':
        raise ValueError(
            '--packet-chunk-size 仅支持 Kitsune 包级标签数据集：定长段标签'
            '依赖包级真值，其他 mini 数据集只有流级标签。')

    audit = {
        'dataset': dataset_name,
        'class': class_name,
        'csv_path': str(csv_path),
        'pcap_path': str(pcap_path),
        'inputs': [describe_input_file(pcap_path)],
        'flow_idle_timeout_seconds': float(flow_idle_timeout_seconds),
        'label_time_tolerance_seconds': float(label_time_tolerance_seconds),
    }

    if adapter == 'kitsune':
        print(f'[{dataset_name}/{class_name}] Kitsune 包级标签模式')
        benign_flows, attack_flows, kitsune_audit = \
            collect_kitsune_labeled_flows(
                pcap_path, csv_path,
                idle_timeout=flow_idle_timeout_seconds,
                min_packets=min_packets,
                chunk_packets=packet_chunk_size,
            )
        audit['label_rule'] = 'kitsune_packet_majority'
        audit.update(kitsune_audit)
        all_flows = benign_flows + attack_flows
        host_windows = build_host_windows(
            all_flows, host_window, show_progress=not no_progress)
        base_ts = all_flows[0].start_ts if all_flows else 0.0
        tasks, benign_kept, attack_kept = [], 0, 0
        for flow in attack_flows:
            if max_attack_per_cat >= 0 and attack_kept >= max_attack_per_cat:
                break
            tasks.append((len(tasks), class_name, class_name, flow))
            attack_kept += 1
        for flow in benign_flows:
            if max_benign >= 0 and benign_kept >= max_benign:
                break
            tasks.append((len(tasks), 'BENIGN', 'benign', flow))
            benign_kept += 1
        benign_items, attack_by_cat = _materialize_labeled_tasks(
            tasks, host_windows, base_ts, segment_len, max_packet_len,
            gap_threshold, host_window, workers, chunksize)
    elif adapter in ('iot23', 'cse2018'):
        print(f'[{dataset_name}/{class_name}] '
              + ('Zeek 行时间窗回放' if adapter == 'iot23' else '指纹配对'))
        labeled_flows, row_audit = load_labeled_mini_flows(
            dataset_name, class_name, csv_path, pcap_path, manifest_path,
            flow_idle_timeout_seconds, min_packets)
        audit.update(row_audit)
        all_flows = [flow for flow, _, _ in labeled_flows]
        host_windows = build_host_windows(
            all_flows, host_window, show_progress=not no_progress)
        base_ts = all_flows[0].start_ts if all_flows else 0.0
        tasks, benign_kept = [], 0
        attack_kept_by_cat = defaultdict(int)
        for flow, label, category in labeled_flows:
            if category == 'benign':
                if max_benign >= 0 and benign_kept >= max_benign:
                    continue
                benign_kept += 1
            else:
                if max_attack_per_cat >= 0 and \
                        attack_kept_by_cat[category] >= max_attack_per_cat:
                    continue
                attack_kept_by_cat[category] += 1
            tasks.append((len(tasks), label, category, flow))
        benign_items, attack_by_cat = _materialize_labeled_tasks(
            tasks, host_windows, base_ts, segment_len, max_packet_len,
            gap_threshold, host_window, workers, chunksize)
    else:
        print(f'[{dataset_name}/{class_name}] 读取标签 CSV: {csv_path.name}')
        label_index, csv_stats = build_mini_label_index(
            dataset_name, class_name, csv_path)
        audit['csv_rows'] = csv_stats

        print(f'[{dataset_name}/{class_name}] 解析 PCAP 流')
        stats: dict = {}
        flows = collect_flows(
            [pcap_path], min_packets=min_packets,
            show_progress=not no_progress,
            flow_idle_timeout_seconds=flow_idle_timeout_seconds,
            skip_malformed_dns=skip_malformed_dns,
            stats=stats,
        )
        audit['pcap_flows'] = len(flows)
        audit['flow_stats'] = stats

        offsets = mini_manifest_offsets(manifest_path)
        offset_source = 'manifest'
        if not offsets:
            inferred = infer_source_time_offset(
                class_name, flows, label_index) if flows else None
            offsets = [inferred] if inferred is not None else [0.0]
            offset_source = 'inferred' if inferred is not None else 'zero'
        audit['time_offsets_seconds'] = offsets
        audit['offset_source'] = offset_source

        resolutions, report = ([], {
            'matched': 0, 'unmatched': 0, 'matched_by_category': {},
            'matched_by_offset': {}, 'rejections': {}})
        if flows:
            resolutions, report = resolve_mini_flow_labels(
                flows, class_name, label_index, offsets,
                label_time_tolerance_seconds)
            # 偏移方向自检：manifest 符号约定若整体零匹配，尝试逐个取反
            if report['matched'] == 0 and \
                    any(off != 0.0 for off in offsets):
                neg_offsets = [-off for off in offsets]
                alt_res, alt_report = resolve_mini_flow_labels(
                    flows, class_name, label_index, neg_offsets,
                    label_time_tolerance_seconds)
                if alt_report['matched'] > 0:
                    offsets, offset_source = neg_offsets, 'manifest_negated'
                    resolutions, report = alt_res, alt_report
                    audit['time_offsets_seconds'] = offsets
                    audit['offset_source'] = offset_source
        audit['label_resolution'] = report

        host_windows = build_host_windows(
            flows, host_window, show_progress=not no_progress)
        base_ts = flows[0].start_ts if flows else 0.0
        tasks, benign_kept = [], 0
        attack_kept_by_cat = defaultdict(int)
        for flow, resolution in zip(flows, resolutions):
            if resolution is None:
                continue
            label, category = resolution
            if category == 'benign':
                if max_benign >= 0 and benign_kept >= max_benign:
                    continue
                benign_kept += 1
            else:
                if max_attack_per_cat >= 0 and \
                        attack_kept_by_cat[category] >= max_attack_per_cat:
                    continue
                attack_kept_by_cat[category] += 1
            tasks.append((len(tasks), label, category, flow))
        benign_items, attack_by_cat = _materialize_labeled_tasks(
            tasks, host_windows, base_ts, segment_len, max_packet_len,
            gap_threshold, host_window, workers, chunksize)

    attack_items = []
    for items in attack_by_cat.values():
        attack_items.extend(items)
    save_json_with_compact(
        {'benign': benign_items, 'attack': attack_items},
        str(out_path), compact=compact)
    audit['outputs'] = {
        'path': str(out_path),
        'benign': len(benign_items),
        'attack': len(attack_items),
    }
    print(f'[{dataset_name}/{class_name}] 完成: benign='
          f'{len(benign_items)}, attack={len(attack_items)}')
    return audit


def _parse_filter(raw):
    if not raw:
        return None
    names = {t.strip() for t in str(raw).split(',') if t.strip()}
    return names or None


def run_mini_mode(mini_root, output_dir, datasets_filter=None,
                  classes_filter=None, max_benign=50000,
                  max_attack_per_cat=50000, segment_len=SEGMENT_LEN,
                  max_packet_len=1500, gap_threshold=1.0, host_window=10.0,
                  min_packets=1, flow_idle_timeout_seconds=120.0,
                  label_time_tolerance_seconds=60.0, compact=False,
                  no_progress=False, workers=1, chunksize=1000,
                  packet_chunk_size=0, skip_malformed_dns=False):
    """遍历 {mini_root}/{Dataset}/{Class}/ 并逐类转换。"""
    mini_root = Path(mini_root)
    if not mini_root.is_dir():
        raise NotADirectoryError(f'mini_root 不存在或不是目录: {mini_root}')
    wanted_datasets = _parse_filter(datasets_filter)
    wanted_classes = _parse_filter(classes_filter)

    available = sorted(
        d.name for d in mini_root.iterdir()
        if d.is_dir() and any(c.is_dir() for c in d.iterdir()))
    selected = [
        name for name in available
        if (wanted_datasets is None or name in wanted_datasets)
        and name in MINI_SUPPORTED_DATASETS
    ]
    skipped_unknown = [
        name for name in available
        if (wanted_datasets is None or name in wanted_datasets)
        and name not in MINI_IGNORED_DIRS
    ]
    for name in skipped_unknown:
        if name not in MINI_SUPPORTED_DATASETS:
            print(f'[WARN] 跳过不支持的数据集 {name}'
                  f'（支持: {", ".join(MINI_SUPPORTED_DATASETS)}）')
    if not selected:
        raise SystemExit(
            f'mini 模式没有可处理的数据集（目录下可用: {available}）')

    # 两类布局共用同一批特征参数；旧布局额外需要偏移校准与 Kitsune 分段参数。
    shared_kwargs = dict(
        max_benign=max_benign, max_attack_per_cat=max_attack_per_cat,
        segment_len=segment_len, max_packet_len=max_packet_len,
        gap_threshold=gap_threshold, host_window=host_window,
        min_packets=min_packets,
        flow_idle_timeout_seconds=flow_idle_timeout_seconds,
        compact=compact, no_progress=no_progress, workers=workers,
        chunksize=chunksize, skip_malformed_dns=skip_malformed_dns,
    )
    convert_kwargs = dict(
        shared_kwargs,
        label_time_tolerance_seconds=label_time_tolerance_seconds,
        packet_chunk_size=packet_chunk_size,
    )
    split_kwargs = dict(shared_kwargs)

    all_audits = []
    for dataset_name in selected:
        class_dirs = sorted(
            (d for d in (mini_root / dataset_name).iterdir() if d.is_dir()),
            key=lambda d: d.name)
        dataset_audits = []
        for class_dir in class_dirs:
            if wanted_classes is not None and \
                    class_dir.name not in wanted_classes:
                continue
            if is_split_layout(class_dir):
                converter, kwargs = convert_split_class, split_kwargs
            elif (class_dir / f'{class_dir.name}.pcap').exists():
                converter, kwargs = convert_mini_class, convert_kwargs
            else:
                print(f'[WARN] {class_dir} 既非 v4 折布局也缺少 '
                      f'{class_dir.name}.pcap，跳过')
                continue
            try:
                audit = converter(
                    dataset_name, class_dir.name, class_dir, output_dir,
                    **kwargs)
            except Exception as exc:      # 单类失败不阻断其余类别
                print(f'[ERROR] {dataset_name}/{class_dir.name}: {exc}')
                dataset_audits.append({
                    'dataset': dataset_name, 'class': class_dir.name,
                    'error': str(exc),
                })
                continue
            dataset_audits.append(audit)
            all_audits.append(audit)

        ds_manifest = {
            'contract_version': 3,
            'mode': 'mini_datasets',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'mini_root': str(mini_root),
            'flow_idle_timeout_seconds': float(flow_idle_timeout_seconds),
            'label_time_tolerance_seconds': float(
                label_time_tolerance_seconds),
            'classes': dataset_audits,
        }
        ds_dir = Path(output_dir) / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        with open(ds_dir / 'conversion_manifest.json', 'w',
                  encoding='utf-8') as target:
            json.dump(ds_manifest, target, ensure_ascii=False, indent=2)

    print('=' * 50)
    print('mini 模式汇总:')
    for audit in all_audits:
        outputs = audit.get('outputs', {})
        if audit.get('layout') == 'v4_split':
            folds = audit.get('folds', {})
            matched = sum(fold.get('matched', 0) for fold in folds.values())
            unmatched = sum(fold.get('unmatched', 0) for fold in folds.values())
            print(f"  {audit['dataset']}/{audit['class']}[v4]: "
                  f"benign={outputs.get('benign')}, "
                  f"attack={outputs.get('attack')}, "
                  f"matched={matched}, unmatched={unmatched}, "
                  f"folds={sorted(folds)}")
            continue
        resolution = audit.get('label_resolution', {})
        print(f"  {audit['dataset']}/{audit['class']}: "
              f"benign={outputs.get('benign')}, "
              f"attack={outputs.get('attack')}, "
              f"matched={resolution.get('matched', audit.get('flows', '-'))}")
    print(f'共转换 {len(all_audits)} 个类别')
    return all_audits


def save_json_with_compact(data, filepath, compact=False):
    """保存 JSON 文件并打印统计"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        if compact:
            json.dump(data, f, separators=(',', ':'))
        else:
            json.dump(data, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    n_benign = len(data['benign'])
    n_attack = len(data['attack'])
    print(f'  保存: {os.path.basename(filepath)} ({size_mb:.1f} MB, benign={n_benign}, attack={n_attack})')


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='CICIDS2017 PCAP + CSV labels → tFusion JSON 转换'
                    '（含 mini_datasets 切分产物模式）')
    parser.add_argument('--pcap', nargs='+', default=None,
                        help='可选：输入 pcap/pcapng 文件。提供后使用真实 PCAP 包特征，并用 --csv-dir 中的 CICIDS CSV 做标签映射。')
    parser.add_argument('--csv-dir', default='data/label/train')
    parser.add_argument('--output-dir', default='data/CICIDS2017/pcap_processed_v2')
    parser.add_argument('--mini-root', nargs='?', const=MINI_DEFAULT_ROOT,
                        default=None,
                        help='mini_datasets 切分产物根目录。提供后进入 mini 模式：遍历 '
                             '{root}/{数据集}/{类}/ 并输出 tFusion JSON。不带路径值时'
                             f'使用默认根 {MINI_DEFAULT_ROOT}。与 --pcap 互斥。')
    parser.add_argument('--datasets', default=None,
                        help='mini 模式数据集过滤，逗号分隔（如 CICDDoS2019,Kitsune）')
    parser.add_argument('--classes', default=None,
                        help='mini 模式类别目录过滤，逗号分隔（如 DrDoS_DNS,Syn）')
    parser.add_argument('--max-benign', type=int, default=50000)
    parser.add_argument('--max-attack-per-cat', type=int, default=50000,
                        help='每个攻击类别最多采样多少条')
    parser.add_argument('--segment-len', type=int, default=SEGMENT_LEN)
    parser.add_argument('--max-packet-len', type=int, default=1500)
    parser.add_argument('--gap-threshold', type=float, default=1.0)
    parser.add_argument('--host-window', type=float, default=10.0)
    parser.add_argument('--min-packets', type=int, default=1,
                        help='丢弃包数少于此值的流。默认 1（不过滤）——实测按 4 过滤会删掉'
                             '全库 32.6%% 的流，且各类别不均衡：CICDDoS2019 攻击占比 48.0%%→17.5%%，'
                             'CSECICIDS2018 近乎清空，IoT-23 有多个类别整体消失。'
                             '仅在明确需要时显式指定。')
    parser.add_argument('--flow-idle-timeout-seconds', type=float, default=120.0,
                        help='相同五元组超过此空闲时间后开始新的 PCAP flow。')
    parser.add_argument('--label-time-tolerance-seconds', type=float, default=60.0,
                        help='校准时钟后 PCAP flow 与 CSV 聚合标签允许的最大起始时间差。')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--unmatched-policy', choices=['skip', 'benign', 'error'], default='skip',
                        help='PCAP flow 没有匹配到 CICIDS 标签时的处理方式。默认 skip，避免把未知流误作良性。')
    parser.add_argument('--workers', type=int, default=1,
                        help='PCAP flow 特征转换进程数。1 表示串行；大文件可设为 CPU 核数附近。')
    parser.add_argument('--chunksize', type=int, default=1000,
                        help='多进程转换时每个 worker 一次领取的 flow 数量。')
    parser.add_argument('--packet-chunk-size', type=int, default=0,
                        help='仅 Kitsune：将每条空闲流再切成连续定长包段（如 100），'
                             '段内按包标签多数表决；输出段携带 group 字段供组级划分。'
                             '0 表示保持整条流。')
    parser.add_argument('--skip-malformed-dns', action='store_true',
                        help='丢弃 53 端口且载荷短于 12 字节的畸形 DNS 包（默认关闭；'
                             '丢弃数与 flow 统计写入该类审计记录）。')
    args = parser.parse_args()

    t_start = time.time()

    if args.mini_root:
        if args.pcap:
            parser.error('--mini-root 与 --pcap 互斥（mini 模式输入来自 mini_root）')
        output_dir = args.output_dir
        if output_dir == 'data/CICIDS2017/pcap_processed_v2':
            output_dir = 'data/mini_processed'
        run_mini_mode(
            mini_root=args.mini_root,
            output_dir=output_dir,
            datasets_filter=args.datasets,
            classes_filter=args.classes,
            max_benign=args.max_benign,
            max_attack_per_cat=args.max_attack_per_cat,
            segment_len=args.segment_len,
            max_packet_len=args.max_packet_len,
            gap_threshold=args.gap_threshold,
            host_window=args.host_window,
            min_packets=args.min_packets,
            flow_idle_timeout_seconds=args.flow_idle_timeout_seconds,
            label_time_tolerance_seconds=args.label_time_tolerance_seconds,
            compact=args.compact,
            no_progress=args.no_progress,
            workers=args.workers,
            chunksize=args.chunksize,
            packet_chunk_size=args.packet_chunk_size,
            skip_malformed_dns=args.skip_malformed_dns,
        )
        total_t = time.time() - t_start
        print(f'\n总耗时: {total_t:.1f}s ({total_t / 60:.1f} min)')
        return

    if not args.pcap:
        parser.error('--pcap is required. CSV-only pseudo seq_feature conversion has been removed.')

    convert_pcaps_with_cicids_labels(
        pcaps=args.pcap,
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        max_benign=args.max_benign,
        max_attack_per_cat=args.max_attack_per_cat,
        segment_len=args.segment_len,
        max_packet_len=args.max_packet_len,
        gap_threshold=args.gap_threshold,
        host_window=args.host_window,
        min_packets=args.min_packets,
        compact=args.compact,
        no_progress=args.no_progress,
        unmatched_policy=args.unmatched_policy,
        workers=args.workers,
        chunksize=args.chunksize,
        flow_idle_timeout_seconds=args.flow_idle_timeout_seconds,
        label_time_tolerance_seconds=args.label_time_tolerance_seconds,
        skip_malformed_dns=args.skip_malformed_dns,
    )
    total_t = time.time() - t_start
    print(f'\n总耗时: {total_t:.1f}s ({total_t / 60:.1f} min)')


if __name__ == '__main__':
    main()
