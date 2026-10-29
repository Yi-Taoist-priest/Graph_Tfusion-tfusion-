"""模态特征构造：包序列、流级统计与主机因果统计。"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Sequence, Tuple

import torch

from .constants import MAX_LOG1P_FLOAT64, VOID_SEQ_FEATURE


def _normalize_seq_features(flows: List[dict], segment_len: int) -> List[List[List[int]]]:
    normalized = []
    for flow in flows:
        seq = [list(x) for x in flow.get("seq_feature", [])]
        if len(seq) < segment_len:
            seq += [VOID_SEQ_FEATURE] * (segment_len - len(seq))
        else:
            seq = seq[:segment_len]
        normalized.append(seq)
    return normalized


def _flow_ids(flows: Sequence[dict]) -> List[List[int]]:
    ids: List[List[int]] = []
    for flow in flows:
        flow_id = flow.get("id", [0, 0])
        if not isinstance(flow_id, list) or len(flow_id) < 2:
            ids.append([0, 0])
            continue
        ids.append([int(flow_id[0]), int(flow_id[1])])
    return ids


def _flow_sizes(flows: Sequence[dict]) -> List[int]:
    return [max(1, int(flow.get("len", 1))) for flow in flows]


def _flow_metrics(flows: Sequence[dict]) -> List[dict]:
    metrics = []
    for index, flow in enumerate(flows):
        nonseq = flow.get("nonseq_feature") or []
        five_tuple = flow.get("five_tuple")
        if len(nonseq) != 6 or not all(math.isfinite(float(value)) for value in nonseq):
            raise ValueError(
                f"Flow {index} has an invalid six-dimensional nonseq_feature; "
                "regenerate this dataset with convert_cicids_pcap.py."
            )
        total_log_bytes = float(nonseq[2])
        if total_log_bytes < 0 or total_log_bytes > MAX_LOG1P_FLOAT64:
            raise ValueError(
                f"Flow {index} has nonseq_feature[2]={nonseq[2]!r}; expected "
                "log1p(total_bytes). Regenerate this dataset with convert_cicids_pcap.py."
            )
        if not isinstance(five_tuple, list) or len(five_tuple) < 5:
            raise ValueError(
                f"Flow {index} has no five_tuple; regenerate this dataset with convert_cicids_pcap.py."
            )
        total_bytes = math.expm1(total_log_bytes)
        start_ts = float(flow.get("start_ts", 0.0))
        end_ts = float(flow.get("end_ts", start_ts))
        metrics.append({
            "total_bytes": max(0.0, total_bytes),
            "duration": max(0.0, end_ts - start_ts),
            "protocol": int(five_tuple[0]),
            "src_port": int(five_tuple[3]),
            "dst_port": int(five_tuple[4]),
        })
    return metrics


def build_causal_host_features(flows: Sequence[dict], host_window: float = 10.0) -> List[List[float]]:
    """Build host statistics using only flows that started earlier in the same window."""
    if not flows:
        return []
    base_ts = min(float(flow.get("start_ts", 0.0)) for flow in flows)
    window = max(float(host_window), 1e-9)
    stats = defaultdict(lambda: {"send": [0.0, 0.0, 0.0], "recv": [0.0, 0.0, 0.0]})
    result = [[0.0] * 12 for _ in flows]
    order = sorted(range(len(flows)), key=lambda i: float(flows[i].get("start_ts", 0.0)))
    metrics = _flow_metrics(flows)
    for index in order:
        flow = flows[index]
        src, dst = _flow_ids([flow])[0]
        wid = int((float(flow.get("start_ts", 0.0)) - base_ts) // window)
        src_stats = stats[(wid, src)]
        dst_stats = stats[(wid, dst)]
        values = src_stats["send"] + src_stats["recv"] + dst_stats["send"] + dst_stats["recv"]
        result[index] = [math.log1p(value) for value in values]

        packet_count = float(max(1, int(flow.get("len", 1))))
        total_bytes = float(metrics[index]["total_bytes"])
        for target, direction in ((src_stats, "send"), (dst_stats, "recv")):
            target[direction][0] += 1.0
            target[direction][1] += packet_count
            target[direction][2] += total_bytes
    return result


def _extract_modal_features(
    flows: List[dict],
    segment_len: int,
    host_window: float = 10.0,
    compute_host: bool = True,
):
    """构造三模态特征；下游使用预计算上下文时跳过 host 统计。"""
    seq_features = _normalize_seq_features(flows, segment_len)
    non_seq_features = [x["nonseq_feature"] for x in flows]
    if compute_host:
        interact_features = build_causal_host_features(flows, host_window=host_window)
    else:
        interact_features = [[0.0] * 12 for _ in flows]
    flow_sizes = _flow_sizes(flows)
    flow_ids = _flow_ids(flows)
    return seq_features, non_seq_features, interact_features, flow_sizes, flow_ids


def _tensor_or_empty(rows: List, shape_tail: Tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
    if rows:
        return torch.tensor(rows, dtype=dtype)
    return torch.empty((0,) + shape_tail, dtype=dtype)
