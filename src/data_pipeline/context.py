"""Label-agnostic causal context features for chronological network flows."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


HOST_FEATURE_NAMES = (
    "src_send_flows_log",
    "src_send_packets_log",
    "src_send_bytes_log",
    "src_receive_flows_log",
    "src_receive_packets_log",
    "src_receive_bytes_log",
    "dst_send_flows_log",
    "dst_send_packets_log",
    "dst_send_bytes_log",
    "dst_receive_flows_log",
    "dst_receive_packets_log",
    "dst_receive_bytes_log",
)

NODE_FEATURE_NAMES = (
    "src_out_flow_count_log",
    "src_in_flow_count_log",
    "src_out_peer_count_log",
    "src_in_peer_count_log",
    "dst_out_flow_count_log",
    "dst_in_flow_count_log",
    "dst_out_peer_count_log",
    "dst_in_peer_count_log",
    "src_short_flow_share",
    "src_long_flow_share",
    "dst_short_flow_share",
    "dst_long_flow_share",
    "src_window_flow_share",
    "dst_window_flow_share",
    "src_window_byte_share",
    "dst_window_byte_share",
)

EDGE_FEATURE_NAMES = (
    "pair_flow_count_log",
    "reverse_flow_count_log",
    "pair_window_flow_share",
    "reverse_window_flow_share",
    "pair_average_packets_log",
    "pair_average_bytes_log",
    "pair_average_duration_log",
    "src_port_diversity_log",
    "dst_port_diversity_log",
    "pair_short_flow_share",
    "pair_long_flow_share",
    "pair_mean_interarrival_log",
    "pair_latest_interarrival_log",
    "bidirectional_flow_ratio",
    "pair_window_packet_share",
    "pair_window_byte_share",
)


def _log1p(value: float) -> float:
    return math.log1p(max(float(value), 0.0))


def _share(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, float(numerator) / float(denominator)))


@dataclass(frozen=True)
class TimestampDiagnostics:
    """Ordering-evidence quality of the flow timestamps fed to the builder."""

    flow_count: int
    distinct_timestamps: int
    largest_tie_group: int
    missing_start_ts: int

    @property
    def tie_ratio(self) -> float:
        if self.flow_count <= 0:
            return 0.0
        return min(1.0, self.largest_tie_group / float(self.flow_count))

    @property
    def degenerate(self) -> bool:
        """Too few distinct anchors: prior context is essentially empty."""
        if self.flow_count <= 0:
            return False
        return self.distinct_timestamps <= max(1, self.flow_count // 10)


@dataclass(frozen=True)
class CausalContextTable:
    host: np.ndarray
    node: np.ndarray
    edge: np.ndarray
    window_ids: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_delta_t: np.ndarray  # float32 [rows, neighbor_count]，无效邻居为 0
    host_feature_names: Tuple[str, ...] = HOST_FEATURE_NAMES
    node_feature_names: Tuple[str, ...] = NODE_FEATURE_NAMES
    edge_feature_names: Tuple[str, ...] = EDGE_FEATURE_NAMES
    timestamp_quality: TimestampDiagnostics = field(
        default_factory=lambda: TimestampDiagnostics(0, 0, 0, 0))


@dataclass
class _HostState:
    send_flows: int = 0
    send_packets: int = 0
    send_bytes: float = 0.0
    receive_flows: int = 0
    receive_packets: int = 0
    receive_bytes: float = 0.0
    out_peers: Set[int] = field(default_factory=set)
    in_peers: Set[int] = field(default_factory=set)
    short_flows: int = 0
    long_flows: int = 0

    @property
    def total_flows(self) -> int:
        return self.send_flows + self.receive_flows

    @property
    def total_bytes(self) -> float:
        return self.send_bytes + self.receive_bytes


@dataclass
class _PairState:
    flow_count: int = 0
    packets: int = 0
    total_bytes: float = 0.0
    duration: float = 0.0
    short_flows: int = 0
    long_flows: int = 0
    src_ports: Set[int] = field(default_factory=set)
    dst_ports: Set[int] = field(default_factory=set)
    interarrival_sum: float = 0.0
    interarrival_count: int = 0
    last_start_ts: float | None = None


class _CausalWindowState:
    def __init__(self, long_flow_packets: int):
        self.long_flow_packets = int(long_flow_packets)
        self.hosts: Dict[int, _HostState] = defaultdict(_HostState)
        self.pairs: Dict[Tuple[int, int], _PairState] = defaultdict(_PairState)
        self.flow_count = 0
        self.packet_count = 0
        self.total_bytes = 0.0

    @staticmethod
    def _flow_values(flow: dict) -> Tuple[int, int, int, int, int, int, float, float, float]:
        flow_id = flow.get("id")
        five_tuple = flow.get("five_tuple")
        nonseq = flow.get("nonseq_feature")
        if not isinstance(flow_id, list) or len(flow_id) < 2:
            raise ValueError("Every flow must contain id=[source, destination].")
        if not isinstance(five_tuple, list) or len(five_tuple) < 5:
            raise ValueError("Every flow must contain a five_tuple with five fields.")
        if not isinstance(nonseq, list) or len(nonseq) != 6:
            raise ValueError("Every flow must contain a six-dimensional nonseq_feature.")
        start_ts = float(flow.get("start_ts",0))
        end_ts = float(flow.get("end_ts", start_ts))
        if not math.isfinite(start_ts) or not math.isfinite(end_ts):
            raise ValueError("Flow timestamps must be finite.")
        total_bytes = math.expm1(float(nonseq[2]))
        if not math.isfinite(total_bytes):
            raise ValueError("nonseq_feature[2] must be finite log1p(total_bytes).")
        return (
            int(flow_id[0]),
            int(flow_id[1]),
            int(five_tuple[3]),
            int(five_tuple[4]),
            max(0, int(flow.get("len", 0))),
            int(five_tuple[0]),
            max(0.0, total_bytes),
            max(0.0, end_ts - start_ts),
            start_ts,
        )

    @staticmethod
    def _host_vector(src_state: _HostState, dst_state: _HostState) -> np.ndarray:
        values = (
            src_state.send_flows,
            src_state.send_packets,
            src_state.send_bytes,
            src_state.receive_flows,
            src_state.receive_packets,
            src_state.receive_bytes,
            dst_state.send_flows,
            dst_state.send_packets,
            dst_state.send_bytes,
            dst_state.receive_flows,
            dst_state.receive_packets,
            dst_state.receive_bytes,
        )
        return np.asarray([_log1p(value) for value in values], dtype=np.float32)

    def _node_vector(self, src_state: _HostState, dst_state: _HostState) -> np.ndarray:
        src_short_long = src_state.short_flows + src_state.long_flows
        dst_short_long = dst_state.short_flows + dst_state.long_flows
        host_touch_count = max(2 * self.flow_count, 1)
        byte_touch_count = max(2.0 * self.total_bytes, 1.0)
        return np.asarray(
            [
                _log1p(src_state.send_flows),
                _log1p(src_state.receive_flows),
                _log1p(len(src_state.out_peers)),
                _log1p(len(src_state.in_peers)),
                _log1p(dst_state.send_flows),
                _log1p(dst_state.receive_flows),
                _log1p(len(dst_state.out_peers)),
                _log1p(len(dst_state.in_peers)),
                _share(src_state.short_flows, src_short_long),
                _share(src_state.long_flows, src_short_long),
                _share(dst_state.short_flows, dst_short_long),
                _share(dst_state.long_flows, dst_short_long),
                _share(src_state.total_flows, host_touch_count),
                _share(dst_state.total_flows, host_touch_count),
                _share(src_state.total_bytes, byte_touch_count),
                _share(dst_state.total_bytes, byte_touch_count),
            ],
            dtype=np.float32,
        )

    def _edge_vector(self, pair: _PairState, reverse: _PairState, start_ts: float) -> np.ndarray:
        count = pair.flow_count
        pair_short_long = pair.short_flows + pair.long_flows
        latest_gap = 0.0 if pair.last_start_ts is None else max(0.0, start_ts - pair.last_start_ts)
        mean_gap = pair.interarrival_sum / max(pair.interarrival_count, 1)
        direction_total = pair.flow_count + reverse.flow_count
        return np.asarray(
            [
                _log1p(pair.flow_count),
                _log1p(reverse.flow_count),
                _share(pair.flow_count, self.flow_count),
                _share(reverse.flow_count, self.flow_count),
                _log1p(pair.packets / max(count, 1)),
                _log1p(pair.total_bytes / max(count, 1)),
                _log1p(pair.duration / max(count, 1)),
                _log1p(max(len(pair.src_ports) - 1, 0)),
                _log1p(max(len(pair.dst_ports) - 1, 0)),
                _share(pair.short_flows, pair_short_long),
                _share(pair.long_flows, pair_short_long),
                _log1p(mean_gap),
                _log1p(latest_gap),
                _share(min(pair.flow_count, reverse.flow_count) * 2, direction_total),
                _share(pair.packets, self.packet_count),
                _share(pair.total_bytes, self.total_bytes),
            ],
            dtype=np.float32,
        )

    def features_before_insert(self, flow: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        src, dst, _, _, _, _, _, _, start_ts = self._flow_values(flow)
        src_state = self.hosts[src]
        dst_state = self.hosts[dst]
        pair = self.pairs[(src, dst)]
        reverse = self.pairs[(dst, src)]
        return (
            self._host_vector(src_state, dst_state),
            self._node_vector(src_state, dst_state),
            self._edge_vector(pair, reverse, start_ts),
        )

    def insert(self, flow: dict) -> None:
        src, dst, src_port, dst_port, packets, _, total_bytes, duration, start_ts = self._flow_values(flow)
        src_state = self.hosts[src]
        dst_state = self.hosts[dst]
        src_state.send_flows += 1
        src_state.send_packets += packets
        src_state.send_bytes += total_bytes
        src_state.out_peers.add(dst)
        dst_state.receive_flows += 1
        dst_state.receive_packets += packets
        dst_state.receive_bytes += total_bytes
        dst_state.in_peers.add(src)
        target_is_long = packets >= self.long_flow_packets
        for state in (src_state, dst_state):
            if target_is_long:
                state.long_flows += 1
            else:
                state.short_flows += 1

        pair = self.pairs[(src, dst)]
        if pair.last_start_ts is not None:
            pair.interarrival_sum += max(0.0, start_ts - pair.last_start_ts)
            pair.interarrival_count += 1
        pair.last_start_ts = start_ts
        pair.flow_count += 1
        pair.packets += packets
        pair.total_bytes += total_bytes
        pair.duration += duration
        pair.src_ports.add(src_port)
        pair.dst_ports.add(dst_port)
        if target_is_long:
            pair.long_flows += 1
        else:
            pair.short_flows += 1

        self.flow_count += 1
        self.packet_count += packets
        self.total_bytes += total_bytes


def _flow_start_ts(flow: dict) -> Tuple[float, int]:
    """Return (start_ts, missing_flag); absent start_ts is treated as 0.0."""
    value = flow.get("start_ts")
    if value is None:
        return 0.0, 1
    ts = float(value)
    if not math.isfinite(ts):
        raise ValueError("Flow start_ts must be finite.")
    return ts, 0


def iter_timestamp_groups(flows: Sequence[dict]) -> List[List[int]]:
    """Group flow indices by identical start_ts, ordered by timestamp.

    Equal timestamps carry no ordering evidence, so every caller must treat a
    whole group as simultaneous: group members read context before any member
    is inserted, and never appear in each other's causal history. Row order
    inside a group stays deterministic but causally irrelevant.
    """
    order = sorted(range(len(flows)), key=lambda i: _flow_start_ts(flows[i])[0])
    groups: List[List[int]] = []
    current: List[int] = []
    current_ts = None
    for index in order:
        ts = _flow_start_ts(flows[index])[0]
        if current and ts != current_ts:
            groups.append(current)
            current = []
        current.append(index)
        current_ts = ts
    if current:
        groups.append(current)
    return groups


def build_causal_context_table(
    flows: Sequence[dict],
    graph_window: float = 10.0,
    long_flow_packets: int = 15,
    neighbor_count: int = 8,
) -> CausalContextTable:
    """Build context and graph-neighbor indices from prior flows in each window."""
    if not math.isfinite(float(graph_window)) or graph_window <= 0:
        raise ValueError("graph_window must be a finite positive number.")
    if int(long_flow_packets) <= 0:
        raise ValueError("long_flow_packets must be positive.")
    if int(neighbor_count) <= 0:
        raise ValueError("neighbor_count must be positive.")

    row_count = len(flows)
    host = np.zeros((row_count, len(HOST_FEATURE_NAMES)), dtype=np.float32)
    node = np.zeros((row_count, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    edge = np.zeros((row_count, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    window_ids = np.zeros(row_count, dtype=np.int64)
    neighbor_indices = np.full((row_count, int(neighbor_count)), -1, dtype=np.int64)
    neighbor_delta_t = np.zeros((row_count, int(neighbor_count)), dtype=np.float32)
    if not flows:
        return CausalContextTable(host, node, edge, window_ids, neighbor_indices, neighbor_delta_t)

    missing_start_ts = 0
    for index, flow in enumerate(flows):
        start_ts, missing = _flow_start_ts(flow)
        missing_start_ts += missing
        window_ids[index] = math.floor(start_ts / graph_window)

    states: Dict[int, _CausalWindowState] = {}
    histories: Dict[int, Dict[str, Dict[object, list[int]]]] = {}
    groups = iter_timestamp_groups(flows)
    diagnostics = TimestampDiagnostics(
        flow_count=row_count,
        distinct_timestamps=len(groups),
        largest_tie_group=max(len(group) for group in groups),
        missing_start_ts=missing_start_ts,
    )
    if diagnostics.degenerate:
        logging.warning(
            "Causal context degraded: %d flows share %d distinct timestamp(s) "
            "(missing_start_ts=%d); no ordering evidence exists, so every flow "
            "receives empty prior context. Fix the upstream converter instead of "
            "interpreting graph features on this capture.",
            row_count, diagnostics.distinct_timestamps, missing_start_ts,
        )
    elif diagnostics.largest_tie_group > 1:
        logging.info(
            "Causal context tie groups: distinct_timestamps=%d largest_tie_group=%d; "
            "same-timestamp flows are treated as simultaneous and never observe "
            "each other as history.",
            diagnostics.distinct_timestamps, diagnostics.largest_tie_group,
        )
    if missing_start_ts:
        logging.warning(
            "%d flow(s) lack start_ts and were placed at ts=0; their causal "
            "context is not trustworthy.", missing_start_ts,
        )

    for group in groups:
        # Read phase: every member observes only flows from strictly earlier
        # groups; group members never appear in each other's history.
        pending = []
        for index in group:
            window_id = int(window_ids[index])
            state = states.setdefault(
                window_id,
                _CausalWindowState(long_flow_packets=long_flow_packets),
            )
            history = histories.setdefault(
                window_id,
                {"host": defaultdict(list), "pair": defaultdict(list)},
            )
            src, dst, _, _, _, _, _, _, _ = state._flow_values(flows[index])

            candidates: list[int] = []
            for history_key in ((src, dst), (dst, src)):
                candidates.extend(reversed(history["pair"][history_key]))
            for endpoint in (src, dst):
                candidates.extend(reversed(history["host"][endpoint]))
            selected: list[int] = []
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) == neighbor_count:
                    break

            host_row, node_row, edge_row = state.features_before_insert(flows[index])
            pending.append((index, window_id, src, dst, host_row, node_row, edge_row, selected))

        # Write phase: materialize features, then insert members and extend
        # histories in deterministic (row) order inside the group.
        for index, _, _, _, host_row, node_row, edge_row, selected in pending:
            host[index] = host_row
            node[index] = node_row
            edge[index] = edge_row
            neighbor_indices[index, :len(selected)] = selected
            current_ts = _flow_start_ts(flows[index])[0]
            for slot, neighbor_idx in enumerate(selected):
                if neighbor_idx >= 0:
                    neighbor_ts = _flow_start_ts(flows[neighbor_idx])[0]
                    neighbor_delta_t[index, slot] = max(0.0, current_ts - neighbor_ts)
        for index, window_id, src, dst, *_ in pending:
            state = states[window_id]
            history = histories[window_id]
            state.insert(flows[index])
            for endpoint in (src, dst):
                history["host"][endpoint].append(index)
                if len(history["host"][endpoint]) > neighbor_count:
                    del history["host"][endpoint][0]
            history["pair"][(src, dst)].append(index)
            if len(history["pair"][(src, dst)]) > neighbor_count:
                del history["pair"][(src, dst)][0]

    for array in (host, node, edge, window_ids, neighbor_indices, neighbor_delta_t):
        array.setflags(write=False)
    return CausalContextTable(host, node, edge, window_ids, neighbor_indices, neighbor_delta_t,
                              timestamp_quality=diagnostics)
