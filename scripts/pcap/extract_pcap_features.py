#!/usr/bin/env python3
"""
Extract tFusion-style features directly from PCAP/PCAPNG files.

This script follows the feature definitions in the tFusion paper as closely as
the released code interface allows:

1. Aggregate packets into flows by the five tuple:
   (transport protocol, source IP, destination IP, source port, destination port).
2. Packet sequence feature:
   v_j = p * M + min(length_j, M) - 1
   u_j = 0 for the first packet, otherwise 2*j or 2*j+1 depending on whether
   the inter-arrival gap is smaller than the configured threshold T.
3. Flow feature, six dimensions:
   log([N, FCT, sum(length), min(length), max(length), mean(length)] + 1).
4. Host interaction feature, 12 dimensions:
   log(Cat(send(src), receive(src), send(dst), receive(dst)) + 1), where
   send/receive(host) = [flow_count, packet_count_sum, byte_count_sum] within
   the same arrival-time window.

The output JSON is compatible with this repository's data_pipeline feature contract:
{
  "benign": [{"seq_feature": ..., "nonseq_feature": ..., "interact_feature": ...}],
  "attack": [...]
}
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


SEGMENT_LEN = 100
MAX_PACKET_LEN = 1500
DEFAULT_GAP_THRESHOLD = 1.0
DEFAULT_HOST_WINDOW = 10.0


FiveTuple = Tuple[int, str, str, int, int]

_WORKER_HOST_WINDOWS = None
_WORKER_BASE_TS = 0.0
_WORKER_SEGMENT_LEN = SEGMENT_LEN
_WORKER_MAX_PACKET_LEN = MAX_PACKET_LEN
_WORKER_GAP_THRESHOLD = DEFAULT_GAP_THRESHOLD
_WORKER_HOST_WINDOW = DEFAULT_HOST_WINDOW
_WORKER_DEFAULT_LABEL = "BENIGN"
_WORKER_LABEL_MAP: Dict[str, str] = {}


class ProgressBar:
    """Small stderr progress bar to avoid adding a runtime dependency."""

    def __init__(
        self,
        label: str,
        total: Optional[int] = None,
        unit: str = "items",
        enabled: bool = True,
        width: int = 28,
        min_interval: float = 0.25,
    ) -> None:
        self.label = label
        self.total = total if total and total > 0 else None
        self.unit = unit
        self.enabled = enabled
        self.width = width
        self.min_interval = min_interval
        self.current = 0
        self.start_ts = time.monotonic()
        self.last_render_ts = 0.0
        self.last_message_len = 0

    def update(
        self,
        current: Optional[int] = None,
        advance: int = 0,
        detail: str = "",
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        if current is not None:
            self.current = max(0, current)
        elif advance:
            self.current += advance

        now = time.monotonic()
        if not force and now - self.last_render_ts < self.min_interval:
            return
        self.last_render_ts = now
        self._render(detail)

    def close(self, detail: str = "") -> None:
        if not self.enabled:
            return
        if self.total is not None:
            self.current = min(max(self.current, self.total), self.total)
        self._render(detail, final=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self, detail: str = "", final: bool = False) -> None:
        elapsed = max(time.monotonic() - self.start_ts, 1e-9)
        rate = self.current / elapsed
        if self.total is None:
            bar = "." * self.width
            progress = f"{self.current:,} {self.unit}"
        else:
            ratio = min(max(self.current / self.total, 0.0), 1.0)
            filled = int(round(self.width * ratio))
            bar = "#" * filled + "-" * (self.width - filled)
            progress = f"{ratio * 100:6.2f}% {self.current:,}/{self.total:,} {self.unit}"
        message = f"\r{self.label} [{bar}] {progress} {rate:,.1f} {self.unit}/s"
        if detail:
            message += f" | {detail}"
        if final:
            message += " | done"
        padding = max(0, self.last_message_len - len(message))
        sys.stderr.write(message + (" " * padding))
        sys.stderr.flush()
        self.last_message_len = len(message)


@dataclass
class PacketPoint:
    ts: float
    length: int
    # 0 = 与所属流首包同向（上行），1 = 反向（下行）。方向在流层确定，
    # 见 ``collect_flows``。
    direction: int = 0


@dataclass
class FlowRecord:
    key: FiveTuple
    packets: List[PacketPoint] = field(default_factory=list)
    # Parent-flow identity for chunked segmentations (e.g. Kitsune fixed-packet
    # chunks). Segments sharing one group must never be split across train/test.
    group: Optional[str] = None

    @property
    def proto(self) -> int:
        return self.key[0]

    @property
    def src(self) -> str:
        return self.key[1]

    @property
    def dst(self) -> str:
        return self.key[2]

    @property
    def sport(self) -> int:
        return self.key[3]

    @property
    def dport(self) -> int:
        return self.key[4]

    @property
    def start_ts(self) -> float:
        return self.packets[0].ts

    @property
    def end_ts(self) -> float:
        return self.packets[-1].ts

    @property
    def packet_count(self) -> int:
        return len(self.packets)

    @property
    def byte_count(self) -> int:
        return sum(p.length for p in self.packets)


def split_packet_points_by_idle_timeout(
    points: Sequence[PacketPoint],
    idle_timeout_seconds: float,
) -> List[List[PacketPoint]]:
    """Split one five-tuple's packets into flows separated by idle time."""
    if idle_timeout_seconds <= 0:
        raise ValueError("idle_timeout_seconds must be positive.")

    groups: List[List[PacketPoint]] = []
    current: List[PacketPoint] = []
    for point in sorted(points, key=lambda item: item.ts):
        if current and point.ts - current[-1].ts > idle_timeout_seconds:
            groups.append(current)
            current = []
        current.append(point)
    if current:
        groups.append(current)
    return groups


def _require_scapy():
    try:
        from scapy.all import IP, IPv6, PcapReader, TCP, UDP  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: scapy. Install it with:\n"
            "  python -m pip install scapy\n"
            "Then rerun this script."
        ) from exc
    return IP, IPv6, PcapReader, TCP, UDP


def protocol_bucket(ip_proto: int) -> int:
    """Map transport protocol to the 0/1/2 buckets used by 3*M token space."""
    if ip_proto == 6:
        return 0  # TCP
    if ip_proto == 17:
        return 1  # UDP
    return 2      # other IP transport protocols


def ip_to_int_for_project(ip: str) -> int:
    """Return an integer IP id for the existing contrastive/topology code."""
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return 0


def canonical_five_tuple(key: FiveTuple) -> FiveTuple:
    """把双向五元组折叠成规范键，使两个方向映射到同一条流。

    与切分端 `canonical_key` / `convert_cicids_pcap._canonical_pcap_key`
    同口径（首序 IP/端口在前），保证转换端与切分端对"同一条流"的认定一致。
    """
    proto, src, dst, sport, dport = key
    if (src, sport) <= (dst, dport):
        return key
    return (proto, dst, src, dport, sport)


def is_malformed_dns_payload(payload: bytes) -> bool:
    """DNS 报文头固定 12 字节，短于该长度视为截断/畸形的 DNS 载荷。"""
    return len(payload) < 12


def _reader_file_position(reader) -> Optional[int]:
    for attr in ("f", "file", "fileobj", "_file", "fd"):
        handle = getattr(reader, attr, None)
        if handle is None or not hasattr(handle, "tell"):
            continue
        try:
            return int(handle.tell())
        except (OSError, ValueError):
            return None
    return None


def packet_iter_from_pcap(
    path: Path,
    progress: Optional[ProgressBar] = None,
    byte_offset: int = 0,
    progress_detail: Optional[Callable[[int], str]] = None,
    packet_counter: Optional[List[int]] = None,
    skip_malformed_dns: bool = False,
    dropped_counter: Optional[List[int]] = None,
):
    IP, IPv6, PcapReader, TCP, UDP = _require_scapy()

    def progress_update(reader, packet_count: int) -> None:
        if progress is None:
            return
        pos = _reader_file_position(reader)
        detail = progress_detail(packet_count) if progress_detail else ""
        if pos is None:
            if progress.total is not None:
                progress.total = None
                progress.unit = "packets"
            current = packet_counter[0] if packet_counter is not None else packet_count
            progress.update(current=current, detail=detail)
        else:
            progress.update(current=byte_offset + pos, detail=detail)

    with PcapReader(str(path)) as reader:
        packet_count = 0
        for pkt in reader:
            packet_count += 1
            if packet_counter is not None:
                packet_counter[0] += 1
            progress_update(reader, packet_count)

            ip_layer = pkt.getlayer(IP) or pkt.getlayer(IPv6)
            if ip_layer is None:
                continue

            ip_proto = int(getattr(ip_layer, "proto", getattr(ip_layer, "nh", 0)))
            src = str(ip_layer.src)
            dst = str(ip_layer.dst)

            sport = 0
            dport = 0
            payload = b""
            if pkt.haslayer(TCP):
                tcp = pkt.getlayer(TCP)
                sport = int(tcp.sport)
                dport = int(tcp.dport)
                payload = bytes(getattr(tcp, "payload", b""))
            elif pkt.haslayer(UDP):
                udp = pkt.getlayer(UDP)
                sport = int(udp.sport)
                dport = int(udp.dport)
                payload = bytes(getattr(udp, "payload", b""))

            # 默认关闭：只有显式开启才丢弃畸形的 53 端口载荷，且计入统计。
            if skip_malformed_dns and (sport == 53 or dport == 53) \
                    and is_malformed_dns_payload(payload):
                if dropped_counter is not None:
                    dropped_counter[0] += 1
                continue

            ts = float(pkt.time)
            # 截断捕获（如 CTU-13 truncated 完整版）每包只保留头部字节，
            # len(ip_layer) 是缓冲区实有字节；IP 头总长字段在截断时仍完整，
            # 取两者较大值还原原始包长（非截断 pcap 两者相等，行为不变）。
            if isinstance(ip_layer, IPv6):
                ip_total = 40 + int(ip_layer.plen)
            else:
                ip_total = int(ip_layer.len)
            length = max(int(len(ip_layer)), ip_total)
            key = (ip_proto, src, dst, sport, dport)
            yield key, ts, length


def _normalise_direction(flow: FlowRecord) -> FlowRecord:
    """让 key 的朝向与该流最早包一致：最早包若是反向，key 与方向整体翻转。"""
    if flow.packets and flow.packets[0].direction == 1:
        proto, src, dst, sport, dport = flow.key
        flow.key = (proto, dst, src, dport, sport)
        for packet in flow.packets:
            packet.direction = 1 - packet.direction
    return flow


def collect_flows(
    paths: Iterable[Path],
    min_packets: int = 1,
    show_progress: bool = True,
    flow_idle_timeout_seconds: float = 120.0,
    skip_malformed_dns: bool = False,
    stats: Optional[Dict] = None,
) -> List[FlowRecord]:
    """按**双向**五元组重建流，并按空闲超时切成多条会话。

    双向：``(proto,A,B,sport,dport)`` 与 ``(proto,B,A,dport,sport)`` 归到同一条流，
    首包方向记为上行（``direction=0``），反向包记为下行；若最早包是下行则整体
    翻转，保证 key 的朝向永远等于最早包方向。这与切分端 `canonical_key`、
    `convert_cicids_pcap` 的 IoT-23 / CSECICIDS2018 路径一致。

    空闲切分保留：五元组会被复用，不做空闲切分会让一次样本横跨任意时长，
    使 ``fct`` / 包数 / 字节数全部失真（预训练脚本去掉了这一步，此处不照搬）。

    ``min_packets`` 默认 1（不过滤）。真实流量里极短流占比很高——实测
    CSECICIDS2018 有 99.8%、Kitsune 73%、CICDDoS2019 51% 的流不足 4 个包，
    取 4 会静默删掉大量合法样本，所以过滤必须显式开启；被过滤的条数记在
    ``stats["flows_dropped_short"]``。

    ``stats`` 传入 dict 时会写入 ``raw_five_tuples`` / ``flows_kept`` /
    ``flows_dropped_short`` / ``min_packets`` / ``malformed_dns_dropped`` 供审计。
    """
    paths = list(paths)
    if flow_idle_timeout_seconds <= 0:
        raise ValueError("flow_idle_timeout_seconds must be positive.")
    total_bytes = sum(path.stat().st_size for path in paths)
    progress = ProgressBar("Reading PCAP", total_bytes, "B", enabled=show_progress)
    bytes_done = 0
    packet_counter = [0]
    dropped_counter = [0]
    dropped_short = 0
    flows: List[FlowRecord] = []
    raw_flow_count = 0

    for file_index, path in enumerate(paths, start=1):
        file_size = path.stat().st_size
        flow_map: Dict[FiveTuple, FlowRecord] = {}

        def _detail(packet_count: int, current_path: Path = path) -> str:
            return f"{file_index}/{len(paths)} {current_path.name} {packet_count:,} packets"

        for observed, ts, length in packet_iter_from_pcap(
            path,
            progress=progress,
            byte_offset=bytes_done,
            progress_detail=_detail,
            packet_counter=packet_counter,
            skip_malformed_dns=skip_malformed_dns,
            dropped_counter=dropped_counter,
        ):
            lookup = canonical_five_tuple(observed)
            flow = flow_map.get(lookup)
            if flow is None:
                # 首包定上行；另一个方向共用同一 lookup 键，记为 direction=1。
                flow = FlowRecord(key=observed)
                flow_map[lookup] = flow
            direction = 0 if observed == flow.key else 1
            flow.packets.append(PacketPoint(ts=ts, length=length, direction=direction))
        raw_flow_count += len(flow_map)
        for flow in flow_map.values():
            flow.packets.sort(key=lambda point: point.ts)
            _normalise_direction(flow)
            for packet_group in split_packet_points_by_idle_timeout(
                flow.packets,
                idle_timeout_seconds=flow_idle_timeout_seconds,
            ):
                if len(packet_group) < min_packets:
                    dropped_short += 1
                    continue
                # 分段后最早包可能变成下行，需要按段重新定向。
                flows.append(_normalise_direction(
                    FlowRecord(key=flow.key, packets=list(packet_group))
                ))
        bytes_done += file_size
        progress.update(
            current=bytes_done,
            detail=f"{file_index}/{len(paths)} {path.name}",
            force=True,
        )
    progress.close(detail=f"{raw_flow_count:,} raw five-tuples")
    flows.sort(key=lambda f: f.start_ts)
    if stats is not None:
        stats["raw_five_tuples"] = raw_flow_count
        stats["flows_kept"] = len(flows)
        stats["flows_dropped_short"] = dropped_short
        stats["min_packets"] = int(min_packets)
        stats["malformed_dns_dropped"] = dropped_counter[0]
        stats["skip_malformed_dns"] = bool(skip_malformed_dns)
    return flows


def sample_packet_points(
    points: Sequence[PacketPoint],
    segment_len: int,
    strategy: str = "head",
) -> List[PacketPoint]:
    """Select a bounded, deterministic observation from a packet sequence."""
    if segment_len <= 0:
        raise ValueError("segment_len must be positive.")
    if strategy not in {"head", "uniform", "head_tail"}:
        raise ValueError("packet sampling strategy must be 'head', 'uniform', or 'head_tail'.")
    points = list(points)
    if len(points) <= segment_len:
        return points
    if strategy == "head":
        return points[:segment_len]
    if strategy == "uniform":
        if segment_len == 1:
            return [points[0]]
        indices = [round(i * (len(points) - 1) / (segment_len - 1)) for i in range(segment_len)]
        return [points[index] for index in indices]
    head_count = (segment_len + 1) // 2
    tail_count = segment_len - head_count
    return points[:head_count] + (points[-tail_count:] if tail_count else [])


def make_seq_feature(
    flow: FlowRecord,
    segment_len: int,
    max_packet_len: int,
    gap_threshold: float,
    packet_sampling_strategy: str = "head",
) -> List[List[int]]:
    seq: List[List[int]] = []
    proto_id = protocol_bucket(flow.proto)
    prev_ts: Optional[float] = None

    sampled_packets = sample_packet_points(
        flow.packets, segment_len, strategy=packet_sampling_strategy
    )
    for idx, pkt in enumerate(sampled_packets):
        # tFusion Eq. (1): packet length/protocol token.  Padding uses the
        # same [0, 0] value as the released tFusion implementation.
        length_token = proto_id * max_packet_len + min(pkt.length, max_packet_len) - 1
        length_token = max(0, min(3 * max_packet_len - 1, length_token))

        if idx == 0 or prev_ts is None:
            time_token = 0
        else:
            gap = pkt.ts - prev_ts
            time_token = 2 * idx + 1 if gap_threshold <= gap else 2 * idx
        time_token = max(0, min(2 * segment_len - 1, time_token))
        seq.append([int(length_token), int(time_token)])
        prev_ts = pkt.ts

    if len(seq) < segment_len:
        seq.extend([[0, 0]] * (segment_len - len(seq)))
    return seq


def make_flow_feature(flow: FlowRecord) -> List[float]:
    lengths = [p.length for p in flow.packets]
    n = len(lengths)
    fct = max(0.0, flow.end_ts - flow.start_ts)
    total = float(sum(lengths))
    min_len = float(min(lengths)) if lengths else 0.0
    max_len = float(max(lengths)) if lengths else 0.0
    mean_len = total / max(n, 1)
    return [math.log1p(x) for x in [n, fct, total, min_len, max_len, mean_len]]


def build_host_windows(
    flows: List[FlowRecord],
    host_window: float,
    show_progress: bool = True,
) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """
    Build per-window host sending/receiving statistics.

    Returned shape:
      window_id -> host -> {
          "send": [flow_count, packet_count_sum, byte_count_sum],
          "recv": [flow_count, packet_count_sum, byte_count_sum],
      }

    按**方向**记账：只有存在正向包时才给源主机的 send / 目标主机的 recv 计数，
    只有存在反向包时才给目标主机的 send / 源主机的 recv 计数；累计的是各自
    方向的包数与字节数。这样"没有回包的流"不会凭空抬高目标主机的 recv。
    """
    windows: Dict[int, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"send": [0.0, 0.0, 0.0], "recv": [0.0, 0.0, 0.0]})
    )
    if not flows:
        return windows

    base_ts = flows[0].start_ts
    window = max(host_window, 1e-9)
    progress = ProgressBar(
        "Building host stats",
        len(flows),
        "flows",
        enabled=show_progress,
    )
    for flow in flows:
        wid = int((flow.start_ts - base_ts) // window)
        forward_packets = forward_bytes = 0
        reverse_packets = reverse_bytes = 0
        for packet in flow.packets:
            if packet.direction == 0:
                forward_packets += 1
                forward_bytes += packet.length
            else:
                reverse_packets += 1
                reverse_bytes += packet.length

        if forward_packets:
            src_send = windows[wid][flow.src]["send"]
            dst_recv = windows[wid][flow.dst]["recv"]
            src_send[0] += 1.0
            src_send[1] += float(forward_packets)
            src_send[2] += float(forward_bytes)
            dst_recv[0] += 1.0
            dst_recv[1] += float(forward_packets)
            dst_recv[2] += float(forward_bytes)
        if reverse_packets:
            dst_send = windows[wid][flow.dst]["send"]
            src_recv = windows[wid][flow.src]["recv"]
            dst_send[0] += 1.0
            dst_send[1] += float(reverse_packets)
            dst_send[2] += float(reverse_bytes)
            src_recv[0] += 1.0
            src_recv[1] += float(reverse_packets)
            src_recv[2] += float(reverse_bytes)
        progress.update(advance=1)
    progress.close(detail=f"{len(windows):,} windows")
    return windows


def make_host_feature(
    flow: FlowRecord,
    window_stats: Dict[int, Dict[str, Dict[str, List[float]]]],
    base_ts: float,
    host_window: float,
) -> List[float]:
    wid = int((flow.start_ts - base_ts) // max(host_window, 1e-9))
    stats = window_stats.get(wid, {})

    src_stats = stats.get(flow.src, {"send": [0.0, 0.0, 0.0], "recv": [0.0, 0.0, 0.0]})
    dst_stats = stats.get(flow.dst, {"send": [0.0, 0.0, 0.0], "recv": [0.0, 0.0, 0.0]})
    values = (
        src_stats["send"]
        + src_stats["recv"]
        + dst_stats["send"]
        + dst_stats["recv"]
    )
    return [math.log1p(x) for x in values]


def make_flow_item(
    flow: FlowRecord,
    host_windows: Dict[int, Dict[str, Dict[str, List[float]]]],
    base_ts: float,
    segment_len: int,
    max_packet_len: int,
    gap_threshold: float,
    host_window: float,
) -> dict:
    return {
        "seq_feature": make_seq_feature(flow, segment_len, max_packet_len, gap_threshold),
        "nonseq_feature": make_flow_feature(flow),
        "interact_feature": make_host_feature(flow, host_windows, base_ts, host_window),
        "len": int(flow.packet_count),
        "id": [ip_to_int_for_project(flow.src), ip_to_int_for_project(flow.dst)],
        "five_tuple": [flow.proto, flow.src, flow.dst, flow.sport, flow.dport],
        "start_ts": flow.start_ts,
        "end_ts": flow.end_ts,
    }


def freeze_host_windows(
    host_windows: Dict[int, Dict[str, Dict[str, List[float]]]]
) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    frozen = {}
    for wid, host_map in host_windows.items():
        frozen_hosts = {}
        for host, stats in host_map.items():
            frozen_hosts[host] = {
                "send": list(stats.get("send", [0.0, 0.0, 0.0])),
                "recv": list(stats.get("recv", [0.0, 0.0, 0.0])),
            }
        frozen[int(wid)] = frozen_hosts
    return frozen


def _init_convert_worker(
    host_windows,
    base_ts,
    segment_len,
    max_packet_len,
    gap_threshold,
    host_window,
    default_label,
    label_map,
):
    global _WORKER_HOST_WINDOWS
    global _WORKER_BASE_TS
    global _WORKER_SEGMENT_LEN
    global _WORKER_MAX_PACKET_LEN
    global _WORKER_GAP_THRESHOLD
    global _WORKER_HOST_WINDOW
    global _WORKER_DEFAULT_LABEL
    global _WORKER_LABEL_MAP

    _WORKER_HOST_WINDOWS = host_windows
    _WORKER_BASE_TS = base_ts
    _WORKER_SEGMENT_LEN = segment_len
    _WORKER_MAX_PACKET_LEN = max_packet_len
    _WORKER_GAP_THRESHOLD = gap_threshold
    _WORKER_HOST_WINDOW = host_window
    _WORKER_DEFAULT_LABEL = default_label
    _WORKER_LABEL_MAP = label_map


def _convert_flow_worker(flow: FlowRecord):
    item = make_flow_item(
        flow,
        _WORKER_HOST_WINDOWS,
        _WORKER_BASE_TS,
        _WORKER_SEGMENT_LEN,
        _WORKER_MAX_PACKET_LEN,
        _WORKER_GAP_THRESHOLD,
        _WORKER_HOST_WINDOW,
    )
    bucket = normalize_label(flow_label(flow, _WORKER_DEFAULT_LABEL, _WORKER_LABEL_MAP))
    return bucket, item


def load_label_map(path: Optional[Path]) -> Dict[str, str]:
    """
    Load optional CSV labels.

    Supported columns:
      proto,src,dst,sport,dport,label

    proto can be either numeric IP protocol id or TCP/UDP/OTHER.
    """
    if path is None:
        return {}
    label_map: Dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            proto_raw = str(row.get("proto", "")).strip().upper()
            if proto_raw == "TCP":
                proto = "6"
            elif proto_raw == "UDP":
                proto = "17"
            elif proto_raw in {"OTHER", "IP"}:
                proto = "2"
            else:
                proto = proto_raw
            key = "|".join([
                proto,
                str(row.get("src", "")).strip(),
                str(row.get("dst", "")).strip(),
                str(row.get("sport", "0")).strip(),
                str(row.get("dport", "0")).strip(),
            ])
            label_map[key] = str(row.get("label", "BENIGN")).strip()
    return label_map


def flow_label(flow: FlowRecord, default_label: str, label_map: Dict[str, str]) -> str:
    if not label_map:
        return default_label
    exact = "|".join([str(flow.proto), flow.src, flow.dst, str(flow.sport), str(flow.dport)])
    if exact in label_map:
        return label_map[exact]
    proto_bucket_key = "|".join([
        str(protocol_bucket(flow.proto)),
        flow.src,
        flow.dst,
        str(flow.sport),
        str(flow.dport),
    ])
    return label_map.get(proto_bucket_key, default_label)


def normalize_label(label: str) -> str:
    return "benign" if label.strip().upper() in {"BENIGN", "NORMAL", "0"} else "attack"


def convert_flows(
    flows: List[FlowRecord],
    default_label: str,
    label_map: Dict[str, str],
    segment_len: int,
    max_packet_len: int,
    gap_threshold: float,
    host_window: float,
    show_progress: bool = True,
    workers: int = 1,
    chunksize: int = 1000,
) -> Dict[str, List[dict]]:
    output = {"benign": [], "attack": []}
    if not flows:
        return output

    base_ts = flows[0].start_ts
    host_windows = freeze_host_windows(build_host_windows(flows, host_window, show_progress=show_progress))
    workers = max(1, int(workers))
    chunksize = max(1, int(chunksize))

    progress = ProgressBar("Converting flows", len(flows), "flows", enabled=show_progress)

    _init_convert_worker(
        host_windows,
        base_ts,
        segment_len,
        max_packet_len,
        gap_threshold,
        host_window,
        default_label,
        label_map,
    )
    if workers == 1 or len(flows) <= 1:
        results = (_convert_flow_worker(flow) for flow in flows)
        for bucket, item in results:
            output[bucket].append(item)
            progress.update(advance=1)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_convert_worker,
            initargs=(
                host_windows,
                base_ts,
                segment_len,
                max_packet_len,
                gap_threshold,
                host_window,
                default_label,
                label_map,
            ),
        ) as executor:
            for bucket, item in executor.map(_convert_flow_worker, flows, chunksize=chunksize):
                output[bucket].append(item)
                progress.update(advance=1)

    progress.close(detail=f"{len(output['benign']):,} benign / {len(output['attack']):,} attack")
    return output


def write_tfusion_json(data: Dict[str, List[dict]], output_path: Path, compact: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, separators=(",", ":"))
        else:
            json.dump(data, f, indent=2)


def output_path_for_pcap(pcap_path: Path, output_dir: Path) -> Path:
    suffix = pcap_path.suffix.lower()
    if suffix == ".gz" and pcap_path.with_suffix("").suffix:
        stem = pcap_path.with_suffix("").stem
    else:
        stem = pcap_path.stem
    return output_dir / f"{stem}.json"


def process_pcap_to_json(job):
    (
        pcap_path,
        output_path,
        default_label,
        label_map,
        segment_len,
        max_packet_len,
        gap_threshold,
        host_window,
        min_packets,
        flow_idle_timeout_seconds,
        compact,
        workers,
        chunksize,
        show_progress,
        skip_malformed_dns,
    ) = job

    stats: dict = {}
    flows = collect_flows(
        [pcap_path],
        min_packets=min_packets,
        show_progress=show_progress,
        flow_idle_timeout_seconds=flow_idle_timeout_seconds,
        skip_malformed_dns=skip_malformed_dns,
        stats=stats,
    )
    data = convert_flows(
        flows=flows,
        default_label=default_label,
        label_map=label_map,
        segment_len=segment_len,
        max_packet_len=max_packet_len,
        gap_threshold=gap_threshold,
        host_window=host_window,
        show_progress=show_progress,
        workers=workers,
        chunksize=chunksize,
    )
    write_tfusion_json(data, output_path, compact=compact)
    return {
        "pcap": str(pcap_path),
        "output": str(output_path),
        "flows": len(flows),
        "benign": len(data["benign"]),
        "attack": len(data["attack"]),
        "malformed_dns_dropped": stats.get("malformed_dns_dropped", 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract paper-aligned tFusion features from pcap/pcapng files."
    )
    parser.add_argument("pcap", nargs="+", help="Input .pcap/.pcapng file(s).")
    parser.add_argument("-o", "--output", help="Output tFusion JSON path for merged mode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write one JSON shard per input PCAP into this directory.",
    )
    parser.add_argument(
        "--default-label",
        default="BENIGN",
        help="Label for flows not found in --label-csv. BENIGN/NORMAL/0 go to benign; others go to attack.",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns proto,src,dst,sport,dport,label.",
    )
    parser.add_argument("--segment-len", type=int, default=SEGMENT_LEN)
    parser.add_argument("--max-packet-len", type=int, default=MAX_PACKET_LEN)
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=DEFAULT_GAP_THRESHOLD,
        help="T in the paper's time-aware position formula, in seconds.",
    )
    parser.add_argument(
        "--host-window",
        type=float,
        default=DEFAULT_HOST_WINDOW,
        help="L in the paper's host-level interaction feature, in seconds.",
    )
    parser.add_argument(
        "--min-packets",
        type=int,
        default=1,
        help="丢弃包数少于此值的流。默认 1（不过滤）——真实流量里极短流占比极高"
             "（实测 CSECICIDS2018 99.8%%、Kitsune 73%%、CICDDoS2019 51%% 的流不足 4 包），"
             "过滤会静默删样本，需要时请显式指定并留意 flows_dropped_short。",
    )
    parser.add_argument(
        "--flow-idle-timeout-seconds",
        type=float,
        default=120.0,
        help="Split a reused five-tuple into a new flow after this idle interval.",
    )
    parser.add_argument(
        "--skip-malformed-dns",
        action="store_true",
        help="丢弃 53 端口且载荷短于 12 字节的畸形 DNS 包（默认关闭，丢弃数会统计并打印）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for flow-to-feature conversion inside each PCAP.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="Chunk size for flow-to-feature worker scheduling.",
    )
    parser.add_argument(
        "--pcap-workers",
        type=int,
        default=1,
        help="Worker processes for processing multiple PCAP files in --output-dir mode.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of pretty JSON.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pcaps = [Path(p) for p in args.pcap]
    missing = [str(p) for p in pcaps if not p.exists()]
    if missing:
        print("Missing input file(s): " + ", ".join(missing), file=sys.stderr)
        return 2

    if args.output_dir is None and not args.output:
        print("Either --output or --output-dir is required.", file=sys.stderr)
        return 2

    if args.output_dir is not None and args.output:
        print("Use either --output or --output-dir, not both.", file=sys.stderr)
        return 2

    show_progress = not args.no_progress
    label_map = load_label_map(args.label_csv)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pcap_workers = max(1, int(args.pcap_workers))
        jobs = [
            (
                pcap,
                output_path_for_pcap(pcap, args.output_dir),
                args.default_label,
                label_map,
                args.segment_len,
                args.max_packet_len,
                args.gap_threshold,
                args.host_window,
                args.min_packets,
                args.flow_idle_timeout_seconds,
                args.compact,
                args.workers,
                args.chunksize,
                show_progress and pcap_workers == 1,
                args.skip_malformed_dns,
            )
            for pcap in pcaps
        ]
        if pcap_workers == 1 or len(jobs) <= 1:
            summaries = [process_pcap_to_json(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=pcap_workers) as executor:
                summaries = list(executor.map(process_pcap_to_json, jobs))

        total_flows = sum(item["flows"] for item in summaries)
        total_benign = sum(item["benign"] for item in summaries)
        total_attack = sum(item["attack"] for item in summaries)
        total_dropped = sum(item["malformed_dns_dropped"] for item in summaries)
        print(f"Read pcaps: {len(pcaps)}")
        print(f"Flows: {total_flows}")
        print(f"Benign: {total_benign}")
        print(f"Attack: {total_attack}")
        if args.skip_malformed_dns:
            print(f"Malformed DNS packets dropped: {total_dropped}")
        print(f"Saved shards: {len(summaries)} in {args.output_dir}")
        for item in summaries:
            print(f"  {item['pcap']} -> {item['output']} ({item['flows']} flows)")
        return 0

    stats: dict = {}
    flows = collect_flows(
        pcaps,
        min_packets=args.min_packets,
        show_progress=show_progress,
        flow_idle_timeout_seconds=args.flow_idle_timeout_seconds,
        skip_malformed_dns=args.skip_malformed_dns,
        stats=stats,
    )
    data = convert_flows(
        flows=flows,
        default_label=args.default_label,
        label_map=label_map,
        segment_len=args.segment_len,
        max_packet_len=args.max_packet_len,
        gap_threshold=args.gap_threshold,
        host_window=args.host_window,
        show_progress=show_progress,
        workers=args.workers,
        chunksize=args.chunksize,
    )

    output_path = Path(args.output)
    write_tfusion_json(data, output_path, compact=args.compact)

    print(f"Read pcaps: {len(pcaps)}")
    print(f"Flows: {len(flows)}")
    print(f"Benign: {len(data['benign'])}")
    print(f"Attack: {len(data['attack'])}")
    if args.skip_malformed_dns:
        print(f"Malformed DNS packets dropped: {stats.get('malformed_dns_dropped', 0)}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
