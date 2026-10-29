"""数据加载入口。

正式运行只有两条路径：

- ``pretrain_only=True``：读取单文件或预切分 train 的良性前缀，只构造
  packet / flow / host 三模态，不构建任何历史图或邻居上下文；
- ``pretrain_only=False``（``command=train``）：必须提供 ``data_split``
  预切分文件，train/test/val 全部使用 ``full_stream_causal`` 因果上下文。

运行期不再做内存切分，也不支持其它 ``graph_context_mode`` / ``host_feature_mode``。
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from common import time_log

from .batching import DEFAULT_CONTRAST_SEED, _build_pretrain_batches
from .constants import DEFAULT_D_GRAPH, MIN_ROWS_PER_CLASS
from .context import build_causal_context_table
from .features import _extract_modal_features, _tensor_or_empty
from .reading import _read_flow_sets, read_pre_split_flow_sets
from .split import warn_if_split_priors_diverge


GRAPH_CONTEXT_MODE = "full_stream_causal"
HOST_FEATURE_MODE = "causal"


def _validate_context_modes(graph_context_mode: str, host_feature_mode: str) -> None:
    if graph_context_mode != GRAPH_CONTEXT_MODE:
        raise ValueError(
            f"Unsupported graph_context_mode={graph_context_mode!r}; "
            f"formal runs only support {GRAPH_CONTEXT_MODE!r}."
        )
    if host_feature_mode != HOST_FEATURE_MODE:
        raise ValueError(
            f"Unsupported host_feature_mode={host_feature_mode!r}; "
            f"formal runs only support {HOST_FEATURE_MODE!r}."
        )


def _build_split_rows(
    split_name: str,
    flows: List[dict],
    d_graph: int,
    segment_len: int,
    host_window: float,
    precomputed_context: Optional[Tuple],
    neighbor_count: int,
) -> Tuple[List[Tuple], List[int], List[List[int]]]:
    """用预计算的全流因果上下文组装一个划分的三模态 + 图张量行。"""
    if not flows:
        return [], [], []
    if precomputed_context is None:
        raise ValueError("Downstream splits require a precomputed causal context.")

    (
        host_rows, node_rows, edge_rows,
        neighbor_node_rows, neighbor_edge_rows, neighbor_mask_rows,
        neighbor_delta_rows,
    ) = precomputed_context
    if not (
        len(host_rows) == len(node_rows) == len(edge_rows)
        == len(neighbor_node_rows) == len(neighbor_edge_rows)
        == len(neighbor_mask_rows) == len(neighbor_delta_rows) == len(flows)
    ):
        raise ValueError("Precomputed context rows must align with split flows.")

    seq, non, _, sizes, flow_ids = _extract_modal_features(
        flows,
        segment_len,
        host_window=host_window,
        compute_host=False,
    )
    host_tensor = torch.as_tensor(host_rows, dtype=torch.float32)
    node_raw = torch.as_tensor(node_rows, dtype=torch.float32)
    edge_raw = torch.as_tensor(edge_rows, dtype=torch.float32)
    neighbor_node_raw = torch.as_tensor(neighbor_node_rows, dtype=torch.float32)
    neighbor_edge_raw = torch.as_tensor(neighbor_edge_rows, dtype=torch.float32)
    neighbor_mask = torch.as_tensor(neighbor_mask_rows, dtype=torch.bool)
    neighbor_delta = torch.as_tensor(neighbor_delta_rows, dtype=torch.float32)

    if host_tensor.ndim != 2 or host_tensor.shape[1] != 12:
        raise ValueError("Precomputed host context must have width 12.")
    if node_raw.ndim != 2 or edge_raw.ndim != 2:
        raise ValueError("Precomputed graph context must be two-dimensional.")
    if (
        neighbor_node_raw.ndim != 3 or neighbor_edge_raw.ndim != 3
        or neighbor_mask.shape != neighbor_node_raw.shape[:2]
        or neighbor_node_raw.shape != neighbor_edge_raw.shape
    ):
        raise ValueError("Precomputed neighbor context must be (rows, neighbors, features).")
    if node_raw.shape[1] > d_graph or edge_raw.shape[1] > d_graph:
        raise ValueError("d_graph is smaller than the precomputed graph feature schema.")
    if neighbor_node_raw.shape[1] != neighbor_count:
        raise ValueError("Precomputed neighbor count does not match loader configuration.")
    if neighbor_node_raw.shape[2] > d_graph:
        raise ValueError("d_graph is smaller than the precomputed neighbor feature schema.")

    node = torch.zeros(len(flows), d_graph, dtype=torch.float32)
    edge = torch.zeros(len(flows), d_graph, dtype=torch.float32)
    node[:, :node_raw.shape[1]] = node_raw
    edge[:, :edge_raw.shape[1]] = edge_raw
    neighbor_node = torch.zeros(len(flows), neighbor_count, d_graph, dtype=torch.float32)
    neighbor_edge = torch.zeros(len(flows), neighbor_count, d_graph, dtype=torch.float32)
    neighbor_node[:, :, :neighbor_node_raw.shape[2]] = neighbor_node_raw
    neighbor_edge[:, :, :neighbor_edge_raw.shape[2]] = neighbor_edge_raw

    inter = host_tensor.tolist()
    rows = list(zip(
        seq, non, inter, list(node), list(edge),
        list(neighbor_node), list(neighbor_edge), list(neighbor_mask),
        list(neighbor_delta),
    ))
    logging.info(f"[{split_name}] {len(rows)} flows")
    return rows, sizes, flow_ids


def _build_modal_only_split_rows(
    split_name: str,
    flows: List[dict],
    segment_len: int,
    host_window: float,
    d_graph: int,
    neighbor_count: int,
) -> Tuple[List[Tuple], List[int], List[List[int]]]:
    """预训练行：只构造三模态，node/edge/邻居为固定零占位。"""
    seq, non, inter, sizes, flow_ids = _extract_modal_features(
        flows,
        segment_len,
        host_window=host_window,
        compute_host=True,
    )
    zero_node = torch.zeros(d_graph)
    zero_neighbor = torch.zeros(neighbor_count, d_graph)
    zero_neighbor_mask = torch.zeros(neighbor_count, dtype=torch.bool)
    zero_neighbor_delta = torch.zeros(neighbor_count)
    rows = [
        (
            seq[index], non[index], inter[index],
            zero_node, zero_node,
            zero_neighbor, zero_neighbor,
            zero_neighbor_mask, zero_neighbor_delta,
        )
        for index in range(len(flows))
    ]
    logging.info(
        f"[{split_name}] Tri-modal features: {len(flows)} flows "
        "(historical graph construction skipped)"
    )
    return rows, sizes, flow_ids


def _ratio_sample(items: List[dict], ratio: float, label: str, seed) -> List[dict]:
    """对训练集按比例抽样；两类同比例以保持先验，并至少保留可评估下限。"""
    if not items or ratio >= 1.0:
        return list(items)
    raw_count = math.floor(ratio * len(items))
    count = min(len(items), max(MIN_ROWS_PER_CLASS, raw_count))
    if count != raw_count:
        logging.warning(
            f"train_sample_ratio={ratio} would leave only {raw_count} {label} "
            f"rows; keeping {count} so the class stays evaluable downstream."
        )
    order = random.Random(f"{seed}:{label}").sample(items, len(items))
    return order[:count]


def _empty_flow_tensors(segment_len: int, d_graph: int) -> Tuple:
    return (
        _tensor_or_empty([], (segment_len, 2)),
        _tensor_or_empty([], (6,)),
        _tensor_or_empty([], (12,)),
        _tensor_or_empty([], (d_graph,)),
        _tensor_or_empty([], (d_graph,)),
        torch.IntTensor([]),
        torch.IntTensor([]),
    )


def _load_pretrain_bundle(
    data_tag: str,
    data_path: str,
    data_split: Optional[dict],
    d_graph: int,
    shuffle_seed,
    segment_len: int,
    pre_train_ratio: float,
    contrast_batch_size: int,
    host_window: float,
    neighbor_count: int,
) -> Tuple:
    """预训练专用数据束：只返回 pre-train 张量，train/test 为空占位。"""
    if segment_len > 100:
        raise ValueError("segment_len above 100 is not supported by the current positional encoder.")
    pre_split = read_pre_split_flow_sets(data_split) if data_split is not None else None
    if pre_split is not None:
        benign_flows = list(pre_split["train"][0])
    else:
        benign_flows, _ = _read_flow_sets(data_tag, data_path)

    order = sorted(
        range(len(benign_flows)),
        key=lambda index: float(benign_flows[index].get("start_ts", 0.0)),
    )
    benign_flows = [benign_flows[index] for index in order]
    pre_train_count = math.floor(float(pre_train_ratio) * len(benign_flows))
    pre_train_flows = benign_flows[:pre_train_count]
    if not pre_train_flows:
        raise ValueError(
            f"[{data_tag}] pre_train_ratio={pre_train_ratio} produced no pre-training flows."
        )

    pre_train_data, _, pre_train_id = _build_modal_only_split_rows(
        f"{data_tag}/pre-train",
        pre_train_flows,
        segment_len,
        host_window,
        d_graph,
        neighbor_count,
    )
    contrast_seed = shuffle_seed if shuffle_seed is not None else DEFAULT_CONTRAST_SEED
    pre_train_batch = _build_pretrain_batches(
        pre_train_data, pre_train_id, contrast_batch_size, seed=contrast_seed,
    )

    pre_seq = [row[0] for row in (item for batch in pre_train_batch for item in batch)]
    pre_non = [row[1] for row in (item for batch in pre_train_batch for item in batch)]
    pre_inter = [row[2] for row in (item for batch in pre_train_batch for item in batch)]
    pre_node = [row[3].tolist() for row in (item for batch in pre_train_batch for item in batch)]
    pre_edge = [row[4].tolist() for row in (item for batch in pre_train_batch for item in batch)]

    if pre_train_batch:
        pre_train_seq_ten = _tensor_or_empty(pre_seq, (segment_len, 2)).view(
            len(pre_train_batch), contrast_batch_size, segment_len, 2
        )
        pre_train_non_seq_ten = _tensor_or_empty(pre_non, (6,)).view(
            len(pre_train_batch), contrast_batch_size, 6
        )
        pre_train_interact_ten = _tensor_or_empty(pre_inter, (12,)).view(
            len(pre_train_batch), contrast_batch_size, 12
        )
        pre_train_node_ten = _tensor_or_empty(pre_node, (d_graph,)).view(
            len(pre_train_batch), contrast_batch_size, d_graph
        )
        pre_train_edge_ten = _tensor_or_empty(pre_edge, (d_graph,)).view(
            len(pre_train_batch), contrast_batch_size, d_graph
        )
    else:
        pre_train_seq_ten = _tensor_or_empty([], (contrast_batch_size, segment_len, 2))
        pre_train_non_seq_ten = _tensor_or_empty([], (contrast_batch_size, 6))
        pre_train_interact_ten = _tensor_or_empty([], (contrast_batch_size, 12))
        pre_train_node_ten = _tensor_or_empty([], (contrast_batch_size, d_graph))
        pre_train_edge_ten = _tensor_or_empty([], (contrast_batch_size, d_graph))

    train_payload = _empty_flow_tensors(segment_len, d_graph)
    test_payload = _empty_flow_tensors(segment_len, d_graph)
    logging.info(
        f"[{data_tag}] Pre-train Batches: {len(pre_train_batch)}, "
        f"Pre-train Flows: {len(pre_train_flows)}"
    )
    return (
        pre_train_seq_ten, pre_train_non_seq_ten, pre_train_interact_ten,
        pre_train_node_ten, pre_train_edge_ten,
        *train_payload,
        *test_payload,
    )


@time_log
def load_graph_tfusion_data(
    data_tag: str,
    data_path: str,
    device: str = "cpu",
    d_graph: int = DEFAULT_D_GRAPH,
    shuffle_seed=None,
    segment_len=100,
    pre_train_ratio=0.2,
    contrast_batch_size=50,
    train_sample_ratio=1.0,
    train_sample_stage: str = "before_context",
    graph_window=10.0,
    host_window=10.0,
    graph_context_mode=GRAPH_CONTEXT_MODE,
    host_feature_mode=HOST_FEATURE_MODE,
    neighbor_count=8,
    context_isolation="none",
    data_split=None,
    pretrain_only: bool = False,
):
    """加载数据并构建张量束。

    ``pretrain_only=True`` 时返回长度 19 的数据束：前 5 项为 pre-train
    三模态批次，后 14 项为空的 train/test 占位（预训练不使用）。
    否则返回长度 32（无 val）或 43（有 val）的下游数据束。
    """
    _validate_context_modes(graph_context_mode, host_feature_mode)
    if pretrain_only:
        return _load_pretrain_bundle(
            data_tag, data_path, data_split, d_graph, shuffle_seed,
            segment_len, pre_train_ratio, contrast_batch_size,
            host_window, neighbor_count,
        )

    pre_split = read_pre_split_flow_sets(data_split)
    if pre_split is None:
        raise ValueError(
            "command=train requires data_split.use_pre_split=true; "
            "in-memory splitting has been removed."
        )

    train_benign_flows, train_attack_flows = (
        list(pre_split["train"][0]), list(pre_split["train"][1])
    )
    test_benign_flows, test_attack_flows = (
        list(pre_split["test"][0]), list(pre_split["test"][1])
    )
    val_benign_flows: List[dict] = []
    val_attack_flows: List[dict] = []
    if pre_split["val"] is not None:
        val_benign_flows, val_attack_flows = (
            list(pre_split["val"][0]), list(pre_split["val"][1])
        )

    ratio = 1.0 if train_sample_ratio is None else float(train_sample_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"train_sample_ratio must be in (0, 1], got {ratio!r}.")
    if train_sample_stage not in ("before_context", "after_context"):
        raise ValueError(
            "train_sample_stage must be 'before_context' or 'after_context', "
            f"got {train_sample_stage!r}."
        )
    sample_after_context = train_sample_stage == "after_context"
    if ratio < 1.0 and not sample_after_context:
        train_benign_flows = _ratio_sample(
            train_benign_flows, ratio, "benign", shuffle_seed
        )
        train_attack_flows = _ratio_sample(
            train_attack_flows, ratio, "attack", shuffle_seed
        )
        logging.info(
            f"[{data_tag}] train_sample_ratio={ratio}: sampled train to "
            f"{len(train_benign_flows)} benign / {len(train_attack_flows)} attack "
            "(before_context)."
        )

    if not train_benign_flows and not train_attack_flows:
        raise ValueError(f"[{data_tag}] data_split train file produced an empty split.")
    if not test_benign_flows and not test_attack_flows:
        raise ValueError(f"[{data_tag}] data_split test file produced an empty split.")
    if pre_split["val"] is not None and not val_benign_flows and not val_attack_flows:
        raise ValueError(
            f"[{data_tag}] data_split val is enabled but its file produced an empty split."
        )

    warn_if_split_priors_diverge(
        data_tag,
        train_benign=len(train_benign_flows),
        train_attack=len(train_attack_flows),
        test_benign=len(test_benign_flows),
        test_attack=len(test_attack_flows),
        split_strategy="pre_split",
    )

    train_flows = train_benign_flows + train_attack_flows
    test_flows = test_benign_flows + test_attack_flows
    val_flows = val_benign_flows + val_attack_flows
    train_label = [0] * len(train_benign_flows) + [1] * len(train_attack_flows)
    test_label = [0] * len(test_benign_flows) + [1] * len(test_attack_flows)
    val_label = [0] * len(val_benign_flows) + [1] * len(val_attack_flows)

    context_flows = train_flows + val_flows + test_flows
    full_stream_context = build_causal_context_table(
        context_flows,
        graph_window=graph_window,
        neighbor_count=neighbor_count,
    )
    full_stream_positions = {id(flow): index for index, flow in enumerate(context_flows)}

    train_side_context = None
    train_side_positions = None
    if context_isolation == "train_only":
        train_side_flows = train_flows + val_flows
        train_side_context = build_causal_context_table(
            train_side_flows,
            graph_window=graph_window,
            neighbor_count=neighbor_count,
        )
        train_side_positions = {
            id(flow): index for index, flow in enumerate(train_side_flows)
        }
    elif context_isolation != "none":
        raise ValueError(
            f"Unsupported context_isolation={context_isolation!r}; "
            "use 'none' or 'train_only'."
        )

    def select_context_from(context, positions_map, flows):
        if context is None or positions_map is None or not flows:
            return None
        positions = [positions_map[id(flow)] for flow in flows]
        neighbor_indices = context.neighbor_indices[positions]
        neighbor_mask = neighbor_indices >= 0
        neighbor_node = torch.zeros(
            len(positions), neighbor_count, context.node.shape[1],
            dtype=torch.float32,
        )
        neighbor_edge = torch.zeros_like(neighbor_node)
        for row_index, neighbors in enumerate(neighbor_indices):
            for slot, source_index in enumerate(neighbors):
                if source_index >= 0:
                    neighbor_node[row_index, slot] = torch.tensor(
                        context.node[int(source_index)], dtype=torch.float32
                    )
                    neighbor_edge[row_index, slot] = torch.tensor(
                        context.edge[int(source_index)], dtype=torch.float32
                    )
        return (
            context.host[positions],
            context.node[positions],
            context.edge[positions],
            neighbor_node.numpy(),
            neighbor_edge.numpy(),
            neighbor_mask,
            context.neighbor_delta_t[positions],
        )

    def train_context(flows):
        if train_side_context is not None:
            return select_context_from(train_side_context, train_side_positions, flows)
        return select_context_from(full_stream_context, full_stream_positions, flows)

    def test_context(flows):
        return select_context_from(full_stream_context, full_stream_positions, flows)

    train_data, train_size, _ = _build_split_rows(
        f"{data_tag}/train", train_flows, d_graph, segment_len, host_window,
        train_context(train_flows), neighbor_count,
    )
    test_data, test_size, _ = _build_split_rows(
        f"{data_tag}/test", test_flows, d_graph, segment_len, host_window,
        test_context(test_flows), neighbor_count,
    )
    val_data, val_size, _ = _build_split_rows(
        f"{data_tag}/val", val_flows, d_graph, segment_len, host_window,
        train_context(val_flows), neighbor_count,
    )

    if ratio < 1.0 and sample_after_context:
        benign_indices = [i for i, label in enumerate(train_label) if label == 0]
        attack_indices = [i for i, label in enumerate(train_label) if label == 1]
        keep = sorted(
            _ratio_sample(benign_indices, ratio, "benign", shuffle_seed)
            + _ratio_sample(attack_indices, ratio, "attack", shuffle_seed)
        )
        train_data = [train_data[index] for index in keep]
        train_size = [train_size[index] for index in keep]
        train_label = [train_label[index] for index in keep]
        logging.info(
            f"[{data_tag}] train_sample_ratio={ratio}: sampled train rows to "
            f"{sum(1 for label in train_label if label == 0)} benign / "
            f"{sum(1 for label in train_label if label == 1)} attack "
            "(after_context; neighbors come from the full train fold)."
        )

    train_seq = [row[0] for row in train_data]
    train_non = [row[1] for row in train_data]
    train_inter = [row[2] for row in train_data]
    train_node = [row[3].tolist() for row in train_data]
    train_edge = [row[4].tolist() for row in train_data]
    train_neighbor_node = [row[5].tolist() for row in train_data]
    train_neighbor_edge = [row[6].tolist() for row in train_data]
    train_neighbor_mask = [row[7].tolist() for row in train_data]
    train_neighbor_delta = [row[8].tolist() for row in train_data]

    test_seq = [row[0] for row in test_data]
    test_non = [row[1] for row in test_data]
    test_inter = [row[2] for row in test_data]
    test_node = [row[3].tolist() for row in test_data]
    test_edge = [row[4].tolist() for row in test_data]
    test_neighbor_node = [row[5].tolist() for row in test_data]
    test_neighbor_edge = [row[6].tolist() for row in test_data]
    test_neighbor_mask = [row[7].tolist() for row in test_data]
    test_neighbor_delta = [row[8].tolist() for row in test_data]

    val_seq = [row[0] for row in val_data]
    val_non = [row[1] for row in val_data]
    val_inter = [row[2] for row in val_data]
    val_node = [row[3].tolist() for row in val_data]
    val_edge = [row[4].tolist() for row in val_data]
    val_neighbor_node = [row[5].tolist() for row in val_data]
    val_neighbor_edge = [row[6].tolist() for row in val_data]
    val_neighbor_mask = [row[7].tolist() for row in val_data]
    val_neighbor_delta = [row[8].tolist() for row in val_data]

    train_data_seq_ten = _tensor_or_empty(train_seq, (segment_len, 2))
    train_data_non_seq_ten = _tensor_or_empty(train_non, (6,))
    train_data_interact_ten = _tensor_or_empty(train_inter, (12,))
    train_node_ten = _tensor_or_empty(train_node, (d_graph,))
    train_edge_ten = _tensor_or_empty(train_edge, (d_graph,))
    train_size_ten = torch.IntTensor(train_size)
    train_label_ten = torch.IntTensor(train_label)
    train_neighbor_node_ten = _tensor_or_empty(train_neighbor_node, (neighbor_count, d_graph))
    train_neighbor_edge_ten = _tensor_or_empty(train_neighbor_edge, (neighbor_count, d_graph))
    train_neighbor_mask_ten = _tensor_or_empty(train_neighbor_mask, (neighbor_count,)).to(torch.bool)
    train_neighbor_delta_ten = _tensor_or_empty(train_neighbor_delta, (neighbor_count,))

    test_data_seq_ten = _tensor_or_empty(test_seq, (segment_len, 2))
    test_data_non_seq_ten = _tensor_or_empty(test_non, (6,))
    test_data_interact_ten = _tensor_or_empty(test_inter, (12,))
    test_node_ten = _tensor_or_empty(test_node, (d_graph,))
    test_edge_ten = _tensor_or_empty(test_edge, (d_graph,))
    test_size_ten = torch.IntTensor(test_size)
    test_label_ten = torch.IntTensor(test_label)
    test_neighbor_node_ten = _tensor_or_empty(test_neighbor_node, (neighbor_count, d_graph))
    test_neighbor_edge_ten = _tensor_or_empty(test_neighbor_edge, (neighbor_count, d_graph))
    test_neighbor_mask_ten = _tensor_or_empty(test_neighbor_mask, (neighbor_count,)).to(torch.bool)
    test_neighbor_delta_ten = _tensor_or_empty(test_neighbor_delta, (neighbor_count,))

    val_data_seq_ten = _tensor_or_empty(val_seq, (segment_len, 2))
    val_data_non_seq_ten = _tensor_or_empty(val_non, (6,))
    val_data_interact_ten = _tensor_or_empty(val_inter, (12,))
    val_node_ten = _tensor_or_empty(val_node, (d_graph,))
    val_edge_ten = _tensor_or_empty(val_edge, (d_graph,))
    val_size_ten = torch.IntTensor(val_size)
    val_label_ten = torch.IntTensor(val_label)
    val_neighbor_node_ten = _tensor_or_empty(val_neighbor_node, (neighbor_count, d_graph))
    val_neighbor_edge_ten = _tensor_or_empty(val_neighbor_edge, (neighbor_count, d_graph))
    val_neighbor_mask_ten = _tensor_or_empty(val_neighbor_mask, (neighbor_count,)).to(torch.bool)
    val_neighbor_delta_ten = _tensor_or_empty(val_neighbor_delta, (neighbor_count,))

    test_window_ids_ten = torch.LongTensor([
        math.floor(float(flow.get("start_ts", 0.0)) / max(float(graph_window), 1e-9))
        for flow in test_flows
    ])

    logging.info(
        f"[{data_tag}] Train Flows: {len(train_data)} [Attack: {sum(train_label)}], "
        f"Test Flows: {len(test_data)} [Attack: {sum(test_label)}], "
        f"Val Flows: {len(val_data)} [Attack: {sum(val_label)}]"
    )

    pre_train_payload = _tensor_or_empty(
        [], (contrast_batch_size, segment_len, 2)
    ), _tensor_or_empty([], (contrast_batch_size, 6)), _tensor_or_empty(
        [], (contrast_batch_size, 12)
    ), _tensor_or_empty([], (contrast_batch_size, d_graph)), _tensor_or_empty(
        [], (contrast_batch_size, d_graph)
    )

    val_payload = (
        (
            val_data_seq_ten,
            val_data_non_seq_ten,
            val_data_interact_ten,
            val_node_ten,
            val_edge_ten,
            val_size_ten,
            val_label_ten,
            val_neighbor_node_ten,
            val_neighbor_edge_ten,
            val_neighbor_mask_ten,
            val_neighbor_delta_ten,
        )
        if val_flows
        else ()
    )

    return (
        *pre_train_payload,
        train_data_seq_ten,
        train_data_non_seq_ten,
        train_data_interact_ten,
        train_node_ten,
        train_edge_ten,
        train_size_ten,
        train_label_ten,
        test_data_seq_ten,
        test_data_non_seq_ten,
        test_data_interact_ten,
        test_node_ten,
        test_edge_ten,
        test_size_ten,
        test_label_ten,
        test_window_ids_ten,
        _tensor_or_empty([], (neighbor_count, d_graph)),
        _tensor_or_empty([], (neighbor_count, d_graph)),
        _tensor_or_empty([], (neighbor_count,)).to(torch.bool),
        train_neighbor_node_ten,
        train_neighbor_edge_ten,
        train_neighbor_mask_ten,
        test_neighbor_node_ten,
        test_neighbor_edge_ten,
        test_neighbor_mask_ten,
        _tensor_or_empty([], (neighbor_count,)),
        train_neighbor_delta_ten,
        test_neighbor_delta_ten,
        *val_payload,
    )


@dataclass(frozen=True)
class GraphTFusionDataBundle:
    """``load_graph_tfusion_data`` 张量束的命名视图。"""

    pre_train_seq: object = None
    pre_train_non_seq: object = None
    pre_train_interact: object = None
    pre_train_node: object = None
    pre_train_edge: object = None
    train_seq: object = None
    train_non_seq: object = None
    train_interact: object = None
    train_node: object = None
    train_edge: object = None
    train_size: object = None
    train_label: object = None
    test_seq: object = None
    test_non_seq: object = None
    test_interact: object = None
    test_node: object = None
    test_edge: object = None
    test_size: object = None
    test_label: object = None
    test_window_ids: object = None
    pre_train_neighbor_node: object = None
    pre_train_neighbor_edge: object = None
    pre_train_neighbor_mask: object = None
    train_neighbor_node: object = None
    train_neighbor_edge: object = None
    train_neighbor_mask: object = None
    test_neighbor_node: object = None
    test_neighbor_edge: object = None
    test_neighbor_mask: object = None
    pre_train_neighbor_delta_t: object = None
    train_neighbor_delta_t: object = None
    test_neighbor_delta_t: object = None
    val_seq: object = None
    val_non_seq: object = None
    val_interact: object = None
    val_node: object = None
    val_edge: object = None
    val_size: object = None
    val_label: object = None
    val_neighbor_node: object = None
    val_neighbor_edge: object = None
    val_neighbor_mask: object = None
    val_neighbor_delta_t: object = None


def unpack_graph_tfusion_data(data) -> GraphTFusionDataBundle:
    """把 ``load_graph_tfusion_data`` 的位置元组转为命名访问。"""
    fields = GraphTFusionDataBundle.__dataclass_fields__
    return GraphTFusionDataBundle(**{
        name: (data[index] if index < len(data) else None)
        for index, name in enumerate(fields)
    })
