"""训练/验证索引切分与上下文特征缩放。"""

from __future__ import annotations

import random

import numpy as np
import torch

from feature_scaling import RobustFeatureScaler


def stratified_train_validation_indices(labels, validation_ratio: float, seed: int):
    """Return deterministic, disjoint train/validation indices for both classes."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")
    by_class = {0: [], 1: []}
    for index, label in enumerate(labels):
        label = int(label)
        if label not in by_class:
            raise ValueError(f"Unsupported binary label: {label!r}")
        by_class[label].append(index)
    if any(len(indices) < 2 for indices in by_class.values()):
        raise ValueError("Both classes need at least two rows for validation splitting.")
    rng = random.Random(seed)
    validation = []
    train = []
    for label in (0, 1):
        indices = list(by_class[label])
        rng.shuffle(indices)
        validation_count = min(
            len(indices) - 1,
            max(1, int(round(validation_ratio * len(indices)))),
        )
        validation.extend(indices[:validation_count])
        train.extend(indices[validation_count:])
    return sorted(train), sorted(validation)


def fit_and_apply_context_scalers(
    train_host: torch.Tensor,
    train_node: torch.Tensor,
    train_edge: torch.Tensor,
    test_host: torch.Tensor,
    test_node: torch.Tensor,
    test_edge: torch.Tensor,
    fit_indices,
):
    """Fit host/node/edge scalers on fit rows and transform all model rows."""
    indices = np.asarray(list(fit_indices), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("fit_indices must be non-empty for context scaling.")
    if indices.min() < 0 or indices.max() >= train_host.size(0):
        raise ValueError("fit_indices contain an out-of-range row.")

    def scale_pair(name: str, train_rows: torch.Tensor, test_rows: torch.Tensor):
        if train_rows.ndim != 2 or test_rows.ndim != 2:
            raise ValueError(f"{name} context tensors must be two-dimensional.")
        if train_rows.size(1) != test_rows.size(1):
            raise ValueError(f"{name} train/test context widths must match.")
        feature_names = tuple(f"{name}_{column}" for column in range(train_rows.size(1)))
        scaler = RobustFeatureScaler(feature_names=feature_names).fit(
            train_rows[indices].detach().cpu().numpy()
        )
        train_scaled = torch.from_numpy(scaler.transform(train_rows.detach().cpu().numpy()))
        test_scaled = torch.from_numpy(scaler.transform(test_rows.detach().cpu().numpy()))
        return train_scaled, test_scaled, scaler.to_dict()

    train_host_scaled, test_host_scaled, host_state = scale_pair(
        "host", train_host, test_host
    )
    train_node_scaled, test_node_scaled, node_state = scale_pair(
        "node", train_node, test_node
    )
    train_edge_scaled, test_edge_scaled, edge_state = scale_pair(
        "edge", train_edge, test_edge
    )
    return (
        train_host_scaled,
        train_node_scaled,
        train_edge_scaled,
        test_host_scaled,
        test_node_scaled,
        test_edge_scaled,
    ), {"host": host_state, "node": node_state, "edge": edge_state}


def apply_context_state(host_state: dict, node_state: dict, edge_state: dict,
                        host: torch.Tensor, node: torch.Tensor, edge: torch.Tensor):
    """用已拟合的 scaler 状态变换额外行（如外部验证集的上下文特征）。

    仅做 ``(rows - center) / scale`` 仿射变换，不重新拟合，因此与训练集
    使用同一坐标系；``fit_and_apply_context_scalers`` 的签名保持不变。
    """

    def apply(state: dict, rows: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(state["center"], dtype=rows.dtype).view(1, -1)
        scale = torch.tensor(state["scale"], dtype=rows.dtype).view(1, -1)
        return (rows - center) / scale

    return apply(host_state, host), apply(node_state, node), apply(edge_state, edge)


def iter_index_batches(indices, batch_size: int):
    """Yield non-empty contiguous batches from an explicit index sequence."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        if len(batch):
            yield batch
