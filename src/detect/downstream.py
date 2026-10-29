"""下游检测器装配。

包含梯度开关（``configure_downstream_mode``）、冻结融合表征抽取
（``_extract_graph_features``）与 sklearn 检测器工厂。
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM

from .constants import (
    CROSSMODAL_MODE,
    SKLEARN_DETECTOR_NAMES,
    TFUSION_MODE,
    canonical_graph_mode,
)
from .fusion_model import TFusionPPModel


def configure_downstream_mode(model: TFusionPPModel, graph_mode: str) -> None:
    """Enable gradients only for modules used by the selected downstream path."""
    mode = canonical_graph_mode(graph_mode)
    for parameter in model.parameters():
        parameter.requires_grad = False

    active_modules = [model.backbone]
    if mode == TFUSION_MODE:
        active_modules.extend([model.base_norm, model.classifier])
    elif mode == CROSSMODAL_MODE:
        active_modules.append(model.graph_attn)
    else:
        raise AssertionError(f"Unreachable canonical graph mode: {mode}")

    for module in active_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def model_parameter_counts(model: nn.Module) -> dict:
    return {
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def _create_sklearn_detector(kind: str, seed: int):
    """Factory for sklearn-based downstream detectors (names aligned with tfusion baseline)."""
    if kind == "rforest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        )
    if kind == "logistic":
        return LogisticRegression(
            class_weight="balanced", random_state=seed, max_iter=2000,
        )
    if kind == "kmeans":
        return KMeans(n_clusters=10, random_state=seed, n_init=10)
    if kind == "bayes":
        return GaussianNB()
    if kind == "iforest":
        return IsolationForest(random_state=seed)
    if kind == "ocsvm":
        return OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=3)
    raise ValueError(f"Unsupported sklearn detector={kind!r}; use one of {SKLEARN_DETECTOR_NAMES}.")


def _extract_graph_features(
    model: TFusionPPModel,
    seq: torch.Tensor,
    non_seq: torch.Tensor,
    interact: torch.Tensor,
    node: torch.Tensor,
    edge: torch.Tensor,
    sizes: torch.Tensor,
    neighbor_node: torch.Tensor | None,
    neighbor_edge: torch.Tensor | None,
    neighbor_mask: torch.Tensor | None,
    device: torch.device,
    batch_size: int = 300,
) -> np.ndarray:
    """Extract frozen graph-enhanced features (384-dim) for sklearn detectors."""
    was_training = model.training
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(seq), batch_size):
            stop = min(start + batch_size, len(seq))
            x_seq = seq[start:stop].to(device)
            x_non = non_seq[start:stop].to(device)
            x_inter = interact[start:stop].to(device)
            n_emb = node[start:stop].to(device)
            e_emb = edge[start:stop].to(device)
            n_nn = neighbor_node[start:stop].to(device) if neighbor_node is not None else None
            n_ne = neighbor_edge[start:stop].to(device) if neighbor_edge is not None else None
            n_mask = neighbor_mask[start:stop].to(device) if neighbor_mask is not None else None
            nn_in, ne_in, mask_in = model._neighbor_inputs(stop - start, device, n_nn, n_ne, n_mask)
            packet = model.backbone.encode_packet(x_seq, flow_sizes=sizes[start:stop].to(device))
            fused = model.graph_attn.encode_fused(
                packet_feat=packet, flow_feat=x_non, host_feat=x_inter,
                neighbor_node=nn_in, neighbor_edge=ne_in, neighbor_mask=mask_in,
            )
            batches.append(fused.cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(batches, axis=0) if batches else np.zeros((0, 384), dtype=np.float32)
