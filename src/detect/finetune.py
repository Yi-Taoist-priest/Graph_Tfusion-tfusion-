"""下游微调阶段：用有标签的训练数据微调编码器，使其学到任务相关的表示。

设计意图（与项目原始设想对齐）：

    预训练（无标注大数据 → 通用表示）
      → **微调**（有标签训练数据 → 攻击流的特殊表示）   ← 本模块
      → 冻结编码器抽取特征
      → 简单 ML 检测器（rforest / logistic / …）
      → 测试

微调只在**训练划分内部**进行（拟合子集训练、验证子集早停），测试集始终不参与，
因此不改变原有防泄露边界。

支持的微调范围（`finetune_scope`）：

    full      backbone + 图注意力 + 分类头（全部可训练）
    graph     只训练图注意力（含分类头），backbone 保持预训练权重
    head      只训练图注意力的分类头
    layerwise backbone 用小学习率、图注意力用全学习率

`finetune_epochs=0` 时本模块不被调用，行为与改动前完全一致。

R2/R3 配方新增的显式控制项（默认值保持旧行为）：

    graph_lr      图分支使用独立学习率；None 时沿用 ``lr``
    max_steps     以梯度步而不是 epoch 计预算；达到后停止
    patience      验证指标连续多少轮不提升就早停
    eval_metric   ``auc``（默认，阈值无关）或 ``loss``

这样低数据档（batch 较小、每 epoch 步数少）和全量档可以共享同一个
梯度步预算，避免"数据越少、优化步数越少"的混杂。
"""

from __future__ import annotations

import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .constants import CROSSMODAL_MODE, TFUSION_MODE, canonical_graph_mode

VALID_SCOPES = ("full", "graph", "head", "layerwise")


def apply_scope(model, scope: str, backbone_lr_scale: float = 0.1) -> list[dict]:
    """按范围设置 requires_grad，并返回带名字的优化器参数组。

    返回的每个组包含 ``name``（backbone/graph/head）与 ``lr_scale``，便于
    调用方给图分支单独设置学习率。
    """
    if scope not in VALID_SCOPES:
        raise ValueError(f"finetune_scope must be one of {VALID_SCOPES}, got {scope!r}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone = list(model.backbone.parameters())
    graph = list(model.graph_attn.parameters())
    head = list(model.graph_attn.classifier.parameters())

    def enable(params):
        for parameter in params:
            parameter.requires_grad = True

    if scope == "full":
        enable(backbone); enable(graph)
        # backbone 与 graph 分成两组，允许图分支使用更大的学习率；
        # 不传 graph_lr 时两组仍等价于旧行为（同一 lr）。
        return [
            {"name": "backbone", "params": backbone, "lr_scale": 1.0},
            {"name": "graph", "params": graph, "lr_scale": 1.0},
        ]
    if scope == "graph":
        enable(graph)
        return [{"name": "graph", "params": graph, "lr_scale": 1.0}]
    if scope == "head":
        enable(head)
        return [{"name": "head", "params": head, "lr_scale": 1.0}]
    # layerwise
    enable(backbone); enable(graph)
    return [
        {"name": "backbone", "params": backbone, "lr_scale": backbone_lr_scale},
        {"name": "graph", "params": graph, "lr_scale": 1.0},
    ]


def _infer_validation_metrics(model, graph_mode, tensors, indices, batch_size, device):
    """返回 (验证损失, 验证 AUC)；单类验证集时 AUC 为 NaN。"""
    (seq, non_seq, interact, node, edge, sizes, labels,
     n_node, n_edge, n_mask) = tensors
    model.eval()
    losses, total = 0.0, 0
    scores, targets = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            logits = model.classify(
                seq[idx].to(device), non_seq[idx].to(device), interact[idx].to(device),
                node[idx].to(device), edge[idx].to(device),
                flow_sizes=sizes[idx].to(device), graph_mode=graph_mode,
                neighbor_node=n_node[idx].to(device) if n_node is not None else None,
                neighbor_edge=n_edge[idx].to(device) if n_edge is not None else None,
                neighbor_mask=n_mask[idx].to(device) if n_mask is not None else None,
            )
            target = labels[idx].to(device).long()
            loss = nn.functional.cross_entropy(logits, target, reduction="sum")
            losses += float(loss.item()); total += int(target.numel())
            scores.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    model.train()
    auc = float("nan")
    if targets:
        y = np.concatenate(targets)
        s = np.concatenate(scores)
        if y.size and np.unique(y).size > 1:
            try:
                auc = float(roc_auc_score(y, s))
            except ValueError:
                auc = float("nan")
    return losses / max(1, total), auc


def finetune_encoder(
    model,
    graph_mode: str,
    *,
    seq, non_seq, interact, node, edge, sizes, labels,
    neighbor_node=None, neighbor_edge=None, neighbor_mask=None,
    fit_indices, validation_indices=None, validation_data=None,
    epochs: int, lr: float, batch_size: int, scope: str = "full",
    backbone_lr_scale: float = 0.1, graph_lr: float | None = None,
    max_steps: int | None = None, patience: int = 8,
    eval_metric: str = "auc", seed: int = 777,
    device: torch.device | None = None, grad_clip: float = 5.0,
    data_tag: str = "",
) -> dict:
    """在训练划分上微调编码器；返回微调摘要（供落盘追溯）。

    ``validation_data`` 提供时（预切分 val 文件）直接在其上算验证指标，
    ``validation_indices`` 作用于训练张量；两者二选一，用于早停。

    早停默认按验证 AUC（阈值无关），避免低数据档被 0.5 阈值或类别不平衡误导。
    ``max_steps`` 给定后，训练在达到该梯度步数时停止；epoch 只是外壳，
    因此不同数据量档位可以共享同一个优化预算。
    """
    device = device or torch.device("cpu")
    mode = canonical_graph_mode(graph_mode)
    if mode not in (CROSSMODAL_MODE, TFUSION_MODE):
        raise AssertionError(f"Unreachable canonical graph mode: {mode}")

    groups = apply_scope(model, scope, backbone_lr_scale)
    param_groups = []
    for group in groups:
        group_lr = lr * group["lr_scale"]
        if graph_lr is not None and group["name"] in ("graph", "head"):
            group_lr = float(graph_lr)
        param_groups.append({"params": group["params"], "lr": group_lr})
    optimizer = torch.optim.Adam(param_groups)

    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    target = labels[fit_indices].to(torch.long)
    counts = torch.bincount(target, minlength=2).to(torch.float32).clamp_min(1.0)
    class_weights = (counts.sum() / (2.0 * counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    tensors = (seq, non_seq, interact, node, edge, sizes, labels,
               neighbor_node, neighbor_edge, neighbor_mask)
    if validation_data is not None:
        tensors = tuple(validation_data)
        validation_indices = np.arange(int(tensors[6].size(0)), dtype=np.int64)
    elif validation_indices is None:
        raise ValueError("finetune_encoder requires validation_indices or validation_data.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_loss = float("inf")
    best_auc = float("nan")
    best_score = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    gradient_steps = 0
    history = []
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        perm = rng.permutation(fit_indices)
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            logits = model.classify(
                seq[idx].to(device), non_seq[idx].to(device), interact[idx].to(device),
                node[idx].to(device), edge[idx].to(device),
                flow_sizes=sizes[idx].to(device), graph_mode=mode,
                neighbor_node=neighbor_node[idx].to(device) if neighbor_node is not None else None,
                neighbor_edge=neighbor_edge[idx].to(device) if neighbor_edge is not None else None,
                neighbor_mask=neighbor_mask[idx].to(device) if neighbor_mask is not None else None,
            )
            optimizer.zero_grad()
            loss = loss_fn(logits, labels[idx].to(device).long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item()); batches += 1
            gradient_steps += 1
            if max_steps is not None and gradient_steps >= int(max_steps):
                break

        validation_loss, validation_auc = _infer_validation_metrics(
            model, mode, tensors, validation_indices, batch_size, device)
        if eval_metric == "auc" and not math.isnan(validation_auc):
            score = validation_auc
        else:
            score = -validation_loss
        improved = score > best_score + 1e-9
        if improved:
            best_score = score
            best_loss = validation_loss
            best_auc = validation_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss / max(1, batches), 6),
            "validation_loss": round(validation_loss, 6),
            "validation_auc": (
                None if math.isnan(validation_auc) else round(validation_auc, 6)
            ),
            "gradient_steps": gradient_steps,
        })
        if max_steps is not None and gradient_steps >= int(max_steps):
            break
        if stale_epochs >= int(patience):
            stopped_early = True
            break

    model.load_state_dict(best_state)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    logging.info(
        f'[{data_tag}] Fine-tuned encoder: scope={scope} epochs={epochs} '
        f'steps={gradient_steps} lr={lr} graph_lr={graph_lr} '
        f'trainable_params={trainable} best_epoch={best_epoch} '
        f'best_validation_loss={best_loss:.6f} best_validation_auc={best_auc:.6f} '
        f'early_stop={stopped_early}')
    return {
        "finetune_scope": scope,
        "finetune_epochs": epochs,
        "finetune_lr": lr,
        "finetune_graph_lr": graph_lr,
        "finetune_batch_size": batch_size,
        "finetune_backbone_lr_scale": (
            backbone_lr_scale if scope == "layerwise" else None
        ),
        "finetune_max_steps": max_steps,
        "finetune_patience": patience,
        "finetune_eval_metric": eval_metric,
        "finetune_gradient_steps": gradient_steps,
        "finetune_best_epoch": best_epoch,
        "finetune_best_validation_loss": (
            None if math.isinf(best_loss) else round(float(best_loss), 6)
        ),
        "finetune_best_validation_auc": (
            None if math.isnan(best_auc) else round(float(best_auc), 6)
        ),
        "finetune_stopped_early": stopped_early,
        "finetune_trainable_parameters": int(trainable),
        "finetune_history": history,
    }
