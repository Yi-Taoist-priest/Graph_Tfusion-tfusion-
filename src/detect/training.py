"""下游训练编排：冻结表征 + sklearn 检测器，或端到端分类器。

``model_factory`` 参数用于注入变体模型类（消融 / 对比实验），默认主线模型。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import *
from .artifacts import save_evaluation_artifacts, save_training_provenance
from .checkpoint import graph_classifier_checkpoint_path, load_model_checkpoint, save_model_checkpoint
from .constants import (
    CROSSMODAL_MODE,
    GRAPH_CLASSIFIER_NAME,
    SKLEARN_DETECTOR_NAMES,
    TFUSION_MODE,
    THRESHOLD_STRATEGY_VALIDATION,
    canonical_graph_mode,
)
from .downstream import (
    _create_sklearn_detector,
    _extract_graph_features,
    configure_downstream_mode,
    model_parameter_counts,
)
from .metrics import compute_binary_metrics, warn_if_test_set_single_class
from .finetune import finetune_encoder
from .fusion_model import TFusionPPModel
from model.graph_enhanced import zero_init_gates
from .thresholds import resolve_threshold_strategy, select_threshold_for_strategy
from .validation import (
    apply_context_state,
    fit_and_apply_context_scalers,
    iter_index_batches,
    stratified_train_validation_indices,
)


def apply_weight_initialization(model: nn.Module, mode: str) -> None:
    """按 ``mode`` 随机初始化模型权重（仅在不加载预训练权重时使用）。

    mode:
      - ``kaiming``/``he``: Linear/Conv 与 MultiheadAttention 的 in_proj_weight
        用 ``kaiming_normal_(fan_in, relu)``，偏置置零；Embedding 用 N(0, 0.02)；
        LayerNorm 复位为 weight=1/bias=0。
      - ``xavier``/``glorot``: 对应权重层改用 ``xavier_uniform_``。
      - ``default``/``none``: 不做修改，沿用框架默认初始化。
    """
    normalized = (mode or "default").strip().lower()
    if normalized in ("", "default", "none"):
        return
    if normalized not in ("kaiming", "he", "xavier", "glorot"):
        raise ValueError(f"Unsupported weight_init={mode!r}.")
    use_kaiming = normalized in ("kaiming", "he")

    def _init_weight(weight: torch.Tensor) -> None:
        if use_kaiming:
            nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(weight)

    linear_like = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                _init_weight(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, linear_like):
            _init_weight(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


@time_log
def train_detection_pp(data_tag: str,
                       detection_model: str,
                       log_path: str,
                       fig_path: str,
                       traffic_model_path: str,
                       traffic_model_tag: str,
                       detection_model_path: str,
                       # Train data
                       train_data_seq: torch.Tensor,
                       train_data_non_seq: torch.Tensor,
                       train_data_interact: torch.Tensor,
                       train_node: torch.Tensor,
                       train_edge: torch.Tensor,
                       train_size: torch.Tensor,
                       train_label: torch.Tensor,
                       test_data_seq: torch.Tensor,
                       test_data_non_seq: torch.Tensor,
                       test_data_interact: torch.Tensor,
                       test_node: torch.Tensor,
                       test_edge: torch.Tensor,
                       test_size: torch.Tensor,
                       test_label: torch.Tensor,
                       gpu_id: int,
                       waterline: float,
                       d_graph: int = 128,
                       d_model: int = 128,
                       num_heads: int = 4,
                       dropout: float = 0.1,
                       d_fused_out: int = 384,
                       lr: float = 0.001,
                       num_epoch: int = 50,
                       validation_ratio: float = 0.2,
                       seed: int = 777,
                       patience: int = 10,
                       grad_clip: float = 5.0,
                       threshold_objective: str = "macro_f1",
                       threshold_strategy: str | None = None,
                       threshold_transfer: str = "absolute",
                       optimize_threshold: bool | None = None,
                       model_mode: str = CROSSMODAL_MODE,
                       checkpoint_load_scope: str = "backbone_only",
                       load_pretrained: bool = True,
                       weight_init: str = "default",
                       finetune_epochs: int = 10,
                       finetune_lr: float = 1e-4,
                       finetune_scope: str = "full",
                       finetune_batch_size: int = 300,
                       finetune_backbone_lr_scale: float = 0.1,
                       finetune_graph_lr: float | None = None,
                       finetune_max_steps: int | None = None,
                       finetune_patience: int = 8,
                       finetune_eval_metric: str = "auc",
                       zero_init_gate: bool = False,
                       gate_init_bias: float = -6.0,
                       test_window_ids: torch.Tensor | None = None,
                       train_neighbor_node: torch.Tensor | None = None,
                       train_neighbor_edge: torch.Tensor | None = None,
                       train_neighbor_mask: torch.Tensor | None = None,
                       test_neighbor_node: torch.Tensor | None = None,
                       test_neighbor_edge: torch.Tensor | None = None,
                       test_neighbor_mask: torch.Tensor | None = None,
                       val_data_seq: torch.Tensor | None = None,
                       val_data_non_seq: torch.Tensor | None = None,
                       val_data_interact: torch.Tensor | None = None,
                       val_node: torch.Tensor | None = None,
                       val_edge: torch.Tensor | None = None,
                       val_size: torch.Tensor | None = None,
                       val_label: torch.Tensor | None = None,
                       val_neighbor_node: torch.Tensor | None = None,
                       val_neighbor_edge: torch.Tensor | None = None,
                       val_neighbor_mask: torch.Tensor | None = None,
                       model_factory: Callable = TFusionPPModel):
    """
    Graph_Tfusion 下游检测器训练 + 测试

    关键创新点：
    1. 使用 Graph_Tfusion backbone 提取 backbone 特征
    2. 使用 Graph-Enhanced Cross-Modal Attention 融合图嵌入 + 多模态特征
    3. 端到端训练分类器，输出异常分数

    ``val_*`` 提供时（data_split.val）作为独立验证集用于早停与阈值选择，
    训练集全量拟合，不再从 train 内部按 validation_ratio 二次切分。

    阈值只保留两种策略，由 ``threshold_strategy`` 指定：

    - ``fixed_0.5``：判定阈值固定为 0.5，不读取验证集分数；
    - ``validation``：验证集三级协议（macro-F1 搜索 → 先验匹配 → 0.5）。

    未提供 ``threshold_strategy`` 时兼容旧字段：``optimize_threshold=False``
    映射为 ``fixed_0.5``，``True`` 映射为 ``validation``。``threshold_transfer``
    已废弃，仅保留参数兼容，不再改变阈值。
    """
    logging.info(f'[{data_tag}] Graph_Tfusion Train+Test is started.')
    fout = open(f'{log_path}/{data_tag}.log', 'w', buffering=1)

    strategy = resolve_threshold_strategy(threshold_strategy, optimize_threshold)
    if threshold_transfer not in (None, "", "absolute"):
        logging.warning(
            f'[{data_tag}] threshold_transfer={threshold_transfer!r} is deprecated '
            'and ignored; the unified strategies are fixed_0.5 and validation.'
        )
    logging.info(f'[{data_tag}] threshold_strategy={strategy}')

    device = resolve_torch_device(gpu_id)
    logging.info(
        f'[{data_tag}] device={device} (gpu_id={gpu_id}, '
        f'cuda_available={torch.cuda.is_available()})'
    )
    # 外部验证集（data_split.val）：提供时全量训练集拟合，早停与阈值选择
    # 都只用该验证集；否则回退到从 train 内部按 validation_ratio 二次切分。
    external_val = val_label is not None and val_label.numel() > 0
    if external_val:
        logging.info(
            f'[{data_tag}] External validation set from data_split.val: '
            f'{int(val_label.numel())} rows; fitting on the full training split.'
        )
    else:
        logging.info(
            f'[{data_tag}] No external validation set; carving validation from '
            f'train (validation_ratio={validation_ratio}).'
        )

    canonical_mode = canonical_graph_mode(model_mode)
    model = model_factory(
        d_graph=d_graph,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        d_fused_out=d_fused_out,
    ).to(device)
    model_path = f'{traffic_model_path}/{traffic_model_tag}_pp_model.pt'
    # print(model_path)
    if not load_pretrained:
        checkpoint_provenance = {
            "path": model_path,
            "partial": None,
            "loaded_key_count": 0,
            "missing": True,
            "disabled_by_config": True,
        }
        apply_weight_initialization(model, weight_init)
        logging.warning(f'[{data_tag}] Scratch control: pre-trained checkpoint '
                        f'loading disabled by config ({model_path} ignored); '
                        f'weight_init={weight_init!r}.')
    elif os.path.exists(model_path):
        print("sssssssssssssss"*10)
        print(model_path)
        checkpoint_provenance = load_model_checkpoint(
            model,
            model_path,
            device,
            allow_partial=True,
            load_scope=checkpoint_load_scope,
        )
        logging.info(f'Loaded Graph_Tfusion pre-trained model from {model_path}')
    else:
        checkpoint_provenance = {
            "path": model_path,
            "partial": None,
            "loaded_key_count": 0,
            "missing": True,
        }
        logging.warning(f'Pre-trained model not found, training from scratch: {model_path}')

    gate_init_count = 0
    if zero_init_gate:
        gate_init_count = zero_init_gates(model.graph_attn, bias=float(gate_init_bias))
        logging.info(
            f'[{data_tag}] Zero-initialized {gate_init_count} graph gates '
            f'(bias={gate_init_bias}); full starts near identity.'
        )

    if detection_model in SKLEARN_DETECTOR_NAMES:
        # ── sklearn detector path: 微调编码器 → 冻结特征 → 轻量分类器 ──
        # 设计意图：预训练给出通用表示，这里用有标签训练数据微调出攻击流的特殊
        # 表示，再冻结抽特征交给简单 ML 检测器。微调只在训练划分内部进行
        # （拟合子集训练、验证子集早停），测试集不参与，防泄露边界不变。
        logging.info(f'[{data_tag}] Using sklearn detector: {detection_model}')
        finetune_summary: dict = {}
        if external_val:
            # 外部验证集：全量训练集参与拟合，早停只看 data_split.val。
            fit_idx = np.arange(int(train_label.numel()), dtype=np.int64)
            val_idx = None
            finetune_validation = (
                val_data_seq, val_data_non_seq, val_data_interact,
                val_node, val_edge, val_size, val_label,
                val_neighbor_node, val_neighbor_edge, val_neighbor_mask,
            )
        else:
            fit_idx, val_idx = stratified_train_validation_indices(
                train_label.cpu().tolist(), validation_ratio=validation_ratio, seed=seed,
            )
            finetune_validation = None
        if int(finetune_epochs) > 0:
            finetune_summary = finetune_encoder(
                model, canonical_mode,
                seq=train_data_seq, non_seq=train_data_non_seq,
                interact=train_data_interact, node=train_node, edge=train_edge,
                sizes=train_size, labels=train_label,
                neighbor_node=train_neighbor_node, neighbor_edge=train_neighbor_edge,
                neighbor_mask=train_neighbor_mask,
                fit_indices=fit_idx, validation_indices=val_idx,
                validation_data=finetune_validation,
                epochs=int(finetune_epochs), lr=float(finetune_lr),
                batch_size=int(finetune_batch_size), scope=str(finetune_scope),
                backbone_lr_scale=float(finetune_backbone_lr_scale),
                graph_lr=finetune_graph_lr,
                max_steps=finetune_max_steps,
                patience=int(finetune_patience),
                eval_metric=str(finetune_eval_metric),
                seed=seed, device=device, grad_clip=float(grad_clip),
                data_tag=data_tag,
            )
            finetune_summary["zero_init_gate"] = bool(zero_init_gate)
            finetune_summary["gate_init_bias"] = float(gate_init_bias)
            finetune_summary["gate_init_count"] = int(gate_init_count)
        else:
            logging.info(f'[{data_tag}] Fine-tuning disabled (finetune_epochs=0); '
                         f'encoder used as loaded.')
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        train_features = _extract_graph_features(
            model, train_data_seq, train_data_non_seq, train_data_interact,
            train_node, train_edge, train_size,
            train_neighbor_node, train_neighbor_edge, train_neighbor_mask,
            device,
        )
        test_features = _extract_graph_features(
            model, test_data_seq, test_data_non_seq, test_data_interact,
            test_node, test_edge, test_size,
            test_neighbor_node, test_neighbor_edge, test_neighbor_mask,
            device,
        )
        val_features = None
        if external_val:
            val_features = _extract_graph_features(
                model, val_data_seq, val_data_non_seq, val_data_interact,
                val_node, val_edge, val_size,
                val_neighbor_node, val_neighbor_edge, val_neighbor_mask,
                device,
            )
        train_labels_np = train_label.cpu().numpy()

        detector = _create_sklearn_detector(detection_model, seed)
        # 所有 sklearn 检测器（含无监督）只在 fit 索引上拟合；validation
        # 行只用于取分数选阈值，绝不进入 fit。
        benign_fit_mask = (train_labels_np[fit_idx] == 0)
        if detection_model == "kmeans":
            # 仅用良性样本拟合，
            # 以到最近簇心的距离作为异常分数。
            detector.fit(train_features[fit_idx][benign_fit_mask])
            centers = detector.cluster_centers_

            def _kmeans_scores(features):
                return np.min(
                    np.linalg.norm(
                        features[:, None, :] - centers[None, :, :], axis=2
                    ), axis=1,
                )

            train_scores = _kmeans_scores(train_features)
            test_scores = _kmeans_scores(test_features)
            val_scores = _kmeans_scores(val_features) if external_val else None
        elif detection_model == "iforest":
            detector.fit(train_features[fit_idx][benign_fit_mask])
            train_scores = -detector.score_samples(train_features)
            test_scores = -detector.score_samples(test_features)
            val_scores = -detector.score_samples(val_features) if external_val else None
        elif detection_model == "ocsvm":
            detector.fit(train_features[fit_idx][benign_fit_mask])
            train_scores = -detector.decision_function(train_features)
            test_scores = -detector.decision_function(test_features)
            val_scores = -detector.decision_function(val_features) if external_val else None
        else:
            # Only fit on the fit indices, never on validation.
            detector.fit(train_features[fit_idx], train_labels_np[fit_idx])
            test_scores = detector.predict_proba(test_features)[:, 1]
            train_scores = detector.predict_proba(train_features)[:, 1]
            val_scores = (
                detector.predict_proba(val_features)[:, 1] if external_val else None
            )

        if external_val:
            val_labels = val_label.cpu().numpy().tolist()
        else:
            val_scores = train_scores[val_idx]
            val_labels = train_labels_np[val_idx].tolist()

        selected_threshold, threshold_source = select_threshold_for_strategy(
            strategy,
            val_labels,
            val_scores.tolist() if val_scores is not None else None,
            objective=threshold_objective,
            data_tag=data_tag,
        )
        logging.info(
            f'[{data_tag}] threshold_strategy={strategy}, '
            f'threshold={selected_threshold:.6f} ({threshold_source})'
        )

        label_res = test_label.cpu().numpy().tolist()
        loss_res = test_scores.tolist()
        metrics = compute_binary_metrics(label_res, loss_res, selected_threshold)
        metrics["model_mode"] = model_mode
        metrics["detection_model"] = detection_model
        metrics["threshold_strategy"] = strategy
        metrics["threshold_source"] = threshold_source
        for key in (
            "finetune_gradient_steps", "finetune_best_epoch",
            "finetune_best_validation_auc", "finetune_stopped_early",
            "finetune_graph_lr", "finetune_max_steps",
            "zero_init_gate", "gate_init_bias",
        ):
            if key in finetune_summary:
                metrics[key] = finetune_summary[key]
        warn_if_test_set_single_class(data_tag, metrics, fout=fout)
        roc_auc = metrics["auc"]
        f1 = metrics["f1"]
        per = metrics["precision"]
        rec = metrics["recall"]
        acc = metrics["accuracy"]
        auprc = metrics["auprc"]

        print(f'[{data_tag}] AUC: {roc_auc:.4f}, F1: {f1:.4f}, P: {per:.4f}, R: {rec:.4f}, Acc: {acc:.4f}, AUPRC: {auprc:.4f}, Attack-F1: {metrics["attack_f1"]:.4f}, Threshold: {selected_threshold:.6f}, Support(benign/attack): {metrics["support_benign"]}/{metrics["support_attack"]}, Attack-P/R: {metrics["attack_precision"]:.4f}/{metrics["attack_recall"]:.4f}, FPR/FNR: {metrics["fpr"]:.4f}/{metrics["fnr"]:.4f}',
              file=fout, flush=True)
        print(f'[{data_tag}] AUC: {roc_auc:.4f}, F1: {f1:.4f}, P: {per:.4f}, R: {rec:.4f}, Acc: {acc:.4f}, AUPRC: {auprc:.4f}, Attack-F1: {metrics["attack_f1"]:.4f}, Threshold: {selected_threshold:.6f}, Support(benign/attack): {metrics["support_benign"]}/{metrics["support_attack"]}, Attack-P/R: {metrics["attack_precision"]:.4f}/{metrics["attack_recall"]:.4f}, FPR/FNR: {metrics["fpr"]:.4f}/{metrics["fnr"]:.4f}')

        if detection_model_path:
            os.makedirs(detection_model_path, exist_ok=True)
            sklearn_path = os.path.join(detection_model_path, f'{data_tag}_{detection_model}.joblib')
            joblib.dump(detector, sklearn_path)
            # 预训练权重的审计留痕：sklearn 路径同样落盘 provenance，
            # 与下方 MLP 路径保持一致，使"是否加载预训练"可被机器校验。
            provenance_path = os.path.join(detection_model_path, "checkpoint_provenance.json")
            with open(provenance_path, "w", encoding="utf-8") as target:
                json.dump(checkpoint_provenance, target, indent=2, sort_keys=True)
                target.write("\n")
            # 微调阶段留痕（未微调时写出空摘要，便于区分"没微调"与"忘了写"）
            with open(os.path.join(detection_model_path, "finetune_summary.json"),
                      "w", encoding="utf-8") as target:
                json.dump(finetune_summary, target, ensure_ascii=False, indent=2)
            save_evaluation_artifacts(
                detection_model_path,
                metrics=metrics,
                scores=loss_res,
                labels=label_res,
                validation_threshold=selected_threshold,
                seed=seed,
            )
            logging.info(f'Save sklearn detector to {sklearn_path}')

        try:
            fig = plt.figure(figsize=(10, 10 * 0.618), constrained_layout=True)
            ax = fig.subplots(1, 1)
            benign_score = [s for s, l in zip(loss_res, label_res) if l == 0]
            attack_score = [s for s, l in zip(loss_res, label_res) if l == 1]
            if benign_score:
                ax.hist(benign_score, 1000, density=True, histtype='step', cumulative=True, label='Benign', color='royalblue')
            if attack_score:
                ax.hist(attack_score, 1000, density=True, histtype='step', cumulative=True, label='Attack', color='firebrick')
            ax.set_xlabel('Score')
            ax.set_ylabel('CDF')
            ax.set_title(f'{detection_model} - {data_tag}')
            ax.vlines(selected_threshold, 0, 1.05, lw=1, color='grey')
            ax.legend(loc='right')
            fig.savefig(f'{fig_path}/{data_tag}_{detection_model}_result.png', dpi=600, format='png')
            plt.close(fig)
        except Exception as e:
            logging.warning(f'Failed to save figure: {e}')

        fout.close()
        return roc_auc, f1, per, rec, acc, auprc

    # ── MLP end-to-end path (graph_classifier) ──
    configure_downstream_mode(model, canonical_mode)
    opt = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(1, patience // 3)
    )

    infer_batch_size = 300
    num_train = train_data_seq.size(0)
    num_test = test_data_seq.size(0)
    if num_train == 0 or num_test == 0:
        logging.warning(f'[{data_tag}] Empty train/test split: train={num_train}, test={num_test}')
        fout.close()
        return float('nan'), 0.0, 0.0, 0.0, 0.0, float('nan')

    if external_val:
        # 全量训练集拟合；验证来自 data_split.val，不在 train 张量内。
        fit_indices = np.arange(int(train_label.numel()), dtype=np.int64)
        validation_indices = None
        validation_indices_t = None
    else:
        train_indices, validation_indices = stratified_train_validation_indices(
            train_label.cpu().tolist(), validation_ratio=validation_ratio, seed=seed
        )
        fit_indices = np.asarray(train_indices, dtype=np.int64)
        validation_indices_t = torch.as_tensor(validation_indices, dtype=torch.long)
    scaled_context, scaler_state = fit_and_apply_context_scalers(
        train_data_interact,
        train_node,
        train_edge,
        test_data_interact,
        test_node,
        test_edge,
        fit_indices,
    )
    (
        train_data_interact,
        train_node,
        train_edge,
        test_data_interact,
        test_node,
        test_edge,
    ) = scaled_context
    def scale_neighbors(rows: torch.Tensor | None, mask: torch.Tensor | None, state: dict):
        if rows is None:
            return None
        if rows.ndim != 3:
            raise ValueError("Neighbor context must have shape (rows, neighbors, features).")
        center = torch.tensor(state["center"], dtype=rows.dtype).view(1, 1, -1)
        scale = torch.tensor(state["scale"], dtype=rows.dtype).view(1, 1, -1)
        scaled = (rows - center) / scale
        return scaled.masked_fill(~mask.to(dtype=torch.bool).unsqueeze(-1), 0.0) if mask is not None else scaled

    train_neighbor_node = scale_neighbors(train_neighbor_node, train_neighbor_mask, scaler_state["node"])
    train_neighbor_edge = scale_neighbors(train_neighbor_edge, train_neighbor_mask, scaler_state["edge"])
    test_neighbor_node = scale_neighbors(test_neighbor_node, test_neighbor_mask, scaler_state["node"])
    test_neighbor_edge = scale_neighbors(test_neighbor_edge, test_neighbor_mask, scaler_state["edge"])
    # 外部验证集用同一 scaler 坐标系做仿射变换，不重新拟合。
    val_indices_t = None
    if external_val:
        val_data_interact, val_node, val_edge = apply_context_state(
            scaler_state["host"], scaler_state["node"], scaler_state["edge"],
            val_data_interact, val_node, val_edge,
        )
        val_neighbor_node = scale_neighbors(val_neighbor_node, val_neighbor_mask, scaler_state["node"])
        val_neighbor_edge = scale_neighbors(val_neighbor_edge, val_neighbor_mask, scaler_state["edge"])
        val_indices_t = torch.arange(int(val_label.numel()), dtype=torch.long)
    effective_graph_mode = canonical_mode
    fit_labels = train_label[fit_indices].to(torch.long)
    class_counts = torch.bincount(fit_labels, minlength=2).to(torch.float32)
    class_weights = class_counts.sum() / (2.0 * class_counts.clamp_min(1.0))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    def infer_scores(indices: torch.Tensor) -> np.ndarray:
        scores = np.zeros(indices.numel(), dtype=np.float64)
        model.eval()
        with torch.no_grad():
            for start in range(0, indices.numel(), infer_batch_size):
                selected = indices[start:min(start + infer_batch_size, indices.numel())]
                x_seq = train_data_seq[selected].to(device)
                x_non_seq = train_data_non_seq[selected].to(device)
                x_interact = train_data_interact[selected].to(device)
                n_emb = train_node[selected].to(device)
                e_emb = train_edge[selected].to(device)
                n_neighbor = train_neighbor_node[selected].to(device) if train_neighbor_node is not None else None
                e_neighbor = train_neighbor_edge[selected].to(device) if train_neighbor_edge is not None else None
                neighbor_valid = train_neighbor_mask[selected].to(device) if train_neighbor_mask is not None else None
                logits = model.classify(
                    x_seq, x_non_seq, x_interact, n_emb, e_emb,
                    flow_sizes=train_size[selected].to(device),
                    graph_mode=effective_graph_mode,
                    neighbor_node=n_neighbor,
                    neighbor_edge=e_neighbor,
                    neighbor_mask=neighbor_valid,
                )
                scores[start:start + selected.numel()] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return scores

    def infer_val_scores(indices: torch.Tensor) -> np.ndarray:
        """在外部验证集（data_split.val）张量上推理异常分数。"""
        scores = np.zeros(indices.numel(), dtype=np.float64)
        model.eval()
        with torch.no_grad():
            for start in range(0, indices.numel(), infer_batch_size):
                selected = indices[start:min(start + infer_batch_size, indices.numel())]
                logits = model.classify(
                    val_data_seq[selected].to(device),
                    val_data_non_seq[selected].to(device),
                    val_data_interact[selected].to(device),
                    val_node[selected].to(device),
                    val_edge[selected].to(device),
                    flow_sizes=val_size[selected].to(device),
                    graph_mode=effective_graph_mode,
                    neighbor_node=val_neighbor_node[selected].to(device) if val_neighbor_node is not None else None,
                    neighbor_edge=val_neighbor_edge[selected].to(device) if val_neighbor_edge is not None else None,
                    neighbor_mask=val_neighbor_mask[selected].to(device) if val_neighbor_mask is not None else None,
                )
                scores[start:start + selected.numel()] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return scores

    # 1. 训练 graph_attn 模块；仅验证集决定最佳 epoch。
    logging.info(f'[{data_tag}] Training Graph-Enhanced Cross-Modal Attention...')
    model.train()
    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    rng = np.random.default_rng(seed)
    training_history = []
    training_started = time.perf_counter()
    for epoch in range(num_epoch):
        perm = rng.permutation(fit_indices)
        epoch_loss = 0.0
        num_batch = 0
        for idx in iter_index_batches(perm, infer_batch_size):
            x_seq = train_data_seq[idx].to(device)
            x_non_seq = train_data_non_seq[idx].to(device)
            x_interact = train_data_interact[idx].to(device)
            n_emb = train_node[idx].to(device)
            e_emb = train_edge[idx].to(device)
            n_neighbor = train_neighbor_node[idx].to(device) if train_neighbor_node is not None else None
            e_neighbor = train_neighbor_edge[idx].to(device) if train_neighbor_edge is not None else None
            neighbor_valid = train_neighbor_mask[idx].to(device) if train_neighbor_mask is not None else None
            y = train_label[idx].to(device)

            opt.zero_grad()
            logits = model.classify(
                x_seq, x_non_seq, x_interact, n_emb, e_emb,
                flow_sizes=train_size[idx].to(device),
                graph_mode=effective_graph_mode,
                neighbor_node=n_neighbor,
                neighbor_edge=e_neighbor,
                neighbor_mask=neighbor_valid,
            )
            # ★ Fix: CrossEntropyLoss 需要 Long 类型的 target
            y = y.long()
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )
            opt.step()
            epoch_loss += loss.item()
            num_batch += 1
        validation_scores = (
            infer_val_scores(val_indices_t) if external_val
            else infer_scores(validation_indices_t)
        )
        validation_labels = (
            val_label.cpu().numpy() if external_val
            else train_label[validation_indices_t].cpu().numpy()
        )
        eps = 1e-7
        clipped = np.clip(validation_scores, eps, 1.0 - eps)
        validation_loss = float(np.mean(
            -(validation_labels * np.log(clipped) + (1 - validation_labels) * np.log(1 - clipped))
        ))
        scheduler.step(validation_loss)
        improved = validation_loss < best_validation_loss - 1e-8
        if improved:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        training_history.append({
            "epoch": epoch + 1,
            "fit_loss": float(epoch_loss / max(num_batch, 1)),
            "validation_loss": validation_loss,
            "learning_rate": float(opt.param_groups[0]["lr"]),
            "best_so_far": bool(improved),
        })
        model.train()
        if (epoch + 1) % 10 == 0:
            print(f'[{data_tag}] Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss / num_batch:.4f}',
                  file=fout, flush=True)
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state, strict=True)
    training_seconds = time.perf_counter() - training_started
    best_epoch = max(
        (
            int(item["epoch"])
            for item in training_history
            if item["best_so_far"]
        ),
        default=0,
    )
    evaluation_started = time.perf_counter()
    if strategy == THRESHOLD_STRATEGY_VALIDATION:
        if external_val:
            validation_scores = infer_val_scores(val_indices_t)
            validation_labels = val_label.cpu().numpy().tolist()
        else:
            validation_scores = infer_scores(validation_indices_t)
            validation_labels = train_label[validation_indices_t].cpu().numpy().tolist()
    else:
        validation_scores = None
        validation_labels = None

    selected_threshold, threshold_source = select_threshold_for_strategy(
        strategy,
        validation_labels,
        validation_scores.tolist() if validation_scores is not None else None,
        objective=threshold_objective,
        data_tag=data_tag,
    )
    logging.info(
        f'[{data_tag}] threshold_strategy={strategy}, '
        f'threshold={selected_threshold:.6f} ({threshold_source})'
    )

    # 2. 测试
    logging.info(f'[{data_tag}] Testing Graph_Tfusion model...')
    model.eval()
    test_scores = np.zeros(num_test)
    with torch.no_grad():
        for i in range(0, num_test, infer_batch_size):
            x_seq = test_data_seq[i:min(i + infer_batch_size, num_test)].to(device)
            x_non_seq = test_data_non_seq[i:min(i + infer_batch_size, num_test)].to(device)
            x_interact = test_data_interact[i:min(i + infer_batch_size, num_test)].to(device)
            n_emb = test_node[i:min(i + infer_batch_size, num_test)].to(device)
            e_emb = test_edge[i:min(i + infer_batch_size, num_test)].to(device)
            n_neighbor = test_neighbor_node[i:min(i + infer_batch_size, num_test)].to(device) if test_neighbor_node is not None else None
            e_neighbor = test_neighbor_edge[i:min(i + infer_batch_size, num_test)].to(device) if test_neighbor_edge is not None else None
            neighbor_valid = test_neighbor_mask[i:min(i + infer_batch_size, num_test)].to(device) if test_neighbor_mask is not None else None
            logits = model.classify(
                x_seq, x_non_seq, x_interact, n_emb, e_emb,
                flow_sizes=test_size[i:min(i + infer_batch_size, num_test)].to(device),
                graph_mode=effective_graph_mode,
                neighbor_node=n_neighbor,
                neighbor_edge=e_neighbor,
                neighbor_mask=neighbor_valid,
            )
            # 用 softmax 概率作为异常分数
            probs = torch.softmax(logits, dim=-1)
            test_scores[i:min(i + infer_batch_size, num_test)] = probs[:, 1].cpu().numpy()

    # 3. 评估指标
    loss_res = test_scores.tolist()
    label_res = test_label.cpu().numpy().tolist()

    metrics = compute_binary_metrics(label_res, loss_res, selected_threshold)
    metrics["model_mode"] = model_mode
    metrics["threshold_strategy"] = strategy
    metrics["threshold_source"] = threshold_source
    warn_if_test_set_single_class(data_tag, metrics, fout=fout)
    metrics.update(model_parameter_counts(model))
    metrics["best_epoch"] = best_epoch
    metrics["epochs_ran"] = len(training_history)
    metrics["training_seconds"] = float(training_seconds)
    metrics["evaluation_seconds"] = float(
        time.perf_counter() - evaluation_started
    )
    roc_auc = metrics["auc"]
    f1 = metrics["f1"]
    per = metrics["precision"]
    rec = metrics["recall"]
    acc = metrics["accuracy"]
    auprc = metrics["auprc"]

    print(f'[{data_tag}] AUC: {roc_auc:.4f}, F1: {f1:.4f}, P: {per:.4f}, R: {rec:.4f}, Acc: {acc:.4f}, AUPRC: {auprc:.4f}, Attack-F1: {metrics["attack_f1"]:.4f}, Threshold: {selected_threshold:.6f}, Support(benign/attack): {metrics["support_benign"]}/{metrics["support_attack"]}, Attack-P/R: {metrics["attack_precision"]:.4f}/{metrics["attack_recall"]:.4f}, FPR/FNR: {metrics["fpr"]:.4f}/{metrics["fnr"]:.4f}',
          file=fout, flush=True)
    print(f'[{data_tag}] AUC: {roc_auc:.4f}, F1: {f1:.4f}, P: {per:.4f}, R: {rec:.4f}, Acc: {acc:.4f}, AUPRC: {auprc:.4f}, Attack-F1: {metrics["attack_f1"]:.4f}, Threshold: {selected_threshold:.6f}, Support(benign/attack): {metrics["support_benign"]}/{metrics["support_attack"]}, Attack-P/R: {metrics["attack_precision"]:.4f}/{metrics["attack_recall"]:.4f}, FPR/FNR: {metrics["fpr"]:.4f}/{metrics["fnr"]:.4f}')

    if detection_model_path:
        os.makedirs(detection_model_path, exist_ok=True)
        classifier_path = graph_classifier_checkpoint_path(detection_model_path, data_tag)
        save_model_checkpoint(model, classifier_path)
        provenance_path = os.path.join(detection_model_path, "checkpoint_provenance.json")
        with open(provenance_path, "w", encoding="utf-8") as target:
            json.dump(checkpoint_provenance, target, indent=2, sort_keys=True)
            target.write("\n")
        save_evaluation_artifacts(
            detection_model_path,
            metrics=metrics,
            scores=loss_res,
            labels=label_res,
            validation_threshold=selected_threshold,
            seed=seed,
            scaler_state=scaler_state,
            window_ids=(
                test_window_ids.cpu().tolist()
                if test_window_ids is not None else None
            ),
        )
        save_training_provenance(
            detection_model_path,
            fit_indices=fit_indices,
            validation_indices=validation_indices,
            train_labels=train_label.cpu().tolist(),
            test_labels=label_res,
            history=training_history,
            scaler_state=scaler_state,
            node=test_node,
            edge=test_edge,
            model_mode=model_mode,
            seed=seed,
        )
        logging.info(f'Save Graph_Tfusion graph classifier to {classifier_path}')

    # 4. CDF 图
    try:
        fig = plt.figure(figsize=(10, 10 * 0.618), constrained_layout=True)
        ax = fig.subplots(1, 1)
        benign_score = [s for s, l in zip(loss_res, label_res) if l == 0]
        attack_score = [s for s, l in zip(loss_res, label_res) if l == 1]
        if benign_score:
            ax.hist(benign_score, 1000, density=True, histtype='step', cumulative=True, label='Benign', color='royalblue')
        if attack_score:
            ax.hist(attack_score, 1000, density=True, histtype='step', cumulative=True, label='Attack', color='firebrick')
        ax.set_xlabel('Graph_Tfusion Score')
        ax.set_ylabel('CDF')
        ax.set_title(f'Graph_Tfusion - {data_tag}')
        ax.vlines(selected_threshold, 0, 1.05, lw=1, color='grey')
        ax.legend(loc='right')
        save_addr = f'{fig_path}/{data_tag}_pp_result.png'
        fig.savefig(save_addr, dpi=600, format='png')
        plt.close(fig)
    except Exception as e:
        logging.warning(f'Failed to save figure: {e}')

    fout.close()
    return roc_auc, f1, per, rec, acc, auprc
