"""模型 checkpoint 的读写与部分加载策略。

``load_model_checkpoint`` 支持 allow_partial，可在特征维度变更时按形状筛选
兼容权重；调用方负责把返回的 provenance 落盘，保证可追溯。

**形状不匹配必须显式暴露**：名字对得上但维度不同的键（典型成因是预训练与
下游的 ``d_graph`` / ``d_model`` 等架构宽度不一致）既不在 ``missing_keys``
（那是"模型缺、checkpoint 无"），也不该混进 ``ignored_keys`` 里被当作普通
多余键忽略——它们会被静默留在随机初始化上。这里单列
``shape_mismatch_keys`` 并发 WARNING，使"预训练权重其实没生效"可被机器校验。
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from .constants import CHECKPOINT_CONTRACT_VERSION


def graph_classifier_checkpoint_path(detection_model_path: str, data_tag: str) -> str:
    return os.path.join(detection_model_path, f"{data_tag}_pp_classifier.pt")


def save_model_checkpoint(model: nn.Module, path: str) -> None:
    torch.save({
        "contract_version": CHECKPOINT_CONTRACT_VERSION,
        "state_dict": model.state_dict(),
    }, path)


def load_model_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    allow_partial: bool = False,
    load_scope: str = "all_compatible",
) -> dict:
    if load_scope not in {"all_compatible", "backbone_only"}:
        raise ValueError("load_scope must be 'all_compatible' or 'backbone_only'.")
    payload = torch.load(path, map_location=device, weights_only=True)
    state_dict = payload.get("state_dict") if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {path} does not contain a model state dictionary.")
    if not allow_partial and load_scope != "all_compatible":
        raise ValueError("A restricted load_scope requires allow_partial=True.")
    if not allow_partial:
        model.load_state_dict(state_dict, strict=True)
        return {
            "path": path,
            "load_scope": load_scope,
            "partial": False,
            "loaded_key_count": len(state_dict),
            "loaded_backbone_key_count": sum(
                key.startswith("backbone.") for key in state_dict
            ),
            "missing_keys": [],
            "ignored_keys": [],
            "shape_mismatch_keys": {},
        }

    target_state = model.state_dict()
    in_scope = {
        key: value for key, value in state_dict.items()
        if load_scope != "backbone_only" or key.startswith("backbone.")
    }
    compatible = {
        key: value for key, value in in_scope.items()
        if key in target_state
        and tuple(value.shape) == tuple(target_state[key].shape)
    }
    # 名字命中但维度不同：会被静默留在随机初始化上，必须单列出来。
    shape_mismatch = {
        key: {"checkpoint": list(value.shape), "model": list(target_state[key].shape)}
        for key, value in in_scope.items()
        if key in target_state
        and tuple(value.shape) != tuple(target_state[key].shape)
    }
    if shape_mismatch:
        detail = "; ".join(
            f"{key} checkpoint{tuple(info['checkpoint'])} != model{tuple(info['model'])}"
            for key, info in sorted(shape_mismatch.items())
        )
        logging.warning(
            "[checkpoint] %d 个键形状不匹配，将保持随机初始化（预训练权重未生效）：%s",
            len(shape_mismatch), detail,
        )

    loaded_backbone = [key for key in compatible if key.startswith("backbone.")]
    if not loaded_backbone:
        raise ValueError(f"Checkpoint {path} contains no compatible tFusion backbone weights.")
    result = model.load_state_dict(compatible, strict=False)
    return {
        "path": path,
        "load_scope": load_scope,
        "partial": True,
        "loaded_key_count": len(compatible),
        "loaded_backbone_key_count": len(loaded_backbone),
        "missing_keys": sorted(result.missing_keys),
        "ignored_keys": sorted(set(state_dict) - set(compatible)),
        "shape_mismatch_keys": {key: shape_mismatch[key] for key in sorted(shape_mismatch)},
    }

