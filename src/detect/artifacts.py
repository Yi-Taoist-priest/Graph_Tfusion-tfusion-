"""评估指标、逐行分数与训练溯源信息的原子落盘。"""

from __future__ import annotations

import hashlib
import json
import os

import torch


def save_evaluation_artifacts(
    output_dir: str,
    metrics: dict,
    scores,
    labels,
    validation_threshold: float,
    seed: int,
    scaler_state: dict | None = None,
    window_ids=None,
) -> None:
    """Persist compact metrics and row-level scores without overwriting via partial writes."""
    os.makedirs(output_dir, exist_ok=True)
    metric_payload = {key: value for key, value in metrics.items() if key != "judge"}
    metric_payload["validation_threshold"] = float(validation_threshold)
    metric_payload["seed"] = int(seed)
    score_payload = {
        "labels": [int(label) for label in labels],
        "scores": [float(score) for score in scores],
    }
    if window_ids is not None:
        if len(window_ids) != len(score_payload["labels"]):
            raise ValueError("window_ids must align with evaluation labels and scores.")
        score_payload["window_ids"] = [int(window_id) for window_id in window_ids]
    for filename, payload in (("metrics.json", metric_payload), ("scores.json", score_payload)):
        final_path = os.path.join(output_dir, filename)
        temporary_path = final_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as target:
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
        os.replace(temporary_path, final_path)
    if scaler_state is not None:
        final_path = os.path.join(output_dir, "scalers.json")
        temporary_path = final_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as target:
            json.dump(scaler_state, target, indent=2, sort_keys=True)
            target.write("\n")
        os.replace(temporary_path, final_path)


def save_training_provenance(
    output_dir: str,
    fit_indices,
    validation_indices,
    train_labels,
    test_labels,
    history,
    scaler_state: dict,
    node: torch.Tensor,
    edge: torch.Tensor,
    model_mode: str,
    seed: int,
) -> None:
    """Save immutable evidence for the split, context schema, and graph rows."""
    os.makedirs(output_dir, exist_ok=True)

    def support(labels):
        values = [int(label) for label in labels]
        return {"benign": values.count(0), "attack": values.count(1)}

    train_labels = [int(label) for label in train_labels]
    fit_indices = [int(index) for index in fit_indices]
    # 外部验证集（data_split.val）不落在 train 张量内，故无索引可言。
    external_validation = validation_indices is None
    split_payload = {
        "fit_indices": fit_indices,
        "validation_indices": (
            None if external_validation
            else [int(index) for index in validation_indices]
        ),
        "external_validation": bool(external_validation),
        "supports": {
            "fit": support(train_labels[index] for index in fit_indices),
            "validation": (
                None if external_validation
                else support(train_labels[index] for index in validation_indices)
            ),
            "test": support(test_labels),
        },
        "seed": int(seed),
    }
    schema_payload = {
        "host_feature_names": list(scaler_state["host"]["feature_names"]),
        "node_feature_names": list(scaler_state["node"]["feature_names"]),
        "edge_feature_names": list(scaler_state["edge"]["feature_names"]),
    }
    digest = hashlib.sha256()
    for tensor in (node, edge):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    history_payload = {
        "model_mode": str(model_mode),
        "seed": int(seed),
        "graph_row_sha256": digest.hexdigest(),
        "epochs": list(history),
    }
    for filename, payload in (
        ("split_manifest.json", split_payload),
        ("context_schema.json", schema_payload),
        ("history.json", history_payload),
    ):
        final_path = os.path.join(output_dir, filename)
        temporary_path = final_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as target:
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
        os.replace(temporary_path, final_path)
