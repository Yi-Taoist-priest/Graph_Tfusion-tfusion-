"""二分类检测指标与验证集阈值搜索。"""

from __future__ import annotations

import logging

from sklearn.metrics import (
    average_precision_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_binary_metrics(label_res, score_res, waterline: float):
    """Compute threshold metrics plus ranking metrics for binary detection."""
    judge = [1 if sc > waterline else 0 for sc in score_res]
    if len(set(label_res)) >= 2:
        fpr, tpr, _ = roc_curve(label_res, score_res)
        roc_auc = auc(fpr, tpr)
        auprc = average_precision_score(label_res, score_res)
    else:
        roc_auc = float('nan')
        auprc = float('nan')

    f1 = f1_score(label_res, judge, average='macro', zero_division=0)
    per = precision_score(label_res, judge, average='macro', zero_division=0)
    rec = recall_score(label_res, judge, average='macro', zero_division=0)
    acc = sum([1 if x == y else 0 for x, y in zip(label_res, judge)]) / max(len(label_res), 1)
    tn = sum(1 for label, predicted in zip(label_res, judge) if label == 0 and predicted == 0)
    fp = sum(1 for label, predicted in zip(label_res, judge) if label == 0 and predicted == 1)
    fn = sum(1 for label, predicted in zip(label_res, judge) if label == 1 and predicted == 0)
    tp = sum(1 for label, predicted in zip(label_res, judge) if label == 1 and predicted == 1)
    attack_precision = tp / max(tp + fp, 1)
    attack_recall = tp / max(tp + fn, 1)
    attack_f1 = (
        2 * attack_precision * attack_recall / max(attack_precision + attack_recall, 1e-12)
    )
    return {
        "auc": roc_auc,
        "f1": f1,
        "precision": per,
        "recall": rec,
        "accuracy": acc,
        "auprc": auprc,
        "evaluable": len(set(label_res)) >= 2,
        "judge": judge,
        "threshold": float(waterline),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
        "macro_f1": f1,
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "support_benign": tn + fp,
        "support_attack": tp + fn,
    }


def select_threshold(labels, scores, objective: str = "attack_f1") -> float:
    """Select a decision threshold from validation labels and scores only."""
    labels = [int(label) for label in labels]
    scores = [float(score) for score in scores]
    if len(labels) != len(scores) or not labels:
        raise ValueError("Validation labels and scores must be non-empty and aligned.")
    if set(labels) != {0, 1}:
        raise ValueError("Validation threshold selection requires both classes.")
    if objective not in {"attack_f1", "macro_f1"}:
        raise ValueError("objective must be 'attack_f1' or 'macro_f1'.")
    unique_scores = sorted(set(scores))
    # 只取相邻唯一值的中点作为候选，排除两端退化解（全判正/全判负=不做检测）。
    # 退化解让多数类判对、macro_f1 虚高，且分数极端分布（单侧离群点）时会
    # 选中该退化阈值导致 tp=fp=0（如 IDS-17 tuesday 良性强离群点 9.318）。
    candidates = [(left + right) / 2.0
                  for left, right in zip(unique_scores, unique_scores[1:])]
    if not candidates:
        return float(unique_scores[0])
    ranked = []
    for threshold in candidates:
        metrics = compute_binary_metrics(labels, scores, threshold)
        ranked.append((
            metrics[objective],
            -metrics["fpr"],
            -abs(threshold - 0.5),
            -threshold,
            threshold,
        ))
    return float(max(ranked)[-1])


def warn_if_test_set_single_class(data_tag: str, metrics: dict, fout=None) -> None:
    """Surface non-evaluable test splits instead of silently emitting nan AUC."""
    if metrics.get("evaluable", True):
        return
    message = (
        f"[{data_tag}] WARNING test set is single-class "
        f"(benign={metrics['support_benign']}, attack={metrics['support_attack']}); "
        "AUC/AUPRC are undefined and macro metrics are invalid. Fix the split "
        "reservation rather than interpreting these numbers."
    )
    logging.warning(message)
    if fout is not None:
        print(message, file=fout, flush=True)
