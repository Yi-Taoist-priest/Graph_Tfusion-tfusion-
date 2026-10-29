"""决策阈值策略。

下游只保留两种策略：

- ``fixed_0.5``：判定阈值固定为 0.5；
- ``validation``：验证集三级协议（macro-F1 搜索 → 先验匹配 → 0.5）。

旧的 ``quantile`` / ``prior`` 迁移只保留为历史 API，不再进入训练路径。
"""

from __future__ import annotations

import logging

from .constants import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_THRESHOLD_STRATEGY,
    MIN_THRESHOLD_VALIDATION_PER_CLASS,
    THRESHOLD_STRATEGIES,
    THRESHOLD_STRATEGY_FIXED,
    THRESHOLD_STRATEGY_VALIDATION,
)
from .metrics import select_threshold


_STRATEGY_ALIASES = {
    "fixed": THRESHOLD_STRATEGY_FIXED,
    "fixed_0.5": THRESHOLD_STRATEGY_FIXED,
    "fixed0.5": THRESHOLD_STRATEGY_FIXED,
    "0.5": THRESHOLD_STRATEGY_FIXED,
    "validation": THRESHOLD_STRATEGY_VALIDATION,
    "val": THRESHOLD_STRATEGY_VALIDATION,
}


def normalize_threshold_strategy(value: str) -> str:
    """把配置中的策略名归一为两个合法值之一。"""
    normalized = str(value).strip().lower()
    strategy = _STRATEGY_ALIASES.get(normalized)
    if strategy is None:
        raise ValueError(
            f"threshold_strategy must be one of {THRESHOLD_STRATEGIES}, "
            f"got {value!r}."
        )
    return strategy


def resolve_threshold_strategy(threshold_strategy=None, optimize_threshold=None) -> str:
    """解析策略；新字段优先，旧 ``optimize_threshold`` 自动映射。

    - 显式 ``threshold_strategy`` 最优先；
    - 否则 ``optimize_threshold=False`` → ``fixed_0.5``，``True`` → ``validation``；
    - 两者都缺省时沿用历史默认 ``validation``。
    """
    if threshold_strategy is not None and str(threshold_strategy).strip():
        return normalize_threshold_strategy(threshold_strategy)
    if optimize_threshold is not None:
        return (
            THRESHOLD_STRATEGY_VALIDATION
            if bool(optimize_threshold)
            else THRESHOLD_STRATEGY_FIXED
        )
    return DEFAULT_THRESHOLD_STRATEGY


def prior_matching_threshold(scores, prior: float) -> float:
    """Cutoff whose predicted-positive rate matches a target class prior.

    Random-forest probabilities on imbalanced traffic are not calibrated, so a
    fixed 0.5 cutoff systematically under-predicts the minority class. Picking
    the cutoff that reproduces the observed prior is class-agnostic,
    dataset-agnostic, and stable when only a handful of rows are available.
    """
    scores = [float(score) for score in scores]
    unique_scores = sorted(set(scores))
    candidates = [
        (left + right) / 2.0
        for left, right in zip(unique_scores, unique_scores[1:])
    ]
    # 全部分数相同时没有非退化候选，只能退回默认阈值。
    if not candidates:
        return DEFAULT_DECISION_THRESHOLD
    total = len(scores)
    best = None
    for threshold in candidates:
        predicted_positive_rate = sum(
            1 for score in scores if score >= threshold
        ) / total
        rank = (
            abs(predicted_positive_rate - prior),
            abs(threshold - DEFAULT_DECISION_THRESHOLD),
            -threshold,
        )
        if best is None or rank < best[0]:
            best = (rank, threshold)
    return float(best[1])


def quantile_threshold_for_rate(scores, rate: float) -> float:
    """Cutoff on ``scores`` whose predicted-positive rate equals ``rate``.

    候选阈值取相邻唯一分数的中点，与 ``select_threshold`` / ``prior_matching_threshold``
    保持同一约定：判定用严格 ``score > threshold``（见 ``compute_binary_metrics``），
    若直接返回分数原值，离散分数（如 kNN 的 n_neighbors=3 只有 4 档取值）会因
    差一个边界而整体判负，出现 AUC 很高但 Attack-F1=0 的退化结果。
    """
    target = sorted(float(score) for score in scores)
    if not target:
        return DEFAULT_DECISION_THRESHOLD
    rate = min(max(float(rate), 0.0), 1.0)
    if rate <= 0.0:
        return target[-1]
    if rate >= 1.0:
        return target[0]
    unique_scores = sorted(set(target))
    candidates = [
        (left + right) / 2.0
        for left, right in zip(unique_scores, unique_scores[1:])
    ]
    # 分数只有单一取值时没有非退化候选，只能退回默认阈值。
    if not candidates:
        return DEFAULT_DECISION_THRESHOLD
    total = len(target)
    best = None
    for threshold in candidates:
        predicted_positive_rate = sum(
            1 for score in target if score > threshold
        ) / total
        rank = (
            abs(predicted_positive_rate - rate),
            abs(threshold - DEFAULT_DECISION_THRESHOLD),
            -threshold,
        )
        if best is None or rank < best[0]:
            best = (rank, threshold)
    return float(best[1])


def transfer_threshold_by_quantile(reference_scores, target_scores,
                                   threshold: float) -> float:
    """Re-express a chosen cutoff as an equal-positive-rate quantile.

    Tree-ensemble probabilities are not calibrated, so the score scale can
    drift between the rows a threshold was tuned on and the evaluated
    population: benign evaluation rows may score 0.84 while benign validation
    rows scored below 0.5, which turns a cutoff of 0.5 into "predict attack".
    The ranking survives (AUC stays high) but the absolute cutoff stops
    meaning anything. Mapping the validation operating point onto the target
    population's matching positive-rate quantile removes that scale
    dependence. Only unlabeled target scores are used, so no class
    information can leak from the evaluation split.
    """
    reference = [float(score) for score in reference_scores]
    if not reference or not target_scores:
        return float(threshold)
    positive_rate = sum(1 for score in reference if score >= threshold) / len(reference)
    return quantile_threshold_for_rate(target_scores, positive_rate)


def assert_formal_threshold_transfer(mode: str) -> None:
    """Formal runs only permit the absolute validation threshold.

    ``prior`` pins the operating point to a class prevalence and ``quantile``
    re-expresses the cutoff on the evaluated population's score distribution;
    both read information from the population under evaluation and are
    therefore forbidden in the formal protocol.
    """
    if mode != "absolute":
        raise ValueError(
            "Formal runs only allow threshold_transfer='absolute'; "
            f"got {mode!r}."
        )


def resolve_decision_threshold(threshold, reference_scores, target_scores,
                               mode: str = "absolute", data_tag: str = "",
                               prior_rate: float | None = None) -> float:
    """Deprecated: 仅保留历史 API；训练路径不再调用。

    Apply the configured threshold-transfer mode to a selected cutoff.

    ``quantile`` keeps whatever positive rate the validation split produced,
    which still inherits that split's class balance. ``prior`` instead pins the
    predicted-positive rate to the class's natural prevalence, so a balanced
    validation subset can no longer mis-set the operating point for a skewed
    evaluation population.
    """
    if mode == "absolute":
        return float(threshold)
    if mode == "quantile":
        transferred = transfer_threshold_by_quantile(
            reference_scores, target_scores, threshold
        )
    elif mode == "prior":
        if prior_rate is None or not 0.0 < float(prior_rate) < 1.0:
            raise ValueError(
                "threshold_transfer='prior' requires an attack prevalence in (0, 1)."
            )
        transferred = quantile_threshold_for_rate(target_scores, prior_rate)
    else:
        raise ValueError(
            "threshold_transfer must be 'absolute', 'quantile' or 'prior', "
            f"got {mode!r}."
        )
    logging.info(
        f"[{data_tag}] threshold_transfer={mode}: validation cutoff "
        f"{threshold:.6f} re-expressed as {transferred:.6f} on the evaluated "
        f"score distribution."
    )
    return transferred


def resolve_validation_threshold(labels, scores, objective: str = "macro_f1",
                                 data_tag: str = ""):
    """验证集三级协议，返回 ``(threshold, source)``。

    - 双类且每类 >= 10：macro-F1 搜索，source=``validation_search``；
    - 双类但少数类 < 10：先验匹配，source=``prior_matching``；
    - 单类或空：0.5，source=``default_0.5``。
    """
    labels = [int(label) for label in labels]
    n_attack = sum(labels)
    n_benign = len(labels) - n_attack

    if min(n_attack, n_benign) >= MIN_THRESHOLD_VALIDATION_PER_CLASS:
        return (
            select_threshold(labels, scores, objective=objective),
            "validation_search",
        )

    if n_attack and n_benign:
        prior = n_attack / len(labels)
        threshold = prior_matching_threshold(scores, prior)
        logging.warning(
            f"[{data_tag}] Validation split has only {n_benign} benign / "
            f"{n_attack} attack rows (< {MIN_THRESHOLD_VALIDATION_PER_CLASS} "
            f"per class); using prior-matching threshold {threshold:.6f} "
            f"(attack prior {prior:.4f}) instead of a blind "
            f"{DEFAULT_DECISION_THRESHOLD} cutoff."
        )
        return threshold, "prior_matching"

    logging.warning(
        f"[{data_tag}] Validation split is single-class "
        f"(benign={n_benign}, attack={n_attack}); no class prior is "
        f"estimable, using default threshold {DEFAULT_DECISION_THRESHOLD}."
    )
    return DEFAULT_DECISION_THRESHOLD, "default_0.5"


def choose_validation_threshold(labels, scores, objective: str = "macro_f1",
                                data_tag: str = "") -> float:
    """兼容旧调用：只返回验证集三级协议的阈值。"""
    threshold, _ = resolve_validation_threshold(
        labels, scores, objective=objective, data_tag=data_tag
    )
    return threshold


def select_threshold_for_strategy(strategy: str, labels=None, scores=None,
                                  objective: str = "macro_f1",
                                  data_tag: str = ""):
    """按两种策略之一返回 ``(threshold, source)``。"""
    strategy = normalize_threshold_strategy(strategy)
    if strategy == THRESHOLD_STRATEGY_FIXED:
        return float(DEFAULT_DECISION_THRESHOLD), "fixed_0.5"
    if labels is None or scores is None:
        raise ValueError(
            "threshold_strategy='validation' requires validation labels and scores."
        )
    return resolve_validation_threshold(
        labels, scores, objective=objective, data_tag=data_tag
    )
