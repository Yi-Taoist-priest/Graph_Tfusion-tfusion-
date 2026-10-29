"""预切分数据的可评估性告警。

正式运行只使用离线切分好的 ``train/test/val`` 文件（见 ``data_split``），
运行期不再做任何内存切分；这里只保留一个只读检查，用于提示训练/测试
类别先验不可迁移或某一类支持数过低的场景。
"""

from __future__ import annotations

import logging

from .constants import (
    MIN_EVALUABLE_TEST_SUPPORT,
    SPLIT_PRIOR_DIVERGENCE_TOLERANCE,
)


def warn_if_split_priors_diverge(
    data_tag: str,
    *,
    train_benign: int,
    train_attack: int,
    test_benign: int,
    test_attack: int,
    split_strategy: str,
    tolerance: float = SPLIT_PRIOR_DIVERGENCE_TOLERANCE,
    min_support: int = MIN_EVALUABLE_TEST_SUPPORT,
) -> None:
    """Flag a split whose class prior is not transferable to the test set.

    A time-ordered split on a capture whose classes are separated in time
    (background traffic recorded first, the attack scenario later) hands the
    model a training prior that does not describe the evaluation population.
    The selected threshold then no longer transports, and macro-F1 collapses
    while AUC stays high - the evaluation measures the protocol instead of
    the detector. Guarding on the prior (rather than on any single dataset)
    keeps the check meaningful for every capture in the suite.
    """
    n_train = train_benign + train_attack
    n_test = test_benign + test_attack
    if not n_train or not n_test:
        return
    train_prior = train_attack / n_train
    test_prior = test_attack / n_test
    divergence = abs(train_prior - test_prior)
    reasons = []
    if divergence > tolerance:
        reasons.append(
            f"attack prior differs by {divergence:.3f} (train {train_prior:.3f} "
            f"vs test {test_prior:.3f}, tolerance {tolerance})"
        )
    if min(test_benign, test_attack) < min_support:
        reasons.append(
            f"test support {test_benign}(benign)/{test_attack}(attack) leaves a "
            f"class below {min_support} rows"
        )
    if not reasons:
        return
    logging.warning(
        f"[{data_tag}] split_strategy={split_strategy!r} produced a "
        "non-transferable evaluation population: " + "; ".join(reasons) +
        ". Treat these metrics as a protocol artefact until the split is "
        "fixed; check whether class labels are confounded with capture time."
    )
