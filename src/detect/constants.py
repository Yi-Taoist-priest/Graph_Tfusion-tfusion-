"""detect 包的常量与下游模式契约。

集中定义下游检测器名称、图模式枚举、checkpoint 契约版本，以及阈值选择
用到的小样本常量；detect 包内其他模块统一从这里引用，避免重复定义。
"""


n_hidden = 100


n_project = 128


d_graph = 128


d_model = 128


GRAPH_CLASSIFIER_NAME = "graph_classifier"


SKLEARN_DETECTOR_NAMES = {"kmeans", "iforest", "ocsvm", "rforest", "knn", "logistic", "bayes"}


ALL_DETECTION_MODELS = {GRAPH_CLASSIFIER_NAME} | SKLEARN_DETECTOR_NAMES


CHECKPOINT_CONTRACT_VERSION = 2


TFUSION_MODE = "tfusion"


CROSSMODAL_MODE = "causal_graph_crossmodal"


SUPPORTED_DOWNSTREAM_MODES = {
    TFUSION_MODE,
    CROSSMODAL_MODE,
}


REMOVED_DOWNSTREAM_MODES = {
    "causal_graph",
    "causal_graph_residual",
    "shuffled_graph",
}


def canonical_graph_mode(mode: str) -> str:
    if mode in REMOVED_DOWNSTREAM_MODES:
        raise ValueError(
            f"graph_mode={mode!r} was removed; use {CROSSMODAL_MODE!r}."
        )
    if mode not in SUPPORTED_DOWNSTREAM_MODES:
        raise ValueError(f"Unsupported graph_mode={mode!r}.")
    return mode


MIN_THRESHOLD_VALIDATION_PER_CLASS = 10


DEFAULT_DECISION_THRESHOLD = 0.5


# 下游只保留两种阈值策略：
# - fixed_0.5：判定阈值固定为 0.5，不读取验证集；
# - validation：验证集三级协议（macro-F1 搜索 → 先验匹配 → 0.5）。
THRESHOLD_STRATEGY_FIXED = "fixed_0.5"

THRESHOLD_STRATEGY_VALIDATION = "validation"

THRESHOLD_STRATEGIES = (
    THRESHOLD_STRATEGY_FIXED,
    THRESHOLD_STRATEGY_VALIDATION,
)

DEFAULT_THRESHOLD_STRATEGY = THRESHOLD_STRATEGY_VALIDATION
