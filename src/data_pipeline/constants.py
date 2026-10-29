"""数据管线的共享常量与最小样本约束。"""

from __future__ import annotations

import math
import sys


VOID_SEQ_FEATURE = [0, 0]


DEFAULT_D_GRAPH = 128


MIN_ATTACK_TRAIN_SIZE = 10


MIN_ATTACK_TEST_SIZE = 10


MIN_BENIGN_TRAIN_SIZE = 10


MIN_BENIGN_TEST_SIZE = 10


MAX_LOG1P_FLOAT64 = math.log(sys.float_info.max)


SPLIT_PRIOR_DIVERGENCE_TOLERANCE = 0.25


MIN_EVALUABLE_TEST_SUPPORT = 10

# 按比例抽样训练集时每类至少保留的行数：下游要先从训练集切出验证集，
# 拟合/验证各至少 1 行，故低于 2 行会让 stratified_train_validation_indices 报错。
MIN_ROWS_PER_CLASS = 2
