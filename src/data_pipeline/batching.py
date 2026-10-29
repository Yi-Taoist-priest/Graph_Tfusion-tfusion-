"""对比学习预训练 batch 的构造（同源正对 + 异源负对）。

抽样必须由显式 ``seed`` 驱动（``random.Random(seed)``），不得依赖全局
``random`` 状态，否则同一数据在不同进程会得到不同批次，破坏同 seed 可比性。
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

DEFAULT_CONTRAST_SEED = 777


def _sample_k(pool: Sequence[int], k: int, rng: random.Random) -> List[int]:
    if k <= 0:
        return []
    if len(pool) < k:
        return []
    if not isinstance(pool, (list, range)):
        pool = list(pool)
    return rng.sample(pool, k)


def _build_pretrain_batches(
    pre_train_data: List[Tuple],
    pre_train_ids: List[List[int]],
    contrast_batch_size: int,
    seed: int = DEFAULT_CONTRAST_SEED,
) -> List[List[Tuple]]:
    if contrast_batch_size <= 0 or len(pre_train_data) < contrast_batch_size:
        return []
    if contrast_batch_size < 4 or contrast_batch_size % 2:
        raise ValueError("contrast_batch_size must be an even integer of at least 4.")
    if seed is None:
        raise ValueError(
            "contrastive batch construction requires an explicit seed so that "
            "identical inputs produce identical batches across processes."
        )
    rng = random.Random(int(seed))

    half = contrast_batch_size // 2
    batch_num = len(pre_train_data) // contrast_batch_size
    pre_train_batch: List[List[Tuple]] = []
    src_ctr: Dict[int, List[int]] = {}
    dst_ctr: Dict[int, List[int]] = {}

    for i, (src, dst) in enumerate(pre_train_ids):
        src_ctr.setdefault(src, []).append(i)
        dst_ctr.setdefault(dst, []).append(i)

    all_indices = list(range(len(pre_train_data)))
    diff_cache: Dict[int, List[int] | None] = {}

    def diff_pool_for(groups: Dict[int, List[int]], key: int) -> List[int] | None:
        """大组的异源池只构造一次；小组用拒绝采样，避免每批 O(N) 扫描。"""
        if key not in diff_cache:
            group = groups[key]
            if len(group) * 2 >= len(pre_train_data):
                group_set = set(group)
                diff_cache[key] = [idx for idx in all_indices if idx not in group_set]
            else:
                diff_cache[key] = None
        return diff_cache[key]

    def add_group_batches(groups: Dict[int, List[int]], rounds: int) -> None:
        keys = [k for k, v in groups.items() if len(v) >= half]
        if not keys:
            return
        for i in range(rounds):
            target_key = keys[i % len(keys)]
            same = _sample_k(groups[target_key], half, rng)
            needed = contrast_batch_size - len(same)
            pool = diff_pool_for(groups, target_key)
            if pool is not None:
                diff = _sample_k(pool, needed, rng)
            else:
                group_set = set(groups[target_key])
                diff = []
                while len(diff) < needed:
                    idx = rng.randrange(len(pre_train_data))
                    if idx not in group_set and idx not in diff:
                        diff.append(idx)
            if len(same) + len(diff) == contrast_batch_size:
                pre_train_batch.append([pre_train_data[x] for x in same + diff])

    add_group_batches(dst_ctr, batch_num // 2)
    add_group_batches(src_ctr, batch_num - len(pre_train_batch))

    return pre_train_batch
