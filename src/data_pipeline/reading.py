"""数据集 JSON 的读取。"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import ujson as json


def _dataset_file(data_path: str, data_tag: str) -> str:
    if data_path.lower().endswith(".json"):
        return data_path
    return os.path.join(data_path, f"{data_tag}.json")


def _read_flow_sets(data_tag: str, data_path: str) -> Tuple[List[dict], List[dict]]:
    """读取单文件数据集（预训练使用），返回 (benign, attack)。"""
    data_target = _dataset_file(data_path, data_tag)
    if not os.path.exists(data_target):
        logging.warning(f"Target dataset not found: {data_target}")
        raise FileNotFoundError(data_target)

    with open(data_target, "r") as f:
        payload = json.load(f)
    benign = payload.get("benign") or []
    attack = payload.get("attack") or []
    logging.info(f"[{data_tag}] All Benign Flow: {len(benign)}, All Attack Flow: {len(attack)}")
    return benign, attack


def read_split_flow_file(path: str) -> Tuple[List[dict], List[dict]]:
    """读取一个预切分文件（train/test/val）的 (benign, attack) 列表。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pre-split dataset file not found: {path}"
        )
    with open(path, "r") as f:
        payload = json.load(f)
    benign = payload.get("benign") or []
    attack = payload.get("attack") or []
    logging.info(f"[pre-split] {path}: Benign {len(benign)}, Attack {len(attack)}")
    return benign, attack


def read_pre_split_flow_sets(
    data_split: Optional[dict],
) -> Optional[dict]:
    """按 data_split 配置读取 train/test/val 三份预切分数据。

    ``data_split`` 为空或 ``use_pre_split`` 为假时返回 None；正式训练入口
    会拒绝该情况，因此运行时只有预切分一条路径。train/test 必读；
    val 仅在 ``enable`` 为真时读取，否则该键为 None。
    """
    if not data_split or not data_split.get("use_pre_split"):
        return None

    def read_split(name: str):
        entry = data_split.get(name) or {}
        if not entry.get("enable"):
            return None
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"data_split.{name}.path must be set when enabled.")
        return read_split_flow_file(path)

    result = {
        "train": read_split("train"),
        "test": read_split("test"),
        "val": read_split("val"),
    }
    if result["train"] is None or result["test"] is None:
        raise ValueError(
            "data_split requires both train and test to be enabled; "
            "only val may be disabled."
        )
    return result
