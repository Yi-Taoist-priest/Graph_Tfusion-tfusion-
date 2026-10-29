#!/usr/bin/env python3
"""Run immutable, full-test, multi-seed Graph_Tfusion experiments.

用法: python scripts/run_formal_matrix.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_ROOT, _ROOT / "src", _ROOT / "scripts", _ROOT / "scripts" / "pcap"):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import argparse
import copy
import hashlib
import json
import os
from typing import Dict, Iterable, List

from run_graph_tfusion import train_task_pp
from detect.constants import THRESHOLD_STRATEGIES


UNIFIED_DATASET_KEYS = {"data_tag", "data_path", "result_namespace"}
# 可选的数据集级键：给出时才随配置透传（data_split 走预切分文件路径）。
UNIFIED_OPTIONAL_DATASET_KEYS = {"data_split"}
FORMAL_MODEL_MODES = {"tfusion", "causal_graph_crossmodal"}


def _config_hash(config: Dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def merge_unified_config(common: Dict, dataset: Dict) -> Dict:
    extra = set(dataset) - UNIFIED_DATASET_KEYS - UNIFIED_OPTIONAL_DATASET_KEYS
    missing = {"data_tag", "data_path", "result_namespace"} - set(dataset)
    if extra or missing:
        raise ValueError(
            f"Unified dataset config keys are invalid; missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )
    merged = copy.deepcopy(common)
    for key in UNIFIED_DATASET_KEYS | UNIFIED_OPTIONAL_DATASET_KEYS:
        if key in dataset:
            merged[key] = copy.deepcopy(dataset[key])
    merged["common_config_hash"] = _config_hash(common)
    _validate_formal_config(merged)
    params = merged["data_construct_param"]
    if (
        merged.get("d_graph") != 16
        or params.get("host_feature_mode") != "causal"
        or params.get("graph_context_mode") != "full_stream_causal"
    ):
        raise ValueError(
            "Unified runs require d_graph=16, causal host context, and "
            "full_stream_causal graph context."
        )
    return merged


def _validate_formal_config(config: Dict) -> None:
    params = config.get("data_construct_param", {})
    if (
        config.get("command") != "train"
        or params.get("test_samp_ratio", 0) != 0
        or params.get("pre_train_ratio") != 0
        or params.get("host_feature_mode") == "stored"
        or params.get("graph_context_mode") == "window_batch"
    ):
        raise ValueError(
            "A formal run requires command=train, full testing, no implicit target pretraining, "
            "and causal or zero context features."
        )
    if len(config.get("data_tag", {})) != 1:
        raise ValueError("A formal base config must contain exactly one data tag.")
    train_param = config.get("train_param", {}) or {}
    strategy = train_param.get("threshold_strategy")
    if strategy is not None and strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(
            f"A formal run requires threshold_strategy in {THRESHOLD_STRATEGIES}; "
            f"got {strategy!r}."
        )
    threshold_transfer = train_param.get("threshold_transfer", "absolute")
    if threshold_transfer != "absolute":
        raise ValueError(
            "threshold_transfer is deprecated and formal runs only accept "
            f"'absolute'; got {threshold_transfer!r}. Use threshold_strategy "
            "instead."
        )


def expand_run_configs(
    base_config: Dict,
    seeds: Iterable[int],
    output_root: str,
    variant: str,
) -> List[Dict]:
    _validate_formal_config(base_config)
    data_tag = next(iter(base_config["data_tag"]))
    runs = []
    for seed in seeds:
        config = copy.deepcopy(base_config)
        run_root = os.path.join(output_root, data_tag, variant, f"seed_{int(seed)}")
        config.setdefault("train_param", {})["seed"] = int(seed)
        config["run_root"] = run_root
        config["log_path"] = os.path.join(run_root, "log")
        config["fig_path"] = os.path.join(run_root, "figure")
        config["detection_model_path"] = os.path.join(run_root, "checkpoint")
        runs.append(config)
    return runs


def expand_unified_matrix(
    dataset_configs: Iterable[Dict],
    modes: Iterable[str],
    seeds: Iterable[int],
    output_root: str,
    variant: str,
) -> List[Dict]:
    configs = [copy.deepcopy(config) for config in dataset_configs]
    mode_list = [str(mode) for mode in modes]
    seed_list = [int(seed) for seed in seeds]
    unknown_modes = set(mode_list) - FORMAL_MODEL_MODES
    if unknown_modes:
        raise ValueError(f"Unsupported formal model modes: {sorted(unknown_modes)}")
    if not configs or not mode_list or not seed_list:
        raise ValueError("Unified matrix requires datasets, modes, and seeds.")
    common_hashes = {config.get("common_config_hash") for config in configs}
    if len(common_hashes) != 1 or None in common_hashes:
        raise ValueError("All unified datasets must share one common_config_hash.")

    runs = []
    for base in configs:
        _validate_formal_config(base)
        data_tag = next(iter(base["data_tag"]))
        namespace = base["result_namespace"]
        for mode in mode_list:
            for seed in seed_list:
                config = copy.deepcopy(base)
                run_root = os.path.join(
                    output_root,
                    variant,
                    namespace,
                    mode,
                    f"seed_{seed}",
                )
                config["model_mode"] = mode
                config.setdefault("train_param", {})["seed"] = seed
                config["run_root"] = run_root
                config["log_path"] = os.path.join(run_root, "log")
                config["fig_path"] = os.path.join(run_root, "figure")
                config["detection_model_path"] = os.path.join(run_root, "checkpoint")
                config["data_tag"] = {data_tag: base["data_tag"][data_tag]}
                runs.append(config)
    if len({run["run_root"] for run in runs}) != len(runs):
        raise ValueError("Unified run matrix generated overlapping output paths.")
    return runs


def _write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, sort_keys=True)
        target.write("\n")
    os.replace(temporary, path)


def run_config(config: Dict) -> None:
    run_root = config["run_root"]
    if os.path.exists(run_root):
        raise FileExistsError(f"Formal run directory already exists: {run_root}")
    checkpoint = os.path.join(
        config["traffic_model_path"],
        f"{config['traffic_model_tag']}_pp_model.pt",
    )
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Required formal pretraining checkpoint not found: {checkpoint}")
    for key in ("log_path", "fig_path", "detection_model_path"):
        os.makedirs(config[key], exist_ok=False)
    _write_json(os.path.join(run_root, "run_config.json"), config)
    data_tag, waterline = next(iter(config["data_tag"].items()))
    gpu_ids = config.get("gpu_enable", [-1])
    gpu_id = gpu_ids[0] if gpu_ids else -1
    # 全部使用关键字参数，避免 train_task_pp 签名演进后再次发生位置错位
    # （历史上 d_fused_out 缺失导致 graph_feature_mode/model_mode/checkpoint_load_scope
    # 整体左移，正式运行静默命中 graph_feature_mode 守卫后空跑）。
    train_task_pp(
        command=config["command"],
        detection_model=config["detection_model"],
        data_tag=data_tag,
        data_path=config["data_path"],
        log_path=config["log_path"],
        fig_path=config["fig_path"],
        traffic_model_path=config.get("traffic_model_path"),
        traffic_model_tag=config.get("traffic_model_tag"),
        detection_model_path=config["detection_model_path"],
        gpu_id=gpu_id,
        waterline=waterline,
        data_set_params=config["data_construct_param"],
        pre_train_params=config["train_param"],
        d_graph=config.get("d_graph", 128),
        d_fused_out=config.get("d_fused_out", 384),
        graph_feature_mode=config.get("graph_feature_mode", "structural"),
        model_mode=config.get("model_mode", "causal_graph_crossmodal"),
        checkpoint_load_scope=config.get("checkpoint_load_scope", "backbone_only"),
        load_pretrained=config.get("load_pretrained", True),
        data_split=config.get("data_split"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config")
    parser.add_argument("--common")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=sorted(FORMAL_MODEL_MODES))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-root", default="runs/formal")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config:
        if args.common or args.datasets or args.modes:
            raise ValueError("Use either --config or the unified --common/--datasets/--modes form.")
        with open(args.config, "r", encoding="utf-8") as source:
            base = json.load(source)
        runs = expand_run_configs(base, args.seeds, args.output_root, args.variant)
    else:
        if not args.common or not args.datasets or not args.modes:
            raise ValueError("Unified runs require --common, --datasets, and --modes.")
        with open(args.common, "r", encoding="utf-8") as source:
            common = json.load(source)
        configs = []
        for path in args.datasets:
            with open(path, "r", encoding="utf-8") as source:
                configs.append(merge_unified_config(common, json.load(source)))
        runs = expand_unified_matrix(
            configs,
            args.modes,
            args.seeds,
            args.output_root,
            args.variant,
        )
    if args.dry_run:
        print(json.dumps(runs, indent=2, sort_keys=True))
        return
    for config in runs:
        run_config(config)


if __name__ == "__main__":
    main()
