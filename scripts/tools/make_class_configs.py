#!/usr/bin/env python
"""Derive one tuning config per attack class from a dataset's shared config.

A dataset's ``all.json`` is the single source of truth for parameters that must
stay identical across classes (feature contract, split strategy, budget). This
script expands it into per-class configs so a single attack type can carry its
own training-stage parameters without diverging from the rest.

The measured need for this: no single threshold policy wins across classes.
On Kitsune chunk100 at seed 777 the per-class Attack-F1 is

    class            absolute   quantile   prior(prevalence)
    OS_Scan            0.5471     0.7195     0.9905
    SYN_DoS            0.9183          -     0.8541
    SSDP_Flood         0.9996          -     0.9884
    Active_Wiretap     0.6322     0.5776     0.5489

so ``threshold_transfer`` has to be chosen per attack type rather than globally.
Same argument applies to ``threshold_objective`` and to budget parameters.

Usage
-----
    python scripts/tools/make_class_configs.py \
        --source config/train-test/IDS2017/all.json \
        --prefix "" \
        --overrides '{"Bot": {"threshold_objective": "attack_f1"}}'

Existing files are never overwritten unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parameters a per-class config may override, and where each one lives.
TRAIN_PARAM_KEYS = {
    "lr", "num_epoch", "validation_ratio", "seed", "patience", "grad_clip",
    "threshold_objective", "threshold_strategy", "threshold_transfer",
}
CONSTRUCT_PARAM_KEYS = {
    "contrast_batch_size", "train_number", "pre_train_ratio", "segment_len",
    "train_sample_stage", "test_samp_ratio", "host_feature_mode", "host_window",
    "graph_context_mode", "graph_window", "split_strategy",
    "group_test_ratio", "chunk_samples_per_group",
    "benign_attack_parse_num", "benign_attack_samp_num",
}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def safe_component(name: str) -> str:
    """Reject class names that would escape the output directory."""
    if not name or name in {".", ".."} or os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(f"Unsafe class name for a file name: {name!r}")
    return name


def apply_overrides(config: dict, overrides: Dict[str, object], class_name: str) -> None:
    """Merge one class's overrides into the copied shared config."""
    for key, value in overrides.items():
        if key in TRAIN_PARAM_KEYS:
            config.setdefault("train_param", {})[key] = value
        elif key in CONSTRUCT_PARAM_KEYS:
            config.setdefault("data_construct_param", {})[key] = value
        else:
            raise ValueError(
                f"Unknown override {key!r} for class {class_name!r}; extend "
                "TRAIN_PARAM_KEYS/CONSTRUCT_PARAM_KEYS to add it."
            )


def build_class_config(source: dict, class_name: str, waterline, overrides: dict) -> dict:
    config = json.loads(json.dumps(source))
    config["data_tag"] = {class_name: waterline}
    # Keep artefacts per class so parallel or repeated runs never collide.
    for key in ("log_path", "fig_path", "detection_model_path"):
        if config.get(key):
            config[key] = os.path.join(config[key], class_name)
    apply_overrides(config, overrides, class_name)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="Shared dataset config, e.g. config/train-test/IDS2017/all.json")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: the source config's directory)")
    parser.add_argument("--prefix", default=None,
                        help="File name prefix (default: the source config's stem)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated class subset to generate")
    parser.add_argument("--overrides", default="{}",
                        help='Inline JSON: {"Class": {"threshold_transfer": "prior"}}')
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing per-class configs")
    args = parser.parse_args()

    source_path = args.source if os.path.isabs(args.source) else os.path.join(ROOT, args.source)
    source = load_config(source_path)
    data_tag = source.get("data_tag")
    if not isinstance(data_tag, dict) or not data_tag:
        raise ValueError(f"{args.source} has no data_tag mapping to expand.")

    out_dir = args.out_dir or os.path.dirname(source_path)
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    # 顶层 config/<mode>/ 下的配置受 tests/test_runtime_config.py 的统一契约约束
    # （除 data_path/data_tag/log_path/fig_path/detection_model_path/result_namespace
    # 外必须逐字段相同）。逐类配置天然违反该契约，只能落在数据集子目录里。
    if os.path.dirname(os.path.abspath(out_dir)) == os.path.join(ROOT, "config"):
        raise SystemExit(
            f"拒绝写入 {out_dir}：顶层 config 目录受统一契约测试约束，"
            "逐类配置请放到数据集子目录（如 config/train-test/IDS2017/）。"
        )
    prefix = os.path.splitext(os.path.basename(source_path))[0] if args.prefix is None else args.prefix
    overrides = json.loads(args.overrides)
    allowed = {name.strip() for name in args.only.split(",")} if args.only else None

    os.makedirs(out_dir, exist_ok=True)
    written, skipped, unknown = [], [], []
    for class_name, waterline in data_tag.items():
        if allowed is not None and class_name not in allowed:
            continue
        if class_name not in overrides and allowed is not None:
            unknown.append(class_name)
        class_overrides = overrides.get(class_name, {})
        if class_overrides is None:
            class_overrides = {}
        safe_component(class_name)
        file_name = f"{prefix}_{class_name}.json" if prefix else f"{class_name}.json"
        target = os.path.join(out_dir, file_name)
        if os.path.exists(target) and not args.force:
            skipped.append(file_name)
            continue
        config = build_class_config(source, class_name, waterline, class_overrides)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        written.append(file_name)

    for name in written:
        print(f"  写入 {name}")
    for name in skipped:
        print(f"  跳过（已存在，未覆盖）{name}")
    if not written and not skipped:
        print("  没有生成任何文件（检查 --only / data_tag）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
