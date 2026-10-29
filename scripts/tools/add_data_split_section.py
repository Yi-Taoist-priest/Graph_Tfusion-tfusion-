#!/usr/bin/env python
"""为逐类配置写入 data_split（预切分 train/test/val 文件）段。

数据集已按 ``data/train-test/<数据根>/<类别>/{train,test,val}.json`` 预切分，
本脚本把对应路径以幂等方式写进 ``config/train-test/<数据集>/<类别>.json``，
并按类别独立开关三个文件。

规则：
- 三个文件齐全时写 ``use_pre_split: true``，train/test/val 全部 enable；
- 数据文件缺失时写 ``use_pre_split: false``（回退旧的单文件内存切分），
  并在输出中列为告警，不静默错配。

可选地把全部配置的预训练权重路径/标签、训练集抽样比例与阈值寻优开关
一并改写（--traffic-model-path / --traffic-model-tag / --train-sample-ratio /
--optimize-threshold），便于整批统一切换。

Usage
-----
    python scripts/tools/add_data_split_section.py --data-root data/train-test
    python scripts/tools/add_data_split_section.py \
        --traffic-model-path ./save/traffic_model/pretrain \
        --traffic-model-tag campus_20260423_Monday_full \
        --train-sample-ratio 1.0 --optimize-threshold true
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置目录名 → 数据根下的叶子目录名（两者不一定同形）。
DATASET_DATA_DIRS = {
    "DDOS_19": "CICDDoS2019",
    "IDS2017": "CICIDS2017",
    "CSECICIDS2018": "CSECICIDS2018",
    "IoT-23": "IoT-23",
    "Kitsune": "Kitsune",
    "UNSW_NB15": "UNSW_NB15",
}
SPLIT_FILES = ("train", "test", "val")


def build_data_split(data_root: str, data_dir: str, tag: str, rel_root: str):
    """返回 (data_split 段, 缺失文件列表)。三个文件齐全才启用预切分。"""
    missing = []
    section = {"use_pre_split": True}
    for name in SPLIT_FILES:
        rel = os.path.join(rel_root, data_dir, tag, f"{name}.json").replace(os.sep, "/")
        abs_path = os.path.join(data_root, data_dir, tag, f"{name}.json")
        section[name] = {"enable": True, "path": f"./{rel}"}
        if not os.path.isfile(abs_path):
            missing.append(abs_path)
    if missing:
        # 正式运行只有预切分一条路径；数据不全时不写入禁用段，交由调用方报告。
        return None, missing
    return section, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-root", default="config/train-test")
    parser.add_argument("--data-root", default="data/train-test")
    parser.add_argument("--traffic-model-path", default=None,
                        help="若提供，改写全部配置的 traffic_model_path")
    parser.add_argument("--traffic-model-tag", default=None,
                        help="若提供，改写全部配置的 traffic_model_tag")
    parser.add_argument("--train-sample-ratio", type=float, default=None,
                        help="若提供，改写 data_construct_param.train_sample_ratio (0, 1]")
    parser.add_argument("--optimize-threshold", choices=["true", "false"], default=None,
                        help="若提供，改写 train_param.optimize_threshold")
    parser.add_argument("--force-data-split", action="store_true",
                        help="已存在 data_split 时也按数据目录重新推导（默认保留逐文件开关）")
    parser.add_argument("--disable-val", action="store_true",
                        help="把 data_split.val.enable 统一置为 false（保留 path，便于日后重启）")
    parser.add_argument("--threshold-transfer", choices=["absolute", "prior", "quantile"],
                        default=None,
                        help="若提供，改写 train_param.threshold_transfer")
    args = parser.parse_args()

    if args.train_sample_ratio is not None and not 0.0 < args.train_sample_ratio <= 1.0:
        raise SystemExit(f"--train-sample-ratio 必须在 (0, 1]，收到 {args.train_sample_ratio}")

    config_root = args.config_root if os.path.isabs(args.config_root) else os.path.join(ROOT, args.config_root)
    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(ROOT, args.data_root)
    rel_root = os.path.relpath(data_root, ROOT).replace(os.sep, "/")

    updated, fallback, missing_report = [], [], []
    for dataset in sorted(os.listdir(config_root)):
        dataset_dir = os.path.join(config_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        data_dir = DATASET_DATA_DIRS.get(dataset)
        if data_dir is None:
            raise SystemExit(f"未登记的数据集目录：{dataset}（请更新 DATASET_DATA_DIRS）")
        for name in sorted(os.listdir(dataset_dir)):
            if not name.endswith(".json") or name.startswith("."):
                continue
            tag = os.path.splitext(name)[0]
            target = os.path.join(dataset_dir, name)
            with open(target, encoding="utf-8") as handle:
                config = json.load(handle)

            if config.get("data_split") is not None and not args.force_data_split:
                # 保留已有（含人工调整过的逐文件开关），不静默覆盖。
                section, missing = config["data_split"], []
            else:
                section, missing = build_data_split(data_root, data_dir, tag, rel_root)
            if section is None:
                fallback.append(f"{dataset}/{name}")
                missing_report.append((f"{dataset}/{name}", missing))
                continue
            if args.disable_val:
                section.setdefault("val", {"path": ""})["enable"] = False
            config["data_split"] = section
            if args.traffic_model_path is not None:
                config["traffic_model_path"] = args.traffic_model_path
            if args.traffic_model_tag is not None:
                config["traffic_model_tag"] = args.traffic_model_tag
            if args.train_sample_ratio is not None:
                config.setdefault("data_construct_param", {})[
                    "train_sample_ratio"
                ] = args.train_sample_ratio
            if args.optimize_threshold is not None:
                optimize = args.optimize_threshold == "true"
                config.setdefault("train_param", {})["optimize_threshold"] = optimize
                config["train_param"]["threshold_strategy"] = (
                    "validation" if optimize else "fixed_0.5"
                )
            if args.threshold_transfer is not None:
                config.setdefault("train_param", {})[
                    "threshold_transfer"
                ] = args.threshold_transfer

            with open(target, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

            label = f"{dataset}/{name}"
            updated.append(label)

    print(f"写入预切分段：{len(updated)} 个配置")
    for label in updated:
        print(f"  [pre-split] {label}")
    if fallback:
        print(f"\n未写入（数据不完整，正式运行要求预切分）：{len(fallback)} 个配置")
        for label, missing in missing_report:
            print(f"  [missing  ] {label}")
            for path in missing:
                print(f"              缺失 {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
