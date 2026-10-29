#!/usr/bin/env python3
"""微调范围择优：在固定场景集上比较 off / full / graph / head / layerwise。

口径（除微调范围外全部相同）：
  数据、划分、seed、检测器（rforest）、阈值、超参均继承主线配置；
  微调只在训练划分内进行（拟合子集训练、验证子集早停），测试集不参与。

用法（在 Graph_Tfusion/ 下执行）：
    python scripts/tools/select_finetune_scope.py                       # 默认 6 场景 × 5 范围
    python scripts/tools/select_finetune_scope.py --scopes off full --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _candidate in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from run_graph_tfusion import train_task_pp  # noqa: E402
from detect.constants import CROSSMODAL_MODE  # noqa: E402

DEFAULT_SCENARIOS = [
    "IDS2017/Bot", "IDS2017/DoS_Hulk", "Kitsune/Mirai_Botnet",
    "Kitsune/ARP_MitM", "UNSW_NB15/Exploits", "DDOS_19/DrDoS_NTP",
]
DEFAULT_SCOPES = ["off", "full", "graph", "head", "layerwise"]
OUT_ROOT = os.path.join(_ROOT, "save", "finetune_scope_selection")
# 默认从「与下游 d_graph 对齐」的预训练权重出发（原 11G checkpoint 无法全键迁移）
DEFAULT_TAG = "campus_traffic_11G_dg16"


def _run(scenario: str, scope: str, epochs: int, lr: float, tag: str) -> dict:
    dataset, cls = scenario.split("/", 1)
    with open(os.path.join(_ROOT, "config", "train-test", dataset, f"{cls}.json"),
              encoding="utf-8") as source:
        config = json.load(source)
    out_dir = os.path.join(OUT_ROOT, scope, dataset, cls)
    log_dir = os.path.join(_ROOT, "log", "finetune_scope_selection", scope, dataset, cls)
    fig_dir = os.path.join(_ROOT, "figure", "finetune_scope_selection", scope, dataset, cls)
    for path in (out_dir, log_dir, fig_dir):
        os.makedirs(path, exist_ok=True)

    seed = int(config["train_param"].get("seed", 777))
    torch.manual_seed(seed); np.random.seed(seed)
    train_param = dict(config["train_param"])
    train_param["finetune_epochs"] = 0 if scope == "off" else epochs
    train_param["finetune_scope"] = "full" if scope == "off" else scope
    train_param["finetune_lr"] = lr
    train_param["finetune_batch_size"] = 300

    started = time.perf_counter()
    for data_tag, waterline in config["data_tag"].items():
        train_task_pp(
            command="train",
            detection_model=config["detection_model"],
            data_tag=data_tag,
            data_path=os.path.join(_ROOT, config["data_path"]),
            log_path=log_dir,
            fig_path=fig_dir,
            traffic_model_path=os.path.join(_ROOT, "save", "traffic_model", tag),
            traffic_model_tag=tag,
            detection_model_path=out_dir,
            gpu_id=config.get("gpu_enable", [-1])[0],
            waterline=waterline,
            data_set_params=dict(config["data_construct_param"]),
            pre_train_params=train_param,
            d_graph=config.get("d_graph", 128),
            d_fused_out=config.get("d_fused_out", 384),
            graph_feature_mode=config.get("graph_feature_mode", "structural"),
            model_mode=config.get("model_mode", CROSSMODAL_MODE),
            checkpoint_load_scope=config.get("checkpoint_load_scope", "backbone_only"),
            load_pretrained=True,
            data_split=config.get("data_split"),
        )
    seconds = round(time.perf_counter() - started, 1)
    with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as source:
        metrics = json.load(source)
    with open(os.path.join(out_dir, "finetune_summary.json"), encoding="utf-8") as source:
        finetune = json.load(source)
    return {"scenario": scenario, "scope": scope, "seconds": seconds,
            "auc": round(float(metrics["auc"]), 4),
            "attack_f1": round(float(metrics["attack_f1"]), 4),
            "macro_f1": round(float(metrics["f1"]), 4),
            "finetune_trainable_params": finetune.get("finetune_trainable_parameters"),
            "finetune_best_val_loss": finetune.get("finetune_best_validation_loss"),
            "finetune_val_loss_first": (finetune.get("finetune_history") or [{}])[0].get("validation_loss"),
            "finetune_val_loss_last": (finetune.get("finetune_history") or [{}])[-1].get("validation_loss")}


def main() -> None:
    parser = argparse.ArgumentParser(description="微调范围择优")
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--scopes", nargs="*", default=DEFAULT_SCOPES)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tag", default=DEFAULT_TAG, help="预训练权重 tag")
    parser.add_argument("--out", default=None,
                        help="汇总 JSON 输出路径（默认 save/finetune_scope_selection/summary.json）")
    args = parser.parse_args()

    print(f"预训练权重: {args.tag} | 微调 epoch={args.epochs} lr={args.lr}\n")
    header = f"{'scenario':26s}" + "".join(f"{s:>11s}" for s in args.scopes)
    print(header + f"{'  Δbest-off':>12s}")
    rows = []
    for scenario in args.scenarios:
        record = {"scenario": scenario}
        line = f"{scenario:26s}"
        for scope in args.scopes:
            try:
                res = _run(scenario, scope, args.epochs, args.lr, args.tag)
            except Exception as error:  # noqa: BLE001
                print(f"  [{scope}] {scenario} 失败: {type(error).__name__}: {error}")
                res = {"auc": float("nan"), "attack_f1": float("nan")}
            record[scope] = res
            rows.append(res)
            line += f"{res.get('auc', float('nan')):11.4f}"
        off = record.get("off", {}).get("auc", float("nan"))
        best = max((v.get("auc", float("nan")) for k, v in record.items()
                    if k != "scenario" and isinstance(v, dict)), default=float("nan"))
        record["delta_best_minus_off"] = round(best - off, 4)
        print(line + f"{record['delta_best_minus_off']:+12.4f}", flush=True)

    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "pretrain_tag": args.tag, "finetune_epochs": args.epochs, "finetune_lr": args.lr,
           "note": "Δ = 最好范围 AUC − 不微调 AUC；检测器与划分继承主线",
           "rows": rows}
    path = args.out or os.path.join(_ROOT, "save", "finetune_scope_selection", "summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as target:
        json.dump(out, target, ensure_ascii=False, indent=2)

    print("\n按范围汇总（AUC 均值）:")
    for scope in args.scopes:
        values = [r["auc"] for r in rows if r["scope"] == scope and r["auc"] == r["auc"]]
        if values:
            print(f"  {scope:10s} 均值 {np.mean(values):.4f}  中位 {np.median(values):.4f}  n={len(values)}")
    print(f"\n-> {os.path.relpath(path, _ROOT)}")


if __name__ == "__main__":
    main()
