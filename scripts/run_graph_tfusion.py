#!/usr/bin/env python3
"""tFusion++ 入口：图增强 Cross-Modal 注意力恶意流量检测。

用法:
    python scripts/run_graph_tfusion.py -c config/pre-train/campus_traffic_11G.json
    python scripts/run_graph_tfusion.py -c config/train-test/IDS2017/Bot.json

模块分层（代码在 src/ 下）：
    数据  src/data_pipeline  划分、模态特征、因果上下文、张量束组装
    模型  src/model          tFusion backbone + 图增强跨模态融合
    下游  src/detect         检测器装配、指标、阈值、产物落盘、训练编排
    运行  src/runner         配置校验与训练进程启动

核心创新：
- 复用 tFusion 的多模态编码 (Packet + Flow + Host)
- 借鉴 HyperVision 的流交互图建模
- 引入 Graph-Enhanced Cross-Modal Attention 作为中层融合
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_ROOT, _ROOT / "src", _ROOT / "scripts", _ROOT / "scripts" / "pcap"):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import argparse
import json
import multiprocessing
import os
import random

import numpy as np
import torch

from common import *
from data_pipeline.loader import load_graph_tfusion_data
from detect.constants import (
    ALL_DETECTION_MODELS,
    CROSSMODAL_MODE,
    GRAPH_CLASSIFIER_NAME,
)
from detect.pretraining import pre_train_pp
from detect.training import train_detection_pp
from runner.config import validate_json_config_pp
from runner.launch import launch_training_processes

seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def extract_test_window_ids(loaded_data):
    """Return test-window provenance from legacy and neighbor-extended bundles."""
    return loaded_data[19] if len(loaded_data) >= 20 else None


def train_task_pp(command,
                  detection_model,
                  data_tag,
                  data_path,
                  log_path,
                  fig_path,
                  traffic_model_path,
                  traffic_model_tag,
                  detection_model_path,
                  gpu_id,
                  waterline,
                  data_set_params,
                  pre_train_params,
                  d_graph: int = 128,
                  d_fused_out: int = 384,
                  graph_feature_mode: str = "structural",
                  model_mode: str = CROSSMODAL_MODE,
                  checkpoint_load_scope: str = "backbone_only",
                  load_pretrained: bool = True,
                  model_factory=None,
                  data_split=None):
    """tFusion++ 训练任务"""

    device = resolve_torch_device(gpu_id)
    # 图特征只支持确定性结构特征路径（可学习图编码器在训练图之外无法生效）
    if graph_feature_mode != "structural":
        logging.error(f'Unsupported graph_feature_mode={graph_feature_mode!r}. Only "structural" is supported.')
        return

    run_seed = int(pre_train_params.get("seed", seed))
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    logging.info(f'[{data_tag}] Loading data...')
    loaded_data = load_graph_tfusion_data(
        data_tag,
        data_path,
        device=str(device),
        d_graph=d_graph,
        shuffle_seed=run_seed,
        data_split=data_split,
        pretrain_only=(command == 'pre-train'),
        **data_set_params,
    )
    # 数据束契约：前 19 项为基础张量（pre-train×5 + train×7 + test×7），
    # 第 20 项为 test 窗口 id（len=20），len>=29 时后 9 项为邻居上下文
    # （pre/train/test 各 node/edge/mask 三组），len=32 时再追加 delta_t，
    # len=43 时末尾 11 项为 data_split.val 外部验证集张量。
    if len(loaded_data) not in (19, 20, 29, 32, 43):
        raise ValueError(f"Unexpected Graph_Tfusion data bundle width: {len(loaded_data)}")
    (
        pre_train_batch_seq,
        pre_train_batch_non_seq,
        pre_train_batch_interact,
        pre_train_batch_node,
        pre_train_batch_edge,
        train_data_seq,
        train_data_non_seq,
        train_data_interact,
        train_node,
        train_edge,
        train_size,
        train_label,
        test_data_seq,
        test_data_non_seq,
        test_data_interact,
        test_node,
        test_edge,
        test_size,
        test_label,
    ) = loaded_data[:19]
    test_window_ids = extract_test_window_ids(loaded_data)
    neighbor_data = (
        loaded_data[20:29] if len(loaded_data) >= 29 else (None,) * 9
    )
    (
        pre_train_neighbor_node, pre_train_neighbor_edge, pre_train_neighbor_mask,
        train_neighbor_node, train_neighbor_edge, train_neighbor_mask,
        test_neighbor_node, test_neighbor_edge, test_neighbor_mask,
    ) = neighbor_data
    # 外部验证集（data_split.val）：仅 train 命令消费，pre-train 忽略
    val_data = (
        loaded_data[32:43] if len(loaded_data) >= 43 else (None,) * 11
    )
    (
        val_data_seq, val_data_non_seq, val_data_interact,
        val_node, val_edge, val_size, val_label,
        val_neighbor_node, val_neighbor_edge, val_neighbor_mask,
        _val_neighbor_delta,
    ) = val_data

    # 变体模型注入：None 表示使用 detect.training 的默认主线模型。
    factory_kwargs = {} if model_factory is None else {'model_factory': model_factory}

    # 3. 分发到不同处理函数
    if command == 'pre-train':
        pre_train_pp(
            data_tag, detection_model, log_path, fig_path, traffic_model_path,
            pre_train_batch_seq, pre_train_batch_non_seq, pre_train_batch_interact,
            train_data_seq, train_data_non_seq, train_data_interact,
            train_node, train_edge, train_size, train_label,
            test_data_seq, test_data_non_seq, test_data_interact,
            test_node, test_edge, test_size, test_label,
            gpu_id, waterline, d_graph=d_graph,
            d_fused_out=d_fused_out,
            **factory_kwargs,
            **pre_train_params,
        )
    elif command == 'train':
        train_detection_pp(
            data_tag, detection_model, log_path, fig_path,
            traffic_model_path, traffic_model_tag, detection_model_path,
            train_data_seq, train_data_non_seq, train_data_interact,
            train_node, train_edge, train_size, train_label,
            test_data_seq, test_data_non_seq, test_data_interact,
            test_node, test_edge, test_size, test_label,
            gpu_id, waterline,
            d_graph=d_graph,
            d_fused_out=d_fused_out,
            model_mode=model_mode,
            checkpoint_load_scope=checkpoint_load_scope,
            load_pretrained=load_pretrained,
            test_window_ids=test_window_ids,
            train_neighbor_node=train_neighbor_node,
            train_neighbor_edge=train_neighbor_edge,
            train_neighbor_mask=train_neighbor_mask,
            test_neighbor_node=test_neighbor_node,
            test_neighbor_edge=test_neighbor_edge,
            test_neighbor_mask=test_neighbor_mask,
            val_data_seq=val_data_seq,
            val_data_non_seq=val_data_non_seq,
            val_data_interact=val_data_interact,
            val_node=val_node,
            val_edge=val_edge,
            val_size=val_size,
            val_label=val_label,
            val_neighbor_node=val_neighbor_node,
            val_neighbor_edge=val_neighbor_edge,
            val_neighbor_mask=val_neighbor_mask,
            **factory_kwargs,
            **pre_train_params,
        )
    else:
        logging.error(f'Command {command} not supported in tFusion++')
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tFusion++: Graph-Enhanced Cross-Modal Feature Fusion.')
    parser.add_argument('-c', '--config', type=str, default='./config.json', help='Configuration file.')
    parser.add_argument(
        '--ablation', type=str, default=None,
        choices=['none', 'no_retrieval', 'no_gated_residual', 'no_modal_attn', 'no_history_context'],
        help='可选的组件消融变体（默认 none = 主线模型）。'
             '预训练消融不走此开关，而是配置 load_pretrained=false。',
    )
    parser.add_argument(
        '--in-process', action='store_true',
        help='在当前进程直接训练（不 fork/spawn），规避子进程 CUDA 不可用。',
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        jin = json.load(f)

    if not validate_json_config_pp(jin):
        exit(-1)

    # 消融变体注入：None 表示主线模型（detect.training 的默认 TFusionPPModel）。
    model_factory = None
    if args.ablation and args.ablation != 'none':
        from ablations import ABLATION_FACTORIES
        model_factory = ABLATION_FACTORIES[args.ablation]
        logging.info(f'Using ablation variant model_factory={model_factory.__name__}.')

    if args.in_process:
        from runner.launch import build_train_args
        gpu_ids = jin.get("gpu_enable") or [-1]
        for index, (data_tag, waterline) in enumerate(jin["data_tag"].items()):
            train_task_pp(*build_train_args(
                jin, data_tag, waterline, gpu_ids[index % len(gpu_ids)], model_factory
            ))
        raise SystemExit(0)

    # CUDA 在父进程中一旦初始化，fork 子进程无法重新初始化并会静默回退 CPU；
    # 统一用 spawn，让每个 data_tag 子进程独立初始化 CUDA。
    multiprocessing.set_start_method("spawn", force=True)
    launch_training_processes(jin, train_task_pp, model_factory=model_factory)
