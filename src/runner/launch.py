"""训练进程的启动。

按 ``data_tag`` 逐个拉起一个执行 ``train_task_pp`` 的子进程，全部 join 后
统一校验退出码；位置参数顺序与 ``train_task_pp`` 的签名严格一致。
"""

from __future__ import annotations

import multiprocessing

from detect.constants import CROSSMODAL_MODE

from .config import ensure_processes_succeeded


def build_train_args(config, data_tag, waterline, gpu_id, model_factory):
    """按 train_task_pp 的形参顺序组装位置参数（含变体模型工厂）。"""
    return (
        config["command"],
        config["detection_model"],
        data_tag,
        config["data_path"],
        config["log_path"],
        config["fig_path"],
        config.get("traffic_model_path", None),
        config.get("traffic_model_tag", None),
        config.get("detection_model_path", None),
        gpu_id,
        waterline,
        config["data_construct_param"],
        config["train_param"],
        config.get("d_graph", 128),
        config.get("d_fused_out", 384),
        config.get("graph_feature_mode", "structural"),
        config.get("model_mode", CROSSMODAL_MODE),
        config.get("checkpoint_load_scope", "backbone_only"),
        config.get("load_pretrained", True),
        model_factory,
        config.get("data_split"),
    )


def launch_training_processes(config, target, *, model_factory=None):
    """为每个 data_tag 启动一个训练进程，并在任一失败时抛错。

    每个 data_tag 按 round-robin 分配 ``gpu_enable`` 中的设备；无可用设备
    （空列表）时传 -1，由下游回退 CPU。
    """
    gpu_ids = config.get("gpu_enable") or [-1]
    processes = []
    for index, (data_tag, waterline) in enumerate(config["data_tag"].items()):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        processes.append(multiprocessing.Process(
            target=target,
            args=build_train_args(config, data_tag, waterline, gpu_id, model_factory),
        ))
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    ensure_processes_succeeded(processes)
