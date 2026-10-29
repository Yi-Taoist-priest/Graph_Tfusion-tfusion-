"""预训练编排：对比学习。

- ``contrastive_loss_enhanced``：批内同源正对 / 异源负对的对比损失；
- ``pre_train_pp``：在预训练数据集上迭代，并按 epoch 平均对比损失保存最佳 checkpoint。
"""

from __future__ import annotations

import copy
import logging

import torch

from common import *
from .checkpoint import save_model_checkpoint
from .fusion_model import TFusionPPModel


def contrastive_loss_enhanced(x_predict, batch_size: int) -> torch.Tensor:
    """
    对比损失（与原 tFusion 批排列一致）。

    前半批来自同一地址组，因此相邻样本构成正对；后半批来自组外，
    ``i + same_line`` 构成负对。最后一个前半批样本没有下一个组内
    样本，所以有效损失项数量是 ``same_line - 1``。
    """
    if batch_size < 4 or batch_size % 2:
        raise ValueError("batch_size must be an even integer of at least 4.")
    if x_predict.size(0) != batch_size:
        raise ValueError(
            f"x_predict contains {x_predict.size(0)} rows, expected batch_size={batch_size}."
        )
    same_line = batch_size // 2

    def sim(x, y):
        return x.dot(y) / (x.norm() * y.norm() + 1e-8)

    def sim_pair(i, j):
        return torch.exp(sim(x_predict[i], x_predict[j]))

    def sim_loss(i):
        return -torch.log(sim_pair(i, i + 1) / (sim_pair(i, i + 1) + sim_pair(i, i + same_line) + 1e-8))

    losses = [sim_loss(i) for i in range(same_line - 1)]
    return torch.stack(losses).mean()


@time_log
def pre_train_pp(data_tag: str,
                 detection_model: str,
                 log_path: str,
                 fig_path: str,
                 traffic_model_path: str,
                 pre_train_batch_seq: torch.Tensor,
                 pre_train_batch_non_seq: torch.Tensor,
                 pre_train_batch_interact: torch.Tensor,
                 train_data_seq: torch.Tensor,
                 train_data_non_seq: torch.Tensor,
                 train_data_interact: torch.Tensor,
                 train_node: torch.Tensor,
                 train_edge: torch.Tensor,
                 train_size: torch.Tensor,
                 train_label: torch.Tensor,
                 test_data_seq: torch.Tensor,
                 test_data_non_seq: torch.Tensor,
                 test_data_interact: torch.Tensor,
                 test_node: torch.Tensor,
                 test_edge: torch.Tensor,
                 test_size: torch.Tensor,
                 test_label: torch.Tensor,
                 gpu_id: int,
                 waterline: float,
                 d_graph: int = 128,
                 d_fused_out: int = 384,
                 lr: float = 0.001,
                 wd: float = 1e-4,
                 num_epoch: int = 15,
                 model_factory: Callable = TFusionPPModel):
    """
    tFusion++ 预训练：三模态对比学习（packet + flow + host）。

    预训练阶段不读取历史图/邻居上下文，只使用 backbone 的三模态表示。
    """
    if pre_train_batch_seq.size(0) == 0:
        raise ValueError(
            f"[{data_tag}] No tri-modal pre-training batches were constructed; "
            "refusing to save an untrained random checkpoint."
        )
    if num_epoch <= 0:
        raise ValueError(f"[{data_tag}] num_epoch must be positive, got {num_epoch!r}.")

    logging.info(f'[{data_tag}] Graph_Tfusion Pre-train is started.')
    fout = open(f'{log_path}/{data_tag}.log', 'w', buffering=1)

    device = resolve_torch_device(gpu_id)
    print(f'[{data_tag}] Use device:{device}.', file=fout, flush=True)

    model = model_factory(d_graph=d_graph, d_fused_out=d_fused_out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    pre_train_batch_size = pre_train_batch_seq.size(1)
    checkpoint_path = f"{traffic_model_path}/{data_tag}_pp_model.pt"

    # 预训练没有验证集，用 epoch 平均对比损失作为“最佳”判据；
    # 只在损失改善时覆盖写，保证落盘的是历史最佳而不是末轮。
    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for e in range(num_epoch):
        train_loss = 0.0
        num_train = 0
        model.train()
        num_pretrain_batch = pre_train_batch_seq.size(0)

        for i in range(0, num_pretrain_batch):
            x_seq = pre_train_batch_seq[i].to(device)
            x_non_seq = pre_train_batch_non_seq[i].to(device)
            x_interact = pre_train_batch_interact[i].to(device)
            num_train += 1
            opt.zero_grad()
            embedding = model.encode_for_contrast(
                (x_seq, x_non_seq, x_interact),
            )
            loss_t = contrastive_loss_enhanced(embedding, pre_train_batch_size)
            train_loss += loss_t.item()
            loss_t.backward()
            opt.step()

        model.eval()
        epoch_loss = train_loss / max(num_train, 1)
        print(f'[{data_tag}] Pre-Training Loss: {epoch_loss:.4f}')
        print(f'[{data_tag}] Pre-Training Loss: {epoch_loss:.4f}', file=fout, flush=True)

        if epoch_loss < best_loss - 1e-8:
            best_loss = epoch_loss
            best_epoch = e + 1
            best_state = copy.deepcopy(model.state_dict())
            save_model_checkpoint(model, checkpoint_path)
            logging.info(
                f'Save best Graph_Tfusion pre-trained model to {checkpoint_path} '
                f'(epoch={best_epoch}, loss={best_loss:.6f})'
            )

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    else:
        # 损失为 NaN 等异常情况下至少保留一份可加载的权重。
        best_epoch = num_epoch
        best_loss = float(train_loss / max(num_train, 1))
        save_model_checkpoint(model, checkpoint_path)
        logging.warning(
            f'No finite pre-training loss improvement; saved final epoch '
            f'weights to {checkpoint_path}'
        )

    logging.info(
        f'[{data_tag}] Pre-training finished: best_epoch={best_epoch}, '
        f'best_loss={best_loss:.6f}'
    )
    fout.close()
