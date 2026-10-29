"""tFusion++ 模型封装。

= tFusion 多模态 backbone + Graph-Enhanced Cross-Modal Attention，
提供对比学习接口（``encode_for_contrast``，仅三模态）与分类接口（``classify``，图增强）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import TFusion, GraphEnhancedCrossModalAttention

from .constants import (
    CROSSMODAL_MODE,
    TFUSION_MODE,
    canonical_graph_mode,
    n_hidden,
    n_project,
)


class TFusionPPModel(nn.Module):
    """
    tFusion++ 完整模型封装
    = tFusion backbone + Graph-Enhanced Cross-Modal Attention
    """

    def __init__(self,
                 d_graph: int = 128,
                 d_flow: int = 6,
                 d_host: int = 12,
                 d_model: int = 128,
                 n_project: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 d_fused_out: int = 384,
                 use_time_encoding: bool = False):
        super().__init__()
        self.d_graph = d_graph
        self.d_fused_out = d_fused_out

        # 复用 tFusion backbone
        self.backbone = TFusion(
            flow_feature_size=d_flow,
            interact_feature_size=d_host,
            hidden=n_hidden,
            n_layers=1,
            attn_heads=1,
            dropout=dropout,
        )

        # Keep baseline initialization independent from graph-only width.
        self.base_norm = nn.LayerNorm(n_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, 2),
        )

        # 对比学习投影头：输入是 backbone 的三模态融合表示（packet + flow + host），
        # 分类路径不使用；预训练因此不依赖任何历史图特征。
        self.graph_proj_head = nn.Sequential(
            nn.Linear(n_hidden, n_project),
            nn.ReLU(),
            nn.Linear(n_project, n_project),
        )

        # Graph-Enhanced Cross-Modal Attention
        self.graph_attn = GraphEnhancedCrossModalAttention(
            d_graph=d_graph,
            d_flow=d_flow,
            d_host=d_host,
            d_packet=n_hidden,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            d_fused_out=d_fused_out,
            use_time_encoding=use_time_encoding,
        )

    def base_embedding(
        self,
        x_seq: torch.Tensor,
        x_non_seq: torch.Tensor,
        x_interact: torch.Tensor,
        flow_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packet = self.backbone.encode_packet(x_seq, flow_sizes=flow_sizes)
        return self.backbone.fus(packet, x_non_seq, x_interact)

    def classify(
        self,
        x_seq: torch.Tensor,
        x_non_seq: torch.Tensor,
        x_interact: torch.Tensor,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        flow_sizes: torch.Tensor | None,
        graph_mode: str,
        neighbor_node: torch.Tensor | None = None,
        neighbor_edge: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode = canonical_graph_mode(graph_mode)
        if mode == CROSSMODAL_MODE:
            packet = self.backbone.encode_packet(x_seq, flow_sizes=flow_sizes)
            neighbor_node, neighbor_edge, neighbor_mask = self._neighbor_inputs(
                packet.size(0), packet.device, neighbor_node, neighbor_edge, neighbor_mask
            )
            return self.graph_attn(
                packet,
                x_non_seq,
                x_interact,
                neighbor_node,
                neighbor_edge,
                neighbor_mask,
            )

        base = self.base_embedding(
            x_seq,
            x_non_seq,
            x_interact,
            flow_sizes=flow_sizes,
        )
        if mode == TFUSION_MODE:
            fused = base
        else:
            raise AssertionError(f"Unreachable canonical graph mode: {mode}")
        return self.classifier(self.base_norm(fused))
    # 处理邻居输入
    def _neighbor_inputs(
        self,
        batch_size: int,
        device: torch.device,
        neighbor_node: torch.Tensor | None,
        neighbor_edge: torch.Tensor | None,
        neighbor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if neighbor_node is None and neighbor_edge is None and neighbor_mask is None:
            empty = torch.zeros(batch_size, 1, self.d_graph, device=device)
            return empty, empty.clone(), torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        if neighbor_node is None or neighbor_edge is None or neighbor_mask is None:
            raise ValueError("neighbor_node, neighbor_edge and neighbor_mask must be supplied together.")
        return (
            neighbor_node.to(device=device, dtype=torch.float32),
            neighbor_edge.to(device=device, dtype=torch.float32),
            neighbor_mask.to(device=device, dtype=torch.bool),
        )

    def pretrain_embedding(self,
                           x_seq: torch.Tensor,
                           x_non_seq: torch.Tensor,
                           x_interact: torch.Tensor) -> torch.Tensor:
        """预训练对比表示：只融合 packet / flow / host 三模态，不读取历史图。"""
        inferred_sizes = torch.expm1(x_non_seq[:, 0]).round().to(torch.long)
        flow_sizes = inferred_sizes if torch.all(inferred_sizes > 0) else None
        base = self.base_embedding(
            x_seq, x_non_seq, x_interact, flow_sizes=flow_sizes,
        )
        return self.graph_proj_head(base)

    def encode_for_contrast(self, x):
        """对比学习接口：三模态表示，不接受 node/edge/邻居上下文。"""
        x_seq, x_non_seq, x_interact = x
        return self.pretrain_embedding(x_seq, x_non_seq, x_interact)
