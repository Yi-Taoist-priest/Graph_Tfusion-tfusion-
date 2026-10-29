"""Causal graph-neighbor retrieval and three-modal traffic fusion."""

from __future__ import annotations

import torch
import torch.nn as nn

from .relative_time import RelativeTimeEncoder


class GatedResidualFusion(nn.Module):
    """Selectively retain the original modality after graph retrieval."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([original, enhanced], dim=-1))
        return self.norm(gate * enhanced + (1.0 - gate) * original)


def zero_init_gates(module: nn.Module, bias: float = -6.0) -> int:
    """把三个门控残差初始化为"接近恒等"（gate≈sigmoid(bias)）。

    默认初始化下 Sigmoid 门的输出约为 0.5，随机初始化的图分支会立即把
    噪声混入预训练 backbone。把最后一层权重置零、bias 设为负值后，
    ``full`` 在初始化时近似等于不使用历史上下文，随后由训练决定是否打开
    门控。返回被修改的门控数量。
    """
    changed = 0
    for name in ("packet_gate", "flow_gate", "host_gate"):
        fusion = getattr(module, name, None)
        if fusion is None:
            continue
        gate = getattr(fusion, "gate", None)
        if not isinstance(gate, nn.Sequential) or not isinstance(gate[0], nn.Linear):
            continue
        nn.init.zeros_(gate[0].weight)
        nn.init.constant_(gate[0].bias, float(bias))
        changed += 1
    return changed


class GraphEnhancedCrossModalAttention(nn.Module):
    """Retrieve strictly prior graph neighbors, then fuse packet/flow/host tokens.

    The target flow is deliberately absent from K/V. A learned null token keeps
    attention numerically well-defined when a flow has no eligible neighbor.
    """

    def __init__(
        self,
        d_graph: int = 128,
        d_flow: int = 6,
        d_host: int = 12,
        d_packet: int = 100,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        d_fused_out: int = 384,
        use_time_encoding: bool = False,
    ):
        super().__init__()
        if d_fused_out not in (d_model, 3 * d_model):
            raise ValueError(
                "d_fused_out must be d_model or 3*d_model "
                f"(got d_fused_out={d_fused_out}, d_model={d_model})."
            )
        self.d_model = d_model
        self.d_fused_out = d_fused_out
        self.use_time_encoding = use_time_encoding
        self.time_encoder = RelativeTimeEncoder(d_model) if use_time_encoding else None
        self.neighbor_proj = nn.Linear(d_graph * 2, d_model)
        self.null_context = nn.Parameter(torch.zeros(1, 1, d_model))
        self.flow_proj = nn.Linear(d_flow, d_model)
        self.host_proj = nn.Linear(d_host, d_model)
        self.packet_proj = nn.Linear(d_packet, d_model)
        self.packet_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.flow_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.host_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.packet_gate = GatedResidualFusion(d_model)
        self.flow_gate = GatedResidualFusion(d_model)
        self.host_gate = GatedResidualFusion(d_model)
        self.modal_type = nn.Parameter(torch.zeros(1, 3, d_model))
        self.modal_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.modal_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_fused_out, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.dropout = nn.Dropout(dropout)

    def _context_tokens(
        self,
        neighbor_node: torch.Tensor,
        neighbor_edge: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_delta_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if neighbor_node.ndim != 3 or neighbor_node.shape != neighbor_edge.shape:
            raise ValueError("neighbor_node and neighbor_edge must share shape (batch, neighbors, d_graph).")
        if neighbor_mask.shape != neighbor_node.shape[:2]:
            raise ValueError("neighbor_mask must have shape (batch, neighbors).")
        neighbor = self.neighbor_proj(torch.cat([neighbor_node, neighbor_edge], dim=-1))
        if self.use_time_encoding:
            if neighbor_delta_t is None:
                raise ValueError("neighbor_delta_t is required when use_time_encoding=True.")
            if neighbor_delta_t.shape != neighbor_node.shape[:2]:
                raise ValueError("neighbor_delta_t must have shape (batch, neighbors).")
            neighbor = neighbor + self.time_encoder(
                neighbor_delta_t.to(device=neighbor.device, dtype=neighbor.dtype)
            )
        null = self.null_context.expand(neighbor.size(0), -1, -1)
        tokens = torch.cat([null, neighbor], dim=1)
        padding = torch.cat([
            torch.zeros(neighbor.size(0), 1, dtype=torch.bool, device=neighbor.device),
            ~neighbor_mask.to(device=neighbor.device, dtype=torch.bool),
        ], dim=1)
        return tokens, padding

    @staticmethod
    def _retrieve(
        query: torch.Tensor,
        attention: nn.MultiheadAttention,
        context_tokens: torch.Tensor,
        context_padding: torch.Tensor,
    ) -> torch.Tensor:
        retrieved, _ = attention(
            query=query.unsqueeze(1), key=context_tokens, value=context_tokens,
            key_padding_mask=context_padding,
        )
        return retrieved.squeeze(1)

    def encode_fused(
        self,
        packet_feat: torch.Tensor,
        flow_feat: torch.Tensor,
        host_feat: torch.Tensor,
        neighbor_node: torch.Tensor,
        neighbor_edge: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a dynamically fused (batch, 3*d_model) representation."""
        context_tokens, context_padding = self._context_tokens(
            neighbor_node, neighbor_edge, neighbor_mask, neighbor_delta_t
        )
        packet = self.packet_proj(packet_feat)
        flow = self.flow_proj(flow_feat)
        host = self.host_proj(host_feat)
        packet = self.packet_gate(packet, self._retrieve(packet, self.packet_attn, context_tokens, context_padding))
        flow = self.flow_gate(flow, self._retrieve(flow, self.flow_attn, context_tokens, context_padding))
        host = self.host_gate(host, self._retrieve(host, self.host_attn, context_tokens, context_padding))
        modalities = torch.stack([packet, flow, host], dim=1) + self.modal_type
        attended, _ = self.modal_attn(modalities, modalities, modalities)
        fused = self.modal_norm(modalities + attended)
        if self.d_fused_out == 3 * self.d_model:
            return fused.reshape(packet.size(0), -1)
        return fused.mean(dim=1)

    def forward(
        self,
        packet_feat: torch.Tensor,
        flow_feat: torch.Tensor,
        host_feat: torch.Tensor,
        neighbor_node: torch.Tensor,
        neighbor_edge: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = self.dropout(self.encode_fused(
            packet_feat, flow_feat, host_feat,
            neighbor_node, neighbor_edge, neighbor_mask,
            neighbor_delta_t=neighbor_delta_t,
        ))
        return self.classifier(fused)
