import torch.nn as nn
import torch

from .transformer import TransformerSelfAttenBlock
from .embedding import MultiModalEmbedding
from .traffusion import TransformerTrafficFusion

class TFusion(nn.Module):

    def __init__(self, 
                 flow_feature_size=6, 
                 interact_feature_size=12, 
                 hidden=100, 
                 n_layers=1, 
                 attn_heads=1, 
                 dropout=0.1):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        
        # embedding for traffic, sum of positional, segment, token embeddings
        self.embedding = MultiModalEmbedding(embed_size=hidden)

        # multi-layers transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerSelfAttenBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        )

        self.packet_pool = nn.Linear(hidden, 1)

        self.fus = TransformerTrafficFusion(hidden, hidden, flow_feature_size, interact_feature_size)

    def forward(self, x):
        x_seq, x_non_seq, x_interact = x
        return self.fus(self.encode_packet(x_seq), x_non_seq, x_interact)

    @staticmethod
    def packet_mask(x_seq, flow_sizes=None):
        """Return valid packet positions; sizes disambiguate real token [0, 0]."""
        if flow_sizes is not None:
            sizes = flow_sizes.to(device=x_seq.device, dtype=torch.long).clamp(
                min=0, max=x_seq.size(1)
            )
            positions = torch.arange(x_seq.size(1), device=x_seq.device).unsqueeze(0)
            mask = positions < sizes.unsqueeze(1)
        else:
            mask = x_seq.ne(0).any(dim=-1)
        if (~mask.any(dim=1)).any():
            raise ValueError("Every flow must contain at least one valid packet token.")
        return mask

    def encode_packet_tokens(self, x_seq, packet_mask=None, flow_sizes=None):
        """Return all transformed valid/invalid packet tokens with their mask.

        ``packet_mask`` 表示有效位置。
        """
        if packet_mask is None:
            packet_mask = self.packet_mask(x_seq, flow_sizes=flow_sizes)
        else:
            packet_mask = packet_mask.to(device=x_seq.device, dtype=torch.bool)
            if packet_mask.shape != x_seq.shape[:2]:
                raise ValueError("packet_mask must have shape (batch, sequence_length).")
            if (~packet_mask.any(dim=1)).any():
                raise ValueError("Every flow must contain at least one valid packet token.")
        _x_seq = self.embedding(x_seq)
        for transformer in self.transformer_blocks:
            _x_seq = transformer.forward(_x_seq, mask=packet_mask)
        return _x_seq, packet_mask

    def pool_packet_tokens(self, packet_tokens, packet_mask):
        """Read a flow vector from every valid packet position."""
        if packet_tokens.ndim != 3:
            raise ValueError("packet_tokens must have shape (batch, sequence_length, hidden).")
        if packet_mask.shape != packet_tokens.shape[:2]:
            raise ValueError("packet_mask must align with packet_tokens.")
        scores = self.packet_pool(packet_tokens).squeeze(-1)
        scores = scores.masked_fill(~packet_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), packet_tokens).squeeze(1)

    def encode_packet(self, x_seq, packet_mask=None, flow_sizes=None):
        """Encode a packet sequence with mask-aware attention pooling."""
        packet_tokens, packet_mask = self.encode_packet_tokens(
            x_seq, packet_mask=packet_mask, flow_sizes=flow_sizes
        )
        return self.pool_packet_tokens(packet_tokens, packet_mask)
