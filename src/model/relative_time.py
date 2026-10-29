"""相对时间编码（Plan A Task 8）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeTimeEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, delta_t):
        return self.net(torch.log1p(delta_t).unsqueeze(-1))
