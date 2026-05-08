"""
Hedging network — He, Sutter & Gonon (2025) / BS_util.RNN_BN_simple.

N SEPARATE sub-networks, one per time step (no shared weights).
Each sub-network maps log(S_t) → δ_t independently.

Architecture per time step:
    BN(1) → Linear(1, width) → BN(width) → ReLU
           → Linear(width, width) → BN(width) → ReLU
           → Linear(width, 1)

Input:  log-prices  (batch, N, 1)
Output: hedge ratios (batch, N)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HedgeNet(nn.Module):
    """
    Args:
        N:     Number of time steps (one sub-network per step).
        width: Hidden layer width. Default 20 matches He et al.
    """

    def __init__(self, N: int, width: int = 20) -> None:
        super().__init__()
        self.N = N

        def _block() -> nn.Sequential:
            return nn.Sequential(
                nn.BatchNorm1d(1),
                nn.Linear(1, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Linear(width, 1),
            )

        self.nets = nn.ModuleList([_block() for _ in range(N)])

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eps = 1e-3
                m.momentum = 0.3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, 1) — log(S_t) at each time step.
        Returns:
            (batch, N) — hedge ratio δ_t at each time step.
        """
        batch = x.shape[0]
        out = torch.empty(batch, self.N, device=x.device, dtype=x.dtype)
        for t in range(self.N):
            out[:, t] = self.nets[t](x[:, t, :]).squeeze(1)
        return out
