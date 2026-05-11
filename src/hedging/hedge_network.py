"""
Hedging networks — He, Sutter & Gonon (2025).

HedgeNet       — GBM (single asset): log(S_t) → δ_t
                 Matches BS_util.RNN_BN_simple with input_size=1.

HestonHedgeNet — Heston (stock + variance swap): [log(S_t), V_t] → [δ_S, δ_V]
                 Matches Heston_util.RNN_BN_simple with input_size=2.

Both use N SEPARATE sub-networks (one per time step, no shared weights, no
recurrence). Architecture per time step:
    BN(d_in) → Linear(d_in, width) → BN(width) → ReLU
             → Linear(width, width) → BN(width) → ReLU
             → Linear(width, d_out)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HedgeNet(nn.Module):
    """
    Args:
        N:     Number of time steps (one sub-network per step).
        width: Hidden layer width. Default 20 matches He et al.

    Input:  log-prices  (batch, N, 1)
    Output: hedge ratios (batch, N)
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


class HestonHedgeNet(nn.Module):
    """
    Hedging network for Heston model — two instruments (stock + variance swap).

    Args:
        N:     Number of time steps (one sub-network per step).
        width: Hidden layer width. Default 20 matches He et al.

    Input:  [log(S_t), V_t]  (batch, N, 2)
    Output: [δ_S_t, δ_V_t]  (batch, N, 2) — hedge ratios per instrument
    """

    def __init__(self, N: int, width: int = 20) -> None:
        super().__init__()
        self.N = N

        def _block() -> nn.Sequential:
            return nn.Sequential(
                nn.BatchNorm1d(2),
                nn.Linear(2, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Linear(width, 2),
            )

        self.nets = nn.ModuleList([_block() for _ in range(N)])

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eps = 1e-3
                m.momentum = 0.3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, 2) — [log(S_t), V_t] at each time step.
        Returns:
            (batch, N, 2) — [δ_S_t, δ_V_t] at each time step.
        """
        batch = x.shape[0]
        out = torch.empty(batch, self.N, 2, device=x.device, dtype=x.dtype)
        for t in range(self.N):
            out[:, t, :] = self.nets[t](x[:, t, :])
        return out
