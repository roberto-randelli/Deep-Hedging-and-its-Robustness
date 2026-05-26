"""
CVaR loss function for the Bates model — He, Sutter & Gonon (2025).

BatesCVaRLoss — CVaR / ES_α loss for Bates (stock + variance swap).
                Mirrors HestonCVaRLoss from loss.py.

PnL = Σ_t (δ_S_t · ΔS_t + δ_V_t · ΔVarPrice_t)
X   = C_T − PnL     (hedging error, positive = under-hedged)

Rockafellar-Uryasev dual representation (differentiable):
    ES_α(X) = min_p0 { p0 + E[max(X − p0, 0)] / (1 − α) }
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BatesCVaRLoss(nn.Module):
    """
    CVaR (ES_α) loss for Bates model.

    Hedges with two instruments simultaneously:
        - stock S
        - variance swap VarPrice

    Args:
        K:     Strike price.
        alpha: Confidence level α ∈ (0, 1). Higher = more tail-focused.
    """

    def __init__(self, K: float, alpha: float = 0.95) -> None:
        super().__init__()
        self.K = K
        self.alpha = alpha

    def terminal_payoff(self, S_T: torch.Tensor) -> torch.Tensor:
        return torch.clamp(S_T - self.K, min=0.0)

    def forward(
        self,
        holding: torch.Tensor,    # (batch, N, 2) — [δ_S, δ_V] per step
        S: torch.Tensor,          # (batch, N+1)  — stock prices
        VarPrice: torch.Tensor,   # (batch, N+1)  — variance swap fair values
        p0: torch.Tensor,         # scalar nn.Parameter — VaR threshold
    ) -> torch.Tensor:
        dS  = S[:, 1:] - S[:, :-1]                          # (batch, N)
        dVP = VarPrice[:, 1:] - VarPrice[:, :-1]            # (batch, N)
        dp  = torch.stack([dS, dVP], dim=-1)                 # (batch, N, 2)
        PnL = (holding * dp).sum(dim=(1, 2))                 # (batch,)
        C_T = self.terminal_payoff(S[:, -1])
        X   = C_T - PnL
        return torch.clamp(X - p0, min=0.0).mean() / (1.0 - self.alpha) + p0
