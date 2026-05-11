"""
Loss functions for deep hedging — He, Sutter & Gonon (2025).

EntropicOCELoss — OCE loss with entropic disutility (GBM / single asset).
CVaRLoss        — CVaR / ES_α loss (GBM / single asset, Buehler et al. 2019).
HestonCVaRLoss  — CVaR / ES_α loss for Heston (stock + variance swap).

OCE loss with entropic disutility — He, Sutter & Gonon (2025) / BS_util.loss_exp_OCE.

PnL and payoff computation are embedded in the loss forward.
p0 (option price proxy) is an external nn.Parameter optimised jointly with
the network weights via Adam — it is NOT stored inside this class.

Loss formula:
    X    = PnL − C_T                        (hedging error, no premium)
    loss = E[ exp(−λ(X + p0)) ] + p0 − (1 + log λ) / λ

Gradient w.r.t. p0 at the optimum satisfies  E[exp(−λ(X+p0*))] = 1/λ,
and the loss at that optimum equals the entropic risk measure  ρ(X).

Numerical stability: X is clamped from below at −10 (X_max=True, default)
to prevent exp overflow during early training with a random-init network.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class EntropicOCELoss(nn.Module):
    """
    Args:
        K:      Strike price.
        sigma:  Volatility (used only for bs_price reference).
        T:      Maturity.
        lamb:   Risk-aversion parameter λ > 0.
        X_max:  If True, clamp hedging error X from below at −10.
    """

    def __init__(
        self,
        K: float,
        sigma: float,
        T: float,
        lamb: float = 1.3,
        X_max: bool = True,
    ) -> None:
        super().__init__()
        self.K = K
        self.sigma = sigma
        self.T = T
        self.lamb = lamb
        self.X_max = X_max

    # ------------------------------------------------------------------
    def terminal_payoff(self, S_T: torch.Tensor) -> torch.Tensor:
        return torch.clamp(S_T - self.K, min=0.0)

    def bs_price(self, S0: torch.Tensor) -> torch.Tensor:
        """Black-Scholes call price — useful for initialising p0."""
        d1 = (torch.log(S0 / self.K) + 0.5 * self.sigma ** 2 * self.T) / (
            self.sigma * self.T ** 0.5
        )
        d2 = d1 - self.sigma * self.T ** 0.5
        N = Normal(0.0, 1.0)
        return S0 * N.cdf(d1) - self.K * N.cdf(d2)

    # ------------------------------------------------------------------
    def forward(
        self,
        holding: torch.Tensor,   # (batch, N)   — hedge ratios
        price: torch.Tensor,     # (batch, N+1) — stock prices
        p0: torch.Tensor,        # scalar nn.Parameter — option price proxy
    ) -> torch.Tensor:
        """Returns scalar OCE loss."""
        PnL = (holding * (price[:, 1:] - price[:, :-1])).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1])
        X = PnL - C_T

        if self.X_max:
            X = torch.clamp(X, min=-10.0)

        x = X + p0
        return torch.exp(-self.lamb * x).mean() + p0 - (1 + np.log(self.lamb)) / self.lamb


# ---------------------------------------------------------------------------

class CVaRLoss(nn.Module):
    """
    Expected Shortfall (CVaR) loss — Buehler et al. (2019), §3.2 / Eq. (3.1).

    Uses the Rockafellar-Uryasev dual representation (differentiable):

        ES_α(X) = min_z { z + E[ max(X − z, 0) ] / (1 − α) }

    where X = C_T − PnL  (loss, positive = bad) and z = p0 is the jointly-
    optimised VaR threshold (same role as p0 in EntropicOCELoss).

    At convergence:  p0* = VaR_α(C_T − PnL)
                     loss* = ES_α(C_T − PnL)

    Args:
        K:     Strike price.
        alpha: Confidence level α ∈ (0, 1). Higher = more tail-focused.
    """

    def __init__(self, K: float, alpha: float = 0.5) -> None:
        super().__init__()
        self.K = K
        self.alpha = alpha

    def terminal_payoff(self, S_T: torch.Tensor) -> torch.Tensor:
        return torch.clamp(S_T - self.K, min=0.0)

    def forward(
        self,
        holding: torch.Tensor,   # (batch, N)   — hedge ratios
        price: torch.Tensor,     # (batch, N+1) — stock prices
        p0: torch.Tensor,        # scalar nn.Parameter — VaR threshold / price proxy
    ) -> torch.Tensor:
        PnL = (holding * (price[:, 1:] - price[:, :-1])).sum(dim=1)
        C_T = self.terminal_payoff(price[:, -1])
        X   = C_T - PnL                              # loss per path
        return torch.clamp(X - p0, min=0.0).mean() / (1.0 - self.alpha) + p0


# ---------------------------------------------------------------------------

class HestonCVaRLoss(nn.Module):
    """
    CVaR (ES_α) loss for Heston model — Buehler et al. (2019) / He et al. (2025).

    Hedges with two instruments simultaneously:
        - stock S
        - variance swap VarPrice

    PnL = Σ_t (δ_S_t · ΔS_t + δ_V_t · ΔVarPrice_t)
    X   = C_T − PnL     (hedging error, positive = under-hedged)

    Rockafellar-Uryasev dual representation (differentiable):
        ES_α(X) = min_p0 { p0 + E[max(X − p0, 0)] / (1 − α) }

    Args:
        K:     Strike price.
        alpha: Confidence level α ∈ (0, 1). Higher = more tail-focused.
    """

    def __init__(self, K: float, alpha: float = 0.5) -> None:
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
