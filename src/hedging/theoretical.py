"""
Analytical / model-based hedging strategies.

GBM
---
bs_delta(S, tau, K, sigma) -> delta

Heston
------
complete_market_hedge(S_t, V_t, tau, K, kappa) -> (delta_S, delta_V)

    Completes the Heston market by pairing the BS delta (stock leg) with a
    variance-swap vega leg.  The formula follows from:

        delta_S = N(d1)   with  sigma = sqrt(V_t)

        delta_V = (dC/dV) / (dVarSwap/dV_t)

    where
        dC/dV       = S * sqrt(tau) * phi(d1) / (2*sqrt(V))   [BS vega in V-units]
        dVarSwap/dV_t = (1 - exp(-kappa*tau)) / kappa          [from Prequel VarSwap formula]

    At tau->0 both legs collapse to the intrinsic hedge (delta_S=1_{S>K}, delta_V=0).
"""

from __future__ import annotations

import torch
from torch.distributions import Normal

_N = Normal(0.0, 1.0)


def bs_delta(S: torch.Tensor, tau: float, K: float, sigma: float) -> torch.Tensor:
    """Black-Scholes call delta.  Returns (S>K).float() at expiry."""
    if tau <= 1e-9:
        return (S > K).float()
    d1 = (torch.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * tau ** 0.5)
    return _N.cdf(d1)


def complete_market_hedge(
    S_t: torch.Tensor,
    V_t: torch.Tensor,
    tau: float,
    K: float,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Delta-vega hedge that completes the Heston market via the variance swap.

    Args:
        S_t:   Stock prices at time t,              shape (M,)
        V_t:   Instantaneous variances at time t,   shape (M,)
        tau:   Time to maturity  T - t  (years)
        K:     Strike price
        kappa: Heston mean-reversion rate

    Returns:
        delta_S: Stock hedge ratio,     shape (M,)
        delta_V: Var-swap hedge ratio,  shape (M,)
    """
    if tau <= 1e-9:
        return (S_t > K).float(), torch.zeros_like(S_t)

    V_c   = torch.clamp(V_t, min=1e-6)
    sigma = V_c.sqrt()
    d1    = (torch.log(S_t / K) + 0.5 * V_c * tau) / (sigma * tau ** 0.5)

    delta_S = _N.cdf(d1)

    # dC/dV  (BS vega expressed in variance units, not vol units)
    vega_V = S_t * (tau ** 0.5) * _N.log_prob(d1).exp() / (2.0 * sigma)

    # dVarSwap/dV_t  (sensitivity of the Prequel VarSwap price to current variance)
    dvs_dv = (1.0 - torch.exp(torch.full_like(V_t, -kappa * tau))) / kappa

    delta_V = vega_V / dvs_dv
    return delta_S, delta_V
