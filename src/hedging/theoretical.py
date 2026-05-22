"""
Analytical / model-based hedging strategies.

GBM
---
bs_delta(S, tau, K, sigma) -> delta

Heston
------
complete_market_hedge(S_t, V_t, tau, K, kappa, theta, xi, rho) -> (delta_S, delta_V)

    Model hedge for the Heston + variance swap market.

    It uses:
        delta_S = ∂_s u(t, S_t, V_t)
        delta_V = (∂_v u(t, S_t, V_t)) / (∂_v L(t, V_t))

    where u is the Heston call price and
        L(t, v) = (v - theta)/kappa * (1 - exp(-kappa * tau)) + theta * tau
    so that
        ∂_v L(t, v) = (1 - exp(-kappa * tau)) / kappa.

    At tau -> 0 both legs collapse to the intrinsic hedge
        delta_S = 1_{S > K}, delta_V = 0.
"""

from __future__ import annotations

import math

import torch
from torch.distributions import Normal

_N = Normal(0.0, 1.0)


def bs_delta(S: torch.Tensor, tau: float, K: float, sigma: float) -> torch.Tensor:
    """Black-Scholes call delta. Returns (S > K).float() at expiry."""
    if tau <= 1e-9:
        return (S > K).float()

    S = torch.clamp(S, min=1e-12)
    sigma = max(float(sigma), 1e-12)

    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * math.sqrt(tau))
    return _N.cdf(d1)


def _heston_cf_log_return(
    u: torch.Tensor,
    v0: torch.Tensor,
    tau: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    r: float = 0.0,
    q: float = 0.0,
) -> torch.Tensor:
    """
    Characteristic function of X_T = log(S_T / S_t) under the Heston model.

    This uses the standard 'little Heston trap' representation.
    """
    if not torch.is_complex(u):
        raise TypeError("u must be a complex tensor")

    complex_dtype = u.dtype
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64

    v0 = torch.as_tensor(v0, dtype=real_dtype, device=u.device)
    kappa = torch.as_tensor(kappa, dtype=real_dtype, device=u.device)
    theta = torch.as_tensor(theta, dtype=real_dtype, device=u.device)
    xi = torch.as_tensor(xi, dtype=real_dtype, device=u.device)
    rho = torch.as_tensor(rho, dtype=real_dtype, device=u.device)
    r = torch.as_tensor(r, dtype=real_dtype, device=u.device)
    q = torch.as_tensor(q, dtype=real_dtype, device=u.device)

    i = torch.tensor(1j, dtype=complex_dtype, device=u.device)
    a = kappa * theta

    iu = i * u
    d = torch.sqrt((rho * xi * iu - kappa) ** 2 + xi**2 * (iu + u**2))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)

    exp_dt = torch.exp(-d * tau)

    C = (r - q) * iu * tau + (a / xi**2) * (
        (kappa - rho * xi * iu - d) * tau
        - 2.0 * torch.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = ((kappa - rho * xi * iu - d) / xi**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

    return torch.exp(C + D * v0)


def heston_call_price(
    S_t: torch.Tensor,
    V_t: torch.Tensor,
    tau: float,
    K: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    r: float = 0.0,
    q: float = 0.0,
    n_u: int = 128,
    u_max: float = 100.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Heston European call price C(t, S_t, V_t; K).

    Uses the semi-closed formula
        C = S_t * P1 - K * exp(-r * tau) * P2

    where P1 and P2 are computed by Fourier inversion.

    Works with scalar or batched S_t, V_t tensors.
    """
    if tau <= 1e-10:
        return torch.clamp(S_t - K, min=0.0)

    orig_shape = S_t.shape
    S_t = torch.as_tensor(S_t)
    V_t = torch.as_tensor(V_t, device=S_t.device, dtype=S_t.dtype)

    S = torch.clamp(S_t.reshape(-1, 1), min=1e-12)
    V = torch.clamp(V_t.reshape(-1, 1), min=1e-12)

    real_dtype = S.dtype
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    u_real = torch.linspace(eps, u_max, n_u, device=S.device, dtype=real_dtype)
    du = u_real[1] - u_real[0]

    weights = torch.ones_like(u_real)
    weights[0] = 0.5
    weights[-1] = 0.5

    u = u_real.to(complex_dtype)[None, :]
    i = torch.tensor(1j, dtype=complex_dtype, device=S.device)

    K_t = torch.as_tensor(K, device=S.device, dtype=real_dtype)
    k = (torch.log(K_t) - torch.log(S)).to(complex_dtype)  # log(K / S_t)

    psi_u = _heston_cf_log_return(u, V, tau, kappa, theta, xi, rho, r=r, q=q)
    psi_u_mi = _heston_cf_log_return(u - i, V, tau, kappa, theta, xi, rho, r=r, q=q)

    # psi(-i) = E[e^{X_T}] = exp((r-q) tau)
    psi_mi = torch.exp(torch.as_tensor((r - q) * tau, device=S.device, dtype=real_dtype)).to(
        complex_dtype
    )

    phase = torch.exp(-i * u * k)

    integrand_p2 = torch.real(phase * psi_u / (i * u))
    integrand_p1 = torch.real(phase * psi_u_mi / (i * u * psi_mi))

    w = weights[None, :]
    P2 = 0.5 + (du / torch.pi) * torch.sum(w * integrand_p2, dim=-1)
    P1 = 0.5 + (du / torch.pi) * torch.sum(w * integrand_p1, dim=-1)

    price = S.squeeze(-1) * P1 - K_t * torch.exp(
        torch.as_tensor(-r * tau, device=S.device, dtype=real_dtype)
    ) * P2
    price = torch.clamp(price, min=0.0)

    return price.reshape(orig_shape)


def complete_market_hedge(
    S_t: torch.Tensor,
    V_t: torch.Tensor,
    tau: float,
    K: float,
    kappa: float,   # paper's alpha
    theta: float,   # paper's b
    xi: float,
    rho: float,
    eps_s: float = 1e-3,
    eps_v: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Model hedge in the Heston + variance swap market.

    delta_S = ∂_s u
    delta_V = (∂_v u) / (∂_v L),   where ∂_v L = (1 - exp(-kappa * tau)) / kappa
    """
    if tau <= 1e-10:
        return (S_t > K).float(), torch.zeros_like(S_t)

    S_t = torch.clamp(S_t, min=1e-10)
    V_t = torch.clamp(V_t, min=1e-10)

    S_up = S_t + eps_s
    S_dn = torch.clamp(S_t - eps_s, min=1e-10)
    V_up = V_t + eps_v
    V_dn = torch.clamp(V_t - eps_v, min=1e-10)

    u_S_up = heston_call_price(S_up, V_t, tau, K, kappa, theta, xi, rho)
    u_S_dn = heston_call_price(S_dn, V_t, tau, K, kappa, theta, xi, rho)
    u_V_up = heston_call_price(S_t, V_up, tau, K, kappa, theta, xi, rho)
    u_V_dn = heston_call_price(S_t, V_dn, tau, K, kappa, theta, xi, rho)

    delta_S = (u_S_up - u_S_dn) / (S_up - S_dn)
    du_dv = (u_V_up - u_V_dn) / (V_up - V_dn)

    dL_dv = (1.0 - torch.exp(torch.tensor(-kappa * tau, device=S_t.device, dtype=S_t.dtype))) / kappa
    delta_V = du_dv / dL_dv

    return delta_S, delta_V