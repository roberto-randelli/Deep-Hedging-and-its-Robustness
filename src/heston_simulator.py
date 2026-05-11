"""
Heston path simulator using the Broadie-Kaya exact scheme.

Dynamics (Heston 1993):
    dS = S * sqrt(V) * dW_S
    dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V,  corr(dW_S, dW_V) = rho

Variance is sampled exactly via the non-central chi-squared representation:
    V(t+dt) | V(t)  ~  c * chi^2(d, lambda_t)
where
    d        = 4*kappa*theta / xi^2          (degrees of freedom)
    c        = xi^2*(1 - exp(-kappa*dt)) / (4*kappa)
    lambda_t = V(t)*exp(-kappa*dt) / c       (non-centrality)

Stock log-increment uses the Broadie-Kaya trapezoidal approximation for the
integrated variance, which avoids Euler discretization error on S.

Three tensors are returned — (S, V, VarPrice) each of shape (M, N+1):
  - S:        stock price path
  - V:        instantaneous variance path
  - VarPrice: fair value of a variance swap at each time step
              (= E_t[∫_t^T V_s ds] under the Heston risk-neutral measure)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class HestonParams:
    S0: float    = 100.0
    v0: float    = 0.04    # initial variance
    kappa: float = 1.0     # mean-reversion rate         (alpha in He et al.)
    theta: float = 0.04    # long-run variance            (b     in He et al.)
    xi: float    = 2.0     # volatility of variance       (sigma in He et al.)
    rho: float   = -0.7    # correlation dW_S · dW_V
    T: float     = 30 / 365
    N: int       = 30      # time steps
    M: int       = 100_000


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate_paths(
    params: HestonParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (S, V) numpy arrays of shape (M, N+1).

    Variance uses the exact Broadie-Kaya chi-squared sampling.
    Stock log-increment uses the trapezoidal integrated-variance approximation.
    """
    M, N = params.M, params.N
    dt = params.T / N
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho

    S = np.zeros((M, N + 1), dtype=np.float64)
    V = np.zeros((M, N + 1), dtype=np.float64)
    S[:, 0] = params.S0
    V[:, 0] = params.v0

    d = 4.0 * kappa * theta / xi ** 2
    c = xi ** 2 * (1.0 - np.exp(-kappa * dt)) / (4.0 * kappa)

    for t in range(N):
        vprev = V[:, t]

        # --- exact variance step (Broadie-Kaya) ---
        lam     = vprev * np.exp(-kappa * dt) / c
        poisson = rng.poisson(lam / 2.0)           # (M,)
        shape   = (d + 2.0 * poisson) / 2.0        # (M,)
        V[:, t + 1] = c * rng.gamma(shape, scale=2.0)

        # --- log-stock increment (trapezoidal int-var approx) ---
        vnext = V[:, t + 1]
        int_v = 0.5 * (vprev + vnext) * dt         # integrated var approximation
        Z     = rng.normal(0.0, np.sqrt(np.maximum(int_v, 0.0)))
        term1 = (rho / xi * kappa - 0.5) * int_v
        term2 = (rho / xi) * (vnext - vprev - kappa * theta * dt)
        term3 = np.sqrt(1.0 - rho ** 2) * Z
        S[:, t + 1] = S[:, t] * np.exp(term1 + term2 + term3)

    return S, V


def _compute_var_swap_prices(
    V: np.ndarray,
    params: HestonParams,
) -> np.ndarray:
    """
    Computes variance swap fair values: VarPrice[i, t] = E_t[ ∫_t^T V_s ds ].

    Under the Heston model (risk-neutral):
        E_t[∫_t^T V_s ds] = (V_t − theta)/kappa * (1 − exp(−kappa*(T−t))) + theta*(T−t)

    For t > 0 we add the already-realised integrated variance (trapezoidal).
    """
    M, N   = V.shape[0], params.N
    dt     = params.T / N
    kappa, theta, T = params.kappa, params.theta, params.T

    VarPrice = np.zeros((M, N + 1), dtype=np.float64)
    VarPrice[:, 0] = (
        (V[:, 0] - theta) / kappa * (1.0 - np.exp(-kappa * T)) + theta * T
    )

    var_int = np.zeros(M, dtype=np.float64)
    for i in range(N):
        var_int += 0.5 * dt * (V[:, i] + V[:, i + 1])
        remaining  = T - (i + 1) * dt
        correction = (
            (V[:, i + 1] - theta) / kappa * (1.0 - np.exp(-kappa * remaining))
            + theta * remaining
        )
        VarPrice[:, i + 1] = var_int + correction

    return VarPrice


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    params: HestonParams,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns float32 tensors (S, V, VarPrice) each of shape (M, N+1) on CPU.
    """
    rng = np.random.default_rng(seed)
    S_np, V_np   = _simulate_paths(params, rng)
    VP_np        = _compute_var_swap_prices(V_np, params)
    return (
        torch.tensor(S_np,  dtype=torch.float32),
        torch.tensor(V_np,  dtype=torch.float32),
        torch.tensor(VP_np, dtype=torch.float32),
    )


def run_and_save(
    params: HestonParams,
    output_dir: str | Path = "data",
    filename: str = "heston_paths",
    seed: int | None = None,
) -> Path:
    """
    Simulates Heston paths and saves:
      <output_dir>/<filename>.pt   — tuple (S, V, VarPrice) float32 tensors
      <output_dir>/<filename>.json — simulation metadata
    Returns the path to the .pt file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    S, V, VarPrice = simulate(params, seed=seed)
    elapsed = time.perf_counter() - t0

    pt_path   = output_dir / f"{filename}.pt"
    meta_path = output_dir / f"{filename}.json"

    torch.save((S, V, VarPrice), pt_path)

    meta = {
        **asdict(params),
        "shape": list(S.shape),
        "dtype": str(S.dtype),
        "elapsed_s": round(elapsed, 4),
        "seed": seed,
        "feller_condition_satisfied": bool(2 * params.kappa * params.theta > params.xi ** 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(
        f"Saved {S.shape[0]:,} paths × {S.shape[1]} steps  "
        f"→ {pt_path}  ({elapsed:.3f}s)"
    )
    return pt_path


if __name__ == "__main__":
    params = HestonParams(
        S0=100.0, v0=0.04,
        kappa=1.0, theta=0.04, xi=2.0, rho=-0.7,
        T=30 / 365, N=30, M=100_000,
    )
    repo_root = Path(__file__).resolve().parent.parent
    run_and_save(params, output_dir=repo_root / "data", seed=42)
