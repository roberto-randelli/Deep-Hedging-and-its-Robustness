"""
GAD parameter estimation — two methods.

Follows He, Sutter & Gonon (2025) and Lütkebohmert–Schmidt–Sester (2022).
γ is fixed at 1 throughout (linear-coefficient SDE).

Model (γ = 1, Euler-Maruyama at daily steps Δt):
    ΔS_t = (b0 + b1·S_{t-1})·Δt + (a0 + a1·S_{t-1})·√Δt·Z_t

Public API
----------
calibrate_gad_mle(prices, dt)      → GADParams  [PREFERRED — matches He et al.]
calibrate_gad(prices, dt)          → GADParams  [OLS fallback — biased drift]
rolling_calibration(prices, window, dt) → pd.DataFrame
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.gad_simulator import GADParams


def _normalise(prices: np.ndarray, S0_target: float = 10.0) -> np.ndarray:
    """Scale price series so that prices[0] == S0_target."""
    return prices * (S0_target / prices[0])


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least-squares: returns coefficients β minimising ‖y − Xβ‖²."""
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def calibrate_gad_mle(
    prices: np.ndarray,
    dt: float = 1 / 252,
    S0_target: float = 10.0,
    T: float = 30 / 252,
    N: int = 30,
    M: int = 100_000,
    gamma_bounds: tuple[float, float] = (1e-6, 1.2),
    method: str = "COBYLA",
    max_iter: int = 1000,
) -> GADParams:
    """
    Estimate GAD parameters via maximum likelihood (He et al. 2025).

    Maximises the Gaussian log-likelihood of the Euler-Maruyama increments:

        ΔS_t | S_{t-1} ~ N((b0+b1·S_{t-1})·Δt,  (a0+a1·S_{t-1})^{2γ}·Δt)

    Operates on the normalised price series (S0 = S0_target). Directly
    matches He et al.'s `compute_max_parameters` function (GAD_generator.ipynb).

    Args:
        prices:       1-D array of daily closing prices (raw).
        dt:           Time step in years (default 1/252).
        S0_target:    Normalised starting price (default 10.0).
        T, N, M:      Simulation metadata (not used in calibration).
        gamma_bounds: (min, max) for the elasticity γ. He et al. bound to
                      (eps, 1.2) to prevent degenerate diffusion.
        method:       scipy.optimize.minimize method (default COBYLA).
        max_iter:     Maximum optimiser iterations.

    Returns:
        GADParams with MLE estimates (a0, a1, b0, b1, gamma, S0=S0_target).
    """
    S = _normalise(np.asarray(prices, dtype=np.float64), S0_target)
    eps = 1e-15

    def neg_log_likelihood(params: np.ndarray) -> float:
        a0, a1, b0, b1, gamma = params
        sigma_t = (a0 + a1 * np.maximum(S[:-1], 0.0)) ** gamma
        mean_t  = (b0 + b1 * S[:-1]) * dt
        var_t   = sigma_t ** 2 * dt
        log_const = np.log(sigma_t * np.sqrt(2 * np.pi * dt) + eps)
        sq_term   = (S[1:] - S[:-1] - mean_t) ** 2 / (2 * var_t + eps)
        return float(np.mean(log_const + sq_term))

    x0 = np.array([0.1, 0.1, 0.0, 0.0, 1.0])
    result = minimize(
        neg_log_likelihood,
        x0,
        method=method,
        options={"maxiter": max_iter, "rhobeg": 0.01},
        bounds=[
            (eps, None),    # a0 > 0
            (eps, None),    # a1 > 0
            (None, None),   # b0 unconstrained
            (None, None),   # b1 unconstrained
            gamma_bounds,   # γ ∈ (eps, 1.2)
        ],
    )
    a0, a1, b0, b1, gamma = result.x
    a0    = max(float(a0), eps)
    a1    = max(float(a1), eps)
    gamma = float(np.clip(gamma, gamma_bounds[0], gamma_bounds[1]))

    return GADParams(
        b0=float(b0), b1=float(b1),
        a0=a0, a1=a1, gamma=gamma,
        S0=S0_target, T=T, N=N, M=M,
    )


def calibrate_gad(
    prices: np.ndarray,
    dt: float = 1 / 252,
    S0_target: float = 10.0,
    T: float = 30 / 252,
    N: int = 30,
    M: int = 100_000,
) -> GADParams:
    """
    Estimate GAD parameters from a 1-D array of daily closing prices.

    The price series is first normalised to start at S0_target = 10.
    Both regressions are therefore in the normalised-price space; the
    returned GADParams can be fed directly into GADSimulator.

    Args:
        prices:    1-D array of daily closing prices (raw, not normalised).
        dt:        Time step in years (default 1/252 for daily data).
        S0_target: Normalised starting price (default 10.0).
        T, N, M:   Simulation horizon, steps, and paths passed through to
                   the returned GADParams (not used in calibration itself).

    Returns:
        GADParams with (b0, b1, a0, a1, gamma=1, S0=S0_target, T, N, M).
    """
    S = _normalise(np.asarray(prices, dtype=np.float64), S0_target)

    dS   = np.diff(S)                       # ΔS_t,  length T-1
    S_lag = S[:-1]                           # S_{t-1}
    sqrt_dt = np.sqrt(dt)

    # ── Regression 1: drift ──────────────────────────────────────────────────
    # ΔS_t = b0·dt + b1·S_{t-1}·dt + η_t
    X1 = np.column_stack([
        np.full(len(dS), dt),
        S_lag * dt,
    ])
    b0, b1 = _ols(X1, dS)
    eta = dS - X1 @ np.array([b0, b1])     # residuals

    # ── Regression 2: diffusion (γ = 1) ────────────────────────────────────
    # |η̂_t| / √dt = a0 + a1·S_{t-1} + ε_t
    y2 = np.abs(eta) / sqrt_dt
    X2 = np.column_stack([np.ones(len(eta)), S_lag])
    a0, a1 = _ols(X2, y2)

    # Enforce a0 ≥ 0, a1 ≥ 0 to keep diffusion non-negative
    a0 = max(float(a0), 0.0)
    a1 = max(float(a1), 0.0)

    return GADParams(
        b0=float(b0),
        b1=float(b1),
        a0=a0,
        a1=a1,
        gamma=1.0,
        S0=S0_target,
        T=T,
        N=N,
        M=M,
    )


def rolling_calibration(
    prices: np.ndarray,
    window: int = 250,
    dt: float = 1 / 252,
    S0_target: float = 10.0,
    use_mle: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling GAD estimates over a full price history.

    For each ending index t ∈ [window, len(prices)], fits GAD on
    prices[t-window : t].

    Args:
        use_mle: If True (default), use MLE calibration per window.
                 Set False to use the faster OLS approximation.

    Returns a DataFrame with columns (b0, b1, a0, a1) and an integer index
    corresponding to the last observation in each rolling window.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    records: list[dict] = []
    cal_fn = calibrate_gad_mle if use_mle else calibrate_gad

    for end in range(window, n + 1):
        window_prices = prices[end - window : end]
        try:
            p = cal_fn(window_prices, dt=dt, S0_target=S0_target)
            records.append({
                "end_idx": end - 1,
                "b0": p.b0,
                "b1": p.b1,
                "a0": p.a0,
                "a1": p.a1,
                "gamma": p.gamma,
            })
        except Exception:
            continue

    return pd.DataFrame(records).set_index("end_idx")


if __name__ == "__main__":
    # Smoke test on synthetic GBM prices (should recover a0≈0, a1≈sigma)
    rng = np.random.default_rng(42)
    sigma = 0.2
    T_test = 5.0
    N_test = int(T_test * 252)
    dt_test = 1 / 252
    S_true = np.zeros(N_test + 1)
    S_true[0] = 350.0  # raw SPY-like level
    for t in range(N_test):
        S_true[t + 1] = S_true[t] * np.exp(
            -0.5 * sigma ** 2 * dt_test + sigma * np.sqrt(dt_test) * rng.standard_normal()
        )

    p = calibrate_gad(S_true, dt=dt_test)
    print("Calibrated GAD params (GBM ground truth: b0=0, b1=0, a0=0, a1=0.2 on S0=10):")
    for k, v in asdict(p).items():
        if k not in ("T", "N", "M", "gamma"):
            print(f"  {k} = {v:.6f}")
