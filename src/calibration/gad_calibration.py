"""
GAD parameter estimation via two sequential OLS regressions.

Follows He, Sutter & Gonon (2025) and Lütkebohmert–Schmidt–Sester (2022).
γ is fixed at 1 throughout (linear-coefficient SDE).

---
Model (γ = 1, Euler-Maruyama at daily steps Δt = 1/252):

    ΔS_t = (b0 + b1·S_{t-1})·Δt + (a0 + a1·S_{t-1})·√Δt·Z_t

Regression 1 — Drift:
    Regressors: X = [Δt, S_{t-1}·Δt]  (column vectors)
    Target:     y = ΔS_t
    OLS → (b̂0, b̂1), residuals η̂_t = ΔS_t − (b̂0 + b̂1·S_{t-1})·Δt

Regression 2 — Diffusion:
    Regressors: X = [1, S_{t-1}]
    Target:     y = |η̂_t| / √Δt    (abs. residuals scaled to per-unit-time)
    OLS → (â0, â1)

Both regressions are performed on the normalised price series (S0 = 10).

Public API
----------
calibrate_gad(prices, dt)          → GADParams
rolling_calibration(prices, window, dt) → pd.DataFrame
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

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
) -> pd.DataFrame:
    """
    Compute rolling GAD estimates over a full price history.

    For each ending index t ∈ [window, len(prices)], fits GAD on
    prices[t-window : t].

    Returns a DataFrame with columns (b0, b1, a0, a1) and an integer index
    corresponding to the last observation in each rolling window.
    Useful for robustness-check plots: do the FIX params lie within the
    historical range of estimates?
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    records: list[dict] = []

    for end in range(window, n + 1):
        window_prices = prices[end - window : end]
        try:
            p = calibrate_gad(window_prices, dt=dt, S0_target=S0_target)
            records.append({
                "end_idx": end - 1,
                "b0": p.b0,
                "b1": p.b1,
                "a0": p.a0,
                "a1": p.a1,
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
