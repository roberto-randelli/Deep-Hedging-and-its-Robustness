"""
Bates path simulator using Euler-Maruyama scheme.

Dynamics (Bates 1996 — Heston SV + compound Poisson jumps):
    dS/S = sqrt(V) dW_S + (J − 1) dN   (J lognormal, N Poisson)
    dV   = kappa*(theta − V) dt + xi*sqrt(V) dW_V,   corr(dW_S, dW_V) = rho

Variance is propagated with full-truncation Euler-Maruyama:
    V_floor    = max(V_t, 0)
    V_{t+1}    = V_t + kappa*(theta − V_floor)*dt + xi*sqrt(V_floor*dt)*Z_V

Stock follows log-Euler with jump compensator:
    drift_comp = lam*(exp(mu_J + 0.5*sigma_J^2) − 1)
    log(S_{t+1}) = log(S_t)
                 + (−0.5*V_floor − drift_comp)*dt
                 + sqrt(V_floor*dt)*Z_S
                 + sum_k log(J_k)           (J_count ~ Poisson(lam*dt))

Correlated Brownians via Cholesky of [[1, rho],[rho, 1]].

Variance swap fair value uses the Kozyra closed-form (continuous SV component only):
    VP[t] = realized_integral(0→t)
            + (V_t − theta)/kappa*(1 − exp(−kappa*(T−t))) + theta*(T−t)
Realized integral accumulated with trapezoidal rule — identical to Heston.

Three tensors are returned — (S, V, VarPrice) each of shape (M, N+1).

Classes:
  BatesSimulator — standard simulation from fixed params
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class BatesParams:
    S0: float      = 100.0
    v0: float      = 0.0574    # initial variance — Kozyra Table 4.1
    kappa: float   = 0.4963    # mean-reversion rate
    theta: float   = 0.0650    # long-run variance
    xi: float      = 0.2286    # vol of vol
    rho: float     = -0.990    # stock-variance correlation
    mu_J: float    = 0.1791    # log-jump mean
    sigma_J: float = 0.1346    # log-jump std
    lam: float     = 0.1382    # Poisson jump intensity
    T: float       = 30 / 365
    N: int         = 30        # time steps
    M: int         = 50_000


class BatesSimulator:
    """Simulates Bates paths using full-truncation Euler-Maruyama."""

    def __init__(self, params: BatesParams) -> None:
        self.params = params

    @staticmethod
    def _simulate_paths(
        params: BatesParams,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (S, V) numpy arrays of shape (M, N+1)."""
        M, N = params.M, params.N
        dt   = params.T / N
        kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
        mu_J, sigma_J, lam    = params.mu_J, params.sigma_J, params.lam

        S = np.zeros((M, N + 1), dtype=np.float64)
        V = np.zeros((M, N + 1), dtype=np.float64)
        S[:, 0] = params.S0
        V[:, 0] = params.v0

        # Cholesky for correlated Brownians: [Z_S, Z_V] where corr(Z_S, Z_V) = rho
        L = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho ** 2)]])

        # Jump compensator (risk-neutral drift adjustment)
        drift_comp = lam * (np.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0)

        for t in range(N):
            v_floor = np.maximum(V[:, t], 0.0)

            # Correlated standard normals
            Z = rng.standard_normal((M, 2)) @ L.T   # (M, 2); Z[:,0]=Z_S, Z[:,1]=Z_V

            # Variance — full-truncation Euler
            V[:, t + 1] = (
                V[:, t]
                + kappa * (theta - v_floor) * dt
                + xi * np.sqrt(v_floor * dt) * Z[:, 1]
            )

            # Poisson jump counts per path, then aggregate lognormal log-sizes.
            # Sum of j i.i.d. N(mu_J, sigma_J^2) is N(j*mu_J, j*sigma_J^2).
            j_count  = rng.poisson(lam * dt, size=M)
            log_jump = np.zeros(M)
            has_jump = j_count > 0
            if has_jump.any():
                jc = j_count[has_jump].astype(float)
                log_jump[has_jump] = rng.normal(mu_J * jc, sigma_J * np.sqrt(jc))

            # Stock — log-Euler with drift compensator
            S[:, t + 1] = S[:, t] * np.exp(
                (-0.5 * v_floor - drift_comp) * dt
                + np.sqrt(v_floor * dt) * Z[:, 0]
                + log_jump
            )

        return S, V

    @staticmethod
    def _compute_var_swap_prices(
        V: np.ndarray,
        params: BatesParams,
    ) -> np.ndarray:
        """
        Returns VarPrice of shape (M, N+1).

        Uses the Kozyra closed-form for the continuous SV component:
            VP[t] = realized_integral(0→t)
                    + (V_t − theta)/kappa*(1 − exp(−kappa*(T−t))) + theta*(T−t)
        """
        M, N = V.shape[0], params.N
        dt   = params.T / N
        kappa, theta, T = params.kappa, params.theta, params.T

        VarPrice = np.zeros((M, N + 1), dtype=np.float64)
        VarPrice[:, 0] = (
            (V[:, 0] - theta) / kappa * (1.0 - np.exp(-kappa * T)) + theta * T
        )

        var_int = np.zeros(M, dtype=np.float64)
        for i in range(N):
            var_int   += 0.5 * dt * (V[:, i] + V[:, i + 1])
            remaining  = T - (i + 1) * dt
            correction = (
                (V[:, i + 1] - theta) / kappa * (1.0 - np.exp(-kappa * remaining))
                + theta * remaining
            )
            VarPrice[:, i + 1] = var_int + correction

        return VarPrice

    def simulate(
        self,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns float32 tensors (S, V, VarPrice) each of shape (M, N+1)."""
        rng       = np.random.default_rng(seed)
        S_np, V_np = self._simulate_paths(self.params, rng)
        VP_np      = self._compute_var_swap_prices(V_np, self.params)
        return (
            torch.tensor(S_np,  dtype=torch.float32),
            torch.tensor(V_np,  dtype=torch.float32),
            torch.tensor(VP_np, dtype=torch.float32),
        )

    def run_and_save(
        self,
        output_dir: str | Path = "data",
        filename: str = "bates_paths",
        seed: int | None = None,
    ) -> Path:
        """
        Simulates and saves:
          <output_dir>/<filename>.pt   — tuple (S, V, VarPrice) float32 tensors
          <output_dir>/<filename>.json — simulation metadata
        Returns the path to the .pt file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        S, V, VarPrice = self.simulate(seed=seed)
        elapsed = time.perf_counter() - t0

        pt_path   = output_dir / f"{filename}.pt"
        meta_path = output_dir / f"{filename}.json"

        torch.save((S, V, VarPrice), pt_path)

        meta = {
            **asdict(self.params),
            "shape": list(S.shape),
            "dtype": str(S.dtype),
            "elapsed_s": round(elapsed, 4),
            "seed": seed,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        print(
            f"Saved {S.shape[0]:,} paths × {S.shape[1]} steps  "
            f"→ {pt_path}  ({elapsed:.3f}s)"
        )
        return pt_path


if __name__ == "__main__":
    params = BatesParams()
    repo_root = Path(__file__).resolve().parent.parent
    run_and_save = BatesSimulator(params).run_and_save
    run_and_save(output_dir=repo_root / "data", seed=42)
