"""
General Affine Diffusion (GAD) path simulator — Euler-Maruyama scheme.

Model (Lütkebohmert–Schmidt–Sester 2022, He et al. 2025):
    dS_t = (b0 + b1 * S_t) dt + (a0 + a1 * S_t)^γ dW_t

γ is calibrated via MLE (typically 0.7–1.0 for equity indices).
With γ = 1 this reduces to the linear-coefficient SDE used in the plan.

Discretised via Euler-Maruyama at daily steps dt = T / N:
    S_{t+1} = S_t + (b0 + b1*S_t)*dt + (a0 + a1*S_t)^γ * sqrt(dt)*Z_t
    Z_t ~ N(0, 1)

Paths are clamped to be strictly positive at each step to avoid degenerate
diffusion coefficients when a0 is small.

This generalises the GBM case (b0=0, b1=r, a0=0, a1=sigma) and matches
He et al.'s calibration target for the market-data experiments.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class GADParams:
    b0: float        # constant drift coefficient
    b1: float        # linear drift coefficient (≈ risk-free rate in GBM)
    a0: float        # constant diffusion coefficient
    a1: float        # linear diffusion coefficient (≈ sigma in GBM)
    gamma: float = 1.0       # elasticity — fixed at 1 per He et al. (2025)
    S0: float = 10.0         # normalised starting price
    T: float = 30 / 252      # maturity in years (30 trading days)
    N: int = 30              # number of time steps
    M: int = 100_000         # number of paths


class GADSimulator:
    """
    Simulates GAD paths via Euler-Maruyama on a uniform time grid.

    Supports arbitrary γ ∈ (0, 1.2] as returned by MLE calibration.
    Paths are clamped below at 1e-6 at every step so that the diffusion
    coefficient (a0 + a1*S)^γ is always non-negative.
    """

    def __init__(self, params: GADParams) -> None:
        self.params = params

    @staticmethod
    def _simulate_paths(
        params: GADParams,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Returns price paths as numpy array of shape (M, N+1)."""
        M, N = params.M, params.N
        dt = params.T / N
        b0, b1 = params.b0, params.b1
        a0, a1 = params.a0, params.a1
        gamma = params.gamma
        sqrt_dt = np.sqrt(dt)

        S = np.empty((M, N + 1), dtype=np.float64)
        S[:, 0] = params.S0

        Z = rng.standard_normal((M, N))

        for t in range(N):
            s = S[:, t]
            drift = (b0 + b1 * s) * dt
            diff  = np.power(np.maximum(a0 + a1 * s, 0.0), gamma) * sqrt_dt * Z[:, t]
            S[:, t + 1] = np.maximum(s + drift + diff, 1e-6)

        return S

    def simulate(self, seed: int | None = None) -> torch.Tensor:
        """Returns float32 tensor of shape (M, N+1)."""
        rng = np.random.default_rng(seed)
        S_np = self._simulate_paths(self.params, rng)
        return torch.tensor(S_np, dtype=torch.float32)

    def run_and_save(
        self,
        output_dir: str | Path = "data",
        filename: str = "gad_paths",
        seed: int | None = None,
    ) -> Path:
        """
        Simulates and saves:
          <output_dir>/<filename>.pt   — float32 tensor (M, N+1)
          <output_dir>/<filename>.json — simulation metadata
        Returns the path to the .pt file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        S = self.simulate(seed=seed)
        elapsed = time.perf_counter() - t0

        pt_path   = output_dir / f"{filename}.pt"
        meta_path = output_dir / f"{filename}.json"

        torch.save(S, pt_path)

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
    # Quick smoke-test: simulate with GBM-equivalent params and check moments
    params = GADParams(
        b0=0.0, b1=0.0, a0=0.0, a1=0.2,   # GBM with sigma=0.2
        S0=10.0, T=30 / 252, N=30, M=100_000,
    )
    repo_root = Path(__file__).resolve().parent.parent
    sim = GADSimulator(params)
    S = sim.simulate(seed=42)
    print(f"Shape:          {S.shape}")
    print(f"S0 (all paths): {S[:, 0].mean():.4f}  (should be 10.0)")
    print(f"S_T mean:       {S[:, -1].mean():.4f}")
    print(f"S_T std:        {S[:, -1].std():.4f}")
    sim.run_and_save(output_dir=repo_root / "data", filename="gad_test_paths", seed=42)
