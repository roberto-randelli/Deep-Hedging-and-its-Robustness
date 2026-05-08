"""
GBM path simulator with MPS (Apple Silicon GPU) acceleration.

Dynamics: dS = mu*S*dt + sigma*S*dW

Uses the exact log-normal solution to avoid Euler discretization error:
  S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),  Z ~ N(0,1)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class GBMParams:
    S0: float = 100.0
    mu: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    N: int = 252
    M: int = 100_000


def simulate(
    params: GBMParams,
    device: torch.device | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Returns a float32 tensor of shape (M, N+1) on CPU.
    Column 0 is S0; columns 1..N are the simulated price path.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if seed is not None:
        torch.manual_seed(seed)

    dt = params.T / params.N
    drift = (params.mu - 0.5 * params.sigma ** 2) * dt
    vol = params.sigma * (dt ** 0.5)

    # All random draws at once — one kernel launch, fully vectorised
    Z = torch.randn(params.M, params.N, dtype=torch.float32, device=device)
    log_increments = drift + vol * Z          # (M, N)
    log_paths = torch.cumsum(log_increments, dim=1)  # (M, N)

    # Prepend log(S0) = 0 column, then exponentiate
    zeros = torch.zeros(params.M, 1, dtype=torch.float32, device=device)
    log_paths = torch.cat([zeros, log_paths], dim=1)  # (M, N+1)
    paths = params.S0 * torch.exp(log_paths)

    return paths.cpu()


def run_and_save(
    params: GBMParams,
    output_dir: str | Path = "data",
    filename: str = "gbm_paths",
    device: torch.device | None = None,
    seed: int | None = None,
) -> Path:
    """
    Simulates GBM paths and saves:
      <output_dir>/<filename>.pt   — float32 tensor (M, N+1)
      <output_dir>/<filename>.json — simulation metadata
    Returns the path to the .pt file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    paths = simulate(params, device=device, seed=seed)
    elapsed = time.perf_counter() - t0

    pt_path = output_dir / f"{filename}.pt"
    meta_path = output_dir / f"{filename}.json"

    torch.save(paths, pt_path)

    meta = {
        **asdict(params),
        "shape": list(paths.shape),
        "dtype": str(paths.dtype),
        "elapsed_s": round(elapsed, 4),
        "device": str(device) if device else "auto",
        "seed": seed,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(
        f"Saved {paths.shape[0]:,} paths × {paths.shape[1]} steps  "
        f"→ {pt_path}  ({elapsed:.3f}s)"
    )
    return pt_path


if __name__ == "__main__":
    params = GBMParams(
        S0=100.0,
        mu=0.05,
        sigma=0.20,
        T=1.0,
        N=252,
        M=100_000,
    )

    repo_root = Path(__file__).resolve().parent.parent
    run_and_save(params, output_dir=repo_root / "data", seed=42)
