"""
GBM path simulators with MPS (Apple Silicon GPU) acceleration.

Dynamics: dS = mu*S*dt + sigma*S*dW

Uses the exact log-normal solution to avoid Euler discretization error:
  S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),  Z ~ N(0,1)

Classes
-------
GBMPathGenerator      — standard in-sample simulator, returns (M, N+1) paths.
GBMPathGeneratorOOSP  — out-of-sample simulator: samples a distribution of
                         realised volatilities by resampling standard normals
                         so that each instance's empirical std matches sigma.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class GBMParams:
    S0: float = 100.0
    mu: float = 0.05
    sigma: float = 0.20
    T: float = 30.0 / 252.0
    N: int = 30
    M: int = 100_000


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GBMPathGenerator(nn.Module):
    """
    Standard GBM path generator.

    Returns paths of shape (M, N+1) where column 0 is S0 and columns 1..N
    are the simulated prices under a fixed volatility sigma.
    """

    def __init__(self, params: GBMParams, device: torch.device | None = None):
        super().__init__()
        self.S0 = params.S0
        self.mu = params.mu
        self.sigma = params.sigma
        self.N = params.N
        self.M = params.M
        self.dt = params.T / params.N
        self.device = device or _default_device()

    def forward(self, M: int | None = None, seed: int | None = None) -> torch.Tensor:
        """
        Args:
            M:    number of paths (defaults to self.M).
            seed: optional RNG seed for reproducibility.

        Returns:
            (M, N+1) float32 tensor on CPU.
        """
        if M is None:
            M = self.M
        if seed is not None:
            torch.manual_seed(seed)

        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        vol = self.sigma * (self.dt ** 0.5)

        Z = torch.randn(M, self.N, dtype=torch.float32, device=self.device)
        log_inc = drift + vol * Z                            # (M, N)
        log_paths = torch.cumsum(log_inc, dim=1)             # (M, N)

        zeros = torch.zeros(M, 1, dtype=torch.float32, device=self.device)
        log_paths = torch.cat([zeros, log_paths], dim=1)     # (M, N+1)

        return (self.S0 * torch.exp(log_paths)).cpu()

    def save(
        self,
        output_dir: str | Path = "data",
        filename: str = "gbm_paths",
        seed: int | None = None,
        params: GBMParams | None = None,
    ) -> Path:
        """Simulate, save .pt + .json metadata, and return the .pt path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        paths = self.forward(seed=seed)
        elapsed = time.perf_counter() - t0

        pt_path = output_dir / f"{filename}.pt"
        meta_path = output_dir / f"{filename}.json"

        torch.save(paths, pt_path)

        meta = {
            **(asdict(params) if params else {}),
            "shape": list(paths.shape),
            "dtype": str(paths.dtype),
            "elapsed_s": round(elapsed, 4),
            "device": str(self.device),
            "seed": seed,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        print(
            f"Saved {paths.shape[0]:,} paths × {paths.shape[1]} steps  "
            f"→ {pt_path}  ({elapsed:.3f}s)"
        )
        return pt_path


class GBMPathGeneratorOOSP(nn.Module):
    """
    Out-of-sample GBM path generator.

    Each instance draws a volatility sigma_i by rescaling a block of standard
    normals so that their empirical std equals `sigma` — the same resampling
    trick from the Distributional Adversarial Attacks repo.  This produces a
    distribution of *realised* sigmas around the nominal value rather than
    fixing sigma exactly, giving genuine out-of-sample coverage.

    Shape convention matches the rest of this codebase: paths are (M, N+1)
    where column 0 is S0 and columns 1..N are simulated prices.
    """

    def __init__(self, params: GBMParams, device: torch.device | None = None):
        super().__init__()
        self.S0 = params.S0
        self.mu = params.mu
        self.sigma = params.sigma
        self.N = params.N
        self.dt = params.T / params.N
        self.device = device or _default_device()

    def _sample_sigmas(self, n_instance: int, calibration_len: int = 300) -> torch.Tensor:
        """
        Draw `n_instance` realised volatilities.
        Each sigma_i = self.sigma / empirical_std(Z_i) where Z_i is a block
        of `calibration_len` standard normals.  Returns shape (n_instance,).
        """
        Z = torch.randn(n_instance, calibration_len, device=self.device)
        empirical_std = Z.std(dim=1, correction=1)
        return self.sigma / empirical_std

    def generate_instances(
        self,
        sigmas: torch.Tensor,
        n_per_instance: int,
    ) -> torch.Tensor:
        """
        Simulate paths for each volatility in `sigmas`.

        Args:
            sigmas:          (n_instance,) volatility per instance.
            n_per_instance:  number of paths per instance.

        Returns:
            (n_instance, n_per_instance, N+1) price paths.
        """
        n_instance = sigmas.shape[0]
        drift = (self.mu - 0.5 * sigmas ** 2) * self.dt     # (n_instance,)

        Z = torch.randn(n_instance, n_per_instance, self.N, device=self.device)

        log_inc = (
            drift[:, None, None]
            + sigmas[:, None, None] * (self.dt ** 0.5) * Z
        )                                                    # (n_i, n_pp, N)

        log_paths = torch.cumsum(log_inc, dim=2)
        zeros = torch.zeros(n_instance, n_per_instance, 1, device=self.device)
        log_paths = torch.cat([zeros, log_paths], dim=2)     # (n_i, n_pp, N+1)

        return self.S0 * torch.exp(log_paths)

    def generate_oosp(
        self,
        n_instance: int = 10,
        n_per_instance: int = 10_000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            paths:  (n_instance, n_per_instance, N+1)
            sigmas: (n_instance,)
        """
        sigmas = self._sample_sigmas(n_instance)
        paths = self.generate_instances(sigmas, n_per_instance)
        return paths, sigmas

    def forward(
        self,
        n_instance: int = 10,
        n_per_instance: int = 10_000,
    ) -> torch.Tensor:
        """
        Generate and flatten to (n_instance * n_per_instance, N+1),
        matching the shape returned by GBMPathGenerator.
        """
        paths, _ = self.generate_oosp(n_instance, n_per_instance)
        return paths.view(-1, self.N + 1)


if __name__ == "__main__":
    params = GBMParams(
        S0=100.0,
        mu=0.05,
        sigma=0.20,
        T=30.0 / 252.0,
        N=30,
        M=100_000,
    )

    repo_root = Path(__file__).resolve().parent.parent
    gen = GBMPathGenerator(params)
    gen.save(output_dir=repo_root / "data", filename="gbm_paths", seed=42, params=params)