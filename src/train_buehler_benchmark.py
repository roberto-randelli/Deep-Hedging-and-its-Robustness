"""
Buehler et al. (2019) §3.2 benchmark — GBM + European call + ES_α loss.

Trains one network per α ∈ {0.5, 0.75, 0.95, 0.99} and saves each to results/.

Run:
    python src/train_buehler_benchmark.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import torch
import torch.nn as nn

from src.gbm_simulator import GBMParams, GBMPathGenerator
from src.hedging.hedge_network import HedgeNet
from src.hedging.loss import CVaRLoss
from src.hedging.trainer import train

# ---------------------------------------------------------------------------
# Parameters (Buehler §3.2)
# ---------------------------------------------------------------------------
S0    = 100.0
K     = 100.0
mu    = 0.0          # risk-neutral drift
sigma = 0.2
N     = 30
dt    = 1 / 365
T     = N * dt

M_train    = 100_000
n_epochs   = 400
batch_size = 10_000
lr         = 5e-3

ALPHAS = [0.5, 0.75, 0.95, 0.99]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Simulate once, reuse for all α
# ---------------------------------------------------------------------------
print("Simulating GBM paths ...")
params = GBMParams(S0=S0, mu=mu, sigma=sigma, T=T, N=N, M=M_train)
paths  = GBMPathGenerator(params)(seed=42)
print(f"  paths: {paths.shape}\n")

# ---------------------------------------------------------------------------
# Train one network per α
# ---------------------------------------------------------------------------
for alpha in ALPHAS:
    print(f"{'='*50}")
    print(f"Training ES_{alpha} network ...")

    network = HedgeNet(N=N, width=20)
    loss_fn = CVaRLoss(K=K, alpha=alpha)

    _d1 = (math.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * T**0.5)
    _d2 = _d1 - sigma * T**0.5
    _N  = torch.distributions.Normal(0, 1).cdf
    p0_init = float(S0 * _N(torch.tensor(_d1)) - K * _N(torch.tensor(_d2)))
    print(f"  fixed capital (BS price): {p0_init:.4f}")

    capital = torch.tensor(p0_init, dtype=torch.float32)
    z_init = 0.0

    losses, z = train(
        network,
        paths,
        loss_fn,
        capital   = p0_init,
        z_init    = 0.0,
        n_epochs  = n_epochs if alpha in (0.5, 0.75) else 1000,
        batch_size= batch_size,
        lr        = lr,
        log_every = 100,
    )

    tag = str(alpha).replace(".", "")
    torch.save(network.cpu().state_dict(),
           RESULTS_DIR / f"buehler_ES{tag}_network.pt")
    torch.save({
        "losses": losses,
        "z": z.item(),
        "capital": p0_init,
        "alpha": alpha}, RESULTS_DIR / f"buehler_ES{tag}_log.pt")

    print(f"  Final z = {z.item():.4f}  → saved as buehler_ES{tag}_*\n")

print("All done.")
