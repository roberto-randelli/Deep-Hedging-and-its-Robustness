"""
Vanilla deep hedging — GBM + European call + entropic OCE loss.
Replicates He, Sutter & Gonon (2025) / BS_train_clean.py.

Run:
    python src/train_vanilla.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.gbm_simulator import GBMParams, GBMPathGenerator
from src.hedging.hedge_network import HedgeNet
from src.hedging.loss import EntropicOCELoss
from src.hedging.trainer import train

# ---------------------------------------------------------------------------
# Parameters  (matching He et al. §B.1)
# ---------------------------------------------------------------------------
S0      = 100.0
K       = 100.0        # ATM strike
mu      = 0.0          # risk-neutral drift for training paths
sigma   = 0.2
N       = 30           # time steps
dt      = 1 / 365
T       = N * dt       # ≈ 30 trading days

M_train = 100_000      # total simulated paths
lamb    = 1.3          # risk-aversion parameter

n_epochs   = 300
batch_size = 10_000
lr         = 5e-3

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Simulate GBM paths
# ---------------------------------------------------------------------------
print("Simulating GBM paths ...")
params = GBMParams(S0=S0, mu=mu, sigma=sigma, T=T, N=N, M=M_train)
paths  = GBMPathGenerator(params)(seed=42)  # (M, N+1) on CPU
print(f"  paths: {paths.shape}")

# ---------------------------------------------------------------------------
# 2. Build network and loss
# ---------------------------------------------------------------------------
network = HedgeNet(N=N, width=20)
loss_fn = EntropicOCELoss(K=K, sigma=sigma, T=T, lamb=lamb, X_max=True)

# Initial p0: Black-Scholes ATM call price
p0_init = loss_fn.bs_price(torch.tensor(S0)).item()
print(f"  BS price (p0 init): {p0_init:.4f}")

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
print("\nTraining ...")
losses, p0 = train(
    network,
    paths,
    loss_fn,
    p0_init   = p0_init,
    n_epochs  = n_epochs,
    batch_size= batch_size,
    lr        = lr,
    log_every = 50,
)

print(f"\nDone.  Final p0 = {p0.item():.4f}  (BS price = {p0_init:.4f})")

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
torch.save(network.cpu().state_dict(), RESULTS_DIR / "vanilla_network.pt")
torch.save({"losses": losses, "p0": p0.item(), "params": vars(params)},
           RESULTS_DIR / "vanilla_training_log.pt")

print(f"Saved to {RESULTS_DIR}/")
