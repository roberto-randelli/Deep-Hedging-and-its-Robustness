"""
Heston deep-hedging benchmark — ES_α loss with stock + variance swap.
Replicates He, Sutter & Gonon (2025) / Heston_train_clean.py.

Trains one network per α ∈ {0.5, 0.75, 0.95, 0.99} and saves each to results/.

Parameters match the prequel (Heston_generator.py / Heston_train_clean.py):
    kappa=1, theta=0.04, xi=2, rho=-0.7
    (note: Feller condition 2*kappa*theta > xi^2 is violated; xi^2 = 4 >> 0.08)

Run:
    python src/train_heston.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.heston_simulator import HestonParams, simulate
from src.hedging.hedge_network import HestonHedgeNet
from src.hedging.loss import HestonCVaRLoss
from src.hedging.heston_trainer import train

# ---------------------------------------------------------------------------
# Parameters (Heston_generator.py / Heston_train_clean.py)
# ---------------------------------------------------------------------------
S0    = 100.0
K     = 100.0
v0    = 0.04
kappa = 1.0     # mean-reversion rate  (alpha in He et al.)
theta = 0.04    # long-run variance    (b     in He et al.)
xi    = 2.0     # volatility of vol    (sigma in He et al.)
rho   = -0.7
N     = 30
dt    = 1 / 365
T     = N * dt

M_train    = 100_000
n_epochs   = 300
batch_size = 10_000
lr         = 5e-3

ALPHAS = [0.5, 0.75, 0.95, 0.99]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Simulate once, reuse for all α
# ---------------------------------------------------------------------------
print("Simulating Heston paths (Broadie-Kaya exact scheme) ...")
params = HestonParams(
    S0=S0, v0=v0,
    kappa=kappa, theta=theta, xi=xi, rho=rho,
    T=T, N=N, M=M_train,
)
S, V, VarPrice = simulate(params, seed=42)
print(f"  S:        {S.shape}")
print(f"  V:        {V.shape}")
print(f"  VarPrice: {VarPrice.shape}\n")

# ---------------------------------------------------------------------------
# Train one network per α
# ---------------------------------------------------------------------------
for alpha in ALPHAS:
    print(f"{'='*50}")
    print(f"Training ES_{alpha} network (Heston) ...")

    network = HestonHedgeNet(N=N, width=20)
    loss_fn = HestonCVaRLoss(K=K, alpha=alpha)

    # p0 init: α-quantile of unhedged terminal loss (no hedge)
    with torch.no_grad():
        C_T_sample = torch.clamp(S[:1000, -1] - K, min=0.0)
        p0_init = float(C_T_sample.quantile(alpha))
    print(f"  p0 init (VaR_{alpha} of unhedged loss): {p0_init:.4f}")

    losses, p0 = train(
        network,
        S,
        V,
        VarPrice,
        loss_fn,
        p0_init   = p0_init,
        n_epochs  = n_epochs,
        batch_size= batch_size,
        lr        = lr,
        log_every = 100,
    )

    tag = str(alpha).replace(".", "")
    torch.save(network.cpu().state_dict(),
               RESULTS_DIR / f"heston_ES{tag}_network.pt")
    torch.save({"losses": losses, "p0": p0.item(), "alpha": alpha, "params": vars(params)},
               RESULTS_DIR / f"heston_ES{tag}_log.pt")
    print(f"  Final p0 = {p0.item():.4f}  → saved as heston_ES{tag}_*\n")

print("All done.")
