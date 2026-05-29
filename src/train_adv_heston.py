"""
Reproduce He, Sutter & Gonon (2025) Sections 5.2–5.3.

Trains three hedging strategies on the Heston model:
  - Clean:    standard CVaR₀.₅ deep hedging (700 clean epochs)
  - S-Attack: adversarial training perturbing stock price S only
  - SV-Attack: adversarial training perturbing S and V jointly

Evaluates on:
  - OOS: 1 M held-out Heston paths (same distribution, seed=42)
  - OOD: 100 perturbed Heston configs with ±10% parameter variation

Outputs:
  results/adv_nets/N{N}_{method}_s{s}_network.pt  — network state_dict
  results/adv_nets/N{N}_{method}_s{s}_p0.pt        — learned p0 scalar
  results/adv_heston_results.pt                    — dict of OOS/OOD ES values
  results/heston_figure1_adversarial.png           — Figure 1 (OOS + OOD)

Usage:
  python src/train_adv_heston.py           # full run
  python src/train_adv_heston.py --resume  # skip already-saved networks
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Allow running from project root: python src/train_adv_heston.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heston_simulator import HestonParams, HestonSimulator, OutOfSampleHestonSimulator
from src.hedging.hedge_network import HestonHedgeNet
from src.hedging.loss import HestonCVaRLoss
from src.hedging.adv_trainer import (
    heston_var_price,
    s_budget_attack,
    sv_budget_attack,
    train_adv_heston,
)

torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SIZES        = [5_000, 10_000, 20_000, 50_000, 100_000]
N_SEEDS        = 3
N_EPOCHS_CLEAN = 300
N_EPOCHS_ADV   = 400
BATCH_SIZE     = 10_000
LR             = 5e-3
ALPHA_CVAR     = 0.5
K              = 100.0
N_STEPS        = 30
T              = 30 / 365
VP_SCALE       = 1000.0

# Heston parameters (He et al. calibration)
S0    = 100.0
v0    = 0.04
kappa = 1.0
theta = 0.04
xi    = 2.0
rho   = -0.7

M_TRAIN    = 100_000
M_TEST     = 1_000_000
M_OOD      = 10_000
N_OOD_CFGS = 100

ATK_RATIO   = 4.0
ATK_N_TRAIN = 5     # attack iterations per mini-batch during training
ATK_N_EVAL  = 20    # attack iterations at evaluation (unused here; stored for reference)

# Optimal hyperparameters from He et al. (2025) Table 5 (Appendix E.3)
OPTIMAL_HPS: dict[str, dict[int, dict[str, float]]] = {
    "S": {
        5_000:   {"delta": 0.3,   "alpha_bal": 1.0},
        10_000:  {"delta": 0.1,   "alpha_bal": 10.0},
        20_000:  {"delta": 0.05,  "alpha_bal": 1.0},
        50_000:  {"delta": 0.03,  "alpha_bal": 0.0},
        100_000: {"delta": 0.01,  "alpha_bal": 0.0},
    },
    "SV": {
        5_000:   {"delta": 0.5,   "alpha_bal": 1.0},
        10_000:  {"delta": 1.0,   "alpha_bal": 10.0},
        20_000:  {"delta": 0.1,   "alpha_bal": 1.0},
        50_000:  {"delta": 0.03,  "alpha_bal": 0.0},
        100_000: {"delta": 0.005, "alpha_bal": 0.0},
    },
}

METHODS = ["clean", "S", "SV"]
COLORS  = {"clean": "steelblue", "S": "darkorange", "SV": "seagreen"}
LABELS  = {"clean": "Clean", "S": "S-Attack", "SV": "SV-Attack"}

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ADV_DIR     = RESULTS_DIR / "adv_nets"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _make_input(S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        torch.log(S[:, :-1]).unsqueeze(-1),
        V[:, :-1].unsqueeze(-1),
    ], dim=-1)


@torch.no_grad()
def compute_errors(
    net: nn.Module,
    S: torch.Tensor,
    V: torch.Tensor,
    VP: torch.Tensor,
    device: torch.device,
    batch_size: int = 10_000,
) -> torch.Tensor:
    """Returns (M,) CPU tensor of per-path hedging errors X = C_T − PnL."""
    net = net.to(device).eval()
    errors = []
    for i in range(0, S.shape[0], batch_size):
        S_b  = S[i : i + batch_size].to(device)
        V_b  = V[i : i + batch_size].to(device)
        VP_b = VP[i : i + batch_size].to(device)
        h    = net(_make_input(S_b, V_b))
        dS   = S_b[:, 1:] - S_b[:, :-1]
        dVP  = VP_b[:, 1:] - VP_b[:, :-1]
        PnL  = (h[:, :, 0] * dS + h[:, :, 1] * dVP).sum(1)
        X    = torch.clamp(S_b[:, -1] - K, min=0.0) - PnL
        errors.append(X.cpu())
    net.cpu()
    return torch.cat(errors)


def empirical_es(errors: torch.Tensor, alpha: float = ALPHA_CVAR) -> float:
    """ES_alpha = mean of the top (1-alpha) fraction of losses."""
    k = max(1, int(math.ceil((1.0 - alpha) * len(errors))))
    return float(torch.topk(errors, k).values.mean())


# ---------------------------------------------------------------------------
# Network persistence
# ---------------------------------------------------------------------------

def _net_path(N: int, method: str, s: int) -> Path:
    return ADV_DIR / f"N{N}_{method}_s{s}_network.pt"


def _p0_path(N: int, method: str, s: int) -> Path:
    return ADV_DIR / f"N{N}_{method}_s{s}_p0.pt"


def save_net(net: HestonHedgeNet, p0: float, N: int, method: str, s: int) -> None:
    ADV_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), _net_path(N, method, s))
    torch.save(torch.tensor(p0), _p0_path(N, method, s))


def load_net(N: int, method: str, s: int) -> tuple[HestonHedgeNet, float] | None:
    np_ = _net_path(N, method, s)
    pp_ = _p0_path(N, method, s)
    if not (np_.exists() and pp_.exists()):
        return None
    net = HestonHedgeNet(N=N_STEPS, width=20)
    net.load_state_dict(torch.load(np_, map_location="cpu", weights_only=True))
    p0  = float(torch.load(pp_, map_location="cpu", weights_only=True))
    return net, p0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_panel(ax: plt.Axes, results: dict, n_sizes: list[int]) -> None:
    for method in METHODS:
        means, mins_, maxs_ = [], [], []
        for N in n_sizes:
            n_runs = min(M_TRAIN // N, N_SEEDS)
            vals   = [results[(N, method, s)] for s in range(n_runs)
                      if (N, method, s) in results]
            if vals:
                means.append(float(np.mean(vals)))
                mins_.append(float(np.min(vals)))
                maxs_.append(float(np.max(vals)))
            else:
                means.append(float("nan"))
                mins_.append(float("nan"))
                maxs_.append(float("nan"))

        ax.plot(n_sizes, means, color=COLORS[method], label=LABELS[method],
                linewidth=2.0, marker="o", markersize=4)
        ax.fill_between(n_sizes, mins_, maxs_, alpha=0.20, color=COLORS[method])

    ax.set_xscale("log")
    ax.set_xlabel("Training Samples N", fontsize=12)
    ax.set_ylabel(r"Hedging Loss (ES$_{0.5}$)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_figure1(
    results_oos: dict,
    results_ood: dict,
    n_sizes: list[int],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_panel(axes[0], results_oos, n_sizes)
    axes[0].set_title("Out-of-sample performance", fontsize=13, fontweight="bold")
    _plot_panel(axes[1], results_ood, n_sizes)
    axes[1].set_title("Out-of-distribution performance", fontsize=13, fontweight="bold")
    fig.suptitle(
        "Heston Adversarial Training — Sections 5.2–5.3\n"
        "He, Sutter & Gonon (2025) — Shaded: min-max range across seeds",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(resume: bool = False) -> None:
    device = _auto_device()
    print(f"Device: {device}")

    # ── Simulate training paths ──────────────────────────────────────────────
    print("\nSimulating training paths (100K, seed=19) …")
    t0 = time.perf_counter()
    base_params = HestonParams(
        S0=S0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        T=T, N=N_STEPS, M=M_TRAIN,
    )
    S_tr, V_tr, VP_tr = HestonSimulator(base_params).simulate(seed=19)
    VP_tr = VP_tr * VP_SCALE
    print(f"  Done in {time.perf_counter()-t0:.1f}s  shape={S_tr.shape}")

    # ── Compute p0 warm-start ────────────────────────────────────────────────
    with torch.no_grad():
        C_T_tr = torch.clamp(S_tr[:, -1] - K, min=0.0)
        p0_init = float(C_T_tr.quantile(ALPHA_CVAR))
    print(f"  p0_init = VaR_{ALPHA_CVAR} = {p0_init:.4f}")

    # ── Simulate test paths (OOS) ────────────────────────────────────────────
    print("\nSimulating OOS test paths (1M, seed=42) …")
    t0 = time.perf_counter()
    test_params = HestonParams(
        S0=S0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        T=T, N=N_STEPS, M=M_TEST,
    )
    S_te, V_te, VP_te = HestonSimulator(test_params).simulate(seed=42)
    VP_te = VP_te * VP_SCALE
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # Move train tensors to device
    S_tr_dev  = S_tr.to(device)
    V_tr_dev  = V_tr.to(device)
    VP_tr_dev = VP_tr.to(device)

    # ── Shared loss function ─────────────────────────────────────────────────
    loss_fn = HestonCVaRLoss(K=K, alpha=ALPHA_CVAR)

    # ── Main experiment loop ─────────────────────────────────────────────────
    results_oos: dict[tuple, float] = {}
    stored_nets: dict[tuple, HestonHedgeNet] = {}

    total_runs = sum(
        min(M_TRAIN // N, N_SEEDS) for N in N_SIZES
    ) * len(METHODS)
    run_idx = 0

    print(f"\nStarting {total_runs} training runs …\n")
    outer = tqdm(total=total_runs, desc="total", unit="run",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]  {postfix}")

    for N in N_SIZES:
        n_seeds_for_N = min(M_TRAIN // N, N_SEEDS)
        for s in range(n_seeds_for_N):
            S_n  = S_tr_dev[s * N : (s + 1) * N]
            V_n  = V_tr_dev[s * N : (s + 1) * N]
            VP_n = VP_tr_dev[s * N : (s + 1) * N]

            for method in METHODS:
                run_idx += 1
                label = f"N={N} {method} s{s}"
                outer.set_postfix_str(label)

                # ── Resume: load existing network if available ───────────────
                if resume:
                    cached = load_net(N, method, s)
                    if cached is not None:
                        net, p0 = cached
                        tqdm.write(f"[{run_idx:2d}/{total_runs}] {label:25s}  RESUMED from disk")
                        errors = compute_errors(net, S_te, V_te, VP_te, device)
                        results_oos[(N, method, s)] = empirical_es(errors)
                        stored_nets[(N, method, s)] = net
                        outer.update(1)
                        continue

                # ── Build attack_fn for adversarial methods ──────────────────
                attack_fn = None
                alpha_bal = 1.0
                if method != "clean":
                    hp = OPTIMAL_HPS[method][N]
                    delta_run   = hp["delta"]
                    alpha_bal   = hp["alpha_bal"]
                    _atk_kwargs = dict(
                        K=K,
                        alpha_cvar=ALPHA_CVAR,
                        kappa=kappa,
                        theta=theta,
                        T=T,
                        N_steps=N_STEPS,
                        VP_scale=VP_SCALE,
                        delta=delta_run,
                        ratio=ATK_RATIO,
                        iters=ATK_N_TRAIN,
                    )
                    _base_attack = s_budget_attack if method == "S" else sv_budget_attack
                    attack_fn = partial(_base_attack, **_atk_kwargs)

                # ── Train ───────────────────────────────────────────────────
                t0 = time.perf_counter()
                net, p0 = train_adv_heston(
                    S_n, V_n, VP_n,
                    attack_fn=attack_fn,
                    loss_fn=loss_fn,
                    n_clean=N_EPOCHS_CLEAN,
                    n_adv=N_EPOCHS_ADV,   # same for all — attack_fn=None skips adv phase
                    batch_size=min(BATCH_SIZE, N),
                    lr=LR,
                    alpha_bal=alpha_bal,
                    atk_ratio=ATK_RATIO,
                    atk_n=ATK_N_TRAIN,
                    p0_init=p0_init,
                    device=device,
                    desc=label,
                )
                elapsed = time.perf_counter() - t0

                # ── Save network immediately ────────────────────────────────
                save_net(net, p0, N, method, s)

                # ── OOS evaluation ──────────────────────────────────────────
                errors = compute_errors(net, S_te, V_te, VP_te, device)
                es_val = empirical_es(errors)
                results_oos[(N, method, s)] = es_val
                stored_nets[(N, method, s)] = net

                outer.update(1)
                tqdm.write(
                    f"[{run_idx:2d}/{total_runs}] {label:25s}  "
                    f"OOS ES={es_val:.4f}  p0={p0:.4f}  ({elapsed:.0f}s)"
                )

    outer.close()

    # ── OOD evaluation ───────────────────────────────────────────────────────
    print("\nRunning OOD evaluation …")
    np.random.seed(0)
    ood_sim = OutOfSampleHestonSimulator(base_params, variation=0.1)
    results_ood: dict[tuple, float] = {}

    ood_tasks = [
        (N, method, s)
        for N in N_SIZES
        for s in range(min(M_TRAIN // N, N_SEEDS))
        for method in METHODS
        if (N, method, s) in stored_nets
    ]

    for key in tqdm(ood_tasks, desc="OOD", unit="net"):
        N, method, s = key
        net = stored_nets[key]
        ood_es_vals = []
        for _ in range(N_OOD_CFGS):
            S_o, V_o, VP_o, _ = ood_sim.simulate(M=M_OOD)
            VP_o = VP_o * VP_SCALE
            err  = compute_errors(net, S_o, V_o, VP_o, device)
            ood_es_vals.append(empirical_es(err))
        results_ood[key] = float(np.mean(ood_es_vals))

    # ── Save aggregate results ───────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "adv_heston_results.pt"
    # Convert tuple keys to strings for portability
    torch.save(
        {
            "oos": {str(k): v for k, v in results_oos.items()},
            "ood": {str(k): v for k, v in results_ood.items()},
            "N_SIZES": N_SIZES,
            "N_SEEDS": N_SEEDS,
            "ALPHA_CVAR": ALPHA_CVAR,
        },
        result_path,
    )
    print(f"\nResults saved → {result_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'':25s}  {'N=5K':>10s}  {'N=100K':>10s}")
    print("-" * 50)
    for label, res in [("OOS", results_oos), ("OOD", results_ood)]:
        for method in METHODS:
            def _mean(N: int) -> str:
                n_runs = min(M_TRAIN // N, N_SEEDS)
                vals = [res.get((N, method, s), float("nan")) for s in range(n_runs)]
                v = float(np.nanmean(vals))
                return f"{v:10.4f}"
            print(f"  {label} {LABELS[method]:20s}  {_mean(5_000)}  {_mean(100_000)}")
        print()

    # ── Plot Figure 1 ────────────────────────────────────────────────────────
    fig_path = RESULTS_DIR / "heston_figure1_adversarial.png"
    plot_figure1(results_oos, results_ood, N_SIZES, fig_path)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heston adversarial training (Sections 5.2–5.3)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip networks whose .pt files already exist in results/adv_nets/")
    args = parser.parse_args()
    main(resume=args.resume)
