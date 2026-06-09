"""
Adversarial training for GAD (General Affine Diffusion) deep hedging.

Implements the L∞ WBPGD attack and a unified clean + adversarial training
loop for the single-asset entropic OCE hedger.

Design follows adv_trainer.py (Heston two-asset) but simplified for the
univariate GAD case:
  - No variance channel (V) or variance swap (VarPrice) — price paths only
  - EntropicOCELoss instead of HestonCVaRLoss
  - L∞ path-distance constraint on perturbations (box projection)
  - PGD with sign-gradient steps (FGSM-style inner loop)

Attack reference:
  He, Sutter & Gonon (2025) — WBPGD S-Attack for the univariate model
  Madry et al. (2018) — PGD for adversarial robustness
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.hedging.hedge_network import HedgeNet
from src.hedging.loss import EntropicOCELoss


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def _make_input(S: torch.Tensor) -> torch.Tensor:
    """(batch, N+1) → (batch, N, 1) = log(S_t) for t = 0..N-1."""
    return torch.log(S[:, :-1]).unsqueeze(-1)


# ---------------------------------------------------------------------------
# L∞ WBPGD attack
# ---------------------------------------------------------------------------

def gad_linf_attack(
    network: nn.Module,
    S: torch.Tensor,            # (batch, N+1)
    loss_fn: EntropicOCELoss,
    p0: torch.Tensor,           # scalar OCE dual variable (detached)
    *,
    delta: float,
    iters: int,
    beta_ratio: float = 4.0,    # step size = (beta_ratio / iters) * delta
) -> torch.Tensor:
    """
    L∞ WBPGD attack on price paths.

    Finds adversarial paths Ŝ within the L∞ ball of radius δ:
        max_t |Ŝ_t − S_t| ≤ δ

    The perturbation at t=0 is always zero (shared starting price across
    all paths, matching the paper's construction).

    Uses sign-gradient PGD (FGSM-style):
        att_{i+1} = att_i + β · sign(∇_att loss)
        att       = clamp(att, −δ, +δ)
        att[:, 0] = 0

    Network weights are NOT updated; torch.autograd.grad is used so that
    .backward() is never called on the network parameters.

    Args:
        network:    Trained or in-training HedgeNet (must be in train mode
                    so BN uses current-batch statistics).
        S:          Price paths (batch, N+1) on the correct device.
        loss_fn:    EntropicOCELoss instance.
        p0:         Current OCE dual variable (scalar tensor, detached).
        delta:      L∞ ball radius.
        iters:      Number of PGD iterations.
        beta_ratio: Step-size multiplier; β = (beta_ratio / iters) * delta.

    Returns:
        S_att: Adversarially perturbed price paths (batch, N+1), detached.
    """
    if delta == 0.0:
        return S.clone().detach()

    beta = (beta_ratio / iters) * delta

    att = torch.zeros_like(S)   # perturbation initialised at zero
    att.requires_grad_(True)

    att_best    = att.detach().clone()
    loss_best   = float("-inf")

    for _ in range(iters):
        S_att_tmp = S + att
        holding   = network(_make_input(S_att_tmp))     # (batch, N)
        loss      = loss_fn(holding, S_att_tmp, p0)

        if loss.item() > loss_best:
            att_best  = att.detach().clone()
            loss_best = loss.item()

        (grad,) = torch.autograd.grad(loss, att)

        with torch.no_grad():
            att_new = att + beta * grad.sign()
            att_new = att_new.clamp(-delta, delta)
            att_new[:, 0] = 0.0                         # never perturb t=0
            att.copy_(att_new)

    # Final check
    S_att_tmp = S + att
    holding   = network(_make_input(S_att_tmp))
    loss      = loss_fn(holding, S_att_tmp, p0)
    if loss.item() > loss_best:
        att_best = att.detach().clone()

    return (S + att_best).detach()


# ---------------------------------------------------------------------------
# Unified training loop
# ---------------------------------------------------------------------------

def train_adv_gad(
    S: torch.Tensor,                # (M, N+1) training paths on device
    attack_fn: Callable | None,
    *,
    loss_fn: EntropicOCELoss,
    n_clean: int,
    n_adv: int,
    batch_size: int,
    lr: float,
    alpha_bal: float,               # weight of clean loss in adversarial phase
    p0_init: float,
    device: torch.device,
    width: int = 20,
    desc: str = "",
) -> tuple[HedgeNet, float]:
    """
    Two-stage adversarial training for GAD deep hedging.

    Stage 1 (epochs 0 .. n_clean−1):
        All methods use clean entropic OCE loss only.

    Stage 2 (epochs n_clean .. n_clean+n_adv−1):
        attack_fn=None     → continue clean training (Clean baseline)
        attack_fn=...      → L = alpha_bal·L_clean + L_adversarial

    Network input: log(S_t) for t = 0..N−1, shape (batch, N, 1).

    LR schedule uses proportional breakpoints (same fractions as adv_trainer.py):
        [0,   28.6%): lr × 1.0
        [28.6%, 71.4%): lr × 0.1
        [71.4%, 85.7%): lr × 0.01
        [85.7%, 100%): lr × 0.001

    Args:
        S:          Training price paths (M, N+1) on device.
        attack_fn:  Callable (network, S_batch, p0_adv) → S_att, or None.
        loss_fn:    EntropicOCELoss instance (shared, not mutated).
        n_clean:    Clean pre-training epochs.
        n_adv:      Adversarial fine-tuning epochs (0 for clean method).
        batch_size: Paths per mini-batch.
        lr:         Initial Adam learning rate.
        alpha_bal:  Clean-loss weight in adversarial phase.
        p0_init:    Initial OCE dual variable (warm-start from BS price).
        device:     Compute device.
        width:      Network hidden layer width (default 20).
        desc:       Label shown in tqdm progress bar.

    Returns:
        net:  Trained HedgeNet on CPU.
        p0:   Learned clean-phase OCE dual variable (float, CPU).
    """
    N_steps = S.shape[1] - 1
    n_total = n_clean + n_adv
    t1 = int(0.286 * n_total)
    t2 = int(0.714 * n_total)
    t3 = int(0.857 * n_total)

    net      = HedgeNet(N=N_steps, width=width).to(device)
    p0_clean = nn.Parameter(torch.tensor(p0_init, device=device))
    p0_adv   = nn.Parameter(torch.tensor(p0_init, device=device))

    opt = torch.optim.Adam(
        [{"params": net.parameters()}, {"params": [p0_clean, p0_adv]}],
        lr=lr,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda e: (
            1.0   if e < t1 else
            0.1   if e < t2 else
            0.01  if e < t3 else
            0.001
        ),
    )

    M = S.shape[0]
    pbar = tqdm(
        range(n_total),
        desc=desc or "train_gad",
        leave=False,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            "  lc={postfix[0]:.4f}  la={postfix[1]:.4f}  p0={postfix[2]:.4f}"
        ),
        postfix=[0.0, 0.0, p0_init],
    )

    for epoch in pbar:
        adv_phase = (attack_fn is not None) and (epoch >= n_clean)

        if adv_phase and epoch == n_clean:
            p0_adv.data.copy_(p0_clean.detach())

        net.train()
        idx  = torch.randperm(M, device=device)[:batch_size]
        S_b  = S[idx]

        # Clean loss
        x_c    = _make_input(S_b)
        h_c    = net(x_c)
        loss_c = loss_fn(h_c, S_b, p0_clean)

        if adv_phase:
            # Attack uses autograd.grad — network weights never updated here
            S_att  = attack_fn(net, S_b, p0=p0_adv.detach())
            x_a    = _make_input(S_att)
            h_a    = net(x_a)
            loss_a = loss_fn(h_a, S_att, p0_adv)
            total  = alpha_bal * loss_c + loss_a
        else:
            total  = loss_c
            loss_a = torch.tensor(0.0)

        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

        pbar.postfix[0] = loss_c.item()
        pbar.postfix[1] = loss_a.item()
        pbar.postfix[2] = p0_clean.item()

    return net.cpu(), float(p0_clean.item())


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_errors(
    net: nn.Module,
    S: torch.Tensor,            # (M, N+1) price paths
    K: float,
    device: torch.device,
    batch_size: int = 10_000,
) -> torch.Tensor:
    """
    Compute per-path hedging errors X = C_T − PnL for a trained network.

    C_T = max(S_T − K, 0)  (European call payoff)
    PnL = Σ_t h_t · (S_{t+1} − S_t)

    Returns (M,) CPU tensor of hedging errors. Positive = net loss.
    """
    net = net.to(device).eval()
    errors = []
    for i in range(0, S.shape[0], batch_size):
        S_b    = S[i : i + batch_size].to(device)
        h      = net(_make_input(S_b))                  # (batch, N)
        PnL    = (h * (S_b[:, 1:] - S_b[:, :-1])).sum(1)
        C_T    = torch.clamp(S_b[:, -1] - K, min=0.0)
        errors.append((C_T - PnL).cpu())
    net.cpu()
    return torch.cat(errors)
