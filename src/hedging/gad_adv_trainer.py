"""
Adversarial training for GAD (General Affine Diffusion) deep hedging.

Implements two attacks and a unified clean + adversarial training loop for
the single-asset entropic OCE hedger.

Attacks
-------
gad_budget_attack  (PRIMARY — He et al. 2025, budget_att algorithm)
    Per-sample non-negative budget b_i ≥ 0 with L2-ball constraint
    mean(b²) = δ².  Attack = b_i · sign_{i,t}.  Budgets are
    redistributed toward exploitable paths (more adversarial damage
    per unit of perturbation capacity than L∞).  p0 is recomputed
    analytically at each inner step ('calculate' mode in He et al.),
    detached from the gradient graph.

gad_linf_attack    (SECONDARY — simple L∞ box PGD)
    FGSM-style sign-gradient steps with clamp to [−δ, +δ].  Equivalent
    perturbation budget for every path and every time step.

Training loop
-------------
Two-stage scheme:
  Stage 1 (epochs 0 .. n_clean−1):       clean OCE loss only.
  Stage 2 (epochs n_clean .. n_total−1):  alpha_bal·L_clean + L_adv.

During Stage 2 the adversarial forward pass is evaluated with the network
in eval() mode (BN uses running statistics), matching He et al. (2025).

References
----------
He, Sutter & Gonon (2025) — budget_att / WBPGD S-Attack
Madry et al. (2018)       — PGD for adversarial robustness
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
# Budget attack — He et al. (2025) primary algorithm
# ---------------------------------------------------------------------------

def gad_budget_attack(
    network: nn.Module,
    S: torch.Tensor,            # (batch, N+1)
    loss_fn: EntropicOCELoss,
    p0: torch.Tensor,           # IGNORED — attack recomputes p0 analytically
    *,
    delta: float,
    iters: int,
    beta_ratio: float = 4.0,    # step size α = (beta_ratio / iters) * delta
    q: float = 2.0,             # norm exponent for budget gradient step
) -> torch.Tensor:
    """
    Budget attack — He et al. (2025), budget_att algorithm.

    Each sample i receives a non-negative budget b_i ≥ 0.  The perturbation
    applied to path i at time t is:
        att_{i,t} = b_i · sign_{i,t},   sign_{i,t} ∈ [−1, 1]

    Constraint:  sqrt(mean(b²)) = δ  (L2-ball on the budget vector).
    This allows the attack to concentrate perturbation energy on the paths
    that are most exploitable, unlike the L∞ attack that treats all paths
    identically.

    At t=0 the perturbation is always zero (shared starting price).

    The p0 argument is IGNORED.  Instead p0 is recomputed analytically at
    every inner step (He et al. 'calculate' mode):
        p0* = log(E[exp(−λX)]) / λ + log(λ) / λ
    and treated as a constant (detached via .item()) so no gradient flows
    through it.

    Network is in eval() mode during the attack (BN uses running stats).

    Args:
        network:    HedgeNet (training mode is temporarily switched to eval).
        S:          Price paths (batch, N+1) on the correct device.
        loss_fn:    EntropicOCELoss instance — supplies K, lamb, X_max, x_max_val.
        p0:         Unused.  Kept for API compatibility with gad_linf_attack.
        delta:      Attack budget δ (L2-norm constraint on the budget vector).
        iters:      Number of inner optimisation steps.
        beta_ratio: Step-size multiplier; α = (beta_ratio / iters) * delta.
        q:          Norm exponent for the budget gradient step (default 2).

    Returns:
        S_att: Adversarially perturbed paths (batch, N+1), detached.
    """
    if delta == 0.0:
        return S.clone().detach()

    alpha = beta_ratio * delta / iters
    lamb  = loss_fn.lamb
    K     = loss_fn.K

    was_training = network.training
    network.eval()

    batch = S.shape[0]

    # --- initialise attack variables ----------------------------------------
    budget = torch.full((batch,), delta, device=S.device)
    budget.requires_grad_(True)

    att_sign = torch.ones_like(S)   # all +1 initially
    att_sign[:, 0] = 0.0            # never perturb t=0
    att_sign.requires_grad_(True)

    att_best  = (budget.detach().unsqueeze(1) * att_sign.detach()).clone()
    loss_best = float("-inf")
    sign_prev = att_sign.detach().clone()   # for momentum tracking

    for _ in range(iters):
        S_att   = S + budget.unsqueeze(1) * att_sign
        holding = network(_make_input(S_att))                   # (batch, N)
        PnL     = (holding * (S_att[:, 1:] - S_att[:, :-1])).sum(1)
        C_T     = torch.clamp(S_att[:, -1] - K, min=0.0)
        X       = PnL - C_T
        if loss_fn.X_max:
            X = torch.clamp(X, min=loss_fn.x_max_val)

        # Analytical optimal p0 — detached from graph (He et al. eq. 'calculate')
        p0_val = float(
            (torch.exp(-lamb * X).mean().log() / lamb).item()
            + math.log(lamb) / lamb
        )
        x    = X + p0_val
        loss = torch.exp(-lamb * x).mean() + p0_val - (1.0 + math.log(lamb)) / lamb

        if loss.item() > loss_best:
            loss_best = loss.item()
            att_best  = (budget.detach().unsqueeze(1) * att_sign.detach()).clone()

        grad_b = torch.autograd.grad(loss, budget,   retain_graph=True)[0]
        grad_a = torch.autograd.grad(loss, att_sign)[0]

        with torch.no_grad():
            # q-norm budget ascent then project onto L2 ball: sqrt(mean(b²)) = δ
            step    = alpha * grad_b.pow(q - 1) * (
                (grad_b.pow(q).mean() + 1e-10).pow(1.0 / q - 1.0)
            )
            bud_new = torch.clamp(budget + step, min=0.0)
            bud_new = bud_new / (bud_new.square().mean().sqrt() + 1e-10) * delta
            budget.copy_(bud_new)

            # Sign update with momentum (He et al. coefficients 0.3 + 0.25)
            sign_curr = att_sign.clone()
            sign_new  = (
                att_sign
                + 0.4 * 0.75 * grad_a.sign()
                + 0.25 * (att_sign - sign_prev)
            ).clamp(-1.0, 1.0)
            sign_new[:, 0] = 0.0
            sign_prev = sign_curr               # save before overwrite
            att_sign.copy_(sign_new)

    # --- final check --------------------------------------------------------
    S_att_f  = S + budget.detach().unsqueeze(1) * att_sign.detach()
    hold_f   = network(_make_input(S_att_f))
    PnL_f    = (hold_f * (S_att_f[:, 1:] - S_att_f[:, :-1])).sum(1)
    C_T_f    = torch.clamp(S_att_f[:, -1] - K, min=0.0)
    X_f      = PnL_f - C_T_f
    if loss_fn.X_max:
        X_f = torch.clamp(X_f, min=loss_fn.x_max_val)
    p0_f  = float(
        (torch.exp(-lamb * X_f).mean().log() / lamb).item() + math.log(lamb) / lamb
    )
    x_f   = X_f + p0_f
    loss_f = torch.exp(-lamb * x_f).mean() + p0_f - (1.0 + math.log(lamb)) / lamb
    if loss_f.item() > loss_best:
        att_best = (budget.detach().unsqueeze(1) * att_sign.detach()).clone()

    if was_training:
        network.train()

    return (S + att_best).detach()


# ---------------------------------------------------------------------------
# L∞ WBPGD attack (secondary — simpler, equal per-path budget)
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

    Unlike gad_budget_attack, this imposes the same budget on every path.
    p0 is used as-is (passed in from the training loop — no analytical
    recomputation).

    Network weights are NOT updated; torch.autograd.grad is used so that
    .backward() is never called on the network parameters.

    Args:
        network:    HedgeNet (in any mode — this function temporarily switches
                    it to eval() so BN uses running statistics, then restores).
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

    was_training = network.training
    network.eval()

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

    if was_training:
        network.train()

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
) -> tuple[HedgeNet, float, dict]:
    """
    Two-stage adversarial training for GAD deep hedging.

    Stage 1 (epochs 0 .. n_clean−1):
        All methods use clean entropic OCE loss only.

    Stage 2 (epochs n_clean .. n_clean+n_adv−1):
        attack_fn=None     → continue clean training (Clean baseline)
        attack_fn=...      → L = alpha_bal·L_clean + L_adversarial

    During Stage 2 the adversarial forward pass runs with the network in
    eval() mode (BatchNorm uses running statistics), matching He et al. (2025).
    The clean forward pass always uses train() mode.

    Network input: log(S_t) for t = 0..N−1, shape (batch, N, 1).

    LR schedule uses proportional breakpoints (same fractions as adv_trainer.py):
        [0,   28.6%): lr × 1.0
        [28.6%, 71.4%): lr × 0.1
        [71.4%, 85.7%): lr × 0.01
        [85.7%, 100%): lr × 0.001

    For n_total = 300 these approximate He et al.'s fixed 100/200/250 breakpoints.

    Args:
        S:          Training price paths (M, N+1) on device.
        attack_fn:  Callable (network, S_batch, p0_adv) → S_att, or None.
                    Use functools.partial(gad_budget_attack, loss_fn=...,
                    delta=..., iters=...) for the He et al. primary attack.
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
        net:     Trained HedgeNet on CPU.
        p0:      Learned clean-phase OCE dual variable (float, CPU).
        history: Dict with keys 'loss_clean', 'loss_adv', 'p0' — one float
                 per epoch.  loss_adv is 0.0 in clean-only epochs.
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
    history: dict[str, list[float]] = {"loss_clean": [], "loss_adv": [], "p0": []}

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

        # Clean forward pass — train() mode so BN accumulates batch stats
        net.train()
        idx  = torch.randperm(M, device=device)[:batch_size]
        S_b  = S[idx]

        x_c    = _make_input(S_b)
        h_c    = net(x_c)
        loss_c = loss_fn(h_c, S_b, p0_clean)

        if adv_phase:
            # Attack uses eval() internally (BN uses running stats)
            S_att  = attack_fn(net, S_b, p0=p0_adv.detach())
            # Adversarial forward also in eval() mode — matches He et al. (2025)
            net.eval()
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

        lc = loss_c.item()
        la = loss_a.item()
        p0_val = p0_clean.item()
        pbar.postfix[0] = lc
        pbar.postfix[1] = la
        pbar.postfix[2] = p0_val
        history["loss_clean"].append(lc)
        history["loss_adv"].append(la)
        history["p0"].append(p0_val)

    return net.cpu(), float(p0_clean.item()), history


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_bn_stats(
    net: nn.Module,
    S: torch.Tensor,
    device: torch.device,
    batch_size: int = 10_000,
) -> nn.Module:
    """
    Recalibrate BatchNorm running statistics to a new price distribution.

    Network weights are NOT updated. Only BN running_mean and running_var
    are recomputed from the provided price paths. This corrects the
    distributional shift between training paths (GAD, low-vol SPY-calibrated)
    and test paths (real S&P 500 individual stocks, wider vol range).

    Mechanism: reset all BN running stats, then run the test paths through
    the network in train() mode. BN accumulates fresh running stats via its
    exponential moving average. After all batches are processed, switch
    back to eval() mode — subsequent inference uses the updated stats.

    Call this once before compute_errors() when the test distribution differs
    from the training distribution (e.g., evaluating on high-vol stocks).

    Args:
        net:        Trained HedgeNet (on any device).
        S:          Reference price paths to calibrate from, shape (M, N+1).
                    Typically the full real test set.
        device:     Compute device.
        batch_size: Mini-batch size for the calibration forward passes.

    Returns:
        net in eval() mode with updated BN running stats.
    """
    net = net.to(device)

    # Reset running stats, then switch to cumulative-average mode (momentum=None)
    # so every forward pass contributes equally regardless of how many batches.
    # This gives exact population statistics in one pass rather than relying on
    # the 0.3 EMA momentum to converge over many batches.
    saved_momentum: dict = {}
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            saved_momentum[id(m)] = (m, m.momentum)
            m.momentum = None   # cumulative mean/var mode

    net.train()
    for i in range(0, S.shape[0], batch_size):
        S_b = S[i : i + batch_size].to(device)
        net(_make_input(S_b))

    # Restore original momentum values before returning
    for _, (m, orig) in saved_momentum.items():
        m.momentum = orig

    net.eval()
    return net


@torch.no_grad()
def compute_errors(
    net: nn.Module,
    S: torch.Tensor,            # (M, N+1) price paths
    K: float,
    device: torch.device,
    batch_size: int = 10_000,
    delta_clip: tuple[float, float] | None = (0.0, 1.0),
) -> torch.Tensor:
    """
    Compute per-path hedging errors X = C_T − PnL for a trained network.

    C_T = max(S_T − K, 0)  (European call payoff)
    PnL = Σ_t h_t · (S_{t+1} − S_t)

    Args:
        delta_clip: If set, clamp hedge ratios to (min, max) before computing
                    PnL. Default (0.0, 1.0) enforces the no-arbitrage bound
                    for a European call — the optimal delta is in [0, 1] by
                    the N(d₁) formula. Set to None to use raw network outputs
                    (may produce extreme values for out-of-distribution inputs).

    Returns:
        (M,) CPU tensor of hedging errors. Positive = net loss.
    """
    net = net.to(device).eval()
    errors = []
    for i in range(0, S.shape[0], batch_size):
        S_b = S[i : i + batch_size].to(device)
        h   = net(_make_input(S_b))                     # (batch, N)
        if delta_clip is not None:
            h = h.clamp(delta_clip[0], delta_clip[1])
        PnL = (h * (S_b[:, 1:] - S_b[:, :-1])).sum(1)
        C_T = torch.clamp(S_b[:, -1] - K, min=0.0)
        errors.append((C_T - PnL).cpu())
    net.cpu()
    return torch.cat(errors)
