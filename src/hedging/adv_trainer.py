"""
Adversarial training primitives for Heston deep hedging.

Implements the S-Attack and SV-Attack budget attacks from He, Sutter & Gonon (2025)
plus a unified training loop that handles both clean and adversarial phases.

Key design choices:
  - SV-Attack perturbs V and rebuilds VarPrice via the Heston closed-form (differentiable),
    rather than perturbing VarPrice directly. This preserves the no-arbitrage relationship
    between V and VarPrice and matches the paper's description.
  - heston_var_price is vectorized and fully differentiable in V, so gradients flow through
    the attack loop's VP reconstruction into the budget/sign updates.
  - LR schedule uses proportional breakpoints so the same function works for any epoch count.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.hedging.hedge_network import HestonHedgeNet
from src.hedging.loss import HestonCVaRLoss


# ---------------------------------------------------------------------------
# Closed-form variance swap price (differentiable in V)
# ---------------------------------------------------------------------------

def heston_var_price(
    V: torch.Tensor,
    kappa: float,
    theta: float,
    T: float,
    N_steps: int,
    VP_scale: float = 100.0,
) -> torch.Tensor:
    """
    Heston closed-form variance swap fair value at each time step.

    E_t[∫_t^T V_s ds] computed analytically for all t = 0, …, N simultaneously.
    Fully differentiable in V; used inside SV-attack to rebuild VarPrice from V_att.

    Args:
        V:        Variance paths (batch, N+1).
        kappa:    Mean-reversion rate.
        theta:    Long-run variance.
        T:        Maturity (years).
        N_steps:  Number of time steps (= N).
        VP_scale: Multiplicative scaling applied to raw VarPrice.

    Returns:
        VarPrice (batch, N+1) scaled by VP_scale.
    """
    dt = T / N_steps
    batch, n_plus_1 = V.shape
    VP = torch.zeros(batch, n_plus_1, device=V.device, dtype=V.dtype)

    # t=0: full horizon
    VP[:, 0] = (V[:, 0] - theta) / kappa * (1.0 - math.exp(-kappa * T)) + theta * T

    # t=1..N: trapezoidal integral of realised variance + forward expectation
    var_int = 0.5 * dt * (V[:, :-1] + V[:, 1:]).cumsum(dim=1)   # (batch, N)
    time_idx   = torch.arange(1, N_steps + 1, device=V.device, dtype=V.dtype)
    tau_t      = T - time_idx * dt                                # remaining time (N,)
    correction = (
        (V[:, 1:] - theta) / kappa * (1.0 - torch.exp(-kappa * tau_t))
        + theta * tau_t
    )                                                              # (batch, N)
    VP[:, 1:] = var_int + correction

    return VP * VP_scale


# ---------------------------------------------------------------------------
# Helper: attack loss (CVaR with analytic p0 = quantile)
# ---------------------------------------------------------------------------

def _attack_loss(
    holding: torch.Tensor,
    S: torch.Tensor,
    VP: torch.Tensor,
    K: float,
    alpha_cvar: float,
) -> torch.Tensor:
    """CVaR loss with p0 set analytically as the alpha-quantile of X."""
    dS  = S[:, 1:]  - S[:, :-1]
    dVP = VP[:, 1:] - VP[:, :-1]
    PnL = (holding[:, :, 0] * dS + holding[:, :, 1] * dVP).sum(1)
    X   = torch.clamp(S[:, -1] - K, min=0.0) - PnL
    p0  = X.quantile(alpha_cvar).item()
    return torch.clamp(X - p0, min=0.0).mean() / (1.0 - alpha_cvar) + p0


def _make_input(S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """(batch, N+1) x 2 → (batch, N, 2) = [log S_t, V_t] for t = 0..N-1."""
    return torch.cat([
        torch.log(S[:, :-1]).unsqueeze(-1),
        V[:, :-1].unsqueeze(-1),
    ], dim=-1)


# ---------------------------------------------------------------------------
# S-only budget attack
# ---------------------------------------------------------------------------

def s_budget_attack(
    network: nn.Module,
    S: torch.Tensor,
    V: torch.Tensor,
    *,
    K: float,
    alpha_cvar: float,
    kappa: float,
    theta: float,
    T: float,
    N_steps: int,
    VP_scale: float,
    delta: float,
    ratio: float,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    S-only budget attack: perturbs stock price S, keeps V fixed.

    VarPrice is rebuilt from the unperturbed V via heston_var_price, so the
    variance swap hedge sees the same VP throughout. This isolates the effect
    of stock price perturbations on the hedging loss.

    Returns:
        S_att:  Adversarial stock prices (batch, N+1), gradients detached.
        V:      Unchanged variance paths (same tensor).
        VP_att: VarPrice from unperturbed V (batch, N+1).
    """
    if delta == 0.0:
        VP = heston_var_price(V, kappa, theta, T, N_steps, VP_scale)
        return S.clone(), V, VP

    alpha_step = delta * ratio / iters
    VP_fixed   = heston_var_price(V, kappa, theta, T, N_steps, VP_scale).detach()

    budget    = torch.ones(S.shape[0], 2, device=S.device) * delta
    att_sign  = torch.ones_like(S)
    att_sign[:, 0] = 0.0          # never perturb t=0
    budget.requires_grad_(True)
    att_sign.requires_grad_(True)

    att_best    = (budget[:, 0].unsqueeze(1) * att_sign).detach().clone()
    result_best = 0.0

    for _ in range(iters):
        S_att_tmp = S + budget[:, 0].unsqueeze(1) * att_sign
        holding   = network(_make_input(S_att_tmp, V))
        loss      = _attack_loss(holding, S_att_tmp, VP_fixed, K, alpha_cvar)

        perf = loss.item()
        if perf > result_best:
            att_best    = (budget[:, 0].unsqueeze(1) * att_sign).detach().clone()
            result_best = perf

        grad_b, grad_a = torch.autograd.grad(loss, [budget, att_sign])

        with torch.no_grad():
            b_new = budget + alpha_step * grad_b / (grad_b.pow(2).mean() + 1e-10).sqrt()
            b_new = b_new / (b_new.square().mean().sqrt() + 1e-10) * delta / math.sqrt(2)
            b_new = b_new.clamp(min=0.0)
            budget.copy_(b_new)

            a_new = grad_a.sign()       # direct replacement, no momentum (matches reference)
            a_new[:, 0] = 0.0
            att_sign.copy_(a_new)

    # Final check
    S_att_tmp = S + budget[:, 0].unsqueeze(1) * att_sign
    holding   = network(_make_input(S_att_tmp, V))
    loss      = _attack_loss(holding, S_att_tmp, VP_fixed, K, alpha_cvar)
    if loss.item() > result_best:
        att_best = (budget[:, 0].unsqueeze(1) * att_sign).detach().clone()

    S_att = (S + att_best).detach()
    return S_att, V.clone(), VP_fixed


# ---------------------------------------------------------------------------
# Joint SV budget attack
# ---------------------------------------------------------------------------

def sv_budget_attack(
    network: nn.Module,
    S: torch.Tensor,
    V: torch.Tensor,
    *,
    K: float,
    alpha_cvar: float,
    kappa: float,
    theta: float,
    T: float,
    N_steps: int,
    VP_scale: float,
    delta: float,
    ratio: float,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint SV budget attack: perturbs S and V simultaneously.

    VarPrice is rebuilt from V_att via heston_var_price at every iteration,
    so the attack jointly optimises over the full market state (S, V, VP).

    V perturbations are scaled by 1/100 to match the magnitude of S perturbations
    (V ≈ 0.04, S ≈ 100, so unit budget has comparable effect on both).

    Returns:
        S_att:  Adversarial stock prices (batch, N+1).
        V_att:  Adversarial variance paths (batch, N+1).
        VP_att: VarPrice rebuilt from V_att (batch, N+1).
    """
    if delta == 0.0:
        VP = heston_var_price(V, kappa, theta, T, N_steps, VP_scale)
        return S.clone(), V.clone(), VP

    alpha_step = delta * ratio / iters

    # budget: (batch, 2) — budget[:, 0] for S, budget[:, 1] for V
    budget   = torch.ones(S.shape[0], 2, device=S.device) * delta
    att_sign = torch.ones(S.shape[0], S.shape[1], 2, device=S.device)
    att_sign[:, 0, :] = 0.0       # never perturb t=0
    budget.requires_grad_(True)
    att_sign.requires_grad_(True)

    att_best    = (budget.unsqueeze(1) * att_sign).detach().clone()
    result_best = 0.0

    for _ in range(iters):
        S_att_tmp = S + budget[:, 0].unsqueeze(1) * att_sign[:, :, 0]
        V_att_tmp = V + budget[:, 1].unsqueeze(1) * att_sign[:, :, 1] / 100.0
        VP_att_tmp = heston_var_price(V_att_tmp, kappa, theta, T, N_steps, VP_scale)
        holding   = network(_make_input(S_att_tmp, V_att_tmp))
        loss      = _attack_loss(holding, S_att_tmp, VP_att_tmp, K, alpha_cvar)

        perf = loss.item()
        if perf > result_best:
            att_best    = (budget.unsqueeze(1) * att_sign).detach().clone()
            result_best = perf

        grad_b, grad_a = torch.autograd.grad(loss, [budget, att_sign])

        with torch.no_grad():
            b_new = budget + alpha_step * grad_b / (grad_b.pow(2).mean() + 1e-10).sqrt()
            b_new = b_new / (b_new.square().mean().sqrt() + 1e-10) * delta / math.sqrt(2)
            b_new = b_new.clamp(min=0.0)
            budget.copy_(b_new)

            # Momentum term (att_sign - att_sign_old) is always 0 in reference because
            # att_sign_old is a reference (not clone) and copy_() updates both in-place.
            # Simplified to just the gradient step.
            a_new = (att_sign + 0.75 * grad_a.sign()).clamp(-1.0, 1.0)
            a_new[:, 0, :] = 0.0
            att_sign.copy_(a_new)

    # Final check
    S_att_tmp  = S + budget[:, 0].unsqueeze(1) * att_sign[:, :, 0]
    V_att_tmp  = V + budget[:, 1].unsqueeze(1) * att_sign[:, :, 1] / 100.0
    VP_att_tmp = heston_var_price(V_att_tmp, kappa, theta, T, N_steps, VP_scale)
    holding    = network(_make_input(S_att_tmp, V_att_tmp))
    loss       = _attack_loss(holding, S_att_tmp, VP_att_tmp, K, alpha_cvar)
    if loss.item() > result_best:
        att_best = (budget.unsqueeze(1) * att_sign).detach().clone()

    S_att  = (S + att_best[:, :, 0]).detach()
    V_att  = (V + att_best[:, :, 1] / 100.0).detach()
    VP_att = heston_var_price(V_att, kappa, theta, T, N_steps, VP_scale).detach()
    return S_att, V_att, VP_att


# ---------------------------------------------------------------------------
# Unified adversarial training loop
# ---------------------------------------------------------------------------

def train_adv_heston(
    S: torch.Tensor,
    V: torch.Tensor,
    VP: torch.Tensor,
    attack_fn: Callable | None,
    *,
    loss_fn: HestonCVaRLoss,
    n_clean: int,
    n_adv: int,
    batch_size: int,
    lr: float,
    alpha_bal: float,
    atk_ratio: float,
    atk_n: int,
    p0_init: float,
    device: torch.device,
    desc: str = "",
) -> tuple[HestonHedgeNet, float]:
    """
    Two-stage adversarial training for Heston deep hedging.

    Stage 1 (epochs 0 .. n_clean-1):
        All methods use clean CVaR loss only.

    Stage 2 (epochs n_clean .. n_clean+n_adv-1):
        - attack_fn=None → continue clean training (Clean baseline)
        - attack_fn=s_budget_attack → L = alpha_bal*L_clean + L_S_adv
        - attack_fn=sv_budget_attack → L = alpha_bal*L_clean + L_SV_adv

    LR schedule uses proportional breakpoints so the schedule is the same
    regardless of total epoch count:
        [0,   28.6%) : lr × 1.0
        [28.6%, 71.4%): lr × 0.1
        [71.4%, 85.7%): lr × 0.01
        [85.7%, 100%): lr × 0.001

    Args:
        S, V, VP:    Training paths (N_train, N+1) on device. VP must already
                     be scaled by VP_scale before calling this function.
        attack_fn:   Callable (network, S_b, V_b) → (S_att, V_att, VP_att),
                     or None for clean training throughout.
        loss_fn:     HestonCVaRLoss instance (shared, not mutated).
        n_clean:     Clean pre-training epochs.
        n_adv:       Adversarial fine-tuning epochs (0 if clean method).
        batch_size:  Paths per mini-batch.
        lr:          Initial Adam learning rate.
        alpha_bal:   Weight of clean loss in adversarial phase (0 = pure adv).
        atk_ratio:   Step-size factor; alpha = delta * ratio / iters.
        atk_n:       Attack iterations per mini-batch during training.
        p0_init:     Initial VaR threshold (warm-start from training data quantile).
        device:      Compute device.
        desc:        Label shown in tqdm progress bar.

    Returns:
        net:   Trained HestonHedgeNet on CPU.
        p0:    Learned clean-phase VaR threshold (float, CPU).
    """
    N_steps = S.shape[1] - 1
    n_total = n_clean + n_adv
    t1 = int(0.286 * n_total)
    t2 = int(0.714 * n_total)
    t3 = int(0.857 * n_total)

    net      = HestonHedgeNet(N=N_steps, width=20).to(device)
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
        desc=desc or "train",
        leave=False,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            "  lc={postfix[0]:.4f}  la={postfix[1]:.4f}  p0={postfix[2]:.3f}"
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
        V_b  = V[idx]
        VP_b = VP[idx]

        x_c    = _make_input(S_b, V_b)
        h_c    = net(x_c)
        loss_c = loss_fn(h_c, S_b, VP_b, p0_clean)

        if adv_phase:
            # Network stays in train mode during the attack so BatchNorm uses
            # current-batch statistics (not stale running stats from clean phase).
            # Attack uses torch.autograd.grad(loss, [budget, att_sign]) — not
            # backward() — so network weights are never updated during the attack.
            S_att, V_att, VP_att = attack_fn(net, S_b, V_b)

            x_a    = _make_input(S_att, V_att)
            h_a    = net(x_a)
            loss_a = loss_fn(h_a, S_att, VP_att, p0_adv)
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
