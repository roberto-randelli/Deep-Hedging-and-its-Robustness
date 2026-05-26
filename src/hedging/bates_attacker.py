"""
Distributional adversarial attacks for Bates deep hedging — He, Sutter & Gonon (2025).

Implements Wasserstein-q (wp) and per-path budget attacks for the Bates model.
Mirrors the Heston methods of DHAttacker in attacker.py.

S and VarPrice are perturbed jointly; V (latent variance used as state observation
by the network) is held fixed throughout, consistent with He et al. Corollary 4.2.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn


@contextmanager
def _eval_mode(network: nn.Module) -> Generator[None, None, None]:
    """Temporarily switch network to eval mode; restore original mode on exit."""
    training = network.training
    network.eval()
    try:
        yield
    finally:
        network.train(training)


def _bates_input(S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """(batch, N+1) × 2 → (batch, N, 2) state input for BatesHedgeNet."""
    return torch.cat([
        torch.log(S[:, :-1]).unsqueeze(-1),
        V[:, :-1].unsqueeze(-1),
    ], dim=-1)


def _wp_step(
    att: torch.Tensor,
    grad: torch.Tensor,
    alpha: float,
    q: float,
) -> torch.Tensor:
    """Frank-Wolfe gradient ascent step for the Wasserstein-q ball."""
    grad_norm = grad.norm(p=1, dim=1, keepdim=True)
    return att + alpha * torch.sign(grad) * grad_norm.pow(q - 1) * (
        (grad_norm.pow(q).mean() + 1e-10).pow(1.0 / q - 1)
    )


def _wp_project(
    att: torch.Tensor,
    delta: float,
    q: float,
) -> torch.Tensor:
    """Project att (batch, D) onto the Wasserstein-q ball of radius delta."""
    dist = att.norm(p=float("inf"), dim=1, keepdim=True)
    p_conj = 1.0 / (1.0 - 1.0 / q)
    r_val = float(delta / (dist.pow(p_conj).mean().pow(1.0 / q).item() + 1e-10))
    r = min(1.0, r_val)
    return att.clamp(-r * dist, r * dist)


class BatesDHAttacker:
    """
    Distributional adversarial attacker for Bates hedging networks.

    Two attack algorithms:
        wp_attack      — Wasserstein-q distributional attack (Frank-Wolfe variant)
        budget_attack  — Per-path budget / direction decomposition attack

    Both perturb S and VarPrice jointly; V (latent variance) is held fixed.
    The S-only variant is available via the s_only flag on wp_attack.
    """

    def wp_attack(
        self,
        network: nn.Module,
        S: torch.Tensor,           # (batch, N+1)
        V: torch.Tensor,           # (batch, N+1) — latent variance, kept fixed
        VarPrice: torch.Tensor,    # (batch, N+1)
        loss_fn: nn.Module,        # BatesCVaRLoss
        p0: torch.Tensor,          # scalar CVaR threshold
        delta: float,
        ratio: float,
        n: int,
        q: float = 2.0,
        s_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wasserstein-q distributional attack on a Bates hedging network.

        Args:
            network:  Trained BatesHedgeNet (not mutated).
            S:        Stock price paths (batch, N+1).
            V:        Variance paths — fixed state observation.
            VarPrice: Variance swap fair-value paths (batch, N+1).
            loss_fn:  BatesCVaRLoss instance.
            p0:       CVaR threshold (scalar tensor).
            delta:    Wasserstein ball radius.
            ratio:    Step-size factor; alpha = delta * ratio / n.
            n:        Number of gradient-ascent iterations.
            q:        Wasserstein order (default 2).
            s_only:   If True, perturb only S (VarPrice channel forced to zero).

        Returns:
            S_att:    Adversarial stock price paths (batch, N+1).
            VP_att:   Adversarial variance swap price paths (batch, N+1).
            X_before: Hedging errors before attack (batch,).
            X_after:  Hedging errors after attack (batch,).
        """
        with _eval_mode(network):
            with torch.no_grad():
                holding = network(_bates_input(S, V))
                dS  = S[:, 1:] - S[:, :-1]
                dVP = VarPrice[:, 1:] - VarPrice[:, :-1]
                PnL = (holding[:, :, 0] * dS + holding[:, :, 1] * dVP).sum(1)
                X_before = loss_fn.terminal_payoff(S[:, -1]) - PnL

            if delta == 0.0:
                return S.clone(), VarPrice.clone(), X_before, X_before.clone()

            # Joint perturbation: (batch, N+1, 2) — [:,:,0]=ΔS, [:,:,1]=ΔVP
            att = torch.zeros(S.shape[0], S.shape[1], 2, device=S.device)
            att = att.requires_grad_(True)
            alpha = delta * ratio / n

            for _ in range(n):
                S_p  = S  + att[:, :, 0]
                VP_p = VarPrice + att[:, :, 1]
                holding = network(_bates_input(S_p, V))
                loss = loss_fn(holding, S_p, VP_p, p0)
                (grad,) = torch.autograd.grad(loss, att)

                with torch.no_grad():
                    if s_only:
                        grad = grad.clone()
                        grad[:, :, 1] = 0.0   # suppress VP gradient

                    grad_flat = grad.reshape(grad.shape[0], -1)
                    grad_norm = grad_flat.norm(p=1, dim=1)

                    step = (
                        alpha
                        * torch.sign(grad)
                        * grad_norm.view(-1, 1, 1).pow(q - 1)
                        * (grad_norm.pow(q).mean() + 1e-10).pow(1.0 / q - 1)
                    )
                    att_new = att + step

                    att_flat = att_new.reshape(att_new.shape[0], -1)
                    dist = att_flat.norm(p=float("inf"), dim=1)
                    p_conj = 1.0 / (1.0 - 1.0 / q)
                    r_val = float(
                        delta / (dist.pow(p_conj).mean().pow(1.0 / q).item() + 1e-10)
                    )
                    r = min(1.0, r_val)
                    att_new = att_new.clamp(
                        -r * dist.view(-1, 1, 1), r * dist.view(-1, 1, 1)
                    )
                    att_new[:, 0, :] = 0.0                              # never perturb t=0
                    att_new[:, :, 0] = att_new[:, :, 0].clamp(min=-(S - 0.01))
                    att_new[:, :, 1] = att_new[:, :, 1].clamp(min=-(VarPrice - 0.01))

                    if s_only:
                        att_new[:, :, 1] = 0.0

                    att = att_new.detach().requires_grad_(True)

            S_att  = (S  + att[:, :, 0]).detach().clone()
            VP_att = (VarPrice + att[:, :, 1]).detach().clone()

            with torch.no_grad():
                holding = network(_bates_input(S_att, V))
                dS  = S_att[:, 1:]  - S_att[:, :-1]
                dVP = VP_att[:, 1:] - VP_att[:, :-1]
                PnL_att = (holding[:, :, 0] * dS + holding[:, :, 1] * dVP).sum(1)
                X_after = loss_fn.terminal_payoff(S_att[:, -1]) - PnL_att

        return S_att, VP_att, X_before.detach(), X_after

    def budget_attack(
        self,
        network: nn.Module,
        S: torch.Tensor,
        V: torch.Tensor,
        VarPrice: torch.Tensor,
        loss_fn: nn.Module,
        p0: torch.Tensor,
        delta: float,
        ratio: float,
        n: int,
        q: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        Per-path budget attack on a Bates hedging network.

        S and VarPrice are perturbed jointly via a shared per-path budget
        and a direction tensor over the joint [S, VarPrice] space.

        Args:
            network:  Trained BatesHedgeNet (not mutated).
            S:        Stock price paths (batch, N+1).
            V:        Variance paths — fixed state observation.
            VarPrice: Variance swap fair-value paths (batch, N+1).
            loss_fn:  BatesCVaRLoss instance.
            p0:       CVaR threshold (scalar tensor).
            delta:    Per-path budget (RMS-normalised L∞ over joint path).
            ratio:    Step-size factor.
            n:        Number of iterations.
            q:        Norm order for budget update.

        Returns:
            att_S_best:  Best stock perturbation (batch, N+1), on CPU.
            att_VP_best: Best VarPrice perturbation (batch, N+1), on CPU.
            loss_max:    Maximum loss value achieved.
            X_after:     Hedging errors under best perturbation (batch,).
        """
        with _eval_mode(network):
            if delta == 0.0:
                zeros = torch.zeros_like(S)
                return zeros.cpu(), zeros.cpu(), 0.0, torch.zeros(S.shape[0])

            alpha = delta * ratio / n

            with torch.no_grad():
                init = delta * torch.ones(S.shape[0], S.shape[1], 2, device=S.device)
                init[:, 0, :] = 0.0
            att = init.requires_grad_(True)
            att_sign_old = att.sign().detach()
            att_best = att.detach().clone()
            loss_max = 0.0

            for _ in range(n):
                S_p  = S  + att[:, :, 0]
                VP_p = VarPrice + att[:, :, 1]
                holding = network(_bates_input(S_p, V))
                loss = loss_fn(holding, S_p, VP_p, p0)

                if loss.item() > loss_max:
                    loss_max = loss.item()
                    att_best = att.detach().clone()

                (grad,) = torch.autograd.grad(loss, att)

                with torch.no_grad():
                    att_flat  = att.reshape(att.shape[0], -1)
                    grad_flat = grad.reshape(grad.shape[0], -1)

                    budget = att_flat.abs().max(dim=1)[0]
                    att_sign_now = (att / budget.view(-1, 1, 1)).nan_to_num(0)
                    grad_b = (grad_flat * att_flat.sign()).sum(dim=1)

                    budget_new = budget + alpha * grad_b.pow(q - 1) * (
                        (grad_b.pow(q).mean() + 1e-10).pow(1.0 / q - 1)
                    )
                    budget_new = (
                        budget_new
                        / (budget_new.square().mean().sqrt() + 1e-10)
                        * delta
                    ).clamp(min=0.0)

                    att_sign_new = (
                        att_sign_now
                        + 0.4 * 0.75 * grad.sign()
                        + 0.25 * (att_sign_now - att_sign_old)
                    ).clamp(-1.0, 1.0)
                    att_sign_old = att_sign_now

                    att_new = budget_new.view(-1, 1, 1) * att_sign_new
                    att_new[:, 0, :] = 0.0
                    att = att_new.detach().requires_grad_(True)

            # Final evaluation
            S_p  = S  + att[:, :, 0]
            VP_p = VarPrice + att[:, :, 1]
            holding = network(_bates_input(S_p, V))
            loss = loss_fn(holding, S_p, VP_p, p0)
            if loss.item() > loss_max:
                loss_max = loss.item()
                att_best = att.detach().clone()

            S_att  = (S  + att_best[:, :, 0]).detach()
            VP_att = (VarPrice + att_best[:, :, 1]).detach()

            with torch.no_grad():
                holding = network(_bates_input(S_att, V))
                dS  = S_att[:, 1:]  - S_att[:, :-1]
                dVP = VP_att[:, 1:] - VP_att[:, :-1]
                PnL_att = (holding[:, :, 0] * dS + holding[:, :, 1] * dVP).sum(1)
                X_after = loss_fn.terminal_payoff(S_att[:, -1]) - PnL_att

        return (
            att_best[:, :, 0].cpu(),
            att_best[:, :, 1].cpu(),
            loss_max,
            X_after.detach(),
        )
