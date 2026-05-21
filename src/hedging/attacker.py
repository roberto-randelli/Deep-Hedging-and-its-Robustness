"""
Distributional adversarial attacks for deep hedging — He, Sutter & Gonon (2025).

Implements Wasserstein-q (wp) and per-path budget attacks from:
  "Distributional Adversarial Attacks and Training in Deep Hedging"

Supports:
  - GBM single-asset model   (HedgeNet + CVaRLoss)
  - Heston two-asset model   (HestonHedgeNet + HestonCVaRLoss)

Reference implementation:
  BS_util.DH_Attacker.Wp_att / budget_att / net_budget_att
  (Distributional-Adversarial-Attacks-and-Training-in-Deep-Hedging)
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


class DHAttacker:
    """
    Distributional adversarial attacker for deep-hedging networks.

    Two attack algorithms:
        wp_attack      — Wasserstein-q distributional attack (Frank-Wolfe variant)
        budget_attack  — Per-path budget / direction decomposition attack

    Both are implemented for:
        *_gbm     — GBM single-asset: price paths (batch, N+1)
        *_heston  — Heston two-asset: S and VarPrice paths (batch, N+1) each;
                    V (latent variance) is held fixed as a state observation.

    Typical usage::

        attacker = DHAttacker()
        price_att, X_before, X_after = attacker.wp_attack_gbm(
            network, price, loss_fn, capital, z,
            delta=0.1, ratio=0.1, n=50,
        )
    """

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gbm_input(price: torch.Tensor) -> torch.Tensor:
        """(batch, N+1) → (batch, N, 1) log-price input for HedgeNet."""
        return torch.log(price[:, :-1]).unsqueeze(-1)

    @staticmethod
    def _heston_input(S: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """(batch, N+1) × 2 → (batch, N, 2) state input for HestonHedgeNet."""
        return torch.cat([
            torch.log(S[:, :-1]).unsqueeze(-1),
            V[:, :-1].unsqueeze(-1),
        ], dim=-1)

    @staticmethod
    def _wp_step(
        att: torch.Tensor,
        grad: torch.Tensor,
        alpha: float,
        q: float,
    ) -> torch.Tensor:
        """
        Frank-Wolfe gradient ascent step for the Wasserstein-q ball.

        grad shape: (batch, D)  [D = N+1 for GBM, (N+1)*2 for Heston]
        att shape:  same as grad.
        """
        grad_norm = grad.norm(p=1, dim=1, keepdim=True)          # (batch, 1)
        return att + alpha * torch.sign(grad) * grad_norm.pow(q - 1) * (
            (grad_norm.pow(q).mean() + 1e-10).pow(1.0 / q - 1)
        )

    @staticmethod
    def _wp_project(
        att: torch.Tensor,
        delta: float,
        q: float,
    ) -> torch.Tensor:
        """
        Project att (batch, D) onto the Wasserstein-q ball of radius delta.

        Uses per-path L∞ norm as the transport cost; Hölder conjugate p = q/(q-1).
        """
        dist = att.norm(p=float("inf"), dim=1, keepdim=True)      # (batch, 1)
        p_conj = 1.0 / (1.0 - 1.0 / q)
        r_val = float(delta / (dist.pow(p_conj).mean().pow(1.0 / q).item() + 1e-10))
        r = min(1.0, r_val)
        return att.clamp(-r * dist, r * dist)

    # ------------------------------------------------------------------ #
    #  GBM / single-asset — Wasserstein-q attack                          #
    # ------------------------------------------------------------------ #

    def wp_attack_gbm(
        self,
        network: nn.Module,
        price: torch.Tensor,       # (batch, N+1)
        loss_fn: nn.Module,        # CVaRLoss
        capital: torch.Tensor,     # scalar
        z: torch.Tensor,           # scalar CVaR threshold
        delta: float,
        ratio: float,
        n: int,
        q: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wasserstein-q distributional attack on a GBM hedging network.

        Args:
            network:  Trained HedgeNet (not mutated).
            price:    Stock price paths (batch, N+1).
            loss_fn:  CVaRLoss instance.
            capital:  Fixed option premium (scalar tensor).
            z:        Learned CVaR threshold (scalar tensor).
            delta:    Wasserstein ball radius.
            ratio:    Step-size factor; alpha = delta * ratio / n.
            n:        Number of gradient-ascent iterations.
            q:        Wasserstein order (default 2).

        Returns:
            price_att:  Adversarial price paths (batch, N+1).
            X_before:   Hedging errors before attack (batch,).
            X_after:    Hedging errors after attack (batch,).
        """
        with _eval_mode(network):
            with torch.no_grad():
                holding = network(self._gbm_input(price))
                PnL = (holding * (price[:, 1:] - price[:, :-1])).sum(1)
                X_before = loss_fn.terminal_payoff(price[:, -1]) - capital - PnL

            if delta == 0.0:
                return price.clone(), X_before, X_before.clone()

            att = torch.zeros_like(price).requires_grad_(True)
            alpha = delta * ratio / n

            for _ in range(n):
                price_p = price + att
                holding = network(self._gbm_input(price_p))
                loss = loss_fn(holding, price_p, capital, z)
                (grad,) = torch.autograd.grad(loss, att)

                with torch.no_grad():
                    att_new = self._wp_step(att, grad, alpha, q)
                    att_new = self._wp_project(att_new, delta, q)
                    att_new[:, 0] = 0.0                        # never perturb S_0
                    att_new = att_new.clamp(min=-(price - 0.01))  # keep S > 0
                    att = att_new.detach().requires_grad_(True)

            price_att = (price + att).detach().clone()

            with torch.no_grad():
                holding = network(self._gbm_input(price_att))
                PnL_att = (holding * (price_att[:, 1:] - price_att[:, :-1])).sum(1)
                X_after = loss_fn.terminal_payoff(price_att[:, -1]) - capital - PnL_att

        return price_att, X_before.detach(), X_after

    # ------------------------------------------------------------------ #
    #  GBM / single-asset — budget attack                                  #
    # ------------------------------------------------------------------ #

    def budget_attack_gbm(
        self,
        network: nn.Module,
        price: torch.Tensor,       # (batch, N+1)
        loss_fn: nn.Module,
        capital: torch.Tensor,
        z: torch.Tensor,
        delta: float,
        ratio: float,
        n: int,
        q: float = 2.0,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        """
        Per-path budget attack on a GBM hedging network.

        The perturbation is decomposed into a per-path budget magnitude and a
        direction in {-1, 0, 1}^(N+1), each updated via separate gradient steps.

        Args:
            network:  Trained HedgeNet (not mutated).
            price:    Stock price paths (batch, N+1).
            loss_fn:  CVaRLoss instance.
            capital:  Fixed option premium.
            z:        CVaR threshold.
            delta:    Per-path budget constraint (L∞ sense, RMS-normalised).
            ratio:    Step-size factor; alpha = delta * ratio / n.
            n:        Number of iterations.
            q:        Norm order for budget update (default 2).

        Returns:
            att_best:  Best perturbation found (batch, N+1), on CPU.
            loss_max:  Maximum loss value achieved.
            X_after:   Hedging errors under best perturbation (batch,).
        """
        with _eval_mode(network):
            if delta == 0.0:
                return torch.zeros_like(price).cpu(), 0.0, torch.zeros(price.shape[0])

            alpha = delta * ratio / n
            att = (delta * torch.ones_like(price)).requires_grad_(True)
            # att[:, 0] = 0 but requires_grad_ prevents in-place on leaf, so:
            with torch.no_grad():
                init = delta * torch.ones_like(price)
                init[:, 0] = 0.0
                att = init.requires_grad_(True)

            att_sign_old = att.sign().detach()
            att_best = att.detach().clone()
            loss_max = 0.0

            for _ in range(n):
                holding = network(self._gbm_input(price + att))
                loss = loss_fn(holding, price + att, capital, z)

                if loss.item() > loss_max:
                    loss_max = loss.item()
                    att_best = att.detach().clone()

                (grad,) = torch.autograd.grad(loss, att)

                with torch.no_grad():
                    budget = att.abs().max(dim=1)[0]               # (batch,)
                    att_sign_now = (att / budget.unsqueeze(1)).nan_to_num(0)
                    grad_b = (grad * att.sign()).sum(dim=1)

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

                    att_new = budget_new.unsqueeze(1) * att_sign_new
                    att_new[:, 0] = 0.0
                    att = att_new.detach().requires_grad_(True)

            # Final evaluation with current att
            holding = network(self._gbm_input(price + att))
            loss = loss_fn(holding, price + att, capital, z)
            if loss.item() > loss_max:
                loss_max = loss.item()
                att_best = att.detach().clone()

            price_att = (price + att_best).detach()
            with torch.no_grad():
                holding = network(self._gbm_input(price_att))
                PnL_att = (holding * (price_att[:, 1:] - price_att[:, :-1])).sum(1)
                X_after = loss_fn.terminal_payoff(price_att[:, -1]) - capital - PnL_att

        return att_best.cpu(), loss_max, X_after.detach()

    # ------------------------------------------------------------------ #
    #  Heston / two-asset — Wasserstein-q attack                          #
    # ------------------------------------------------------------------ #

    def wp_attack_heston(
        self,
        network: nn.Module,
        S: torch.Tensor,           # (batch, N+1)
        V: torch.Tensor,           # (batch, N+1) — latent variance, kept fixed
        VarPrice: torch.Tensor,    # (batch, N+1)
        loss_fn: nn.Module,        # HestonCVaRLoss
        p0: torch.Tensor,          # scalar CVaR threshold
        delta: float,
        ratio: float,
        n: int,
        q: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wasserstein-q distributional attack on a Heston hedging network.

        S and VarPrice are perturbed jointly; V (latent variance used as state
        observation by the network) is held fixed throughout.

        Args:
            network:   Trained HestonHedgeNet (not mutated).
            S:         Stock price paths (batch, N+1).
            V:         Variance paths (batch, N+1) — fixed, not perturbed.
            VarPrice:  Variance swap fair-value paths (batch, N+1).
            loss_fn:   HestonCVaRLoss instance.
            p0:        CVaR threshold parameter (scalar tensor).
            delta:     Wasserstein ball radius applied to the joint [S, VarPrice] path.
            ratio:     Step-size factor; alpha = delta * ratio / n.
            n:         Number of gradient-ascent iterations.
            q:         Wasserstein order (default 2).

        Returns:
            S_att:    Adversarial stock price paths (batch, N+1).
            VP_att:   Adversarial variance swap price paths (batch, N+1).
            X_before: Hedging errors before attack (batch,).
            X_after:  Hedging errors after attack (batch,).
        """
        with _eval_mode(network):
            with torch.no_grad():
                holding = network(self._heston_input(S, V))
                dS  = S[:, 1:] - S[:, :-1]
                dVP = VarPrice[:, 1:] - VarPrice[:, :-1]
                PnL = (holding[:, :, 0] * dS + holding[:, :, 1] * dVP).sum(1)
                X_before = loss_fn.terminal_payoff(S[:, -1]) - PnL

            if delta == 0.0:
                return S.clone(), VarPrice.clone(), X_before, X_before.clone()

            # Joint perturbation tensor: (batch, N+1, 2)  — [:,:,0]=ΔS, [:,:,1]=ΔVP
            att = torch.zeros(S.shape[0], S.shape[1], 2, device=S.device)
            att = att.requires_grad_(True)
            alpha = delta * ratio / n

            for _ in range(n):
                S_p  = S  + att[:, :, 0]
                VP_p = VarPrice + att[:, :, 1]
                holding = network(self._heston_input(S_p, V))
                loss = loss_fn(holding, S_p, VP_p, p0)
                (grad,) = torch.autograd.grad(loss, att)

                with torch.no_grad():
                    # Flatten to (batch, (N+1)*2) for norm computation, then restore shape
                    grad_flat = grad.reshape(grad.shape[0], -1)
                    grad_norm = grad_flat.norm(p=1, dim=1)             # (batch,)

                    step = (
                        alpha
                        * torch.sign(grad)
                        * grad_norm.view(-1, 1, 1).pow(q - 1)
                        * (grad_norm.pow(q).mean() + 1e-10).pow(1.0 / q - 1)
                    )
                    att_new = att + step

                    # Project using flattened per-path L∞ norm
                    att_flat = att_new.reshape(att_new.shape[0], -1)
                    dist = att_flat.norm(p=float("inf"), dim=1)        # (batch,)
                    p_conj = 1.0 / (1.0 - 1.0 / q)
                    r_val = float(
                        delta / (dist.pow(p_conj).mean().pow(1.0 / q).item() + 1e-10)
                    )
                    r = min(1.0, r_val)
                    att_new = att_new.clamp(
                        -r * dist.view(-1, 1, 1), r * dist.view(-1, 1, 1)
                    )
                    att_new[:, 0, :] = 0.0                             # never perturb t=0
                    att_new[:, :, 0] = att_new[:, :, 0].clamp(min=-(S - 0.01))
                    att_new[:, :, 1] = att_new[:, :, 1].clamp(min=-(VarPrice - 0.01))
                    att = att_new.detach().requires_grad_(True)

            S_att  = (S  + att[:, :, 0]).detach().clone()
            VP_att = (VarPrice + att[:, :, 1]).detach().clone()

            with torch.no_grad():
                holding = network(self._heston_input(S_att, V))
                dS  = S_att[:, 1:]  - S_att[:, :-1]
                dVP = VP_att[:, 1:] - VP_att[:, :-1]
                PnL_att = (holding[:, :, 0] * dS + holding[:, :, 1] * dVP).sum(1)
                X_after = loss_fn.terminal_payoff(S_att[:, -1]) - PnL_att

        return S_att, VP_att, X_before.detach(), X_after

    # ------------------------------------------------------------------ #
    #  Heston / two-asset — budget attack                                  #
    # ------------------------------------------------------------------ #

    def budget_attack_heston(
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
        Per-path budget attack on a Heston hedging network.

        S and VarPrice are perturbed jointly via a shared per-path budget
        and a direction tensor over the joint [S, VarPrice] space.

        Args:
            network:   Trained HestonHedgeNet (not mutated).
            S:         Stock price paths (batch, N+1).
            V:         Variance paths — fixed state observation.
            VarPrice:  Variance swap fair-value paths (batch, N+1).
            loss_fn:   HestonCVaRLoss instance.
            p0:        CVaR threshold parameter.
            delta:     Per-path budget (RMS-normalised L∞ over joint path).
            ratio:     Step-size factor.
            n:         Number of iterations.
            q:         Norm order for budget update.

        Returns:
            att_S_best:   Best stock perturbation (batch, N+1), on CPU.
            att_VP_best:  Best VarPrice perturbation (batch, N+1), on CPU.
            loss_max:     Maximum loss value achieved.
            X_after:      Hedging errors under best perturbation (batch,).
        """
        with _eval_mode(network):
            if delta == 0.0:
                zeros = torch.zeros_like(S)
                return zeros.cpu(), zeros.cpu(), 0.0, torch.zeros(S.shape[0])

            alpha = delta * ratio / n

            # Joint att: (batch, N+1, 2)
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
                holding = network(self._heston_input(S_p, V))
                loss = loss_fn(holding, S_p, VP_p, p0)

                if loss.item() > loss_max:
                    loss_max = loss.item()
                    att_best = att.detach().clone()

                (grad,) = torch.autograd.grad(loss, att)

                with torch.no_grad():
                    # Flatten (N+1, 2) → (N+1)*2 for budget extraction
                    att_flat  = att.reshape(att.shape[0], -1)
                    grad_flat = grad.reshape(grad.shape[0], -1)

                    budget = att_flat.abs().max(dim=1)[0]              # (batch,)
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
            holding = network(self._heston_input(S_p, V))
            loss = loss_fn(holding, S_p, VP_p, p0)
            if loss.item() > loss_max:
                loss_max = loss.item()
                att_best = att.detach().clone()

            S_att  = (S  + att_best[:, :, 0]).detach()
            VP_att = (VarPrice + att_best[:, :, 1]).detach()

            with torch.no_grad():
                holding = network(self._heston_input(S_att, V))
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
