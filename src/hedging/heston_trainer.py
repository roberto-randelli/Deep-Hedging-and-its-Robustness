"""
Deep-hedging training loop for Heston model — He, Sutter & Gonon (2025).

HestonHedgeNet and p0 are jointly optimised with Adam + LR schedule.
Network input at each step: [log(S_t), V_t], shape (batch, N, 2).
Network output:             [δ_S_t, δ_V_t], shape (batch, N, 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.hedging.hedge_network import HestonHedgeNet
from src.hedging.loss import HestonCVaRLoss


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    network: HestonHedgeNet,
    S: torch.Tensor,            # (M, N+1) — stock price paths, CPU
    V: torch.Tensor,            # (M, N+1) — variance paths, CPU
    VarPrice: torch.Tensor,     # (M, N+1) — variance swap price paths, CPU
    loss_fn: HestonCVaRLoss,
    *,
    p0_init: float = 1.96,
    n_epochs: int = 300,
    batch_size: int = 10_000,
    lr: float = 5e-3,
    device: torch.device | None = None,
    log_every: int = 50,
) -> tuple[list[float], torch.Tensor]:
    """
    Jointly optimises HestonHedgeNet weights and p0 via Adam.

    LR schedule (mirrors GBM trainer / He et al.):
        epochs   0–99  : lr × 1
        epochs 100–199 : lr × 0.1
        epochs 200–249 : lr × 0.01
        epochs 250+    : lr × 0.001

    Args:
        network:    HestonHedgeNet. Mutated in-place.
        S:          Stock price paths (M, N+1).
        V:          Variance paths (M, N+1).
        VarPrice:   Variance swap fair values (M, N+1).
        loss_fn:    HestonCVaRLoss instance.
        p0_init:    Initial value for the VaR threshold parameter.
        n_epochs:   Training epochs.
        batch_size: Paths per mini-batch.
        lr:         Initial Adam learning rate.
        device:     Compute device. Auto-selects cuda → mps → cpu if None.
        log_every:  Print frequency.

    Returns:
        losses: Per-epoch scalar loss values.
        p0:     Learned VaR threshold (detached, on CPU).
    """
    if device is None:
        device = _auto_device()

    network  = network.to(device)
    S        = S.to(device)
    V        = V.to(device)
    VarPrice = VarPrice.to(device)

    p0 = nn.Parameter(torch.tensor(p0_init, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(
        [{"params": network.parameters()}, {"params": [p0]}],
        lr=lr,
    )

    def _lr_lambda(epoch: int) -> float:
        # Matches Prequel Heston_train_clean.py schedule; extended for longer runs
        if epoch < 200:   return 1.0
        if epoch < 400:   return 0.1
        if epoch < 600:   return 0.01
        if epoch < 1000:  return 0.001
        return 0.0001

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    M = S.shape[0]
    losses: list[float] = []

    for epoch in range(n_epochs):
        network.train()

        idx   = torch.randperm(M, device=device)[:batch_size]
        S_b   = S[idx]           # (batch, N+1)
        V_b   = V[idx]           # (batch, N+1)
        VP_b  = VarPrice[idx]    # (batch, N+1)

        # Network input: [log(S_t), V_t] for t = 0 … N-1, shape (batch, N, 2)
        x = torch.cat([
            torch.log(S_b[:, :-1]).unsqueeze(-1),
            V_b[:, :-1].unsqueeze(-1),
        ], dim=-1)
        holding = network(x)     # (batch, N, 2)

        loss = loss_fn(holding, S_b, VP_b, p0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  loss={loss.item():.6f}  p0={p0.item():.4f}")

    return losses, p0.detach().cpu()
