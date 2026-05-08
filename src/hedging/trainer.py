"""
Vanilla deep-hedging training loop — He, Sutter & Gonon (2025) / BS_train_clean.

Network and p0 are optimised jointly with Adam + LR schedule.
Input to the network: log(S_t) at each time step, shape (batch, N, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.hedging.hedge_network import HedgeNet
from src.hedging.loss import EntropicOCELoss


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    network: HedgeNet,
    paths: torch.Tensor,          # (M, N+1) — price paths on CPU
    loss_fn: EntropicOCELoss,
    *,
    p0_init: float = 7.96,        # initial option price proxy (≈ BS ATM call, T=1)
    n_epochs: int = 300,
    batch_size: int = 10_000,
    lr: float = 5e-3,
    device: torch.device | None = None,
    log_every: int = 50,
) -> tuple[list[float], torch.Tensor]:
    """
    Vanilla deep-hedging training — He et al. (2025).

    Jointly optimises network weights θ and option price proxy p0 via Adam.
    Learning rate is decayed by schedule matching the reference implementation:
        epochs   0–99  : lr × 1
        epochs 100–199 : lr × 0.1
        epochs 200–249 : lr × 0.01
        epochs 250+    : lr × 0.001

    Args:
        network:    HedgeNet (N separate sub-networks). Mutated in-place.
        paths:      Price paths (M, N+1) on CPU.
        loss_fn:    EntropicOCELoss instance.
        p0_init:    Initial value for the option price parameter.
        n_epochs:   Training epochs.
        batch_size: Paths per mini-batch.
        lr:         Initial Adam learning rate.
        device:     Compute device. Auto-selects cuda → mps → cpu if None.
        log_every:  Print frequency.

    Returns:
        losses: Per-epoch scalar loss values.
        p0:     Learned option price proxy (detached, on CPU).
    """
    if device is None:
        device = _auto_device()

    network = network.to(device)
    paths   = paths.to(device)

    p0 = nn.Parameter(torch.tensor(p0_init, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(
        [{"params": network.parameters()}, {"params": [p0]}],
        lr=lr,
    )

    def _lr_lambda(epoch: int) -> float:
        if epoch < 100:   return 1.0
        if epoch < 200:   return 0.1
        if epoch < 250:   return 0.01
        return 0.001

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    M = paths.shape[0]
    losses: list[float] = []

    for epoch in range(n_epochs):
        network.train()

        idx   = torch.randperm(M, device=device)[:batch_size]
        batch = paths[idx]                                          # (batch, N+1)

        # Network input: log(S_t) for t = 0 … N-1, shape (batch, N, 1)
        x       = torch.log(batch[:, :-1]).unsqueeze(-1)
        holding = network(x)                                        # (batch, N)

        loss = loss_fn(holding, batch, p0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  loss={loss.item():.6f}  p0={p0.item():.4f}")

    return losses, p0.detach().cpu()
