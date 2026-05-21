"""
CVaR deep-hedging training loop.

Network and the CVaR threshold z are optimised with Adam + LR schedule.
Capital / premium is fixed and passed separately.
Input to the network: log(S_t) at each time step, shape (batch, N, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.hedging.hedge_network import HedgeNet
from src.hedging.loss import CVaRLoss


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    network: HedgeNet,
    paths: torch.Tensor,               # (M, N+1) — price paths on CPU
    loss_fn: CVaRLoss,
    *,
    capital: float | torch.Tensor,     # fixed initial premium / capital
    z_init: float = 0.0,               # initial CVaR threshold
    n_epochs: int = 300,
    batch_size: int = 10_000,
    lr: float = 5e-3,
    device: torch.device | None = None,
    log_every: int = 50,
) -> tuple[list[float], torch.Tensor]:
    """
    Jointly optimises network weights θ and CVaR threshold z via Adam.
    Capital is fixed.

    Returns:
        losses: Per-epoch scalar loss values.
        z:      Learned CVaR threshold (detached, on CPU).
    """
    if device is None:
        device = _auto_device()

    network = network.to(device)
    paths   = paths.to(device)

    if torch.is_tensor(capital):
        capital_t = capital.to(device=device, dtype=torch.float32)
    else:
        capital_t = torch.tensor(capital, dtype=torch.float32, device=device)

    z = nn.Parameter(torch.tensor(z_init, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(
        [{"params": network.parameters()}, {"params": [z]}],
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
        batch = paths[idx]                                  # (batch, N+1)

        x       = torch.log(batch[:, :-1]).unsqueeze(-1)    # (batch, N, 1)
        holding = network(x)                                # (batch, N)

        loss = loss_fn(holding, batch, capital_t, z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:4d}  loss={loss.item():.6f}  z={z.item():.4f}")

    return losses, z.detach().cpu()