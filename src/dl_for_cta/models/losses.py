from __future__ import annotations

import torch


def negative_sharpe_loss(
    positions: torch.Tensor,
    future_returns: torch.Tensor,
    *,
    turnover_penalty: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    strategy_returns = positions * future_returns
    mean = strategy_returns.mean()
    std = strategy_returns.std(unbiased=False).clamp_min(eps)
    loss = -mean / std
    if turnover_penalty:
        turnover = (positions[:, 1:] - positions[:, :-1]).abs().mean()
        loss = loss + turnover_penalty * turnover
    return loss
