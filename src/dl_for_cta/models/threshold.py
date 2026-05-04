from __future__ import annotations

import torch
from torch import nn


class LearnableThreshold(nn.Module):
    def __init__(self, initial_value: float, min_value: float = 0.0, max_value: float = 1.0) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        span = max(max_value - min_value, 1e-6)
        p = min(max((initial_value - min_value) / span, 1e-6), 1.0 - 1e-6)
        self.raw = nn.Parameter(torch.logit(torch.tensor(float(p))))

    def value(self) -> torch.Tensor:
        return self.min_value + (self.max_value - self.min_value) * torch.sigmoid(self.raw)

    def forward(self, raw_positions: torch.Tensor, temperature: float = 0.02) -> torch.Tensor:
        theta = self.value()
        prev = torch.zeros_like(raw_positions)
        prev[:, 1:] = raw_positions[:, :-1]
        delta = raw_positions - prev
        gate = torch.sigmoid((delta.abs() - theta) / temperature)
        return prev + gate * delta


class HardThreshold:
    def __init__(self, theta: float) -> None:
        self.theta = float(theta)

    def apply(self, raw_positions) -> list[float]:
        actual = []
        prev = 0.0
        for raw in raw_positions:
            value = float(raw)
            if abs(value - prev) > self.theta:
                prev = value
            actual.append(prev)
        return actual
