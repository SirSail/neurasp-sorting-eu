"""Neural comparator network for pairwise integer comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurasp.core.types import Probability

__all__ = ["Comparator", "ComparatorConfig"]


@dataclass(slots=True, frozen=True)
class ComparatorConfig:
    input_dim: int = 64
    hidden_dim: int = 64
    output_dim: int = 3


class Comparator(nn.Module):
    """Neural network for pairwise integer comparison."""

    OUTPUT_LESS: Final[int] = 0
    OUTPUT_EQUAL: Final[int] = 1
    OUTPUT_GREATER: Final[int] = 2

    __slots__ = ("_config", "fc1", "fc2", "fc3")

    def __init__(self, config: ComparatorConfig | None = None) -> None:
        super().__init__()
        self._config = config or ComparatorConfig()
        self.fc1 = nn.Linear(2 * self._config.input_dim, self._config.hidden_dim)
        self.fc2 = nn.Linear(self._config.hidden_dim, self._config.hidden_dim)
        self.fc3 = nn.Linear(self._config.hidden_dim, self._config.output_dim)

    @property
    def config(self) -> ComparatorConfig:
        return self._config

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1).squeeze(0)

    def get_comparison_probs(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[Probability, Probability, Probability]:
        with torch.no_grad():
            probs = self.forward(a, b)
            return (
                Probability(probs[self.OUTPUT_LESS].item()),
                Probability(probs[self.OUTPUT_EQUAL].item()),
                Probability(probs[self.OUTPUT_GREATER].item()),
            )
