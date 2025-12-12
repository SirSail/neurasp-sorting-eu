"""Integer encoder for neural processing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn

from neurasp.core.types import IntegerValue

__all__ = ["IntegerEncoder", "EncoderConfig"]


@dataclass(slots=True, frozen=True)
class EncoderConfig:
    embed_dim: int = 64
    max_value: int = 1000


class IntegerEncoder(nn.Module):
    """Encodes integers into dense vector embeddings."""

    MIN_VALUE: Final[int] = 0

    __slots__ = ("_config", "embedding")

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self._config = config or EncoderConfig()
        self.embedding = nn.Embedding(self._config.max_value, self._config.embed_dim)

    @property
    def config(self) -> EncoderConfig:
        return self._config

    @property
    def embed_dim(self) -> int:
        return self._config.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        if x.dim() == 0:
            x = x.unsqueeze(0)
        max_idx = self.embedding.num_embeddings - 1
        x = torch.clamp(x, self.MIN_VALUE, max_idx)
        return self.embedding(x)

    def encode_single(self, value: IntegerValue) -> torch.Tensor:
        return self.forward(torch.tensor([value]))
