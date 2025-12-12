"""NeurASP Core - Neural Networks with Answer Set Programming."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import clingo
import torch
import torch.nn as nn

from neurasp.core.types import SortingConfig, SortingResult, ScoreComponents, SortingState
from neurasp.neural.comparator import Comparator, ComparatorConfig
from neurasp.neural.encoder import IntegerEncoder, EncoderConfig

__all__ = ["NeurASPSorter", "SorterWeights", "SortingConfig"]

ASP_DIR: Final[Path] = Path(__file__).parent / "asp"
SORTING_RULES: Final[Path] = ASP_DIR / "sorting.lp"
LIST_SIZE: Final[int] = 5


def iter_pairs(n: int) -> Iterator[tuple[int, int]]:
    """Yield all (i, j) pairs where i != j."""
    for i in range(n):
        for j in range(n):
            if i != j:
                yield i, j


@dataclass(slots=True, frozen=True)
class SorterWeights:
    comparison_weight: float = 1.0
    ordering_weight: float = 0.5
    constraint_weight: float = 0.1


class NeurASPSorter(nn.Module):
    """Neural-Symbolic sorter using NeurASP."""

    PROB_THRESHOLD: Final[float] = 0.5
    LIST_SIZE: Final[int] = 5

    def __init__(
        self,
        config: SortingConfig | None = None,
        weights: SorterWeights | None = None,
    ) -> None:
        super().__init__()
        self._config = config or SortingConfig()
        self._weights = weights or SorterWeights()

        encoder_cfg = EncoderConfig(
            embed_dim=self._config.hidden_dim,
            max_value=self._config.max_integer_value,
        )
        comparator_cfg = ComparatorConfig(
            input_dim=self._config.hidden_dim,
            hidden_dim=self._config.hidden_dim,
        )
        self.encoder = IntegerEncoder(encoder_cfg)
        self.comparator = Comparator(comparator_cfg)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._config.learning_rate)
        self._asp_rules = SORTING_RULES.read_text(encoding="utf-8") if SORTING_RULES.exists() else ""

    @property
    def config(self) -> SortingConfig:
        return self._config

    @property
    def weights(self) -> SorterWeights:
        return self._weights

    def compare(self, a: int, b: int) -> torch.Tensor:
        a_emb = self.encoder(torch.tensor([a]))
        b_emb = self.encoder(torch.tensor([b]))
        return self.comparator(a_emb, b_emb)

    def forward(self, integers: list[int]) -> torch.Tensor:
        n = len(integers)
        embeddings = self.encoder(torch.tensor(integers))
        comparisons = torch.zeros(n, n, 3)
        for i, j in iter_pairs(n):
            comparisons[i, j] = self.comparator(embeddings[i:i+1], embeddings[j:j+1])
        return comparisons

    def solve_asp(self, comparison_probs: torch.Tensor) -> list[int]:
        n = comparison_probs.shape[0]
        program = self._asp_rules + "\n"
        for i, j in iter_pairs(n):
            if comparison_probs[i, j, 0].item() > self.PROB_THRESHOLD:
                program += f"less({i}, {j}).\n"

        ctl = clingo.Control()
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        result: list[int] = list(range(n))
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                for atom in model.symbols(shown=True):
                    if atom.name == "position":
                        idx, pos = atom.arguments
                        result[pos.number] = idx.number
                break
        return result

    def sort(self, integers: list[int]) -> SortingResult:
        with torch.no_grad():
            comparison_probs = self.forward(integers)
            order = self.solve_asp(comparison_probs)
            sorted_list = tuple(integers[i] for i in order)
            expected = tuple(sorted(integers))
            state: SortingState = "SAT" if sorted_list == expected else "VIOL"
            return SortingResult(
                input_list=tuple(integers),
                sorted_list=sorted_list,
                state=state,
                metrics={"order": order},
            )

    def train_step(self, integers: list[int], target: list[int]) -> tuple[float, ScoreComponents]:
        self.optimizer.zero_grad()
        comparison_probs = self.forward(integers)
        comp = self._compute_loss(integers, target, comparison_probs)
        loss = torch.tensor(comp.total(), requires_grad=True)
        loss.backward()
        self.optimizer.step()
        return comp.total(), comp

    def _compute_loss(
        self, integers: list[int], target: list[int], comparison_probs: torch.Tensor
    ) -> ScoreComponents:
        comp = ScoreComponents()
        n = len(integers)
        int_to_target_pos = {v: i for i, v in enumerate(target)}

        for i, j in iter_pairs(n):
            pos_i, pos_j = int_to_target_pos[integers[i]], int_to_target_pos[integers[j]]
            if pos_i < pos_j:
                comp.comparison_loss += -torch.log(comparison_probs[i, j, 0] + 1e-8).item()
            elif pos_i > pos_j:
                comp.comparison_loss += -torch.log(comparison_probs[i, j, 2] + 1e-8).item()

        comp.comparison_loss /= (n * (n - 1))
        return comp

    def train(self, epochs: int = 100, verbose: bool = True) -> list[tuple[float, ScoreComponents]]:
        import random
        results: list[tuple[float, ScoreComponents]] = []
        for epoch in range(epochs):
            integers = random.sample(range(1, 100), self._config.list_size)
            target = sorted(integers)
            loss, comp = self.train_step(integers, target)
            results.append((loss, comp))
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        return results
