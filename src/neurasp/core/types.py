"""Type definitions for NeurASP."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, Mapping, NewType, Sequence

IntegerValue = NewType('IntegerValue', int)
Position = NewType('Position', int)
Probability = NewType('Probability', float)

ComparisonResult = Literal["less", "equal", "greater"]
SortingState = Literal["SAT", "VIOL", "UNK"]

__all__ = [
    "IntegerValue",
    "Position",
    "Probability",
    "ComparisonResult",
    "SortingState",
    "ComparisonOutcome",
    "SortingConfig",
    "SortingResult",
    "ScoreComponents",
]


@dataclass(slots=True, frozen=True)
class ComparisonOutcome:
    prob_less: Probability
    prob_equal: Probability
    prob_greater: Probability

    def argmax(self) -> ComparisonResult:
        probs = [self.prob_less, self.prob_equal, self.prob_greater]
        labels: list[ComparisonResult] = ["less", "equal", "greater"]
        return labels[probs.index(max(probs))]


@dataclass(slots=True, frozen=True)
class SortingConfig:
    list_size: int = 5
    hidden_dim: int = 64
    learning_rate: float = 0.001
    device: str = "cpu"
    max_integer_value: int = 1000


@dataclass(slots=True, frozen=True)
class SortingResult:
    input_list: tuple[int, ...]
    sorted_list: tuple[int, ...]
    state: SortingState
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreComponents:
    comparison_loss: float = 0.0
    ordering_loss: float = 0.0
    constraint_penalty: float = 0.0

    def total(self) -> float:
        return sum(getattr(self, f.name) for f in fields(self))

    def to_dict(self) -> dict[str, float]:
        return {f.name: round(getattr(self, f.name), 4) for f in fields(self)}
