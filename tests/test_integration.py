"""
Integration tests for NeurASPSorter.

Following Caissis testing patterns.
"""
from __future__ import annotations

import pytest
import torch

from neurasp import NeurASPSorter, SorterWeights
from neurasp.core.types import SortingConfig


class TestNeurASPSorter:
    """Integration tests for the NeurASP sorter."""

    def test_sorter_initialization(self) -> None:
        """Test that sorter initializes correctly."""
        sorter = NeurASPSorter()

        assert sorter.config.list_size == 5
        assert sorter.encoder is not None
        assert sorter.comparator is not None

    def test_sorter_with_custom_config(self) -> None:
        """Test sorter with custom configuration."""
        config = SortingConfig(
            list_size=5,
            hidden_dim=32,
            learning_rate=0.01
        )
        sorter = NeurASPSorter(config)

        assert sorter.config.hidden_dim == 32
        assert sorter.config.learning_rate == 0.01

    def test_sorter_with_custom_weights(self) -> None:
        """Test sorter with custom weights."""
        weights = SorterWeights(
            comparison_weight=2.0,
            ordering_weight=0.8,
        )
        sorter = NeurASPSorter(weights=weights)

        assert sorter.weights.comparison_weight == 2.0
        assert sorter.weights.ordering_weight == 0.8

    def test_compare_output_shape(self) -> None:
        """Test that compare outputs correct shape."""
        sorter = NeurASPSorter()

        result = sorter.compare(3, 7)

        assert result.shape == (3,)

    def test_forward_output_shape(self) -> None:
        """Test that forward outputs correct shape."""
        sorter = NeurASPSorter()

        integers = [5, 2, 8, 1, 9]
        result = sorter.forward(integers)

        assert result.shape == (5, 5, 3)

    def test_train_step_returns_loss_and_components(self) -> None:
        """Test that train_step returns a valid loss and components."""
        sorter = NeurASPSorter()

        integers = [5, 2, 8, 1, 9]
        target = [1, 2, 5, 8, 9]

        loss, comp = sorter.train_step(integers, target)

        assert isinstance(loss, float)
        assert loss >= 0
        assert comp.total() == loss

    def test_sort_returns_sorting_result(self) -> None:
        """Test that sort returns SortingResult."""
        sorter = NeurASPSorter()

        integers = [5, 2, 8, 1, 9]
        result = sorter.sort(integers)

        assert result.input_list == tuple(integers)
        assert len(result.sorted_list) == 5
        assert result.state in ("SAT", "VIOL", "UNK")


class TestScoreComponents:
    """Tests for ScoreComponents dataclass."""

    def test_total(self) -> None:
        """Test total method."""
        from neurasp.core.types import ScoreComponents

        comp = ScoreComponents(
            comparison_loss=0.5,
            ordering_loss=0.3,
            constraint_penalty=0.1,
        )

        assert abs(comp.total() - 0.9) < 1e-6

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        from neurasp.core.types import ScoreComponents

        comp = ScoreComponents(comparison_loss=0.12345)

        d = comp.to_dict()

        assert "comparison_loss" in d
        assert d["comparison_loss"] == 0.1235  # Rounded to 4 decimals
