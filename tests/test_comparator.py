"""
Tests for the neural comparator network.

Following Caissis testing patterns.
"""
from __future__ import annotations

import pytest
import torch

from neurasp.neural.comparator import Comparator, ComparatorConfig
from neurasp.neural.encoder import IntegerEncoder, EncoderConfig


class TestComparator:
    """Tests for the Comparator class."""

    def test_comparator_output_shape(self) -> None:
        """Test that comparator outputs correct shape."""
        config = ComparatorConfig(input_dim=64, hidden_dim=64)
        comparator = Comparator(config)

        a = torch.randn(1, 64)
        b = torch.randn(1, 64)

        output = comparator(a, b)
        assert output.shape == (3,), f"Expected shape (3,), got {output.shape}"

    def test_comparator_probabilities_sum_to_one(self) -> None:
        """Test that output probabilities sum to 1."""
        comparator = Comparator()

        a = torch.randn(1, 64)
        b = torch.randn(1, 64)

        output = comparator(a, b)
        assert torch.isclose(output.sum(), torch.tensor(1.0), atol=1e-5)

    def test_comparator_all_positive(self) -> None:
        """Test that all probabilities are positive."""
        comparator = Comparator()

        a = torch.randn(1, 64)
        b = torch.randn(1, 64)

        output = comparator(a, b)
        assert (output >= 0).all()

    def test_comparator_config_property(self) -> None:
        """Test that config property returns configuration."""
        config = ComparatorConfig(input_dim=32, hidden_dim=16)
        comparator = Comparator(config)

        assert comparator.config.input_dim == 32
        assert comparator.config.hidden_dim == 16


class TestIntegerEncoder:
    """Tests for the IntegerEncoder class."""

    def test_encoder_output_shape(self) -> None:
        """Test that encoder outputs correct shape."""
        config = EncoderConfig(embed_dim=64)
        encoder = IntegerEncoder(config)

        x = torch.tensor([42])
        output = encoder(x)

        assert output.shape == (1, 64), f"Expected shape (1, 64), got {output.shape}"

    def test_encoder_batch(self) -> None:
        """Test encoder with batch input."""
        encoder = IntegerEncoder()

        x = torch.tensor([1, 2, 3, 4, 5])
        output = encoder(x)

        assert output.shape == (5, 64), f"Expected shape (5, 64), got {output.shape}"

    def test_encoder_handles_large_values(self) -> None:
        """Test that encoder handles values larger than max_value."""
        config = EncoderConfig(embed_dim=64, max_value=100)
        encoder = IntegerEncoder(config)

        x = torch.tensor([999])  # Larger than max_value
        output = encoder(x)

        assert output.shape == (1, 64)  # Should not crash

    def test_encoder_embed_dim_property(self) -> None:
        """Test embed_dim property."""
        config = EncoderConfig(embed_dim=128)
        encoder = IntegerEncoder(config)

        assert encoder.embed_dim == 128

    def test_encode_single(self) -> None:
        """Test encode_single method."""
        encoder = IntegerEncoder()

        output = encoder.encode_single(42)  # type: ignore

        assert output.shape == (1, 64)
