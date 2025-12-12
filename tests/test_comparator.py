"""
Tests for the neural comparator network.
"""

import pytest
import torch

from neurasp.neural.comparator import Comparator
from neurasp.neural.encoder import IntegerEncoder


class TestComparator:
    """Tests for the Comparator class."""
    
    def test_comparator_output_shape(self) -> None:
        """Test that comparator outputs correct shape."""
        comparator = Comparator(input_dim=64, hidden_dim=64)
        
        a = torch.randn(1, 64)
        b = torch.randn(1, 64)
        
        output = comparator(a, b)
        
        assert output.shape == (3,), f"Expected shape (3,), got {output.shape}"
    
    def test_comparator_probabilities_sum_to_one(self) -> None:
        """Test that output probabilities sum to 1."""
        comparator = Comparator(input_dim=64, hidden_dim=64)
        
        a = torch.randn(1, 64)
        b = torch.randn(1, 64)
        
        output = comparator(a, b)
        
        assert torch.isclose(output.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_comparator_all_positive(self) -> None:
        """Test that all probabilities are positive."""
        comparator = Comparator(input_dim=64, hidden_dim=64)
        
        a = torch.randn(1, 64)
        b = torch.randn(1, 64)
        
        output = comparator(a, b)
        
        assert (output >= 0).all()


class TestIntegerEncoder:
    """Tests for the IntegerEncoder class."""
    
    def test_encoder_output_shape(self) -> None:
        """Test that encoder outputs correct shape."""
        encoder = IntegerEncoder(embed_dim=64)
        
        x = torch.tensor([42])
        output = encoder(x)
        
        assert output.shape == (1, 64), f"Expected shape (1, 64), got {output.shape}"
    
    def test_encoder_batch(self) -> None:
        """Test encoder with batch input."""
        encoder = IntegerEncoder(embed_dim=64)
        
        x = torch.tensor([1, 2, 3, 4, 5])
        output = encoder(x)
        
        assert output.shape == (5, 64), f"Expected shape (5, 64), got {output.shape}"
    
    def test_encoder_handles_large_values(self) -> None:
        """Test that encoder handles values larger than max_value."""
        encoder = IntegerEncoder(embed_dim=64, max_value=100)
        
        x = torch.tensor([999])  # Larger than max_value
        output = encoder(x)
        
        assert output.shape == (1, 64)  # Should not crash
