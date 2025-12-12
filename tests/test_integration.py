"""
Integration tests for NeurASPSorter.
"""

import pytest
import torch

from neurasp.core import NeurASPSorter, SortingConfig


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
    
    def test_train_step_returns_loss(self) -> None:
        """Test that train_step returns a valid loss."""
        sorter = NeurASPSorter()
        
        integers = [5, 2, 8, 1, 9]
        target = [1, 2, 5, 8, 9]
        
        loss = sorter.train_step(integers, target)
        
        assert isinstance(loss, float)
        assert loss >= 0
