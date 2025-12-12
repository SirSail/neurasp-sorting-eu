"""
Neural Comparator Network

Learns pairwise integer comparison: given (a, b), outputs
probabilities for the relations: less, equal, greater.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Comparator(nn.Module):
    """
    Neural network for pairwise integer comparison.
    
    Input: Two integer embeddings (a, b)
    Output: Probability distribution over {less, equal, greater}
    """
    
    __slots__ = ("fc1", "fc2", "fc3")
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 64) -> None:
        """
        Initialize the comparator network.
        
        Args:
            input_dim: Dimension of integer embeddings
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Concatenate two embeddings -> 2 * input_dim
        self.fc1 = nn.Linear(2 * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # less, equal, greater
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compare two integers based on their embeddings.
        
        Args:
            a: Embedding of first integer, shape (batch, embed_dim)
            b: Embedding of second integer, shape (batch, embed_dim)
            
        Returns:
            Probabilities for [less, equal, greater], shape (batch, 3)
        """
        # Concatenate embeddings
        x = torch.cat([a, b], dim=-1)
        
        # MLP with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Softmax for probabilities
        return F.softmax(x, dim=-1).squeeze(0)
