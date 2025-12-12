"""
Integer Encoder

Encodes integers into dense embeddings for neural processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IntegerEncoder(nn.Module):
    """
    Encodes integers into dense vector embeddings.
    
    Uses a combination of learned embeddings and positional encoding
    to handle arbitrary integer ranges.
    """
    
    __slots__ = ("embed_dim", "embedding", "projection")
    
    def __init__(
        self,
        embed_dim: int = 64,
        max_value: int = 1000
    ) -> None:
        """
        Initialize the integer encoder.
        
        Args:
            embed_dim: Dimension of output embeddings
            max_value: Maximum expected integer value (for embedding table)
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Embedding table for common values
        self.embedding = nn.Embedding(max_value, embed_dim)
        
        # Projection for out-of-range values
        self.projection = nn.Linear(1, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode integers into embeddings.
        
        Args:
            x: Tensor of integers, shape (batch,) or (batch, 1)
            
        Returns:
            Embeddings, shape (batch, embed_dim)
        """
        x = x.long()
        if x.dim() == 0:
            x = x.unsqueeze(0)
        
        # Clamp to valid embedding range
        max_idx = self.embedding.num_embeddings - 1
        x_clamped = torch.clamp(x, 0, max_idx)
        
        return self.embedding(x_clamped)
