"""
NeurASP Core - Integration of Neural Networks with Answer Set Programming

This module provides the main NeurASPSorter class that combines:
- Neural comparator network for pairwise integer comparison
- ASP solver for enforcing sorting constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import clingo
import torch
import torch.nn as nn

from neurasp.neural.comparator import Comparator
from neurasp.neural.encoder import IntegerEncoder

# Constants
ASP_DIR: Final[Path] = Path(__file__).parent / "asp"
SORTING_RULES: Final[Path] = ASP_DIR / "sorting.lp"
LIST_SIZE: Final[int] = 5


@dataclass
class SortingConfig:
    """Configuration for the NeurASP sorter."""
    
    list_size: int = LIST_SIZE
    hidden_dim: int = 64
    learning_rate: float = 0.001
    device: str = "cpu"


class NeurASPSorter(nn.Module):
    """
    Neural-Symbolic sorter using NeurASP.
    
    Combines a neural comparator network with ASP rules
    to learn sorting of integer lists.
    """
    
    __slots__ = ("config", "encoder", "comparator", "optimizer", "_asp_rules")
    
    def __init__(self, config: SortingConfig | None = None) -> None:
        super().__init__()
        self.config = config or SortingConfig()
        
        # Neural components
        self.encoder = IntegerEncoder(embed_dim=self.config.hidden_dim)
        self.comparator = Comparator(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate
        )
        
        # Load ASP rules
        self._asp_rules = self._load_asp_rules()
    
    def _load_asp_rules(self) -> str:
        """Load ASP sorting rules from file."""
        if SORTING_RULES.exists():
            return SORTING_RULES.read_text(encoding="utf-8")
        return ""
    
    def compare(self, a: int, b: int) -> torch.Tensor:
        """
        Compare two integers using the neural comparator.
        
        Returns:
            Tensor of shape (3,) with probabilities for [less, equal, greater]
        """
        a_emb = self.encoder(torch.tensor([a]))
        b_emb = self.encoder(torch.tensor([b]))
        return self.comparator(a_emb, b_emb)
    
    def forward(self, integers: list[int]) -> torch.Tensor:
        """
        Compute comparison probabilities for all pairs.
        
        Args:
            integers: List of integers to sort
            
        Returns:
            Tensor of shape (n, n, 3) with pairwise comparison probabilities
        """
        n = len(integers)
        embeddings = self.encoder(torch.tensor(integers))
        
        # Compute all pairwise comparisons
        comparisons = torch.zeros(n, n, 3)
        for i in range(n):
            for j in range(n):
                if i != j:
                    comparisons[i, j] = self.comparator(
                        embeddings[i:i+1],
                        embeddings[j:j+1]
                    )
        
        return comparisons
    
    def solve_asp(self, comparison_probs: torch.Tensor) -> list[int]:
        """
        Use ASP to find the sorted permutation given comparison probabilities.
        
        Args:
            comparison_probs: Tensor of shape (n, n, 3) with probabilities
            
        Returns:
            List of indices representing the sorted order
        """
        n = comparison_probs.shape[0]
        
        # Build ASP program with neural predicates
        program = self._asp_rules + "\n"
        
        # Add neural facts as weighted choices
        for i in range(n):
            for j in range(n):
                if i != j:
                    less_prob = comparison_probs[i, j, 0].item()
                    if less_prob > 0.5:
                        program += f"less({i}, {j}).\n"
        
        # Solve
        ctl = clingo.Control()
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        
        result: list[int] = list(range(n))
        
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                # Extract position assignments
                for atom in model.symbols(shown=True):
                    if atom.name == "position":
                        idx, pos = atom.arguments
                        result[pos.number] = idx.number
                break
        
        return result
    
    def sort(self, integers: list[int]) -> list[int]:
        """
        Sort a list of integers using NeurASP.
        
        Args:
            integers: List of integers to sort
            
        Returns:
            Sorted list of integers
        """
        with torch.no_grad():
            comparison_probs = self.forward(integers)
            order = self.solve_asp(comparison_probs)
            return [integers[i] for i in order]
    
    def train_step(
        self,
        integers: list[int],
        target: list[int]
    ) -> float:
        """
        Perform one training step.
        
        Args:
            integers: Input list of integers
            target: Expected sorted output
            
        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        comparison_probs = self.forward(integers)
        
        # Compute semantic loss based on target ordering
        loss = self._compute_semantic_loss(
            integers, target, comparison_probs
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _compute_semantic_loss(
        self,
        integers: list[int],
        target: list[int],
        comparison_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute semantic loss based on target ordering.
        
        The loss encourages the neural comparator to output
        probabilities consistent with the correct ordering.
        """
        loss = torch.tensor(0.0, requires_grad=True)
        n = len(integers)
        
        # Build index mapping
        int_to_target_pos = {v: i for i, v in enumerate(target)}
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    pos_i = int_to_target_pos[integers[i]]
                    pos_j = int_to_target_pos[integers[j]]
                    
                    if pos_i < pos_j:
                        # i should be before j, so less(i, j) should be high
                        loss = loss - torch.log(comparison_probs[i, j, 0] + 1e-8)
                    elif pos_i > pos_j:
                        # i should be after j, so greater(i, j) should be high
                        loss = loss - torch.log(comparison_probs[i, j, 2] + 1e-8)
        
        return loss / (n * (n - 1))
    
    def train(self, epochs: int = 100, verbose: bool = True) -> list[float]:
        """
        Train on random examples.
        
        Args:
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            List of loss values per epoch
        """
        import random
        
        losses: list[float] = []
        
        for epoch in range(epochs):
            # Generate random example
            integers = random.sample(range(1, 100), self.config.list_size)
            target = sorted(integers)
            
            # Train step
            loss = self.train_step(integers, target)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
