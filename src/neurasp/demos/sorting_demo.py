"""
NeurASP Sorting Demo

Demonstrates the use of NeurASP for learning to sort a list of 5 integers.
"""

from __future__ import annotations

import random


def main() -> None:
    """Run the sorting demonstration."""
    from neurasp import NeurASPSorter
    from neurasp.core import SortingConfig
    
    print("=" * 60)
    print("NeurASP Sorting Demo - Team EU")
    print("Task: Learning to sort a list of 5 integers")
    print("=" * 60)
    print()
    
    # Initialize the sorter
    config = SortingConfig(
        list_size=5,
        hidden_dim=64,
        learning_rate=0.001
    )
    sorter = NeurASPSorter(config)
    
    print("Training the neural-symbolic sorter...")
    print("-" * 40)
    
    # Train
    losses = sorter.train(epochs=50, verbose=True)
    
    print()
    print("-" * 40)
    print("Training complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print()
    
    # Test on random examples
    print("Testing on random examples:")
    print("-" * 40)
    
    for i in range(5):
        test_input = random.sample(range(1, 100), 5)
        expected = sorted(test_input)
        result = sorter.sort(test_input)
        
        status = "✓" if result == expected else "✗"
        print(f"  Input:    {test_input}")
        print(f"  Output:   {result}")
        print(f"  Expected: {expected}  {status}")
        print()
    
    print("=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
