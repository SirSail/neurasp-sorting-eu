"""NeurASP Sorting Demo."""
from __future__ import annotations

import random
from typing import Final

DEMO_EPOCHS: Final[int] = 50
TEST_SAMPLES: Final[int] = 5
VALUE_RANGE: Final[tuple[int, int]] = (1, 100)


def main() -> None:
    from neurasp import NeurASPSorter
    from neurasp.core.types import SortingConfig

    print("=" * 60)
    print("NeurASP Sorting Demo - Team EU")
    print("=" * 60)
    print()

    config = SortingConfig(list_size=5, hidden_dim=64, learning_rate=0.001)
    sorter = NeurASPSorter(config)

    print("Training...")
    results = sorter.train(epochs=DEMO_EPOCHS, verbose=True)
    print(f"\nFinal loss: {results[-1][0]:.4f}\n")

    print("Testing:")
    for _ in range(TEST_SAMPLES):
        test_input = random.sample(range(*VALUE_RANGE), 5)
        result = sorter.sort(test_input)
        status = "✓" if result.state == "SAT" else "✗"
        print(f"  {list(result.input_list)} -> {list(result.sorted_list)} {status}")

    print("\nDone!")


if __name__ == "__main__":
    main()
