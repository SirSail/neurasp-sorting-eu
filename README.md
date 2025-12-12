# neurasp-sorting-eu

> Neural-Symbolic AI with NeurASP: Learning to Sort a List of 5 Integers

**Team:** EU

**Based on:** [NeurASP: Embracing Neural Networks into Answer Set Programming](https://arxiv.org/abs/2307.07700) by Yang et al.

---

## Project Overview

This project demonstrates the application of **NeurASP** (Neural Answer Set Programming) to a classic symbolic-neural task: **learning to sort a list of 5 integers**. NeurASP combines:

- **Neural Networks** – for perception and pattern recognition
- **Answer Set Programming (ASP)** – for symbolic reasoning and constraint satisfaction

The sorting task is a canonical example that showcases how neural components can learn comparison operations while symbolic rules enforce the sorting constraints.

---

## Features

- **Hybrid Architecture:**
  - Neural network learns pairwise comparison: `compare(x, y) → {less, equal, greater}`
  - ASP rules define sorting constraints and permutation validity

- **End-to-End Training:**
  - Joint optimization of neural and symbolic components
  - Gradient-based learning through differentiable ASP

- **Interpretable Results:**
  - Symbolic rules provide explainable reasoning
  - Neural weights are isolated to comparison logic

---

## Installation

### Requirements

- Python >= 3.10
- [PyTorch](https://pytorch.org/) (CPU is enough)
- [Clingo](https://potassco.org/clingo/) (ASP solver)
- `clingo` Python bindings (`pip install clingo`)
- `pytest` (for tests)

### Setup

```bash
# Clone the repository
git clone https://github.com/EU/neurasp-sorting-eu.git
cd neurasp-sorting-eu

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## Quick Start

### Example: Sorting 5 integers

```python
from neurasp import NeurASPSorter

# Initialize the sorter
sorter = NeurASPSorter()

# Train on random examples
sorter.train(epochs=100)

# Test sorting
result = sorter.sort([5, 2, 8, 1, 9])
print(result)  # [1, 2, 5, 8, 9]
```

---

## Running demos

```bash
# Main sorting demo
python -m neurasp.demos.sorting_demo

# Training visualization
python -m neurasp.demos.training_viz
```

---

## Running tests

```bash
pytest tests/
```

---

## Project Structure

```
neurasp-sorting-eu/
├── src/neurasp/
│   ├── __init__.py          # Package metadata
│   ├── asp/
│   │   ├── sorting.lp       # ASP rules for sorting
│   │   └── constraints.lp   # Constraint definitions
│   ├── neural/
│   │   ├── comparator.py    # Neural comparison network
│   │   └── encoder.py       # Integer encoding
│   ├── core.py              # NeurASP integration
│   └── demos/
│       ├── sorting_demo.py  # Main demonstration
│       └── training_viz.py  # Training visualization
├── tests/
│   ├── test_comparator.py   # Neural component tests
│   ├── test_asp_rules.py    # ASP logic tests
│   └── test_integration.py  # End-to-end tests
├── data/
│   └── training/            # Training datasets
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Architecture

### Neural Component

The neural network learns pairwise integer comparison:

```
Input: (a, b) encoded as one-hot or embedding
Output: P(a < b), P(a = b), P(a > b)
```

### Symbolic Component (ASP)

ASP rules define sorting semantics:

```prolog
% A permutation is a valid sort if every adjacent pair is ordered
sorted :- ordered(1..4).
ordered(I) :- position(X, I), position(Y, I+1), less(X, Y).

% Neural predicate: less(X, Y) is true with probability from NN
#neural less(X, Y) : comparator [X, Y].
```

---

## NeurASP in a Nutshell

NeurASP extends Answer Set Programming to incorporate neural network predictions as probabilistic facts:

1. **Neural Predicates** – Neural networks output probabilities for logical atoms
2. **Stable Model Semantics** – ASP computes stable models using these probabilities
3. **Semantic Loss** – Gradient flows through the probability of desired stable models
4. **Joint Optimization** – Neural weights are updated to maximize desired outcomes

This allows learning from high-level specifications (e.g., "the output should be sorted") rather than explicit labels for each neural component.

---

## References

- Yang, Z., Ishay, A., & Lee, J. (2023). *NeurASP: Embracing Neural Networks into Answer Set Programming*. arXiv:2307.07700. [[PDF]](https://arxiv.org/abs/2307.07700)
- Lifschitz, V. (2019). *Answer Set Programming*. Springer.

---

## License

MIT

---

## Author

Team EU - [GitHub](https://github.com/EU)

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/EU/neurasp-sorting-eu/issues)
