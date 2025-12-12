"""
NeurASP - Neural Answer Set Programming for Sorting

A neural-symbolic AI framework combining:
- Neural networks for learning comparison operations
- Answer Set Programming for symbolic reasoning

Team: EU
Task: Ordering a list of 5 integers
"""

__version__ = "0.1.0"
__author__ = "Team EU"

from neurasp.core import NeurASPSorter
from neurasp.neural.comparator import Comparator
from neurasp.neural.encoder import IntegerEncoder

__all__ = [
    "NeurASPSorter",
    "Comparator",
    "IntegerEncoder",
]
