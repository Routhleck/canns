"""Fixed point finder for BrainState RNN models.

This module provides tools for identifying and analyzing fixed points
in recurrent neural networks using JAX/BrainState.
"""

from .fixed_points import FixedPoints
from .finder import FixedPointFinder

__all__ = ["FixedPoints", "FixedPointFinder"]
