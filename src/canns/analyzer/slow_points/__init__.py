"""Fixed point finder for BrainPy RNN models.

This module provides tools for identifying and analyzing fixed points
in recurrent neural networks using JAX/BrainPy.
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .finder import FixedPointFinder
from .fixed_points import FixedPoints
from .visualization import plot_fixed_points_2d, plot_fixed_points_3d

__all__ = [
    "FixedPoints",
    "FixedPointFinder",
    "save_checkpoint",
    "load_checkpoint",
    "plot_fixed_points_2d",
    "plot_fixed_points_3d",
]
