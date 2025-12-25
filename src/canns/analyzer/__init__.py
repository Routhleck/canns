"""Analyzer utilities for inspecting CANNs models and simulations.

NEW STRUCTURE:
├── metrics/           - Computational analysis (no matplotlib)
├── visualization/     - Plotting and animation (matplotlib-based)
├── slow_points/       - Fixed point analysis
└── model_specific/    - Specialized model analyzers
"""

from . import metrics, model_specific, slow_points, visualization

__all__ = [
    "metrics",
    "visualization",
    "slow_points",
    "model_specific",
]
