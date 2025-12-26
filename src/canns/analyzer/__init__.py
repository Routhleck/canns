"""Analyzer utilities for inspecting CANNs models and simulations.

NEW STRUCTURE:
├── metrics/           - Model metrics computation
├── visualization/     - Model visualization and animation
├── data/              - Data analysis (experimental & synthetic)
├── slow_points/       - Fixed point analysis
└── model_specific/    - Specialized model analyzers
"""

from . import data, metrics, model_specific, slow_points, visualization

__all__ = [
    "metrics",
    "visualization",
    "data",
    "slow_points",
    "model_specific",
]
