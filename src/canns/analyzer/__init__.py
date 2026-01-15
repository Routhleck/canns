"""Analyzer utilities for inspecting CANNs models and simulations.

This namespace groups analysis helpers such as model metrics, visualization,
experimental or synthetic data analysis, slow-point (fixed-point) analysis,
and model-specific tools.

Examples:
    >>> from canns import analyzer
    >>> print(analyzer.__all__)
"""

from . import data, metrics, model_specific, slow_points, visualization

__all__ = [
    "metrics",
    "visualization",
    "data",
    "slow_points",
    "model_specific",
]
