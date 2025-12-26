"""Data analysis utilities for experimental and synthetic neural data.

This module provides tools for analyzing neural recordings (spike trains,
firing rates) and extracting features like bump positions, topological
structures, and population dynamics.
"""

from . import cann1d, cann2d

__all__ = [
    "cann1d",
    "cann2d",
]
