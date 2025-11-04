"""
General utilities for CANNs.

This module provides general-purpose utilities that don't fit into specific
domain modules, such as benchmarking and performance measurement tools.
"""

from .benchmark import benchmark

__all__ = [
    "benchmark",
]
