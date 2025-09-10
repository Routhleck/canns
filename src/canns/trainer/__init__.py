"""
Training modules for CANNS models.

This module provides training utilities for different types of neural network models,
including brain-inspired learning algorithms and traditional optimization methods.
"""

from .hebbian import HebbianTrainer

__all__ = [
    "HebbianTrainer",
]
