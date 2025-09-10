"""
Training modules for CANNS models.

This module provides training utilities for different types of neural network models,
including brain-inspired learning algorithms and traditional optimization methods.
"""

from .hebbian import HebbianTrainer
from .progress import (
    ProgressReporter,
    SilentProgressReporter,
    TqdmProgressReporter,
    create_progress_reporter,
)

__all__ = [
    "HebbianTrainer",
    "ProgressReporter",
    "SilentProgressReporter",
    "TqdmProgressReporter",
    "create_progress_reporter",
]
