"""
Training utilities for CANNS models.

Currently exposes a unified ``HebbianTrainer`` with built-in progress reporting.
"""

from .hebbian import HebbianTrainer

__all__ = [
    "HebbianTrainer",
]
