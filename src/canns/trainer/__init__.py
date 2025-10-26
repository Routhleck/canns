"""
Training utilities for CANNS models.

The module exposes the abstract ``Trainer`` base class and concrete implementations
such as ``HebbianTrainer`` and ``AntiHebbianTrainer``.
"""

from ._base import Trainer
from .hebbian import AntiHebbianTrainer, HebbianTrainer

__all__ = [
    "Trainer",
    "HebbianTrainer",
    "AntiHebbianTrainer",
]
