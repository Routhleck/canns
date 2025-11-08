"""
Training utilities for CANNs models.

The module exposes the abstract ``Trainer`` base class and concrete implementations
of classic brain-inspired learning algorithms: ``HebbianTrainer``, ``AntiHebbianTrainer``,
``OjaTrainer``, ``BCMTrainer``, ``SangerTrainer``, and ``STDPTrainer``.
"""

from ._base import Trainer
from .bcm import BCMTrainer
from .hebbian import AntiHebbianTrainer, HebbianTrainer
from .oja import OjaTrainer
from .sanger import SangerTrainer
from .stdp import STDPTrainer

__all__ = [
    "Trainer",
    "HebbianTrainer",
    "AntiHebbianTrainer",
    "OjaTrainer",
    "BCMTrainer",
    "SangerTrainer",
    "STDPTrainer",
]
