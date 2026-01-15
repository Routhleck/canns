"""Training utilities for CANNs models.

This namespace exposes the abstract ``Trainer`` base class and concrete
implementations of classic brain-inspired learning algorithms such as
``HebbianTrainer`` and ``OjaTrainer``.

Examples:
    >>> from canns import trainer
    >>> print(trainer.Trainer)
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
