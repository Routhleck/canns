"""
Training utilities for CANNS models.

The module exposes the abstract ``Trainer`` base class and concrete implementations
such as ``HebbianTrainer``, ``AntiHebbianTrainer``, ``OjaTrainer``, ``BCMTrainer``,
``STDPTrainer``, ``HopfieldEnergyTrainer``, ``ContrastiveDivergenceTrainer``,
and ``WakeSleepTrainer``.
"""

from ._base import Trainer
from .bcm import BCMTrainer
from .contrastive_divergence import ContrastiveDivergenceTrainer
from .hebbian import AntiHebbianTrainer, HebbianTrainer
from .hopfield_energy import HopfieldEnergyTrainer
from .oja import OjaTrainer
from .stdp import STDPTrainer
from .wake_sleep import WakeSleepTrainer

__all__ = [
    "Trainer",
    "HebbianTrainer",
    "AntiHebbianTrainer",
    "OjaTrainer",
    "BCMTrainer",
    "STDPTrainer",
    "HopfieldEnergyTrainer",
    "ContrastiveDivergenceTrainer",
    "WakeSleepTrainer",
]
