"""
Brain-inspired neural network models.

This module contains biologically plausible neural network models that incorporate
principles from neuroscience and cognitive science, including associative memory,
Hebbian learning, and other brain-inspired mechanisms.
"""

from ._base import BrainInspiredModel, BrainInspiredModelGroup
from .bcm import BCMLayer
from .helmholtz import HelmholtzMachine
from .hopfield import AmariHopfieldNetwork
from .linear_hebb import LinearHebbLayer
from .rbm import RestrictedBoltzmannModel
from .spiking import LIFSpikingNetwork

__all__ = [
    # Base classes
    "BrainInspiredModel",
    "BrainInspiredModelGroup",
    # Specific models
    "AmariHopfieldNetwork",
    "LinearHebbLayer",
    "BCMLayer",
    "LIFSpikingNetwork",
    "RestrictedBoltzmannModel",
    "HelmholtzMachine",
]
