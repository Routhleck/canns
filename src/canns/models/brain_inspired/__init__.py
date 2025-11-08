"""
Brain-inspired neural network models.

This module contains biologically plausible neural network models that incorporate
principles from neuroscience and cognitive science, including associative memory,
Hebbian learning, and other brain-inspired mechanisms.
"""

from ._base import BrainInspiredModel, BrainInspiredModelGroup
from .hopfield import AmariHopfieldNetwork
from .linear import LinearLayer
from .spiking import SpikingLayer

__all__ = [
    # Base classes
    "BrainInspiredModel",
    "BrainInspiredModelGroup",
    # Specific models
    "AmariHopfieldNetwork",
    "LinearLayer",
    "SpikingLayer",
]
