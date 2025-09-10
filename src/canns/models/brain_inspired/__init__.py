"""
Brain-inspired neural network models.

This module contains biologically plausible neural network models that incorporate
principles from neuroscience and cognitive science, including associative memory,
Hebbian learning, and other brain-inspired mechanisms.
"""

from ._base import BrainInspiredGroup, BrainInspiredModel
from .hopfield import AmariHopfieldNetwork

__all__ = [
    # Base classes
    "BrainInspiredModel",
    "BrainInspiredGroup",
    # Specific models
    "AmariHopfieldNetwork",
]
