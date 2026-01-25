"""
CANNs Pipeline Module

High-level pipelines for common analysis workflows, designed to make CANN models
accessible to experimental researchers without requiring detailed knowledge of
the underlying implementations.
"""

from ._base import Pipeline
from .asa import ASAApp, main as asa_main

__all__ = [
    "Pipeline",
    "ASAApp",
    "asa_main",
]
