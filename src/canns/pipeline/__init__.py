"""
CANNS Pipeline Module

High-level pipelines for common analysis workflows, designed to make CANN models
accessible to experimental researchers without requiring detailed knowledge of
the underlying implementations.
"""

from .theta_sweep import (
    ThetaSweepPipeline,
    batch_process_trajectories,
    load_trajectory_from_csv,
)

__all__ = [
    "ThetaSweepPipeline",
    "load_trajectory_from_csv",
    "batch_process_trajectories",
]
