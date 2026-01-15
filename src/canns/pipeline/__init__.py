"""
CANNs Pipeline Module

High-level pipelines for common analysis workflows, designed to make CANN models
accessible to experimental researchers without requiring detailed knowledge of
the underlying implementations.

This module provides ready-to-use pipelines that orchestrate multi-step workflows
including data preparation, model execution, and visualization. Pipelines abstract
away implementation complexity while maintaining scientific rigor.

Available Pipelines
-------------------
- **Pipeline**: Abstract base class for creating custom pipelines
- **ThetaSweepPipeline**: Theta oscillation analysis for spatial navigation data

Convenience Functions
---------------------
- **load_trajectory_from_csv**: Load and analyze trajectory data from CSV files
- **batch_process_trajectories**: Process multiple trajectories in parallel

Example:
    Basic usage with trajectory data:
    
    >>> import numpy as np
    >>> from canns.pipeline import ThetaSweepPipeline
    >>> 
    >>> # Create synthetic trajectory (circular path)
    >>> t = np.linspace(0, 4*np.pi, 200)
    >>> trajectory = np.column_stack([
    ...     np.cos(t) * 0.5 + 0.5,  # x coordinates
    ...     np.sin(t) * 0.5 + 0.5   # y coordinates
    >>> ])
    >>> 
    >>> # Run theta sweep analysis
    >>> pipeline = ThetaSweepPipeline(trajectory, env_size=2.0)
    >>> results = pipeline.run(output_dir="my_analysis", verbose=True)
    >>> print(f"Animation saved to: {results['animation_path']}")

    Load from CSV file:
    
    >>> from canns.pipeline import load_trajectory_from_csv
    >>> 
    >>> # Load and analyze trajectory from CSV
    >>> results = load_trajectory_from_csv(
    ...     "path/to/trajectory.csv",
    ...     x_col="x_position",
    ...     y_col="y_position",
    ...     time_col="timestamp"
    ... )
"""

from ._base import Pipeline
from .theta_sweep import (
    ThetaSweepPipeline,
    batch_process_trajectories,
    load_trajectory_from_csv,
)

__all__ = [
    "Pipeline",
    "ThetaSweepPipeline",
    "load_trajectory_from_csv",
    "batch_process_trajectories",
]
