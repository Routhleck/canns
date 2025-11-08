"""
Data utilities for CANNs.

This module provides dataset management, loading, and downloading utilities.
It consolidates data-related functionality previously scattered across the codebase.
"""

from .datasets import (
    DATASETS,
    DEFAULT_DATA_DIR,
    HUGGINGFACE_REPO,
    download_dataset,
    get_data_dir,
    get_dataset_path,
    get_huggingface_upload_guide,
    list_datasets,
    load,
    quick_setup,
)
from .loaders import load_grid_data, load_roi_data

__all__ = [
    # Dataset registry and management
    "DATASETS",
    "HUGGINGFACE_REPO",
    "DEFAULT_DATA_DIR",
    "get_data_dir",
    "list_datasets",
    "download_dataset",
    "get_dataset_path",
    "quick_setup",
    "get_huggingface_upload_guide",
    # Generic loading
    "load",
    # Specialized loaders
    "load_roi_data",
    "load_grid_data",
]
