"""Data utilities for CANNs.

This namespace provides dataset registry, download helpers, and convenience
loaders for common CANNs datasets.

Examples:
    >>> from canns import data
    >>> print(list(data.DATASETS))
"""

from .datasets import (
    DATASETS,
    DEFAULT_DATA_DIR,
    HUGGINGFACE_REPO,
    download_dataset,
    get_data_dir,
    get_dataset_path,
    get_huggingface_upload_guide,
    get_left_right_data_session,
    get_left_right_npz,
    list_datasets,
    load,
    quick_setup,
)
from .loaders import load_grid_data, load_left_right_npz, load_roi_data

__all__ = [
    # Dataset registry and management
    "DATASETS",
    "HUGGINGFACE_REPO",
    "DEFAULT_DATA_DIR",
    "get_data_dir",
    "list_datasets",
    "download_dataset",
    "get_dataset_path",
    "get_left_right_data_session",
    "get_left_right_npz",
    "quick_setup",
    "get_huggingface_upload_guide",
    # Generic loading
    "load",
    # Specialized loaders
    "load_roi_data",
    "load_grid_data",
    "load_left_right_npz",
]
