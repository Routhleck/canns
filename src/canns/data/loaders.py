"""
Experimental data processing utilities for CANNs.

This module provides specialized functions for processing experimental data
typically used in CANN analyses, including ROI data, grid cell data, and
other neurophysiological _datasets.
"""

from pathlib import Path
from typing import Any

import numpy as np

from . import datasets as _datasets


def load_roi_data(source: str | Path | None = None) -> np.ndarray | None:
    """
    Load ROI data for 1D CANN analysis.

    Parameters
    ----------
    source : str, Path, or None
        Data source. Can be:
        - URL string: downloads and loads from URL
        - Path: loads from local file
        - None: uses default CANNs dataset

    Returns
    -------
    ndarray or None
        ROI data array if successful, None otherwise.

    Examples
    --------
    >>> # Load default dataset
    >>> roi_data = load_roi_data()
    >>>
    >>> # Load from URL
    >>> roi_data = load_roi_data('https://example.com/roi_data.txt')
    >>>
    >>> # Load from local file
    >>> roi_data = load_roi_data('./my_roi_data.txt')
    """

    # Handle different source types
    if source is None:
        # Use default CANNs dataset
        dataset_path = _datasets.get_dataset_path("roi_data")
        if dataset_path is None:
            return None

        try:
            data = np.loadtxt(dataset_path)
            print(f"Loaded ROI data: shape {data.shape}")
            return data
        except Exception as e:
            print(f"Failed to load ROI data: {e}")
            return None

    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        # Load from URL
        try:
            data = _datasets.load(source, file_type="text")
            if isinstance(data, str):
                # If loaded as text, try to parse as numpy array
                lines = data.strip().split("\n")
                data = np.array([[float(x) for x in line.split()] for line in lines])
            print(f"Loaded ROI data from URL: shape {data.shape}")
            return data
        except Exception as e:
            print(f"Failed to load ROI data from URL: {e}")
            return None

    else:
        # Load from local path
        source_path = Path(source)
        if not source_path.exists():
            print(f"File not found: {source_path}")
            return None

        try:
            data = np.loadtxt(source_path)
            print(f"Loaded ROI data from {source_path}: shape {data.shape}")
            return data
        except Exception as e:
            print(f"Failed to load ROI data from {source_path}: {e}")
            return None


def load_grid_data(
    source: str | Path | None = None, dataset_key: str = "grid_1"
) -> dict[str, Any] | None:
    """
    Load grid cell data for 2D CANN analysis.

    Parameters
    ----------
    source : str, Path, or None
        Data source. Can be:
        - URL string: downloads and loads from URL
        - Path: loads from local file
        - None: uses default CANNs dataset
    dataset_key : str
        Which default dataset to use ('grid_1' or 'grid_2') when source is None.

    Returns
    -------
    dict or None
        Dictionary containing spike data and metadata if successful, None otherwise.
        Expected keys: 'spike', 't', and optionally 'x', 'y' for position data.

    Examples
    --------
    >>> # Load default dataset
    >>> grid_data = load_grid_data()
    >>>
    >>> # Load from URL
    >>> grid_data = load_grid_data('https://example.com/grid_data.npz')
    >>>
    >>> # Load specific default dataset
    >>> grid_data = load_grid_data(dataset_key='grid_2')
    """

    # Handle different source types
    if source is None:
        # Use default CANNs dataset
        dataset_path = _datasets.get_dataset_path(dataset_key)
        if dataset_path is None:
            return None

        try:
            data = np.load(dataset_path, allow_pickle=True)
            result = {
                "spike": data["spike"],
                "t": data["t"],
            }

            # Add position data if available
            if "x" in data:
                result["x"] = data["x"]
            if "y" in data:
                result["y"] = data["y"]

            # Handle different spike data formats
            if hasattr(result["spike"], "item") and isinstance(result["spike"].item(), dict):
                # Spike data is stored as a dictionary inside numpy array
                spike_dict = result["spike"].item()
                print(f"Loaded {dataset_key}: {len(spike_dict)} neurons")
            else:
                print(f"Loaded {dataset_key}: {len(result['spike'])} neurons")
            if "x" in result:
                print(f"Position data available: {len(result['x'])} time points")

            return result
        except Exception as e:
            print(f"Failed to load {dataset_key}: {e}")
            return None

    elif isinstance(source, str) and source.startswith(("http://", "https://")):
        # Load from URL
        try:
            data = _datasets.load(source, file_type="numpy")
            if isinstance(data, dict):
                result = {}
                if "spike" in data:
                    result["spike"] = data["spike"]
                if "t" in data:
                    result["t"] = data["t"]
                if "x" in data:
                    result["x"] = data["x"]
                if "y" in data:
                    result["y"] = data["y"]

                print(f"Loaded grid data from URL: {len(result.get('spike', []))} neurons")
                return result
            else:
                print("Grid data must be in .npz format with 'spike' and 't' arrays")
                return None
        except Exception as e:
            print(f"Failed to load grid data from URL: {e}")
            return None

    else:
        # Load from local path
        source_path = Path(source)
        if not source_path.exists():
            print(f"File not found: {source_path}")
            return None

        try:
            data = np.load(source_path, allow_pickle=True)
            result = {
                "spike": data["spike"],
                "t": data["t"],
            }

            # Add position data if available
            if "x" in data:
                result["x"] = data["x"]
            if "y" in data:
                result["y"] = data["y"]

            print(f"Loaded grid data from {source_path}: {len(result['spike'])} neurons")
            if "x" in result:
                print(f"Position data available: {len(result['x'])} time points")

            return result
        except Exception as e:
            print(f"Failed to load grid data from {source_path}: {e}")
            return None


def validate_roi_data(data: np.ndarray) -> bool:
    """
    Validate ROI data format for 1D CANN analysis.

    Parameters
    ----------
    data : ndarray
        ROI data array.

    Returns
    -------
    bool
        True if data is valid, False otherwise.
    """

    if not isinstance(data, np.ndarray):
        print("ROI data must be a numpy array")
        return False

    if data.ndim not in [1, 2]:
        print(f"ROI data must be 1D or 2D, got {data.ndim}D")
        return False

    if data.size == 0:
        print("ROI data is empty")
        return False

    if not np.isfinite(data).all():
        print("ROI data contains non-finite values")
        return False

    return True


def validate_grid_data(data: dict[str, Any]) -> bool:
    """
    Validate grid data format for 2D CANN analysis.

    Parameters
    ----------
    data : dict
        Grid data dictionary.

    Returns
    -------
    bool
        True if data is valid, False otherwise.
    """

    if not isinstance(data, dict):
        print("Grid data must be a dictionary")
        return False

    # Check required keys
    required_keys = ["spike", "t"]
    for key in required_keys:
        if key not in data:
            print(f"Grid data missing required key: {key}")
            return False

    # Validate spike data
    spike_data = data["spike"]
    if not isinstance(spike_data, list | np.ndarray):
        print("Spike data must be a list or numpy array")
        return False

    if len(spike_data) == 0:
        print("Spike data is empty")
        return False

    # Validate time data
    t_data = data["t"]
    if not isinstance(t_data, np.ndarray):
        print("Time data must be a numpy array")
        return False

    if t_data.size == 0:
        print("Time data is empty")
        return False

    if not np.isfinite(t_data).all():
        print("Time data contains non-finite values")
        return False

    # Validate position data if present
    for pos_key in ["x", "y"]:
        if pos_key in data:
            pos_data = data[pos_key]
            if not isinstance(pos_data, np.ndarray):
                print(f"Position data '{pos_key}' must be a numpy array")
                return False

            if pos_data.size == 0:
                print(f"Position data '{pos_key}' is empty")
                return False

            if not np.isfinite(pos_data).all():
                print(f"Position data '{pos_key}' contains non-finite values")
                return False

            if pos_data.shape != t_data.shape:
                print(
                    f"Position data '{pos_key}' shape {pos_data.shape} doesn't match time data shape {t_data.shape}"
                )
                return False

    return True


def preprocess_spike_data(
    spike_data: list | np.ndarray,
    time_window: tuple[float, float] | None = None,
    min_spike_count: int = 10,
) -> np.ndarray | None:
    """
    Preprocess spike data for analysis.

    Parameters
    ----------
    spike_data : list or ndarray
        Raw spike data.
    time_window : tuple, optional
        (start, end) time window to filter spikes.
    min_spike_count : int
        Minimum number of spikes required per neuron.

    Returns
    -------
    ndarray or None
        Processed spike data, or None if processing fails.
    """

    # Convert to numpy array if needed
    if isinstance(spike_data, list):
        spike_data = np.array(spike_data, dtype=object)

    if spike_data.size == 0:
        print("Spike data is empty")
        return None

    # Filter by time window if specified
    if time_window is not None:
        start_time, end_time = time_window
        filtered_spikes = []

        for neuron_spikes in spike_data:
            if len(neuron_spikes) > 0:
                mask = (neuron_spikes >= start_time) & (neuron_spikes <= end_time)
                filtered_spikes.append(neuron_spikes[mask])
            else:
                filtered_spikes.append(np.array([]))

        spike_data = np.array(filtered_spikes, dtype=object)

    # Filter neurons with insufficient spikes
    valid_neurons = []
    for neuron_spikes in spike_data:
        if len(neuron_spikes) >= min_spike_count:
            valid_neurons.append(neuron_spikes)

    if len(valid_neurons) == 0:
        print(f"No neurons meet minimum spike count requirement ({min_spike_count})")
        return None

    if len(valid_neurons) < len(spike_data):
        print(f"Filtered {len(spike_data) - len(valid_neurons)} neurons with insufficient spikes")

    return np.array(valid_neurons, dtype=object)


def get_data_summary(data: np.ndarray | dict[str, Any]) -> dict[str, Any]:
    """
    Get summary statistics for experimental data.

    Parameters
    ----------
    data : ndarray or dict
        ROI data (ndarray) or grid data (dict).

    Returns
    -------
    dict
        Summary statistics.
    """

    summary = {}

    if isinstance(data, np.ndarray):
        # ROI data summary
        summary["type"] = "roi_data"
        summary["shape"] = data.shape
        summary["size"] = data.size
        summary["dtype"] = str(data.dtype)
        summary["min"] = float(np.min(data))
        summary["max"] = float(np.max(data))
        summary["mean"] = float(np.mean(data))
        summary["std"] = float(np.std(data))
        summary["has_nan"] = bool(np.isnan(data).any())
        summary["has_inf"] = bool(np.isinf(data).any())

    elif isinstance(data, dict):
        # Grid data summary
        summary["type"] = "grid_data"
        summary["keys"] = list(data.keys())

        if "spike" in data:
            spike_data = data["spike"]
            summary["n_neurons"] = len(spike_data)

            spike_counts = [len(neuron_spikes) for neuron_spikes in spike_data]
            summary["spike_counts"] = {
                "min": int(np.min(spike_counts)),
                "max": int(np.max(spike_counts)),
                "mean": float(np.mean(spike_counts)),
                "total": int(np.sum(spike_counts)),
            }

        if "t" in data:
            t_data = data["t"]
            summary["time_data"] = {
                "length": len(t_data),
                "duration": float(t_data.max() - t_data.min()),
                "sampling_rate": float(1.0 / np.mean(np.diff(t_data))) if len(t_data) > 1 else None,
            }

        for pos_key in ["x", "y"]:
            if pos_key in data:
                pos_data = data[pos_key]
                summary[f"{pos_key}_data"] = {
                    "min": float(np.min(pos_data)),
                    "max": float(np.max(pos_data)),
                    "range": float(np.max(pos_data) - np.min(pos_data)),
                }

    else:
        summary["type"] = "unknown"
        summary["error"] = "Unsupported data type"

    return summary
