"""Shared utility functions for ASA (Attractor State Analysis) modules."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ...visualization.core import PlotConfig
from .path import find_coords_matrix, find_times_box


def _ensure_plot_config(
    config: PlotConfig | None,
    factory,
    *args,
    kwargs: dict | None = None,
    **defaults,
) -> PlotConfig:
    """Ensure a PlotConfig exists, creating one from factory if needed.

    Args:
        config: Optional existing PlotConfig.
        factory: Factory function to create PlotConfig if config is None.
        *args: Positional arguments for factory.
        kwargs: Optional dict to merge into config.kwargs.
        **defaults: Keyword arguments for factory.

    Returns:
        PlotConfig instance.
    """
    if config is None:
        if kwargs:
            defaults.update({"kwargs": kwargs})
        return factory(*args, **defaults)

    # If config exists and kwargs provided, merge them
    if kwargs:
        config_kwargs = config.kwargs or {}
        config_kwargs.update(kwargs)
        config.kwargs = config_kwargs
    return config


def _ensure_parent_dir(save_path: str | None) -> None:
    """Create parent directory for save_path if it doesn't exist.

    Args:
        save_path: Optional file path. If provided, creates parent directory.
    """
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def _circmean(x: np.ndarray) -> float:
    """Compute circular mean of angles.

    Args:
        x: Array of angles in radians.

    Returns:
        Circular mean in radians.
    """
    return float(np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x))))


def _smooth_circular_map(
    mtot: np.ndarray,
    smooth_sigma: float,
    *,
    fill_nan: bool = False,
    fill_sigma: float | None = None,
    fill_min_weight: float = 1e-3,
) -> np.ndarray:
    """Smooth a circular phase map using Gaussian filtering in sin/cos space.

    Args:
        mtot: Phase map array (angles in radians).
        smooth_sigma: Gaussian smoothing sigma.
        fill_nan: Whether to fill NaN values using weighted interpolation.
        fill_sigma: Sigma for NaN filling (defaults to smooth_sigma).
        fill_min_weight: Minimum weight threshold for valid interpolation.

    Returns:
        Smoothed phase map.
    """
    mtot = np.asarray(mtot, dtype=float)
    nans = np.isnan(mtot)
    mask = (~nans).astype(float)
    sintot = np.sin(mtot)
    costot = np.cos(mtot)
    sintot[nans] = 0.0
    costot[nans] = 0.0

    if fill_nan:
        if fill_sigma is None:
            fill_sigma = smooth_sigma if smooth_sigma and smooth_sigma > 0 else 1.0
        weight = gaussian_filter(mask, fill_sigma)
        sintot = gaussian_filter(sintot * mask, fill_sigma)
        costot = gaussian_filter(costot * mask, fill_sigma)
        min_weight = max(float(fill_min_weight), 0.0)
        valid = weight > min_weight
        sintot = np.divide(sintot, weight, out=np.zeros_like(sintot), where=valid)
        costot = np.divide(costot, weight, out=np.zeros_like(costot), where=valid)
        mtot = np.arctan2(sintot, costot)
        if fill_min_weight > 0:
            mtot[~valid] = np.nan
        return mtot

    if smooth_sigma and smooth_sigma > 0:
        sintot = gaussian_filter(sintot, smooth_sigma)
        costot = gaussian_filter(costot, smooth_sigma)
    mtot = np.arctan2(sintot, costot)
    mtot[nans] = np.nan
    return mtot


def _extract_coords_and_times(
    decoding_result: dict[str, Any],
    coords_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract coordinates and time indices from decoding result.

    Args:
        decoding_result: Dictionary containing decoding results.
        coords_key: Optional key for coordinates (defaults to 'coordsbox' or auto-detect).

    Returns:
        Tuple of (coords, times_box) where coords is (T, 2) and times_box is optional.
    """
    if coords_key is not None:
        if coords_key not in decoding_result:
            raise KeyError(f"coords_key '{coords_key}' not found in decoding_result.")
        coords = np.asarray(decoding_result[coords_key])
    elif "coordsbox" in decoding_result:
        coords = np.asarray(decoding_result["coordsbox"])
    else:
        coords, _ = find_coords_matrix(decoding_result)

    times_box, _ = find_times_box(decoding_result)
    return coords, times_box


def _phase_map_valid_fraction(phase_map: np.ndarray) -> float:
    """Calculate fraction of valid (non-NaN) values in phase map.

    Args:
        phase_map: Phase map array.

    Returns:
        Fraction of valid values (0.0 to 1.0).
    """
    valid = np.isfinite(phase_map)
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))
