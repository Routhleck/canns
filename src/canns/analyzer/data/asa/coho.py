"""Shared helpers for CohoMap/CohoSpace analysis."""

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
    **defaults,
) -> PlotConfig:
    if config is None:
        return factory(*args, **defaults)
    return config


def _ensure_parent_dir(save_path: str | None) -> None:
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def _circmean(x: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x))))


def _smooth_circular_map(
    mtot: np.ndarray,
    smooth_sigma: float,
    *,
    fill_nan: bool = False,
    fill_sigma: float | None = None,
    fill_min_weight: float = 1e-3,
) -> np.ndarray:
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
    valid = np.isfinite(phase_map)
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))
