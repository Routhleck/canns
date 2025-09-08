"""
Place Cell Analysis Module

This module provides comprehensive analysis tools for Place Cell models,
including place field detection, spatial information calculation, and
place field characterization.

Functions adapted and optimized from funcs_.py with numpy backend.
"""

import numpy as np
from numba import njit, prange


def compute_firing_field(
    activity: np.ndarray,
    positions: np.ndarray,
    width: float,
    height: float,
    n_bins_x: int = 50,
    n_bins_y: int = 50,
) -> np.ndarray:
    """
    Compute spatial firing rate heatmaps for neurons based on activity and position data.

    Optimized version using numba for high-performance computation.

    Args:
        activity (np.ndarray): Shape (T, N) - firing rates over time for N neurons
        positions (np.ndarray): Shape (T, 2) - [x, y] positions over time
        width (float): Environment width
        height (float): Environment height
        n_bins_x (int): Number of spatial bins in x-direction
        n_bins_y (int): Number of spatial bins in y-direction

    Returns:
        np.ndarray: Shape (n_bins_x, n_bins_y, N) - firing rate heatmaps
    """
    return _compute_firing_field_numba(activity, positions, width, height, n_bins_x, n_bins_y)


@njit(parallel=True)
def _compute_firing_field_numba(
    activity: np.ndarray,
    positions: np.ndarray,
    width: float,
    height: float,
    n_bins_x: int,
    n_bins_y: int,
) -> np.ndarray:
    """Numba-optimized firing field computation."""
    T, N = activity.shape
    heatmaps = np.zeros((n_bins_x, n_bins_y, N))
    bin_counts = np.zeros((n_bins_x, n_bins_y))

    # Determine bin sizes
    bin_width = width / n_bins_x
    bin_height = height / n_bins_y

    # Assign positions to bins
    x_bins = np.clip(((positions[:, 0]) // bin_width).astype(np.int32), 0, n_bins_x - 1)
    y_bins = np.clip(((positions[:, 1]) // bin_height).astype(np.int32), 0, n_bins_y - 1)

    # Accumulate activity in each bin
    for t in prange(T):
        x_bin = x_bins[t]
        y_bin = y_bins[t]
        heatmaps[x_bin, y_bin, :] += activity[t, :]
        bin_counts[x_bin, y_bin] += 1

    # Compute average firing rate per bin (avoid division by zero)
    for n in range(N):
        for i in range(n_bins_x):
            for j in range(n_bins_y):
                if bin_counts[i, j] > 0:
                    heatmaps[i, j, n] /= bin_counts[i, j]

    return heatmaps


def compute_place_score(
    activity: np.ndarray,
    positions: np.ndarray,
    heatmap: np.ndarray | None = None,
    width: float = 1.0,
    height: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute place scores for neurons based on Gaussian place field similarity.

    Args:
        activity (np.ndarray): Shape (T, N) - neural activity over time
        positions (np.ndarray): Shape (T, 2) - [x, y] positions over time
        heatmap (np.ndarray, optional): Precomputed firing rate heatmap
        width (float): Environment width
        height (float): Environment height

    Returns:
        Tuple[np.ndarray, np.ndarray]: (place_scores, sigma_values)
            - place_scores: Shape (N,) - place field quality scores
            - sigma_values: Shape (N,) - place field width estimates
    """
    T, N = activity.shape

    if heatmap is None:
        heatmap = compute_firing_field(activity, positions, width, height)

    place_scores = np.zeros(N)
    sigma_values = np.zeros(N)

    # Create coordinate grids
    M, K, _ = heatmap.shape
    x_bins = np.linspace(0, width, M, endpoint=False) + width / (2 * M)
    y_bins = np.linspace(0, height, K, endpoint=False) + height / (2 * K)
    x, y = np.meshgrid(x_bins, y_bins, indexing="ij")

    for neuron in range(N):
        firing_map = heatmap[:, :, neuron]

        # Find peak firing location
        max_bin = np.unravel_index(np.argmax(firing_map), firing_map.shape)
        peak_x = x[max_bin]
        peak_y = y[max_bin]

        # Calculate weighted standard deviations
        fr_sum = np.sum(firing_map)
        if fr_sum > 0:
            sigma_x = np.sqrt(np.sum(firing_map * (x - peak_x) ** 2) / fr_sum)
            sigma_y = np.sqrt(np.sum(firing_map * (y - peak_y) ** 2) / fr_sum)
            sigma_mean = (sigma_x + sigma_y) / 2

            # Constrain sigma to reasonable range
            sigma_mean = max(0.025, min(sigma_mean, 0.5))
            sigma_values[neuron] = sigma_mean

            # Generate ideal Gaussian place field
            gaussian = np.exp(-((x - peak_x) ** 2 + (y - peak_y) ** 2) / (2 * sigma_mean**2))

            # Compute similarity (cosine similarity weighted by max firing rate)
            firing_flat = firing_map.flatten()
            gaussian_flat = gaussian.flatten()

            # Normalize
            firing_norm = (
                firing_flat / np.sum(firing_flat) if np.sum(firing_flat) > 0 else firing_flat
            )
            gaussian_norm = gaussian_flat / np.sum(gaussian_flat)

            # Cosine similarity weighted by peak firing rate
            place_scores[neuron] = (
                np.max(firing_map)
                * np.dot(firing_norm, gaussian_norm)
                / (np.linalg.norm(firing_norm) * np.linalg.norm(gaussian_norm) + 1e-10)
            )

    return place_scores, sigma_values


def select_place_cells(
    activity: np.ndarray,
    positions: np.ndarray,
    threshold: float = 0.002,
    width: float = 1.0,
    height: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Select place cells based on place field quality scores.

    Args:
        activity (np.ndarray): Shape (T, N) - neural activity over time
        positions (np.ndarray): Shape (T, 2) - positions over time
        threshold (float): Minimum place score threshold
        width (float): Environment width
        height (float): Environment height

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: (place_scores, place_indices, num_place_cells)
            - place_scores: All place scores
            - place_indices: Indices of neurons passing threshold
            - num_place_cells: Number of selected place cells
    """
    place_scores, _ = compute_place_score(activity, positions, width=width, height=height)
    place_indices = np.where(place_scores > threshold)[0]
    num_place_cells = len(place_indices)

    return place_scores, place_indices, num_place_cells


def compute_place_field_centers(
    activity: np.ndarray,
    place_indices: np.ndarray,
    positions: np.ndarray,
    method: str = "peak",
    width: float = 1.0,
    height: float = 1.0,
) -> np.ndarray:
    """
    Compute place field centers for selected place cells.

    Args:
        activity (np.ndarray): Shape (T, N) - neural activity
        place_indices (np.ndarray): Indices of place cells
        positions (np.ndarray): Shape (T, 2) - positions
        method (str): Method for center computation ("peak" or "centroid")
        width (float): Environment width
        height (float): Environment height

    Returns:
        np.ndarray: Shape (num_place_cells, 2) - place field centers [x, y]
    """
    heatmap = compute_firing_field(activity, positions, width, height)
    num_place_cells = len(place_indices)
    centers = np.zeros((num_place_cells, 2))

    # Create coordinate grids
    M, K, _ = heatmap.shape
    x_bins = np.linspace(0, width, M, endpoint=False) + width / (2 * M)
    y_bins = np.linspace(0, height, K, endpoint=False) + height / (2 * K)
    x, y = np.meshgrid(x_bins, y_bins, indexing="ij")

    for i, neuron_idx in enumerate(place_indices):
        firing_map = heatmap[:, :, neuron_idx]

        if method == "peak":
            # Find peak location
            max_bin = np.unravel_index(np.argmax(firing_map), firing_map.shape)
            centers[i, 0] = x[max_bin]
            centers[i, 1] = y[max_bin]

        elif method == "centroid":
            # Compute weighted centroid
            total_activity = np.sum(firing_map)
            if total_activity > 0:
                centers[i, 0] = np.sum(firing_map * x) / total_activity
                centers[i, 1] = np.sum(firing_map * y) / total_activity

    return centers


def compute_spatial_information(
    activity: np.ndarray,
    positions: np.ndarray,
    width: float = 1.0,
    height: float = 1.0,
    n_bins: int = 50,
) -> np.ndarray:
    """
    Compute spatial information content for each neuron.

    Spatial information quantifies how much information a neuron's firing
    conveys about the animal's location.

    Args:
        activity (np.ndarray): Shape (T, N) - neural activity
        positions (np.ndarray): Shape (T, 2) - positions
        width (float): Environment width
        height (float): Environment height
        n_bins (int): Number of spatial bins per dimension

    Returns:
        np.ndarray: Shape (N,) - spatial information bits per spike
    """
    heatmap = compute_firing_field(activity, positions, width, height, n_bins, n_bins)
    T, N = activity.shape
    spatial_info = np.zeros(N)

    # Compute occupancy probability
    bin_width = width / n_bins
    bin_height = height / n_bins
    occupancy = np.zeros((n_bins, n_bins))

    for t in range(T):
        x_bin = int(np.clip(positions[t, 0] / bin_width, 0, n_bins - 1))
        y_bin = int(np.clip(positions[t, 1] / bin_height, 0, n_bins - 1))
        occupancy[x_bin, y_bin] += 1

    occupancy = occupancy / T  # Convert to probability

    for neuron in range(N):
        firing_map = heatmap[:, :, neuron]
        mean_rate = np.mean(activity[:, neuron])

        if mean_rate > 0:
            info_sum = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if occupancy[i, j] > 0 and firing_map[i, j] > 0:
                        # Information = rate_in_bin * P(bin) * log2(rate_in_bin / mean_rate)
                        info_sum += (
                            firing_map[i, j]
                            * occupancy[i, j]
                            * np.log2(firing_map[i, j] / mean_rate)
                        )
            spatial_info[neuron] = info_sum / mean_rate

    return spatial_info


def compute_place_field_stability(
    activity1: np.ndarray,
    activity2: np.ndarray,
    positions1: np.ndarray,
    positions2: np.ndarray,
    place_indices: np.ndarray,
    width: float = 1.0,
    height: float = 1.0,
) -> np.ndarray:
    """
    Compute place field stability between two recording sessions.

    Args:
        activity1 (np.ndarray): Activity from session 1
        activity2 (np.ndarray): Activity from session 2
        positions1 (np.ndarray): Positions from session 1
        positions2 (np.ndarray): Positions from session 2
        place_indices (np.ndarray): Indices of place cells to analyze
        width (float): Environment width
        height (float): Environment height

    Returns:
        np.ndarray: Shape (num_place_cells,) - stability correlations
    """
    heatmap1 = compute_firing_field(activity1, positions1, width, height)
    heatmap2 = compute_firing_field(activity2, positions2, width, height)

    num_place_cells = len(place_indices)
    stability = np.zeros(num_place_cells)

    for i, neuron_idx in enumerate(place_indices):
        map1 = heatmap1[:, :, neuron_idx].flatten()
        map2 = heatmap2[:, :, neuron_idx].flatten()

        # Compute Pearson correlation
        if np.std(map1) > 0 and np.std(map2) > 0:
            stability[i] = np.corrcoef(map1, map2)[0, 1]
        else:
            stability[i] = 0.0

    return stability


def analyze_place_field_properties(
    activity: np.ndarray,
    positions: np.ndarray,
    place_indices: np.ndarray,
    width: float = 1.0,
    height: float = 1.0,
) -> dict:
    """
    Comprehensive analysis of place field properties.

    Args:
        activity (np.ndarray): Shape (T, N) - neural activity
        positions (np.ndarray): Shape (T, 2) - positions
        place_indices (np.ndarray): Indices of place cells
        width (float): Environment width
        height (float): Environment height

    Returns:
        dict: Dictionary containing place field analysis results
    """
    heatmap = compute_firing_field(activity, positions, width, height)

    results = {
        "place_indices": place_indices,
        "num_place_cells": len(place_indices),
        "centers": compute_place_field_centers(
            activity, place_indices, positions, width=width, height=height
        ),
        "spatial_info": compute_spatial_information(
            activity, positions, width=width, height=height
        )[place_indices],
        "heatmaps": heatmap[:, :, place_indices],
    }

    # Compute additional properties
    place_scores, sigma_values = compute_place_score(activity, positions, heatmap, width, height)
    results["place_scores"] = place_scores[place_indices]
    results["field_sizes"] = sigma_values[place_indices]

    # Peak firing rates
    peak_rates = []
    for idx in place_indices:
        peak_rates.append(np.max(heatmap[:, :, idx]))
    results["peak_rates"] = np.array(peak_rates)

    return results
