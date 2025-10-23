"""Spatial analysis utilities for neural activity data.

This module provides functions for analyzing spatial patterns in neural data,
particularly for computing firing fields and spatial smoothing operations.
"""

import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter

__all__ = ["compute_firing_field", "gaussian_smooth_heatmaps"]


@njit(parallel=True)
def compute_firing_field(A, positions, width, height, M, K):
    """Compute spatial firing fields for neural population activity.

    This function bins neural activity into a 2D spatial grid based on the
    animal's position, creating a heatmap for each neuron showing where it
    fires most strongly. Uses Numba JIT compilation for high performance.

    Args:
        A (np.ndarray): Neural activity array of shape (T, N) where T is the
            number of time steps and N is the number of neurons.
        positions (np.ndarray): Position data of shape (T, 2) containing
            (x, y) coordinates at each time step.
        width (float): Width of the spatial environment.
        height (float): Height of the spatial environment.
        M (int): Number of bins along the width dimension.
        K (int): Number of bins along the height dimension.

    Returns:
        np.ndarray: Heatmaps array of shape (N, M, K) containing the average
            firing rate of each neuron in each spatial bin.

    Example:
        >>> activity = np.random.rand(1000, 30)  # 1000 timesteps, 30 neurons
        >>> positions = np.random.rand(1000, 2) * 5.0  # Random walk in 5x5 space
        >>> heatmaps = compute_firing_field(activity, positions, 5.0, 5.0, 50, 50)
        >>> heatmaps.shape
        (30, 50, 50)
    """
    T, N = A.shape  # Number of time steps and neurons
    # Initialize the heatmaps and bin counters
    heatmaps = np.zeros((N, M, K))
    bin_counts = np.zeros((M, K))

    # Determine bin sizes
    bin_width = width / M
    bin_height = height / K
    # Assign positions to bins
    x_bins = np.clip(((positions[:, 0]) // bin_width).astype(np.int32), 0, M - 1)
    y_bins = np.clip(((positions[:, 1]) // bin_height).astype(np.int32), 0, K - 1)

    # Accumulate activity in each bin
    for t in prange(T):
        x_bin = x_bins[t]
        y_bin = y_bins[t]
        heatmaps[:, x_bin, y_bin] += A[t, :]
        bin_counts[x_bin, y_bin] += 1

    # Compute average firing rate per bin (avoid division by zero)
    for n in range(N):
        heatmaps[n] = np.where(bin_counts > 0, heatmaps[n] / bin_counts, 0)

    return heatmaps


def gaussian_smooth_heatmaps(heatmaps: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to spatial heatmaps without mixing channels.

    This function applies Gaussian filtering to each heatmap independently,
    preserving zero values (unvisited spatial bins) and only smoothing regions
    with activity.

    Args:
        heatmaps (np.ndarray): Array of shape (N, M, K) where N is the number
            of neurons/channels and (M, K) is the spatial grid size.
        sigma (float, optional): Standard deviation for Gaussian kernel.
            Defaults to 1.0.

    Returns:
        np.ndarray: Smoothed heatmaps with the same shape as input. Zero values
            in the original heatmaps are preserved.

    Example:
        >>> heatmaps = np.random.rand(30, 50, 50)
        >>> smoothed = gaussian_smooth_heatmaps(heatmaps, sigma=1.5)
        >>> smoothed.shape
        (30, 50, 50)
    """
    filtered = gaussian_filter(heatmaps, sigma=(0, sigma, sigma))
    return np.where(heatmaps == 0, 0, filtered)
