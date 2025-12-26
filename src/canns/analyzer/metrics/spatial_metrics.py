"""Spatial analysis utilities for neural activity data.

This module provides functions for analyzing spatial patterns in neural data,
particularly for computing firing fields and spatial smoothing operations.
Includes specialized functions for grid cell analysis such as spatial
autocorrelation, grid scores, and spacing measurements.
"""

import numpy as np
from numba import njit, prange
from scipy import ndimage, signal

__all__ = [
    "compute_firing_field",
    "gaussian_smooth_heatmaps",
    "compute_spatial_autocorrelation",
    "compute_grid_score",
    "find_grid_spacing",
]


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
    filtered = ndimage.gaussian_filter(heatmaps, sigma=(0, sigma, sigma))
    return np.where(heatmaps == 0, 0, filtered)


def compute_spatial_autocorrelation(rate_map: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Compute 2D spatial autocorrelation of a firing rate map.

    For grid cells, spatial autocorrelation reveals hexagonal periodicity patterns.
    This is the gold standard method for identifying grid cells in neuroscience.
    Uses periodic boundary conditions (wrap) appropriate for toroidal grid cell topology.

    Args:
        rate_map (np.ndarray): 2D firing rate map of shape (M, K).
        max_lag (int | None): Maximum lag for output cropping. If None, returns
            full autocorrelation map. Defaults to None.

    Returns:
        np.ndarray: 2D autocorrelation map normalized to [-1, 1]. For grid cells,
            this will show a characteristic hexagonal pattern of peaks.

    Example:
        >>> from canns.analyzer.metrics.spatial_metrics import compute_firing_field, compute_spatial_autocorrelation
        >>> # After computing firing fields
        >>> rate_map = firing_fields[0]  # First neuron
        >>> autocorr = compute_spatial_autocorrelation(rate_map)
        >>> # For grid cells, autocorr will show hexagonal pattern
        >>> autocorr.shape == rate_map.shape
        True

    References:
        Sargolini et al. (2006). Conjunctive representation of position, direction,
        and velocity in entorhinal cortex. Science, 312(5774), 758-762.
    """
    # Normalize rate map (zero mean, unit variance)
    rate_map_norm = rate_map - np.mean(rate_map)
    rate_map_std = np.std(rate_map)
    if rate_map_std > 1e-10:  # Avoid division by zero
        rate_map_norm = rate_map_norm / rate_map_std

    # Compute 2D autocorrelation with periodic boundary
    # wrap boundary is critical for grid cells (toroidal space)
    autocorr = signal.correlate2d(rate_map_norm, rate_map_norm, mode="same", boundary="wrap")

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(autocorr))
    if max_val > 1e-10:
        autocorr = autocorr / max_val

    # Optionally crop to max_lag around center
    if max_lag is not None:
        center = np.array(autocorr.shape) // 2
        autocorr = autocorr[
            center[0] - max_lag : center[0] + max_lag + 1,
            center[1] - max_lag : center[1] + max_lag + 1,
        ]

    return autocorr


def compute_grid_score(
    autocorr: np.ndarray, annulus_inner: float = 0.3, annulus_outer: float = 0.7
) -> tuple[float, dict[int, float]]:
    """Compute grid score from spatial autocorrelation.

    Grid score quantifies hexagonal symmetry by comparing correlations at 60°/120°
    (hexagonal angles) versus 30°/90°/150° (non-hexagonal angles). Developed by
    Sargolini et al. (2006), this is the standard metric for grid cell identification.

    Formula:
        grid_score = min(r60, r120) - max(r30, r90, r150)

    where rX is the Pearson correlation between the original and rotated autocorrelation
    within an annulus region.

    Args:
        autocorr (np.ndarray): 2D spatial autocorrelation map.
        annulus_inner (float): Inner radius of annulus as fraction of map size.
            Defaults to 0.3.
        annulus_outer (float): Outer radius of annulus as fraction of map size.
            Defaults to 0.7.

    Returns:
        grid_score (float): Grid score value. Values > 0.3 typically indicate grid cells.
        rotated_corrs (dict): Dictionary mapping rotation angles to correlation values.
            Keys: 30, 60, 90, 120, 150 (degrees).

    Example:
        >>> from canns.analyzer.metrics.spatial_metrics import compute_spatial_autocorrelation, compute_grid_score
        >>> autocorr = compute_spatial_autocorrelation(rate_map)
        >>> grid_score, rotated_corrs = compute_grid_score(autocorr)
        >>> print(f"Grid score: {grid_score:.3f}")
        Grid score: 0.456
        >>> if grid_score > 0.3:
        ...     print("This is a grid cell!")

    References:
        Sargolini et al. (2006). Conjunctive representation of position, direction,
        and velocity in entorhinal cortex. Science, 312(5774), 758-762.
    """
    center = np.array(autocorr.shape) // 2
    max_radius = min(center)

    # Create annulus mask
    y, x = np.ogrid[: autocorr.shape[0], : autocorr.shape[1]]
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2) / max_radius
    annulus = (r >= annulus_inner) & (r <= annulus_outer)

    # Rotate autocorr at each angle and compute correlation
    angles = [30, 60, 90, 120, 150]
    rotated_corrs = {}

    for angle in angles:
        # Rotate autocorrelation map
        rotated = ndimage.rotate(autocorr, angle, reshape=False, order=1)

        # Compute Pearson correlation in the annulus region
        orig_vals = autocorr[annulus].flatten()
        rot_vals = rotated[annulus].flatten()

        # Pearson correlation coefficient
        if len(orig_vals) > 0 and np.std(orig_vals) > 1e-10 and np.std(rot_vals) > 1e-10:
            corr = np.corrcoef(orig_vals, rot_vals)[0, 1]
        else:
            corr = 0.0

        rotated_corrs[angle] = corr

    # Compute grid score: min(60°, 120°) - max(30°, 90°, 150°)
    # Hexagonal symmetry should have high correlation at 60° and 120°
    corr_60_120 = min(rotated_corrs[60], rotated_corrs[120])
    corr_30_90_150 = max(rotated_corrs[30], rotated_corrs[90], rotated_corrs[150])
    grid_score = corr_60_120 - corr_30_90_150

    return grid_score, rotated_corrs


def find_grid_spacing(
    autocorr: np.ndarray, bin_size: float | None = None
) -> tuple[float, float | None]:
    """Estimate grid spacing from spatial autocorrelation.

    Finds the distance from center to the first major peak in the autocorrelation,
    which corresponds to the grid field spacing.

    Args:
        autocorr (np.ndarray): 2D autocorrelation map.
        bin_size (float | None): Size of spatial bins in real units (e.g., meters).
            If provided, returns spacing in real units. Defaults to None.

    Returns:
        spacing_bins (float): Grid spacing in number of bins.
        spacing_real (float | None): Grid spacing in real units if bin_size provided,
            otherwise None.

    Example:
        >>> from canns.analyzer.metrics.spatial_metrics import compute_spatial_autocorrelation, find_grid_spacing
        >>> autocorr = compute_spatial_autocorrelation(rate_map)
        >>> spacing_bins, spacing_m = find_grid_spacing(autocorr, bin_size=0.06)  # 6cm bins
        >>> print(f"Grid spacing: {spacing_bins:.1f} bins = {spacing_m:.3f} m")
        Grid spacing: 52.3 bins = 3.138 m
    """
    center = np.array(autocorr.shape) // 2

    # Mask out the center peak (radius ~3 bins to exclude self-correlation)
    y, x = np.ogrid[: autocorr.shape[0], : autocorr.shape[1]]
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    autocorr_masked = autocorr.copy()
    autocorr_masked[r < 3] = -1  # Mask center with low value

    # Find global maximum (first peak)
    peak_idx = np.unravel_index(np.argmax(autocorr_masked), autocorr.shape)

    # Compute Euclidean distance from center to peak
    spacing_bins = float(np.sqrt((peak_idx[0] - center[0]) ** 2 + (peak_idx[1] - center[1]) ** 2))

    # Convert to real units if bin_size provided
    spacing_real = spacing_bins * bin_size if bin_size is not None else None

    return spacing_bins, spacing_real
