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

    This function bins neural activity into a 2D spatial grid based on
    (x, y) positions. The input shapes match the usage patterns in analyzer
    tests: activity is ``(T, N)`` and positions is ``(T, 2)``.

    Args:
        A (np.ndarray): Neural activity of shape ``(T, N)``.
        positions (np.ndarray): Positions of shape ``(T, 2)``.
        width (float): Environment width.
        height (float): Environment height.
        M (int): Number of bins along width.
        K (int): Number of bins along height.

    Returns:
        np.ndarray: Heatmaps of shape ``(N, M, K)``.

    Examples:
        >>> import numpy as np
        >>> from canns.analyzer.metrics.spatial_metrics import compute_firing_field
        >>>
        >>> # Dummy inputs (T timesteps, N neurons)
        >>> activity = np.random.rand(100, 3)
        >>> positions = np.column_stack(
        ...     [np.linspace(0, 1.0, 100), np.linspace(0, 1.0, 100)]
        ... )
        >>>
        >>> heatmaps = compute_firing_field(activity, positions, 1.0, 1.0, 10, 10)
        >>> print(heatmaps.shape)
        (3, 10, 10)
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

    Args:
        heatmaps (np.ndarray): Array of shape ``(N, M, K)``.
        sigma (float, optional): Gaussian kernel width. Defaults to ``1.0``.

    Returns:
        np.ndarray: Smoothed heatmaps with the same shape as input.

    Examples:
        >>> import numpy as np
        >>> from canns.analyzer.metrics.spatial_metrics import gaussian_smooth_heatmaps
        >>>
        >>> heatmaps = np.random.rand(2, 5, 5)
        >>> smoothed = gaussian_smooth_heatmaps(heatmaps, sigma=1.0)
        >>> print(smoothed.shape)
        (2, 5, 5)
    """
    filtered = ndimage.gaussian_filter(heatmaps, sigma=(0, sigma, sigma))
    return np.where(heatmaps == 0, 0, filtered)


def compute_spatial_autocorrelation(rate_map: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Compute 2D spatial autocorrelation of a firing rate map.

    Args:
        rate_map (np.ndarray): 2D firing rate map of shape ``(M, K)``.
        max_lag (int | None): Optional max lag for cropping around the center.

    Returns:
        np.ndarray: Autocorrelation map normalized to ``[-1, 1]``.

    Examples:
        >>> import numpy as np
        >>> from canns.analyzer.metrics.spatial_metrics import compute_spatial_autocorrelation
        >>>
        >>> rate_map = np.random.rand(10, 10)
        >>> autocorr = compute_spatial_autocorrelation(rate_map)
        >>> print(autocorr.shape)
        (10, 10)
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

    Args:
        autocorr (np.ndarray): 2D spatial autocorrelation map.
        annulus_inner (float): Inner radius of annulus (fraction of map size).
        annulus_outer (float): Outer radius of annulus (fraction of map size).

    Returns:
        tuple: ``(grid_score, rotated_corrs)`` where ``rotated_corrs`` maps
        angles ``{30, 60, 90, 120, 150}`` to correlation values.

    Examples:
        >>> import numpy as np
        >>> from canns.analyzer.metrics.spatial_metrics import compute_grid_score
        >>>
        >>> autocorr = np.random.rand(15, 15)
        >>> grid_score, rotated_corrs = compute_grid_score(autocorr)
        >>> print(sorted(rotated_corrs.keys()))
        [30, 60, 90, 120, 150]
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

    Args:
        autocorr (np.ndarray): 2D autocorrelation map.
        bin_size (float | None): Spatial bin size in real units. If provided,
            the function also returns spacing in real units.

    Returns:
        tuple: ``(spacing_bins, spacing_real)``.

    Examples:
        >>> import numpy as np
        >>> from canns.analyzer.metrics.spatial_metrics import find_grid_spacing
        >>>
        >>> autocorr = np.random.rand(20, 20)
        >>> spacing_bins, spacing_m = find_grid_spacing(autocorr, bin_size=0.05)
        >>> print(spacing_m is not None)
        True
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
