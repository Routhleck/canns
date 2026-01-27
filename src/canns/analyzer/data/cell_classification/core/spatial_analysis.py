"""
Spatial Analysis Functions

Functions for computing spatial firing rate maps and related metrics.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


def compute_rate_map(
    spike_times: np.ndarray,
    positions: np.ndarray,
    time_stamps: np.ndarray,
    spatial_bins: int = 20,
    position_range: Optional[Tuple[float, float]] = None,
    smoothing_sigma: float = 2.0,
    min_occupancy: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D spatial firing rate map.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times in seconds
    positions : np.ndarray
        Animal positions, shape (N, 2) where columns are (x, y) coordinates
    time_stamps : np.ndarray
        Time stamps for position samples
    spatial_bins : int or tuple, optional
        Number of spatial bins. If int, uses same for both dimensions.
        Default is 20.
    position_range : tuple of float, optional
        (min, max) for position coordinates. If None, inferred from data.
    smoothing_sigma : float, optional
        Standard deviation of Gaussian smoothing kernel. Default is 2.0.
    min_occupancy : float, optional
        Minimum occupancy (seconds) for valid bins. Default is 0.0.

    Returns
    -------
    rate_map : np.ndarray
        2D firing rate map (Hz), shape (spatial_bins, spatial_bins)
    occupancy_map : np.ndarray
        Time spent in each bin (seconds)
    x_edges : np.ndarray
        Bin edges for x coordinate
    y_edges : np.ndarray
        Bin edges for y coordinate

    Examples
    --------
    >>> # Simulate data
    >>> time_stamps = np.linspace(0, 100, 10000)
    >>> positions = np.column_stack([
    ...     np.sin(time_stamps * 0.1),
    ...     np.cos(time_stamps * 0.1)
    ... ])
    >>> spike_times = time_stamps[::50]  # Some spikes
    >>> rate_map, occ, x_edges, y_edges = compute_rate_map(
    ...     spike_times, positions, time_stamps
    ... )
    """
    # Handle bin specification
    if isinstance(spatial_bins, int):
        n_bins_x = n_bins_y = spatial_bins
    else:
        n_bins_x, n_bins_y = spatial_bins

    # Determine position range
    if position_range is None:
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    else:
        x_min, x_max = position_range
        y_min, y_max = position_range

    # Create bin edges
    x_edges = np.linspace(x_min, x_max, n_bins_x + 1)
    y_edges = np.linspace(y_min, y_max, n_bins_y + 1)

    # Compute occupancy map
    dt = np.median(np.diff(time_stamps))
    occupancy_map, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=[x_edges, y_edges]
    )
    occupancy_map = occupancy_map.T * dt  # Transpose to match image orientation

    # Get spike positions
    spike_positions = np.column_stack([
        np.interp(spike_times, time_stamps, positions[:, 0]),
        np.interp(spike_times, time_stamps, positions[:, 1])
    ])

    # Compute spike count map
    spike_map, _, _ = np.histogram2d(
        spike_positions[:, 0], spike_positions[:, 1],
        bins=[x_edges, y_edges]
    )
    spike_map = spike_map.T  # Transpose

    # Compute rate map
    rate_map = np.zeros_like(occupancy_map)
    valid = occupancy_map > min_occupancy
    rate_map[valid] = spike_map[valid] / occupancy_map[valid]

    # Apply Gaussian smoothing
    if smoothing_sigma > 0:
        rate_map = ndimage.gaussian_filter(rate_map, sigma=smoothing_sigma)

    return rate_map, occupancy_map, x_edges, y_edges


def compute_rate_map_from_binned(
    x: np.ndarray,
    y: np.ndarray,
    spike_counts: np.ndarray,
    bins: int = 35,
    min_occupancy: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a 2D rate map from binned spike counts aligned to positions.

    Parameters
    ----------
    x : np.ndarray
        X positions aligned to spike_counts (same length).
    y : np.ndarray
        Y positions aligned to spike_counts (same length).
    spike_counts : np.ndarray
        Spike counts per time bin (same length as x/y).
    bins : int, optional
        Number of spatial bins per dimension. Default is 35.
    min_occupancy : float, optional
        Minimum occupancy count for valid bins. Default is 0.

    Returns
    -------
    rate_map : np.ndarray
        2D firing rate map, shape (bins, bins).
    occupancy_map : np.ndarray
        Occupancy counts per bin.
    x_edges : np.ndarray
        Bin edges for x coordinate.
    y_edges : np.ndarray
        Bin edges for y coordinate.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    spike_counts = np.asarray(spike_counts, dtype=float).ravel()

    T = min(len(x), len(y), len(spike_counts))
    if T == 0:
        raise ValueError("x, y, and spike_counts must be non-empty and aligned.")

    x = x[:T]
    y = y[:T]
    spike_counts = spike_counts[:T]

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(spike_counts)
    x = x[valid]
    y = y[valid]
    spike_counts = spike_counts[valid]

    occupancy_map, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    spike_map, _, _ = np.histogram2d(
        x, y, bins=[x_edges, y_edges], weights=spike_counts
    )

    occupancy_map = occupancy_map.T
    spike_map = spike_map.T

    rate_map = np.zeros_like(occupancy_map)
    mask = occupancy_map > float(min_occupancy)
    rate_map[mask] = spike_map[mask] / occupancy_map[mask]

    return rate_map, occupancy_map, x_edges, y_edges


def compute_spatial_information(
    rate_map: np.ndarray,
    occupancy_map: np.ndarray,
    mean_rate: Optional[float] = None
) -> float:
    """
    Compute spatial information score (bits per spike).

    Spatial information quantifies how much information about the animal's
    location is conveyed by each spike.

    Parameters
    ----------
    rate_map : np.ndarray
        2D firing rate map (Hz)
    occupancy_map : np.ndarray
        Time spent in each bin (seconds)
    mean_rate : float, optional
        Mean firing rate. If None, computed from rate_map and occupancy_map.

    Returns
    -------
    spatial_info : float
        Spatial information in bits per spike

    Examples
    --------
    >>> rate_map = np.random.rand(20, 20) * 10
    >>> occupancy_map = np.ones((20, 20))
    >>> info = compute_spatial_information(rate_map, occupancy_map)

    Notes
    -----
    Formula: I = Σ_i p_i * (r_i / r_mean) * log2(r_i / r_mean)
    where:
    - p_i is probability of occupancy in bin i
    - r_i is firing rate in bin i
    - r_mean is mean firing rate

    References
    ----------
    Skaggs et al. (1993). "An information-theoretic approach to deciphering
    the hippocampal code." NIPS.
    """
    # Compute occupancy probability
    total_time = np.sum(occupancy_map)
    if total_time == 0:
        return 0.0

    prob_occupancy = occupancy_map / total_time

    # Compute mean firing rate if not provided
    if mean_rate is None:
        mean_rate = np.sum(rate_map * occupancy_map) / total_time

    if mean_rate == 0:
        return 0.0

    # Compute spatial information
    spatial_info = 0.0
    for i in range(rate_map.shape[0]):
        for j in range(rate_map.shape[1]):
            if prob_occupancy[i, j] > 0 and rate_map[i, j] > 0:
                ratio = rate_map[i, j] / mean_rate
                spatial_info += prob_occupancy[i, j] * ratio * np.log2(ratio)

    return spatial_info


def compute_field_statistics(
    rate_map: np.ndarray,
    threshold: float = 0.2,
    min_area: int = 9
) -> dict:
    """
    Extract firing field statistics from a rate map.

    Parameters
    ----------
    rate_map : np.ndarray
        2D firing rate map (Hz)
    threshold : float, optional
        Threshold as fraction of peak rate. Default is 0.2 (20% of peak).
    min_area : int, optional
        Minimum field size in pixels. Default is 9.

    Returns
    -------
    stats : dict
        Dictionary with:
        - num_fields: number of detected fields
        - field_sizes: list of field areas
        - field_peaks: list of peak firing rates
        - field_centers: list of field centers (x, y)

    Examples
    --------
    >>> rate_map = np.random.rand(50, 50) * 10
    >>> stats = compute_field_statistics(rate_map)
    >>> print(f"Found {stats['num_fields']} firing fields")
    """
    from ..utils.image_processing import label_connected_components, regionprops

    # Threshold rate map
    peak_rate = np.max(rate_map)
    threshold_value = threshold * peak_rate
    binary_map = rate_map > threshold_value

    # Label connected components
    labels, num_labels = label_connected_components(binary_map)

    # Get region properties
    props = regionprops(labels, intensity_image=rate_map)

    # Filter by minimum area
    valid_props = [p for p in props if p.area >= min_area]

    # Extract statistics
    stats = {
        'num_fields': len(valid_props),
        'field_sizes': [p.area for p in valid_props],
        'field_peaks': [np.max(rate_map[p.coords[:, 0], p.coords[:, 1]])
                       for p in valid_props],
        'field_centers': [p.centroid for p in valid_props]
    }

    return stats


def compute_grid_spacing(
    rate_map: np.ndarray,
    method: str = 'autocorr'
) -> Optional[float]:
    """
    Estimate grid spacing from a rate map.

    Parameters
    ----------
    rate_map : np.ndarray
        2D firing rate map
    method : str, optional
        Method for estimation: 'autocorr' (default) or 'fft'

    Returns
    -------
    spacing : float or None
        Estimated grid spacing in bins, or None if cannot be determined

    Notes
    -----
    This is a simplified implementation. For full grid analysis,
    use GridnessAnalyzer.
    """
    if method == 'autocorr':
        from .grid_cells import compute_2d_autocorrelation
        from ..utils.image_processing import find_regional_maxima

        # Compute autocorrelation
        autocorr = compute_2d_autocorrelation(rate_map)

        # Find peaks
        maxima = find_regional_maxima(autocorr)

        # Find distances from center
        center = np.array(autocorr.shape) // 2
        coords = np.argwhere(maxima)

        if len(coords) < 2:
            return None

        # Remove center peak
        distances = np.linalg.norm(coords - center, axis=1)
        non_center = distances > 5  # Exclude central peak
        if np.sum(non_center) == 0:
            return None

        # Median distance to peaks
        spacing = np.median(distances[non_center])
        return float(spacing)

    elif method == 'fft':
        # FFT-based spacing estimation
        fft = np.fft.fft2(rate_map)
        fft_shift = np.fft.fftshift(fft)
        power = np.abs(fft_shift) ** 2

        # Find peak in power spectrum (excluding DC component)
        center = np.array(power.shape) // 2
        power[center[0]-2:center[0]+3, center[1]-2:center[1]+3] = 0

        peak_idx = np.unravel_index(np.argmax(power), power.shape)
        distance = np.linalg.norm(np.array(peak_idx) - center)

        if distance > 0:
            spacing = power.shape[0] / distance
            return float(spacing)

    return None


if __name__ == "__main__":
    print("Testing spatial analysis functions...")

    # Simulate trajectory and spikes
    print("\nSimulating data...")
    time_stamps = np.linspace(0, 100, 10000)  # 100 seconds
    t = time_stamps

    # Circular trajectory
    positions = np.column_stack([
        0.5 * np.sin(t * 0.1),
        0.5 * np.cos(t * 0.1)
    ])

    # Place cell: fires at (0, 0.5)
    place_field_center = np.array([0.0, 0.5])
    place_field_width = 0.15

    distances = np.linalg.norm(positions - place_field_center, axis=1)
    firing_prob = np.exp(-(distances ** 2) / (2 * place_field_width ** 2))
    firing_prob = firing_prob * 0.1  # Max 10% per time bin

    spike_mask = np.random.rand(len(t)) < firing_prob
    spike_times = t[spike_mask]

    print(f"Generated {len(spike_times)} spikes")

    # Compute rate map
    print("\nComputing rate map...")
    rate_map, occupancy, x_edges, y_edges = compute_rate_map(
        spike_times, positions, time_stamps,
        spatial_bins=20,
        position_range=(-0.75, 0.75),
        smoothing_sigma=1.5
    )

    print(f"Rate map shape: {rate_map.shape}")
    print(f"Peak firing rate: {rate_map.max():.2f} Hz")
    print(f"Mean firing rate: {rate_map[occupancy > 0].mean():.2f} Hz")

    # Compute spatial information
    print("\nComputing spatial information...")
    spatial_info = compute_spatial_information(rate_map, occupancy)
    print(f"Spatial information: {spatial_info:.3f} bits/spike")

    # Extract field statistics
    print("\nExtracting firing fields...")
    field_stats = compute_field_statistics(rate_map, threshold=0.3)
    print(f"Number of fields: {field_stats['num_fields']}")
    for i, (size, peak, center) in enumerate(zip(
        field_stats['field_sizes'],
        field_stats['field_peaks'],
        field_stats['field_centers']
    )):
        print(f"  Field {i+1}: size={size} pixels, peak={peak:.2f} Hz, "
              f"center=({center[0]:.1f}, {center[1]:.1f})")

    print("\nSpatial analysis tests completed!")
