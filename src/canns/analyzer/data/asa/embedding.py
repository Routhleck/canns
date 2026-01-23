from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import DataLoadError, ProcessingError, SpikeEmbeddingConfig
from .filters import _gaussian_filter1d


def embed_spike_trains(spike_trains, config: SpikeEmbeddingConfig | None = None, **kwargs):
    """
    Load and preprocess spike train data from npz file.

    This function converts raw spike times into a time-binned spike matrix,
    optionally applying Gaussian smoothing and filtering based on animal movement speed.

    Parameters
    ----------
    spike_trains : dict
        Dictionary containing ``'spike'`` and ``'t'``, and optionally ``'x'``/``'y'``.
        ``'spike'`` can be a dict of neuron->spike_times, a list/array of arrays, or
        a numpy object array from ``np.load``.
    config : SpikeEmbeddingConfig, optional
        Configuration object controlling binning, smoothing, and speed filtering.
    **kwargs : Any
        Legacy keyword parameters (``res``, ``dt``, ``sigma``, ``smooth0``, ``speed0``,
        ``min_speed``). Prefer ``config`` in new code.

    Returns
    -------
    tuple
        ``(spikes_bin, xx, yy, tt)`` where:
        - ``spikes_bin`` is a (T, N) binned spike matrix.
        - ``xx``, ``yy``, ``tt`` are position/time arrays when ``speed_filter=True``,
          otherwise ``None``.

    Examples
    --------
    >>> from canns.analyzer.data import SpikeEmbeddingConfig, embed_spike_trains
    >>> cfg = SpikeEmbeddingConfig(smooth=False, speed_filter=False)
    >>> spikes, xx, yy, tt = embed_spike_trains(mock_data, config=cfg)  # doctest: +SKIP
    >>> spikes.ndim
    2
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = SpikeEmbeddingConfig(
            res=kwargs.get("res", 100000),
            dt=kwargs.get("dt", 1000),
            sigma=kwargs.get("sigma", 5000),
            smooth=kwargs.get("smooth0", True),
            speed_filter=kwargs.get("speed0", True),
            min_speed=kwargs.get("min_speed", 2.5),
        )

    try:
        # Step 1: Extract and filter spike data
        spikes_filtered = _extract_spike_data(spike_trains, config)

        # Step 2: Create time bins
        time_bins = _create_time_bins(spike_trains["t"], config)

        # Step 3: Bin spike data
        spikes_bin = _bin_spike_data(spikes_filtered, time_bins, config)

        # Step 4: Apply temporal smoothing if requested
        if config.smooth:
            spikes_bin = _apply_temporal_smoothing(spikes_bin, config)

        # Step 5: Apply speed filtering if requested
        if config.speed_filter:
            return _apply_speed_filtering(spikes_bin, spike_trains, config)

        return spikes_bin, None, None, None

    except Exception as e:
        raise ProcessingError(f"Failed to embed spike trains: {e}") from e


def _extract_spike_data(
    spike_trains: dict[str, Any], config: SpikeEmbeddingConfig
) -> dict[int, np.ndarray]:
    """Extract and filter spike data within time window."""
    try:
        # Handle different spike data formats
        spike_data = spike_trains["spike"]
        if hasattr(spike_data, "item") and callable(spike_data.item):
            # numpy array with .item() method (from npz file)
            spikes_all = spike_data[()]
        elif isinstance(spike_data, dict):
            # Already a dictionary
            spikes_all = spike_data
        elif isinstance(spike_data, (list, np.ndarray)):
            # List or array format
            spikes_all = spike_data
        else:
            # Try direct access
            spikes_all = spike_data

        t = spike_trains["t"]

        min_time0 = np.min(t)
        max_time0 = np.max(t)

        # Extract spike intervals for each cell
        if isinstance(spikes_all, dict):
            # Dictionary format
            spikes = {}
            for i, key in enumerate(spikes_all.keys()):
                s = np.array(spikes_all[key])
                spikes[i] = s[(s >= min_time0) & (s < max_time0)]
        else:
            # List/array format
            cell_inds = np.arange(len(spikes_all))
            spikes = {}

            for i, m in enumerate(cell_inds):
                s = np.array(spikes_all[m]) if len(spikes_all[m]) > 0 else np.array([])
                # Filter spikes within time window
                if len(s) > 0:
                    spikes[i] = s[(s >= min_time0) & (s < max_time0)]
                else:
                    spikes[i] = np.array([])

        return spikes

    except KeyError as e:
        raise DataLoadError(f"Missing required data key: {e}") from e
    except Exception as e:
        raise ProcessingError(f"Error extracting spike data: {e}") from e


def _create_time_bins(t: np.ndarray, config: SpikeEmbeddingConfig) -> np.ndarray:
    """Create time bins for spike discretization."""
    min_time0 = np.min(t)
    max_time0 = np.max(t)

    min_time = min_time0 * config.res
    max_time = max_time0 * config.res

    return np.arange(np.floor(min_time), np.ceil(max_time) + 1, config.dt)


def _bin_spike_data(
    spikes: dict[int, np.ndarray], time_bins: np.ndarray, config: SpikeEmbeddingConfig
) -> np.ndarray:
    """Convert spike times to binned spike matrix."""
    min_time = time_bins[0]
    max_time = time_bins[-1]

    spikes_bin = np.zeros((len(time_bins), len(spikes)), dtype=int)

    for n in spikes:
        spike_times = np.array(spikes[n] * config.res - min_time, dtype=int)
        # Filter valid spike times
        spike_times = spike_times[(spike_times < (max_time - min_time)) & (spike_times > 0)]
        spike_times = np.array(spike_times / config.dt, int)

        # Bin spikes
        for j in spike_times:
            if j < len(time_bins):
                spikes_bin[j, n] += 1

    return spikes_bin


def _apply_temporal_smoothing(spikes_bin: np.ndarray, config: SpikeEmbeddingConfig) -> np.ndarray:
    """Apply Gaussian temporal smoothing to spike matrix."""
    # Calculate smoothing parameters (legacy implementation used custom kernel)
    # Current implementation uses scipy's gaussian_filter1d for better performance

    # Apply smoothing (simplified version - could be further optimized)
    smoothed = np.zeros((spikes_bin.shape[0], spikes_bin.shape[1]))

    # Use scipy's gaussian_filter1d for better performance

    sigma_bins = config.sigma / config.dt

    for n in range(spikes_bin.shape[1]):
        smoothed[:, n] = gaussian_filter1d(
            spikes_bin[:, n].astype(float), sigma=sigma_bins, mode="constant"
        )

    # Normalize
    normalization_factor = 1 / np.sqrt(2 * np.pi * (config.sigma / config.res) ** 2)
    return smoothed * normalization_factor


def _apply_speed_filtering(
    spikes_bin: np.ndarray, spike_trains: dict[str, Any], config: SpikeEmbeddingConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply speed-based filtering to spike data."""
    try:
        xx, yy, tt_pos, speed = _load_pos(
            spike_trains["t"], spike_trains["x"], spike_trains["y"], res=config.res, dt=config.dt
        )

        valid = speed > config.min_speed

        return (spikes_bin[valid, :], xx[valid], yy[valid], tt_pos[valid])

    except KeyError as e:
        raise DataLoadError(f"Missing position data for speed filtering: {e}") from e
    except Exception as e:
        raise ProcessingError(f"Error in speed filtering: {e}") from e


def _load_pos(t, x, y, res=100000, dt=1000):
    """
    Compute animal position and speed from spike data file.

    Interpolates animal positions to match spike time bins and computes smoothed velocity vectors and speed.

    Parameters:
        t (ndarray): Time points of the spikes (in seconds).
        x (ndarray): X coordinates of the animal's position.
        y (ndarray): Y coordinates of the animal's position.
        res (int): Time scaling factor to align with spike resolution.
        dt (int): Temporal bin size in microseconds.

    Returns:
        xx (ndarray): Interpolated x positions.
        yy (ndarray): Interpolated y positions.
        tt (ndarray): Corresponding time points (in seconds).
        speed (ndarray): Speed at each time point (in cm/s).
    """

    min_time0 = np.min(t)
    max_time0 = np.max(t)

    times = np.where((t >= min_time0) & (t < max_time0))
    x = x[times]
    y = y[times]
    t = t[times]

    min_time = min_time0 * res
    max_time = max_time0 * res

    tt = np.arange(np.floor(min_time), np.ceil(max_time) + 1, dt) / res

    idt = np.concatenate(([0], np.digitize(t[1:-1], tt[:]) - 1, [len(tt) + 1]))
    idtt = np.digitize(np.arange(len(tt)), idt) - 1

    idx = np.concatenate((np.unique(idtt), [np.max(idtt) + 1]))
    divisor = np.bincount(idtt)
    steps = 1.0 / divisor[divisor > 0]
    N = np.max(divisor)
    ranges = np.multiply(np.arange(N)[np.newaxis, :], steps[:, np.newaxis])
    ranges[ranges >= 1] = np.nan

    rangesx = x[idx[:-1], np.newaxis] + np.multiply(
        ranges, (x[idx[1:]] - x[idx[:-1]])[:, np.newaxis]
    )
    xx = rangesx[~np.isnan(ranges)]

    rangesy = y[idx[:-1], np.newaxis] + np.multiply(
        ranges, (y[idx[1:]] - y[idx[:-1]])[:, np.newaxis]
    )
    yy = rangesy[~np.isnan(ranges)]

    xxs = _gaussian_filter1d(xx - np.min(xx), sigma=100)
    yys = _gaussian_filter1d(yy - np.min(yy), sigma=100)
    dx = (xxs[1:] - xxs[:-1]) * 100
    dy = (yys[1:] - yys[:-1]) * 100
    speed = np.sqrt(dx**2 + dy**2) / 0.01
    speed = np.concatenate(([speed[0]], speed))
    return xx, yy, tt, speed
