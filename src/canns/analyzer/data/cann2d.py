import logging
import multiprocessing as mp
import numbers
import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from canns_lib.ripser import ripser
from matplotlib import animation, cm, gridspec
from numpy.exceptions import AxisError

# from ripser import ripser
from scipy import signal
from scipy.ndimage import (
    _nd_image,
    _ni_support,
    binary_closing,
    gaussian_filter,
    gaussian_filter1d,
)
from scipy.ndimage._filters import _invalid_origin
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import binned_statistic_2d, multivariate_normal
from sklearn import preprocessing
from tqdm import tqdm

# Import PlotConfig for unified plotting
from ..visualization import PlotConfig
from ..visualization.core.jupyter_utils import (
    display_animation_in_jupyter,
    is_jupyter_environment,
)


# ==================== Configuration Classes ====================
@dataclass
class SpikeEmbeddingConfig:
    """Configuration for spike train embedding."""

    res: int = 100000
    dt: int = 1000
    sigma: int = 5000
    smooth: bool = True
    speed_filter: bool = True
    min_speed: float = 2.5


@dataclass
class TDAConfig:
    """Configuration for Topological Data Analysis."""

    dim: int = 6
    num_times: int = 5
    active_times: int = 15000
    k: int = 1000
    n_points: int = 1200
    metric: str = "cosine"
    nbs: int = 800
    maxdim: int = 1
    coeff: int = 47
    show: bool = True
    do_shuffle: bool = False
    num_shuffles: int = 1000
    progress_bar: bool = True


@dataclass
class CANN2DPlotConfig(PlotConfig):
    """Specialized PlotConfig for CANN2D visualizations."""

    # 3D projection specific parameters
    zlabel: str = "Component 3"
    dpi: int = 300

    # Torus animation specific parameters
    numangsint: int = 51
    r1: float = 1.5  # Major radius
    r2: float = 1.0  # Minor radius
    window_size: int = 300
    frame_step: int = 5
    n_frames: int = 20

    @classmethod
    def for_projection_3d(cls, **kwargs) -> "CANN2DPlotConfig":
        """Create configuration for 3D projection plots."""
        defaults = {
            "title": "3D Data Projection",
            "xlabel": "Component 1",
            "ylabel": "Component 2",
            "zlabel": "Component 3",
            "figsize": (10, 8),
            "dpi": 300,
        }
        defaults.update(kwargs)
        return cls.for_static_plot(**defaults)

    @classmethod
    def for_torus_animation(cls, **kwargs) -> "CANN2DPlotConfig":
        """Create configuration for 3D torus bump animations."""
        defaults = {
            "title": "3D Bump on Torus",
            "figsize": (8, 8),
            "fps": 5,
            "repeat": True,
            "show_progress_bar": True,
            "numangsint": 51,
            "r1": 1.5,
            "r2": 1.0,
            "window_size": 300,
            "frame_step": 5,
            "n_frames": 20,
        }
        defaults.update(kwargs)
        time_steps = kwargs.get("time_steps_per_second", 1000)
        config = cls.for_animation(time_steps, **defaults)
        # Add torus-specific attributes
        config.numangsint = defaults["numangsint"]
        config.r1 = defaults["r1"]
        config.r2 = defaults["r2"]
        config.window_size = defaults["window_size"]
        config.frame_step = defaults["frame_step"]
        config.n_frames = defaults["n_frames"]
        return config


# ==================== Constants ====================
class Constants:
    """Constants used throughout CANN2D analysis."""

    DEFAULT_FIGSIZE = (10, 8)
    DEFAULT_DPI = 300
    GAUSSIAN_SIGMA_FACTOR = 100
    SPEED_CONVERSION_FACTOR = 100
    TIME_CONVERSION_FACTOR = 0.01
    MULTIPROCESSING_CORES = 4


# ==================== Custom Exceptions ====================
class CANN2DError(Exception):
    """Base exception for CANN2D analysis errors."""

    pass


class DataLoadError(CANN2DError):
    """Raised when data loading fails."""

    pass


class ProcessingError(CANN2DError):
    """Raised when data processing fails."""

    pass


try:
    from numba import jit, njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print(
        "Using numba for FAST CANN2D analysis, now using pure numpy implementation.",
        "Try numba by `pip install numba` to speed up the process.",
    )

    # Create dummy decorators if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(x):
        return range(x)


def embed_spike_trains(spike_trains, config: SpikeEmbeddingConfig | None = None, **kwargs):
    """
    Load and preprocess spike train data from npz file.

    This function converts raw spike times into a time-binned spike matrix,
    optionally applying Gaussian smoothing and filtering based on animal movement speed.

    Parameters:
        spike_trains : dict containing 'spike', 't', and optionally 'x', 'y'.
        config : SpikeEmbeddingConfig, optional configuration object
        **kwargs : backward compatibility parameters

    Returns:
        spikes_bin (ndarray): Binned and optionally smoothed spike matrix of shape (T, N).
        xx (ndarray, optional): X coordinates (if speed_filter=True).
        yy (ndarray, optional): Y coordinates (if speed_filter=True).
        tt (ndarray, optional): Time points (if speed_filter=True).
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

        return spikes_bin

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
        elif isinstance(spike_data, list | np.ndarray):
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


def plot_projection(
    reduce_func,
    embed_data,
    config: CANN2DPlotConfig | None = None,
    title="Projection (3D)",
    xlabel="Component 1",
    ylabel="Component 2",
    zlabel="Component 3",
    save_path=None,
    show=True,
    dpi=300,
    figsize=(10, 8),
    **kwargs,
):
    """
    Plot a 3D projection of the embedded data.

    Parameters:
        reduce_func (callable): Function to reduce the dimensionality of the data.
        embed_data (ndarray): Data to be projected.
        config (PlotConfig, optional): Configuration object for unified plotting parameters
        **kwargs: backward compatibility parameters
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        zlabel (str): Label for the z-axis.
        save_path (str, optional): Path to save the plot. If None, plot will not be saved.
        show (bool): Whether to display the plot.
        dpi (int): Dots per inch for saving the figure.
        figsize (tuple): Size of the figure.

    Returns:
        fig: The created figure object.
    """

    # Handle backward compatibility and configuration
    if config is None:
        config = CANN2DPlotConfig.for_projection_3d(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            save_path=save_path,
            show=show,
            figsize=figsize,
            dpi=dpi,
            **kwargs,
        )

    reduced_data = reduce_func(embed_data[::5])

    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], s=1, alpha=0.5)

    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_zlabel(config.zlabel)

    if config.save_path is None and config.show is None:
        raise ValueError("Either save path or show must be provided.")
    if config.save_path:
        plt.savefig(config.save_path, dpi=config.dpi)
    if config.show:
        plt.show()

    plt.close(fig)

    return fig


def tda_vis(embed_data: np.ndarray, config: TDAConfig | None = None, **kwargs) -> dict[str, Any]:
    """
    Topological Data Analysis visualization with optional shuffle testing.

    Parameters:
        embed_data : ndarray
            Embedded spike train data.
        config : TDAConfig, optional
            Configuration object with all TDA parameters
        **kwargs : backward compatibility parameters

    Returns:
        dict : Dictionary containing:
            - persistence: persistence diagrams from real data
            - indstemp: indices of sampled points
            - movetimes: selected time points
            - n_points: number of sampled points
            - shuffle_max: shuffle analysis results (if do_shuffle=True, otherwise None)
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = TDAConfig(
            dim=kwargs.get("dim", 6),
            num_times=kwargs.get("num_times", 5),
            active_times=kwargs.get("active_times", 15000),
            k=kwargs.get("k", 1000),
            n_points=kwargs.get("n_points", 1200),
            metric=kwargs.get("metric", "cosine"),
            nbs=kwargs.get("nbs", 800),
            maxdim=kwargs.get("maxdim", 1),
            coeff=kwargs.get("coeff", 47),
            show=kwargs.get("show", True),
            do_shuffle=kwargs.get("do_shuffle", False),
            num_shuffles=kwargs.get("num_shuffles", 1000),
            progress_bar=kwargs.get("progress_bar", True),
        )

    try:
        # Compute persistent homology for real data
        print("Computing persistent homology for real data...")
        real_persistence = _compute_real_persistence(embed_data, config)

        # Perform shuffle analysis if requested
        shuffle_max = None
        if config.do_shuffle:
            shuffle_max = _perform_shuffle_analysis(embed_data, config)

        # Visualization
        _handle_visualization(real_persistence["persistence"], shuffle_max, config)

        # Return results as dictionary
        return {
            "persistence": real_persistence["persistence"],
            "indstemp": real_persistence["indstemp"],
            "movetimes": real_persistence["movetimes"],
            "n_points": real_persistence["n_points"],
            "shuffle_max": shuffle_max,
        }

    except Exception as e:
        raise ProcessingError(f"TDA analysis failed: {e}") from e


def _compute_real_persistence(embed_data: np.ndarray, config: TDAConfig) -> dict[str, Any]:
    """Compute persistent homology for real data with progress tracking."""

    logging.info("Processing real data - Starting TDA analysis (5 steps)")

    # Step 1: Time point downsampling
    logging.info("Step 1/5: Time point downsampling")
    times_cube = _downsample_timepoints(embed_data, config.num_times)

    # Step 2: Select most active time points
    logging.info("Step 2/5: Selecting active time points")
    movetimes = _select_active_timepoints(embed_data, times_cube, config.active_times)

    # Step 3: PCA dimensionality reduction
    logging.info("Step 3/5: PCA dimensionality reduction")
    dimred = _apply_pca_reduction(embed_data, movetimes, config.dim)

    # Step 4: Point cloud sampling (denoising)
    logging.info("Step 4/5: Point cloud denoising")
    indstemp = _apply_denoising(dimred, config)

    # Step 5: Compute persistent homology
    logging.info("Step 5/5: Computing persistent homology")
    persistence = _compute_persistence_homology(dimred, indstemp, config)

    logging.info("TDA analysis completed successfully")

    # Return all necessary data in dictionary format
    return {
        "persistence": persistence,
        "indstemp": indstemp,
        "movetimes": movetimes,
        "n_points": config.n_points,
    }


def _downsample_timepoints(embed_data: np.ndarray, num_times: int) -> np.ndarray:
    """Downsample timepoints for computational efficiency."""
    return np.arange(0, embed_data.shape[0], num_times)


def _select_active_timepoints(
    embed_data: np.ndarray, times_cube: np.ndarray, active_times: int
) -> np.ndarray:
    """Select most active timepoints based on total activity."""
    activity_scores = np.sum(embed_data[times_cube, :], 1)
    # Match external TDAvis: sort indices first, then map to times_cube
    movetimes = np.sort(np.argsort(activity_scores)[-active_times:])
    return times_cube[movetimes]


def _apply_pca_reduction(embed_data: np.ndarray, movetimes: np.ndarray, dim: int) -> np.ndarray:
    """Apply PCA dimensionality reduction."""
    scaled_data = preprocessing.scale(embed_data[movetimes, :])
    dimred, *_ = _pca(scaled_data, dim=dim)
    return dimred


def _apply_denoising(dimred: np.ndarray, config: TDAConfig) -> np.ndarray:
    """Apply point cloud denoising."""
    indstemp, *_ = _sample_denoising(
        dimred,
        k=config.k,
        num_sample=config.n_points,
        omega=1,  # Match external TDAvis: uses 1, not default 0.2
        metric=config.metric,
    )
    return indstemp


def _compute_persistence_homology(
    dimred: np.ndarray, indstemp: np.ndarray, config: TDAConfig
) -> dict[str, Any]:
    """Compute persistent homology using ripser."""
    d = _second_build(dimred, indstemp, metric=config.metric, nbs=config.nbs)
    np.fill_diagonal(d, 0)

    return ripser(
        d,
        maxdim=config.maxdim,
        coeff=config.coeff,
        do_cocycles=True,
        distance_matrix=True,
        progress_bar=config.progress_bar,
    )


def _perform_shuffle_analysis(embed_data: np.ndarray, config: TDAConfig) -> dict[int, Any]:
    """Perform shuffle analysis with progress tracking."""
    print(f"\nStarting shuffle analysis with {config.num_shuffles} iterations...")

    # Create parameters dict for shuffle analysis
    shuffle_params = {
        "dim": config.dim,
        "num_times": config.num_times,
        "active_times": config.active_times,
        "k": config.k,
        "n_points": config.n_points,
        "metric": config.metric,
        "nbs": config.nbs,
        "maxdim": config.maxdim,
        "coeff": config.coeff,
    }

    shuffle_max = _run_shuffle_analysis(
        embed_data,
        num_shuffles=config.num_shuffles,
        num_cores=Constants.MULTIPROCESSING_CORES,
        progress_bar=config.progress_bar,
        **shuffle_params,
    )

    # Print shuffle analysis summary
    _print_shuffle_summary(shuffle_max)

    return shuffle_max


def _print_shuffle_summary(shuffle_max: dict[int, Any]) -> None:
    """Print summary of shuffle analysis results."""
    print("\nSummary of shuffle-based analysis:")
    for dim_idx in [0, 1, 2]:
        if shuffle_max and dim_idx in shuffle_max and shuffle_max[dim_idx]:
            values = shuffle_max[dim_idx]
            print(
                f"H{dim_idx}: {len(values)} valid iterations | "
                f"Mean maximum persistence: {np.mean(values):.4f} | "
                f"99.9th percentile: {np.percentile(values, 99.9):.4f}"
            )


def _handle_visualization(
    real_persistence: dict[str, Any], shuffle_max: dict[int, Any] | None, config: TDAConfig
) -> None:
    """Handle visualization based on configuration."""
    if config.show:
        if config.do_shuffle and shuffle_max is not None:
            _plot_barcode_with_shuffle(real_persistence, shuffle_max)
        else:
            _plot_barcode(real_persistence)
        plt.show()
    else:
        plt.close()


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


def _gaussian_filter1d(
    input,
    sigma,
    axis=-1,
    order=0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
    *,
    radius=None,
):
    """1-D Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode_reflect)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    radius : None or int, optional
        Radius of the Gaussian kernel. If specified, the size of
        the kernel will be ``2*radius + 1``, and `truncate` is ignored.
        Default is None.

    Returns
    -------
    gaussian_filter1d : ndarray

    Notes
    -----
    The Gaussian kernel will have size ``2*radius + 1`` along each axis. If
    `radius` is None, a default ``radius = round(truncate * sigma)`` will be
    used.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter1d
    >>> import numpy as np
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
    >>> _gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(101).cumsum()
    >>> y3 = _gaussian_filter1d(x, 3)
    >>> y6 = _gaussian_filter1d(x, 6)
    >>> plt.plot(x, 'k', label='original data')
    >>> plt.plot(y3, '--', label='filtered, sigma=3')
    >>> plt.plot(y6, ':', label='filtered, sigma=6')
    >>> plt.legend()
    >>> plt.grid()
    >>> plt.show()

    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError("Radius must be a nonnegative integer.")
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return _correlate1d(input, weights, axis, output, mode, cval, 0)


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def _correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D correlation along the given axis.

    The lines of the array along the given axis are correlated with the
    given weights.

    Parameters
    ----------
    %(input)s
    weights : array
        1-D sequence of numbers.
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s

    Returns
    -------
    result : ndarray
        Correlation result. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy.ndimage import correlate1d
    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([ 8, 26,  8, 12,  7, 28, 36,  9])
    """
    input = np.asarray(input)
    weights = np.asarray(weights)
    complex_input = input.dtype.kind == "c"
    complex_weights = weights.dtype.kind == "c"
    if complex_input or complex_weights:
        if complex_weights:
            weights = weights.conj()
            weights = weights.astype(np.complex128, copy=False)
        kwargs = dict(axis=axis, mode=mode, origin=origin)
        output = _ni_support._get_output(output, input, complex_output=True)
        return _complex_via_real_components(_correlate1d, input, weights, output, cval, **kwargs)

    output = _ni_support._get_output(output, input)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError("no filter weights given")
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = _normalize_axis_index(axis, input.ndim)
    if _invalid_origin(origin, len(weights)):
        raise ValueError(
            "Invalid origin; origin must satisfy "
            "-(len(weights) // 2) <= origin <= "
            "(len(weights)-1) // 2"
        )
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate1d(input, weights, axis, output, mode, cval, origin)
    return output


def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    """Complex convolution via a linear combination of real convolutions."""
    complex_input = input.dtype.kind == "c"
    complex_weights = weights.dtype.kind == "c"
    if complex_input and complex_weights:
        # real component of the output
        func(input.real, weights.real, output=output.real, cval=np.real(cval), **kwargs)
        output.real -= func(input.imag, weights.imag, output=None, cval=np.imag(cval), **kwargs)
        # imaginary component of the output
        func(input.real, weights.imag, output=output.imag, cval=np.real(cval), **kwargs)
        output.imag += func(input.imag, weights.real, output=None, cval=np.imag(cval), **kwargs)
    elif complex_input:
        func(input.real, weights, output=output.real, cval=np.real(cval), **kwargs)
        func(input.imag, weights, output=output.imag, cval=np.imag(cval), **kwargs)
    else:
        if np.iscomplexobj(cval):
            raise ValueError("Cannot provide a complex-valued cval when the input is real.")
        func(input, weights.real, output=output.real, cval=cval, **kwargs)
        func(input, weights.imag, output=output.imag, cval=cval, **kwargs)
    return output


def _normalize_axis_index(axis, ndim):
    # Check if `axis` is in the correct range and normalize it
    if axis < -ndim or axis >= ndim:
        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        raise AxisError(msg)

    if axis < 0:
        axis = axis + ndim
    return axis


def _compute_persistence(
    sspikes,
    dim=6,
    num_times=5,
    active_times=15000,
    k=1000,
    n_points=1200,
    metric="cosine",
    nbs=800,
    maxdim=1,
    coeff=47,
    progress_bar=True,
):
    # Time point downsampling
    times_cube = np.arange(0, sspikes.shape[0], num_times)

    # Select most active time points
    movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube, :], 1))[-active_times:])
    movetimes = times_cube[movetimes]

    # PCA dimensionality reduction
    scaled_data = preprocessing.scale(sspikes[movetimes, :])
    dimred, *_ = _pca(scaled_data, dim=dim)

    # Point cloud sampling (denoising)
    indstemp, *_ = _sample_denoising(dimred, k, n_points, 1, metric)

    # Build distance matrix
    d = _second_build(dimred, indstemp, metric=metric, nbs=nbs)
    np.fill_diagonal(d, 0)

    # Compute persistent homology
    persistence = ripser(
        d,
        maxdim=maxdim,
        coeff=coeff,
        do_cocycles=True,
        distance_matrix=True,
        progress_bar=progress_bar,
    )

    return persistence


def _pca(data, dim=2):
    """
    Perform PCA (Principal Component Analysis) for dimensionality reduction.

    Parameters:
        data (ndarray): Input data matrix of shape (N_samples, N_features).
        dim (int): Target dimension for PCA projection.

    Returns:
        components (ndarray): Projected data of shape (N_samples, dim).
        var_exp (list): Variance explained by each principal component.
        evals (ndarray): Eigenvalues corresponding to the selected components.
    """
    if dim < 2:
        return data, [0]
    m, n = data.shape
    # mean center the data
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eig(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dim]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    tot = np.sum(evals)
    var_exp = [(i / tot) * 100 for i in sorted(evals[:dim], reverse=True)]
    components = np.dot(evecs.T, data.T).T
    return components, var_exp, evals[:dim]


def _sample_denoising(data, k=10, num_sample=500, omega=0.2, metric="euclidean"):
    """
    Perform denoising and greedy sampling based on mutual k-NN graph.

    Parameters:
        data (ndarray): High-dimensional point cloud data.
        k (int): Number of neighbors for local density estimation.
        num_sample (int): Number of samples to retain.
        omega (float): Suppression factor during greedy sampling.
        metric (str): Distance metric used for kNN ('euclidean', 'cosine', etc).

    Returns:
        inds (ndarray): Indices of sampled points.
        d (ndarray): Pairwise similarity matrix of sampled points.
        Fs (ndarray): Sampling scores at each step.
    """
    if HAS_NUMBA:
        return _sample_denoising_numba(data, k, num_sample, omega, metric)
    else:
        return _sample_denoising_numpy(data, k, num_sample, omega, metric)


def _sample_denoising_numpy(data, k=10, num_sample=500, omega=0.2, metric="euclidean"):
    """Original numpy implementation for fallback."""
    n = data.shape[0]
    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = _smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = result + transpose - prod_matrix
    result.eliminate_zeros()
    X = result.toarray()
    F = np.sum(X, 1)
    Fs = np.zeros(num_sample)
    Fs[0] = np.max(F)
    i = np.argmax(F)
    inds_all = np.arange(n)
    inds_left = inds_all > -1
    inds_left[i] = False
    inds = np.zeros(num_sample, dtype=int)
    inds[0] = i
    for j in np.arange(1, num_sample):
        F -= omega * X[i, :]
        Fmax = np.argmax(F[inds_left])
        # Exactly match external TDAvis implementation (including the indexing logic)
        Fs[j] = F[Fmax]
        i = inds_all[inds_left][Fmax]

        inds_left[i] = False
        inds[j] = i
    d = np.zeros((num_sample, num_sample))

    for j, i in enumerate(inds):
        d[j, :] = X[i, inds]
    return inds, d, Fs


def _sample_denoising_numba(data, k=10, num_sample=500, omega=0.2, metric="euclidean"):
    """Optimized numba implementation."""
    n = data.shape[0]
    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = _smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)

    # Build symmetric adjacency matrix using optimized function
    X_adj = _build_adjacency_matrix_numba(rows, cols, vals, n)

    # Greedy sampling using optimized function
    inds, Fs = _greedy_sampling_numba(X_adj, num_sample, omega)

    # Build final distance matrix
    d = _build_distance_matrix_numba(X_adj, inds)

    return inds, d, Fs


@njit(fastmath=True)
def _build_adjacency_matrix_numba(rows, cols, vals, n):
    """Build symmetric adjacency matrix efficiently with numba.

    This matches the scipy sparse matrix operations:
    result = result + transpose - prod_matrix
    where prod_matrix = result.multiply(transpose)
    """
    # Initialize matrices
    X = np.zeros((n, n), dtype=np.float64)
    X_T = np.zeros((n, n), dtype=np.float64)

    # Build adjacency matrix and its transpose simultaneously
    for i in range(len(rows)):
        X[rows[i], cols[i]] = vals[i]
        X_T[cols[i], rows[i]] = vals[i]  # Transpose

    # Apply the symmetrization formula: A = A + A^T - A ⊙ A^T (vectorized)
    # This matches scipy's: result + transpose - prod_matrix
    X[:, :] = X + X_T - X * X_T

    return X


@njit(fastmath=True)
def _greedy_sampling_numba(X, num_sample, omega):
    """Optimized greedy sampling with numba."""
    n = X.shape[0]
    F = np.sum(X, axis=1)
    Fs = np.zeros(num_sample)
    inds = np.zeros(num_sample, dtype=np.int64)
    inds_left = np.ones(n, dtype=np.bool_)

    # Initialize with maximum F
    i = np.argmax(F)
    Fs[0] = F[i]
    inds[0] = i
    inds_left[i] = False

    # Greedy sampling loop
    for j in range(1, num_sample):
        # Update F values
        for k in range(n):
            F[k] -= omega * X[i, k]

        # Find maximum among remaining points (matching numpy logic exactly)
        max_val = -np.inf
        max_idx = -1
        for k in range(n):
            if inds_left[k] and F[k] > max_val:
                max_val = F[k]
                max_idx = k

        # Record the F value using the selected index (matching external TDAvis)
        i = max_idx
        Fs[j] = F[i]
        inds[j] = i
        inds_left[i] = False

    return inds, Fs


@njit(fastmath=True)
def _build_distance_matrix_numba(X, inds):
    """Build final distance matrix efficiently with numba."""
    num_sample = len(inds)
    d = np.zeros((num_sample, num_sample))

    for j in range(num_sample):
        for k in range(num_sample):
            d[j, k] = X[inds[j], inds[k]]

    return d


@njit(fastmath=True)
def _smooth_knn_dist(distances, k, n_iter=64, local_connectivity=0.0, bandwidth=1.0):
    """
    Compute smoothed local distances for kNN graph with entropy balancing.

    Parameters:
        distances (ndarray): kNN distance matrix.
        k (int): Number of neighbors.
        n_iter (int): Number of binary search iterations.
        local_connectivity (float): Minimum local connectivity.
        bandwidth (float): Bandwidth parameter.

    Returns:
        sigmas (ndarray): Smoothed sigma values for each point.
        rhos (ndarray): Minimum distances (connectivity cutoff) for each point.
    """
    target = np.log2(k) * bandwidth
    #    target = np.log(k) * bandwidth
    #    target = k

    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = np.inf
        mid = 1.0

        # Vectorized computation of non-zero distances
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > 1e-5:
                    rho[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1])
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        # Vectorized binary search loop - compute all at once instead of loop
        for _ in range(n_iter):
            # Vectorized computation: compute all distances at once
            d_array = distances[i, 1:] - rho[i]
            # Vectorized conditional: use np.where for conditional computation
            psum = np.sum(np.where(d_array > 0, np.exp(-(d_array / mid)), 1.0))

            if np.fabs(psum - target) < 1e-5:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        # Optimized mean computation - reuse ith_distances
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < 1e-3 * mean_ith_distances:
                result[i] = 1e-3 * mean_ith_distances
        else:
            if result[i] < 1e-3 * mean_distances:
                result[i] = 1e-3 * mean_distances

    return result, rho


@njit(parallel=True, fastmath=True)
def _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """
    Compute membership strength matrix from smoothed kNN graph.

    Parameters:
        knn_indices (ndarray): Indices of k-nearest neighbors.
        knn_dists (ndarray): Corresponding distances.
        sigmas (ndarray): Local bandwidths.
        rhos (ndarray): Minimum distance thresholds.

    Returns:
        rows (ndarray): Row indices for sparse matrix.
        cols (ndarray): Column indices for sparse matrix.
        vals (ndarray): Weight values for sparse matrix.
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                # val = ((knn_dists[i, j] - rhos[i]) / (sigmas[i]))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def _second_build(data, indstemp, nbs=800, metric="cosine"):
    """
    Reconstruct distance matrix after denoising for persistent homology.

    Parameters:
        data (ndarray): PCA-reduced data matrix.
        indstemp (ndarray): Indices of sampled points.
        nbs (int): Number of neighbors in reconstructed graph.
        metric (str): Distance metric ('cosine', 'euclidean', etc).

    Returns:
        d (ndarray): Symmetric distance matrix used for persistent homology.
    """
    # Filter the data using the sampled point indices
    data = data[indstemp, :]

    # Compute the pairwise distance matrix
    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :nbs]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    # Compute smoothed kernel widths
    sigmas, rhos = _smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
    rows, cols, vals = _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)

    # Construct a sparse graph
    result = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = result + transpose - prod_matrix
    result.eliminate_zeros()

    # Build the final distance matrix
    d = result.toarray()
    # Match external TDAvis: direct negative log without epsilon handling
    # Temporarily suppress divide by zero warning to match external behavior
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -np.log(d)
    np.fill_diagonal(d, 0)

    return d


def _run_shuffle_analysis(sspikes, num_shuffles=1000, num_cores=4, progress_bar=True, **kwargs):
    """Perform shuffle analysis with optimized computation."""
    return _run_shuffle_analysis_multiprocessing(
        sspikes, num_shuffles, num_cores, progress_bar, **kwargs
    )


def _run_shuffle_analysis_multiprocessing(
    sspikes, num_shuffles=1000, num_cores=4, progress_bar=True, **kwargs
):
    """Original multiprocessing implementation for fallback."""
    # Use numpy arrays with NaN for failed results (more efficient than None filtering)
    max_lifetimes = {
        0: np.full(num_shuffles, np.nan),
        1: np.full(num_shuffles, np.nan),
        2: np.full(num_shuffles, np.nan),
    }

    # Estimate runtime with a test iteration
    logging.info("Running test iteration to estimate runtime...")

    _ = _process_single_shuffle((0, sspikes, kwargs))

    # Prepare task list
    tasks = [(i, sspikes, kwargs) for i in range(num_shuffles)]
    logging.info(
        f"Starting shuffle analysis with {num_shuffles} iterations using {num_cores} cores..."
    )

    # Use multiprocessing pool for parallel processing
    with mp.Pool(processes=num_cores) as pool:
        results = list(pool.imap(_process_single_shuffle, tasks))
        logging.info("Shuffle analysis completed")

    # Collect results - use indexing instead of append for better performance
    for idx, res in enumerate(results):
        for dim, lifetime in res.items():
            max_lifetimes[dim][idx] = lifetime

    # Filter out NaN values (failed results) - convert to list for consistency
    for dim in max_lifetimes:
        max_lifetimes[dim] = max_lifetimes[dim][~np.isnan(max_lifetimes[dim])].tolist()

    return max_lifetimes


@njit(fastmath=True)
def _fast_pca_transform(data, components):
    """Fast PCA transformation using numba."""
    return np.dot(data, components.T)


def _process_single_shuffle(args):
    """Process a single shuffle task."""
    i, sspikes, kwargs = args
    try:
        shuffled_data = _shuffle_spike_trains(sspikes)
        persistence = _compute_persistence(shuffled_data, **kwargs)

        dim_max_lifetimes = {}
        for dim in [0, 1, 2]:
            if dim < len(persistence["dgms"]):
                # Filter out infinite values
                valid_bars = [bar for bar in persistence["dgms"][dim] if not np.isinf(bar[1])]
                if valid_bars:
                    lifetimes = [bar[1] - bar[0] for bar in valid_bars]
                    if lifetimes:
                        dim_max_lifetimes[dim] = max(lifetimes)
        return dim_max_lifetimes
    except Exception as e:
        print(f"Shuffle {i} failed: {str(e)}")
        return {}


def _shuffle_spike_trains(sspikes):
    """Perform random circular shift on spike trains."""
    shuffled = sspikes.copy()
    num_neurons = shuffled.shape[1]

    # Independent shift for each neuron
    for n in range(num_neurons):
        shift = np.random.randint(0, int(shuffled.shape[0] * 0.1))
        shuffled[:, n] = np.roll(shuffled[:, n], shift)

    return shuffled


def _plot_barcode(persistence):
    """
    Plot barcode diagram from persistent homology result.

    Parameters:
        persistence (dict): Persistent homology result with 'dgms' key.
    """
    cs = np.repeat([[0, 0.55, 0.2]], 3).reshape(3, 3).T  # RGB color for each dimension
    alpha = 1
    inf_delta = 0.1
    colormap = cs
    dgms = persistence["dgms"]
    maxdim = len(dgms) - 1
    dims = np.arange(maxdim + 1)
    labels = ["$H_0$", "$H_1$", "$H_2$"]

    # Determine axis range
    min_birth, max_death = 0, 0
    for dim in dims:
        persistence_dim = dgms[dim][~np.isinf(dgms[dim][:, 1]), :]
        if persistence_dim.size > 0:
            min_birth = min(min_birth, np.min(persistence_dim))
            max_death = max(max_death, np.max(persistence_dim))

    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta
    axis_start = min_birth - delta

    # Create plot
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(len(dims), 1)

    for dim in dims:
        axes = plt.subplot(gs[dim])
        axes.axis("on")
        axes.set_yticks([])
        axes.set_ylabel(labels[dim], rotation=0, labelpad=20, fontsize=12)

        d = np.copy(dgms[dim])
        d[np.isinf(d[:, 1]), 1] = infinity
        dlife = d[:, 1] - d[:, 0]

        # Select top 30 bars by lifetime
        dinds = np.argsort(dlife)[-30:]
        if dim > 0:
            dinds = dinds[np.flip(np.argsort(d[dinds, 0]))]

        axes.barh(
            0.5 + np.arange(len(dinds)),
            dlife[dinds],
            height=0.8,
            left=d[dinds, 0],
            alpha=alpha,
            color=colormap[dim],
            linewidth=0,
        )

        axes.plot([0, 0], [0, len(dinds)], c="k", linestyle="-", lw=1)
        axes.plot([0, len(dinds)], [0, 0], c="k", linestyle="-", lw=1)
        axes.set_xlim([axis_start, infinity])

    plt.tight_layout()
    return fig


def _plot_barcode_with_shuffle(persistence, shuffle_max):
    """
    Plot barcode with shuffle region markers.
    """
    # Handle case where shuffle_max is None
    if shuffle_max is None:
        shuffle_max = {}

    cs = np.repeat([[0, 0.55, 0.2]], 3).reshape(3, 3).T
    alpha = 1
    inf_delta = 0.1
    colormap = cs
    maxdim = len(persistence["dgms"]) - 1
    dims = np.arange(maxdim + 1)

    min_birth, max_death = 0, 0
    for dim in dims:
        # Filter out infinite values
        valid_bars = [bar for bar in persistence["dgms"][dim] if not np.isinf(bar[1])]
        if valid_bars:
            min_birth = min(min_birth, np.min(valid_bars))
            max_death = max(max_death, np.max(valid_bars))

    # Handle case with no valid bars
    if max_death == 0 and min_birth == 0:
        min_birth = 0
        max_death = 1

    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(len(dims), 1)

    # Get shuffle thresholds (99.9th percentile for each dimension)
    thresholds = {}
    for dim in dims:
        if dim in shuffle_max and shuffle_max[dim]:
            thresholds[dim] = np.percentile(shuffle_max[dim], 99.9)
        else:
            thresholds[dim] = 0

    for _, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes.axis("off")

        # Add gray background to represent shuffle region
        if dim in thresholds:
            axes.axvspan(0, thresholds[dim], alpha=0.2, color="gray", zorder=-3)
            axes.axvline(x=thresholds[dim], color="gray", linestyle="--", alpha=0.7)

        # Do not pre-filter out infinite bars; copy the full diagram instead
        d = np.copy(persistence["dgms"][dim])
        if d.size == 0:
            d = np.zeros((0, 2))

        # Map infinite death values to a finite upper bound for visualization
        d[np.isinf(d[:, 1]), 1] = infinity
        dlife = d[:, 1] - d[:, 0]

        # Select top 30 longest-lived bars
        if len(dlife) > 0:
            dinds = np.argsort(dlife)[-30:]
            if dim > 0:
                dinds = dinds[np.flip(np.argsort(d[dinds, 0]))]

            # Mark significant bars
            significant_bars = []
            for idx in dinds:
                if dlife[idx] > thresholds.get(dim, 0):
                    significant_bars.append(idx)

            # Draw bars
            for i, idx in enumerate(dinds):
                color = "red" if idx in significant_bars else colormap[dim]
                axes.barh(
                    0.5 + i,
                    dlife[idx],
                    height=0.8,
                    left=d[idx, 0],
                    alpha=alpha,
                    color=color,
                    linewidth=0,
                )

            indsall = len(dinds)
        else:
            indsall = 0

        axes.plot([0, 0], [0, indsall], c="k", linestyle="-", lw=1)
        axes.plot([0, indsall], [0, 0], c="k", linestyle="-", lw=1)
        axes.set_xlim([0, infinity])
        axes.set_title(f"$H_{dim}$", loc="left")

    plt.tight_layout()
    return fig


def decode_circular_coordinates(
    persistence_result: dict[str, Any],
    spike_data: dict[str, Any],
    real_ground: bool = True,
    real_of: bool = True,
    save_path: str | None = None,
) -> dict[str, Any]:
    """
    Decode circular coordinates (bump positions) from cohomology.

    Parameters:
        persistence_result : dict containing persistence analysis results with keys:
            - 'persistence': persistent homology result
            - 'indstemp': indices of sampled points
            - 'movetimes': selected time points
            - 'n_points': number of sampled points
        spike_data : dict, optional
            Spike data dictionary containing 'spike', 't', and optionally 'x', 'y'
        real_ground : bool
            Whether x, y, t ground truth exists
        real_of : bool
            Whether experiment was performed in open field
        save_path : str, optional
            Path to save decoding results. If None, saves to 'Results/spikes_decoding.npz'

    Returns:
        dict : Dictionary containing decoding results with keys:
            - 'coords': decoded coordinates for all timepoints
            - 'coordsbox': decoded coordinates for box timepoints
            - 'times': time indices for coords
            - 'times_box': time indices for coordsbox
            - 'centcosall': cosine centroids
            - 'centsinall': sine centroids
    """
    ph_classes = [0, 1]  # Decode the ith most persistent cohomology class
    num_circ = len(ph_classes)
    dec_tresh = 0.99
    coeff = 47

    # Extract persistence analysis results
    persistence = persistence_result["persistence"]
    indstemp = persistence_result["indstemp"]
    movetimes = persistence_result["movetimes"]
    n_points = persistence_result["n_points"]

    diagrams = persistence["dgms"]  # the multiset describing the lives of the persistence classes
    cocycles = persistence["cocycles"][1]  # the cocycle representatives for the 1-dim classes
    dists_land = persistence["dperm2all"]  # the pairwise distance between the points
    births1 = diagrams[1][:, 0]  # the time of birth for the 1-dim classes
    deaths1 = diagrams[1][:, 1]  # the time of death for the 1-dim classes
    deaths1[np.isinf(deaths1)] = 0
    lives1 = deaths1 - births1  # the lifetime for the 1-dim classes
    iMax = np.argsort(lives1)
    coords1 = np.zeros((num_circ, len(indstemp)))
    threshold = births1[iMax[-2]] + (deaths1[iMax[-2]] - births1[iMax[-2]]) * dec_tresh

    for c in ph_classes:
        cocycle = cocycles[iMax[-(c + 1)]]
        coords1[c, :], inds = _get_coords(cocycle, threshold, len(indstemp), dists_land, coeff)

    if real_ground:  # 用户所提供的数据是否有真实的xyt
        sspikes, xx, yy, tt = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=True)
        )
    else:
        sspikes = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=False)
        )

    num_neurons = sspikes.shape[1]
    centcosall = np.zeros((num_neurons, 2, n_points))
    centsinall = np.zeros((num_neurons, 2, n_points))
    dspk = preprocessing.scale(sspikes[movetimes[indstemp], :])

    for neurid in range(num_neurons):
        spktemp = dspk[:, neurid].copy()
        centcosall[neurid, :, :] = np.multiply(np.cos(coords1[:, :] * 2 * np.pi), spktemp)
        centsinall[neurid, :, :] = np.multiply(np.sin(coords1[:, :] * 2 * np.pi), spktemp)

    if real_ground:  # 用户所提供的数据是否有真实的xyt
        sspikes, xx, yy, tt = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=True)
        )
        spikes, __, __, __ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=False, speed_filter=True)
        )
    else:
        sspikes = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=False)
        )
        spikes = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=False, speed_filter=False)
        )

    times = np.where(np.sum(spikes > 0, 1) >= 1)[0]
    dspk = preprocessing.scale(sspikes)
    sspikes = sspikes[times, :]
    dspk = dspk[times, :]

    a = np.zeros((len(sspikes[:, 0]), 2, num_neurons))
    for n in range(num_neurons):
        a[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centcosall[n, :, :], 1))

    c = np.zeros((len(sspikes[:, 0]), 2, num_neurons))
    for n in range(num_neurons):
        c[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centsinall[n, :, :], 1))

    mtot2 = np.sum(c, 2)
    mtot1 = np.sum(a, 2)
    coords = np.arctan2(mtot2, mtot1) % (2 * np.pi)

    if real_of:  # 用户的数据是否是来自真实的OF场地
        coordsbox = coords.copy()
        times_box = times.copy()
    else:
        sspikes, xx, yy, tt = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=True, speed_filter=True)
        )
        spikes, __, __, __ = embed_spike_trains(
            spike_data, config=SpikeEmbeddingConfig(smooth=False, speed_filter=True)
        )
        dspk = preprocessing.scale(sspikes)
        times_box = np.where(np.sum(spikes > 0, 1) >= 1)[0]
        dspk = dspk[times_box, :]

        a = np.zeros((len(times_box), 2, num_neurons))
        for n in range(num_neurons):
            a[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centcosall[n, :, :], 1))

        c = np.zeros((len(times_box), 2, num_neurons))
        for n in range(num_neurons):
            c[:, :, n] = np.multiply(dspk[:, n : n + 1], np.sum(centsinall[n, :, :], 1))

        mtot2 = np.sum(c, 2)
        mtot1 = np.sum(a, 2)
        coordsbox = np.arctan2(mtot2, mtot1) % (2 * np.pi)

    # Prepare results dictionary
    results = {
        "coords": coords,
        "coordsbox": coordsbox,
        "times": times,
        "times_box": times_box,
        "centcosall": centcosall,
        "centsinall": centsinall,
    }

    # Save results
    if save_path is None:
        os.makedirs("Results", exist_ok=True)
        save_path = "Results/spikes_decoding.npz"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **results)

    return results


def plot_cohomap(
    decoding_result: dict[str, Any],
    position_data: dict[str, Any],
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (10, 4),
    dpi: int = 300,
    subsample: int = 10,
) -> plt.Figure:
    """
    Visualize CohoMap 1.0: decoded circular coordinates mapped onto spatial trajectory.

    Creates a two-panel visualization showing how the two decoded circular coordinates
    vary across the animal's spatial trajectory. Each panel displays the spatial path
    colored by the cosine of one circular coordinate dimension.

    Parameters:
        decoding_result : dict
            Dictionary from decode_circular_coordinates() containing:
            - 'coordsbox': decoded coordinates for box timepoints (n_times x n_dims)
            - 'times_box': time indices for coordsbox
        position_data : dict
            Position data containing 'x' and 'y' arrays for spatial coordinates
        save_path : str, optional
            Path to save the visualization. If None, no save performed
        show : bool, default=False
            Whether to display the visualization
        figsize : tuple[int, int], default=(10, 4)
            Figure size (width, height) in inches
        dpi : int, default=300
            Resolution for saved figure
        subsample : int, default=10
            Subsampling interval for plotting (plot every Nth timepoint)

    Returns:
        plt.Figure : The matplotlib figure object

    Raises:
        KeyError : If required keys are missing from input dictionaries
        ValueError : If data dimensions are inconsistent
        IndexError : If time indices are out of bounds

    Examples:
        >>> # Decode coordinates
        >>> decoding = decode_circular_coordinates(persistence_result, spike_data)
        >>> # Visualize with trajectory data
        >>> fig = plot_cohomap(
        ...     decoding,
        ...     position_data={'x': xx, 'y': yy},
        ...     save_path='cohomap.png',
        ...     show=True
        ... )
    """
    try:
        # Extract data
        coordsbox = decoding_result["coordsbox"]
        times_box = decoding_result["times_box"]
        xx = position_data["x"]
        yy = position_data["y"]

        # Subsample time indices for plotting
        plot_times = np.arange(0, len(coordsbox), subsample)

        # Create a two-panel figure (one per cohomology dimension)
        plt.set_cmap("viridis")
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Plot for the first circular coordinate
        ax[0].axis("off")
        ax[0].set_aspect("equal", "box")
        im0 = ax[0].scatter(
            xx[times_box][plot_times],
            yy[times_box][plot_times],
            c=np.cos(coordsbox[plot_times, 0]),
            s=8,
            cmap="viridis",
        )
        plt.colorbar(im0, ax=ax[0], label="cos(coord)")
        ax[0].set_title("CohoMap Dim 1", fontsize=10)

        # Plot for the second circular coordinate
        ax[1].axis("off")
        ax[1].set_aspect("equal", "box")
        im1 = ax[1].scatter(
            xx[times_box][plot_times],
            yy[times_box][plot_times],
            c=np.cos(coordsbox[plot_times, 1]),
            s=8,
            cmap="viridis",
        )
        plt.colorbar(im1, ax=ax[1], label="cos(coord)")
        ax[1].set_title("CohoMap Dim 2", fontsize=10)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=dpi)
                print(f"CohoMap visualization saved to {save_path}")
            except Exception as e:
                print(f"Error saving CohoMap visualization: {e}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    except (KeyError, ValueError, IndexError) as e:
        print(f"CohoMap visualization failed: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error in CohoMap visualization: {e}")
        raise


def plot_3d_bump_on_torus(
    decoding_result: dict[str, Any] | str,
    spike_data: dict[str, Any],
    config: CANN2DPlotConfig | None = None,
    save_path: str | None = None,
    numangsint: int = 51,
    r1: float = 1.5,
    r2: float = 1.0,
    window_size: int = 300,
    frame_step: int = 5,
    n_frames: int = 20,
    fps: int = 5,
    show_progress: bool = True,
    show: bool = True,
    figsize: tuple[int, int] = (8, 8),
    **kwargs,
) -> animation.FuncAnimation:
    """
    Visualize the movement of the neural activity bump on a torus using matplotlib animation.

    This function follows the canns.analyzer.plotting patterns for animation generation
    with progress tracking and proper resource cleanup.

    Parameters:
        decoding_result : dict or str
            Dictionary containing decoding results with 'coordsbox' and 'times_box' keys,
            or path to .npz file containing these results
        spike_data : dict, optional
            Spike data dictionary containing spike information
        config : PlotConfig, optional
            Configuration object for unified plotting parameters
        **kwargs : backward compatibility parameters
        save_path : str, optional
            Path to save the animation (e.g., 'animation.gif' or 'animation.mp4')
        numangsint : int
            Grid resolution for the torus surface
        r1 : float
            Major radius of the torus
        r2 : float
            Minor radius of the torus
        window_size : int
            Time window (in number of time points) for each frame
        frame_step : int
            Step size to slide the time window between frames
        n_frames : int
            Total number of frames in the animation
        fps : int
            Frames per second for the output animation
        show_progress : bool
            Whether to show progress bar during generation
        show : bool
            Whether to display the animation
        figsize : tuple[int, int]
            Figure size for the animation

    Returns:
        matplotlib.animation.FuncAnimation : The animation object
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = CANN2DPlotConfig.for_torus_animation(**kwargs)

    # Override config with any explicitly passed parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Extract configuration values
    save_path = config.save_path if config.save_path else save_path
    show = config.show
    figsize = config.figsize
    fps = config.fps
    show_progress = config.show_progress_bar
    numangsint = config.numangsint
    r1 = config.r1
    r2 = config.r2
    window_size = config.window_size
    frame_step = config.frame_step
    n_frames = config.n_frames

    # Load decoding results if path is provided
    if isinstance(decoding_result, str):
        f = np.load(decoding_result, allow_pickle=True)
        coords = f["coordsbox"]
        times = f["times_box"]
        f.close()
    else:
        coords = decoding_result["coordsbox"]
        times = decoding_result["times_box"]

    spk, *_ = embed_spike_trains(
        spike_data, config=SpikeEmbeddingConfig(smooth=False, speed_filter=True)
    )

    # Pre-compute torus geometry (constant across frames - optimization)
    # Create grid for torus surface
    x_edge = np.linspace(0, 2 * np.pi, numangsint)
    y_edge = np.linspace(0, 2 * np.pi, numangsint)
    X_grid, Y_grid = np.meshgrid(x_edge, y_edge)
    X_transformed = (X_grid + np.pi / 5) % (2 * np.pi)

    # Pre-compute torus geometry (only done once!)
    torus_x = (r1 + r2 * np.cos(X_transformed)) * np.cos(Y_grid)
    torus_y = (r1 + r2 * np.cos(X_transformed)) * np.sin(Y_grid)
    torus_z = -r2 * np.sin(X_transformed)  # Flip torus surface orientation

    # Prepare animation data (now only stores colors, not geometry)
    frame_data = []
    prev_m = None

    for frame_idx in tqdm(range(n_frames), desc="Processing frames"):
        start_idx = frame_idx * frame_step
        end_idx = start_idx + window_size
        if end_idx > np.max(times):
            break

        mask = (times >= start_idx) & (times < end_idx)
        coords_window = coords[mask]
        if len(coords_window) == 0:
            continue

        spk_window = spk[times[mask], :]
        activity = np.sum(spk_window, axis=1)

        m, _, _, _ = binned_statistic_2d(
            coords_window[:, 0],
            coords_window[:, 1],
            activity,
            statistic="sum",
            bins=np.linspace(0, 2 * np.pi, numangsint - 1),
        )
        m = np.nan_to_num(m)
        m = _smooth_tuning_map(m, numangsint - 1, sig=4.0, bClose=True)
        m = gaussian_filter(m, sigma=1.0)

        if prev_m is not None:
            m = 0.7 * prev_m + 0.3 * m
        prev_m = m

        # Store only activity map (m) and metadata, reuse geometry
        frame_data.append({"m": m, "time": start_idx * frame_step})

    if not frame_data:
        raise ProcessingError("No valid frames generated for animation")

    # Create figure and animation with optimized geometry reuse
    fig = plt.figure(figsize=figsize)

    try:
        ax = fig.add_subplot(111, projection="3d")
        # Batch set axis properties (reduces overhead)
        ax.set_zlim(-2, 2)
        ax.view_init(-125, 135)
        ax.axis("off")

        # Initialize with first frame
        first_frame = frame_data[0]
        surface = ax.plot_surface(
            torus_x,  # Pre-computed geometry
            torus_y,  # Pre-computed geometry
            torus_z,  # Pre-computed geometry
            facecolors=cm.viridis(first_frame["m"] / (np.max(first_frame["m"]) + 1e-9)),
            alpha=1,
            linewidth=0.1,
            antialiased=True,
            rstride=1,
            cstride=1,
            shade=False,
        )

        def animate(frame_idx):
            """Optimized animation update - reuses pre-computed geometry."""
            if frame_idx >= len(frame_data):
                return (surface,)

            frame = frame_data[frame_idx]

            # 3D surfaces require clear (no blitting support), but minimize overhead
            ax.clear()

            # Batch axis settings together (reduces function call overhead)
            ax.set_zlim(-2, 2)
            ax.view_init(-125, 135)
            ax.axis("off")

            # Reuse pre-computed geometry, only update colors
            new_surface = ax.plot_surface(
                torus_x,  # Pre-computed, not recalculated!
                torus_y,  # Pre-computed, not recalculated!
                torus_z,  # Pre-computed, not recalculated!
                facecolors=cm.viridis(frame["m"] / (np.max(frame["m"]) + 1e-9)),
                alpha=1,
                linewidth=0.1,
                antialiased=True,
                rstride=1,
                cstride=1,
                shade=False,
            )

            # Update time text
            time_text = ax.text2D(
                0.05,
                0.95,
                f"Frame: {frame_idx + 1}/{len(frame_data)}",
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.7),
            )

            return new_surface, time_text

        # Create animation (blit=False due to 3D limitation)
        interval_ms = 1000 / fps
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(frame_data),
            interval=interval_ms,
            blit=False,
            repeat=True,  # 3D does not support blitting
        )

        # Save animation if path provided
        if save_path:
            # Warn if both saving and showing (causes double rendering)
            if show and len(frame_data) > 50:
                from ...visualization.core import warn_double_rendering

                warn_double_rendering(len(frame_data), save_path, stacklevel=2)

            if show_progress:
                pbar = tqdm(total=len(frame_data), desc=f"Saving animation to {save_path}")

                def progress_callback(current_frame, total_frames):
                    pbar.update(1)

                try:
                    writer = animation.PillowWriter(fps=fps)
                    ani.save(save_path, writer=writer, progress_callback=progress_callback)
                    pbar.close()
                    print(f"\nAnimation saved to: {save_path}")
                except Exception as e:
                    pbar.close()
                    print(f"\nError saving animation: {e}")
            else:
                try:
                    writer = animation.PillowWriter(fps=fps)
                    ani.save(save_path, writer=writer)
                    print(f"Animation saved to: {save_path}")
                except Exception as e:
                    print(f"Error saving animation: {e}")

        if show:
            # Automatically detect Jupyter and display as HTML/JS
            if is_jupyter_environment():
                display_animation_in_jupyter(ani)
                plt.close(fig)  # Close after HTML conversion to prevent auto-display
            else:
                plt.show()
        else:
            plt.close(fig)  # Close if not showing

        # Return None in Jupyter when showing to avoid double display
        if show and is_jupyter_environment():
            return None
        return ani

    except Exception as e:
        plt.close(fig)
        raise ProcessingError(f"Failed to create torus animation: {e}") from e


def plot_2d_bump_on_manifold(
    decoding_result: dict[str, Any] | str,
    spike_data: dict[str, Any],
    save_path: str | None = None,
    fps: int = 20,
    show: bool = True,
    mode: str = "fast",
    window_size: int = 10,
    frame_step: int = 5,
    numangsint: int = 20,
    figsize: tuple[int, int] = (8, 6),
    show_progress: bool = False,
):
    """
    Create 2D projection animation of CANN2D bump activity with full blitting support.

    This function provides a fast 2D heatmap visualization as an alternative to the
    3D torus animation. It achieves 10-20x speedup using matplotlib blitting
    optimization, making it ideal for rapid prototyping and daily analysis.

    Args:
        decoding_result: Decoding results containing coords and times (dict or file path)
        spike_data: Dictionary containing spike train data
        save_path: Path to save animation (None to skip saving)
        fps: Frames per second
        show: Whether to display the animation
        mode: Visualization mode - 'fast' for 2D heatmap (default), '3d' falls back to 3D
        window_size: Time window for activity aggregation
        frame_step: Time step between frames
        numangsint: Number of angular bins for spatial discretization
        figsize: Figure size (width, height) in inches
        show_progress: Show progress bar during processing

    Returns:
        FuncAnimation object (or None in Jupyter when showing)

    Raises:
        ProcessingError: If mode is invalid or animation generation fails

    Example:
        >>> # Fast 2D visualization (recommended for daily use)
        >>> ani = plot_2d_bump_on_manifold(
        ...     decoding_result, spike_data,
        ...     save_path='bump_2d.mp4', mode='fast'
        ... )
        >>> # For publication-ready 3D visualization, use mode='3d'
        >>> ani = plot_2d_bump_on_manifold(
        ...     decoding_result, spike_data, mode='3d'
        ... )
    """
    import matplotlib.animation as animation

    from ..visualization.core.jupyter_utils import (
        display_animation_in_jupyter,
        is_jupyter_environment,
    )

    # Validate inputs
    if mode == "3d":
        # Fall back to 3D visualization
        return plot_3d_bump_on_torus(
            decoding_result=decoding_result,
            spike_data=spike_data,
            save_path=save_path,
            fps=fps,
            show=show,
            window_size=window_size,
            frame_step=frame_step,
            numangsint=numangsint,
            figsize=figsize,
            show_progress=show_progress,
        )

    if mode != "fast":
        raise ProcessingError(f"Invalid mode '{mode}'. Must be 'fast' or '3d'.")

    # Load decoding results
    if isinstance(decoding_result, str):
        f = np.load(decoding_result, allow_pickle=True)
        coords = f["coordsbox"]
        times = f["times_box"]
        f.close()
    else:
        coords = decoding_result["coordsbox"]
        times = decoding_result["times_box"]

    # Process spike data for 2D projection
    spk, *_ = embed_spike_trains(
        spike_data, config=SpikeEmbeddingConfig(smooth=False, speed_filter=True)
    )

    # Process frames
    n_frames = (np.max(times) - window_size) // frame_step
    frame_activity_maps = []
    prev_m = None

    for frame_idx in tqdm(range(n_frames), desc="Processing frames", disable=not show_progress):
        start_idx = frame_idx * frame_step
        end_idx = start_idx + window_size
        if end_idx > np.max(times):
            break

        mask = (times >= start_idx) & (times < end_idx)
        coords_window = coords[mask]
        if len(coords_window) == 0:
            continue

        spk_window = spk[times[mask], :]
        activity = np.sum(spk_window, axis=1)

        m, _, _, _ = binned_statistic_2d(
            coords_window[:, 0],
            coords_window[:, 1],
            activity,
            statistic="sum",
            bins=np.linspace(0, 2 * np.pi, numangsint - 1),
        )
        m = np.nan_to_num(m)
        m = _smooth_tuning_map(m, numangsint - 1, sig=4.0, bClose=True)
        m = gaussian_filter(m, sigma=1.0)

        if prev_m is not None:
            m = 0.7 * prev_m + 0.3 * m
        prev_m = m

        frame_activity_maps.append(m)

    if not frame_activity_maps:
        raise ProcessingError("No valid frames generated for animation")

    # Create 2D visualization with blitting
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Manifold Dimension 1 (rad)", fontsize=12)
    ax.set_ylabel("Manifold Dimension 2 (rad)", fontsize=12)
    ax.set_title("CANN2D Bump Activity (2D Projection)", fontsize=14, fontweight="bold")

    # Pre-create artists for blitting
    # Heatmap
    im = ax.imshow(
        frame_activity_maps[0].T,  # Transpose for correct orientation
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
        origin="lower",
        cmap="viridis",
        animated=True,
        aspect="auto",
    )
    # Colorbar (static)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Activity", fontsize=11)

    # Time text
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        animated=True,
    )

    def init():
        """Initialize animation"""
        im.set_array(frame_activity_maps[0].T)
        time_text.set_text("")
        return im, time_text

    def update(frame_idx):
        """Update function - only modify data using blitting"""
        if frame_idx >= len(frame_activity_maps):
            return im, time_text

        # Update heatmap data
        im.set_array(frame_activity_maps[frame_idx].T)

        # Update time text
        time_text.set_text(f"Frame: {frame_idx + 1}/{len(frame_activity_maps)}")

        return im, time_text

    # Check blitting support
    use_blitting = True
    try:
        if not fig.canvas.supports_blit:
            use_blitting = False
            print("Warning: Backend does not support blitting. Using slower mode.")
    except AttributeError:
        use_blitting = False

    # Create animation with blitting enabled for 10-20x speedup
    interval_ms = 1000 / fps
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_activity_maps),
        init_func=init,
        interval=interval_ms,
        blit=use_blitting,
        repeat=True,
    )

    # Save animation if path provided
    if save_path:
        # Warn if both saving and showing (causes double rendering)
        if show and len(frame_activity_maps) > 50:
            from ...visualization.core import warn_double_rendering

            warn_double_rendering(len(frame_activity_maps), save_path, stacklevel=2)

        if show_progress:
            pbar = tqdm(total=len(frame_activity_maps), desc=f"Saving animation to {save_path}")

            def progress_callback(current_frame, total_frames):
                pbar.update(1)

            try:
                if save_path.endswith(".mp4"):
                    from matplotlib.animation import FFMpegWriter

                    writer = FFMpegWriter(
                        fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
                    )
                else:
                    from matplotlib.animation import PillowWriter

                    writer = PillowWriter(fps=fps)

                ani.save(save_path, writer=writer, progress_callback=progress_callback)
                pbar.close()
                print(f"\nAnimation saved to: {save_path}")
            except Exception as e:
                pbar.close()
                print(f"\nError saving animation: {e}")
                raise
        else:
            try:
                if save_path.endswith(".mp4"):
                    from matplotlib.animation import FFMpegWriter

                    writer = FFMpegWriter(
                        fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
                    )
                else:
                    from matplotlib.animation import PillowWriter

                    writer = PillowWriter(fps=fps)

                ani.save(save_path, writer=writer)
                print(f"Animation saved to: {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                raise

    if show:
        # Automatically detect Jupyter and display as HTML/JS
        if is_jupyter_environment():
            display_animation_in_jupyter(ani)
            plt.close(fig)  # Close after HTML conversion to prevent auto-display
        else:
            plt.show()
    else:
        plt.close(fig)  # Close if not showing

    # Return None in Jupyter when showing to avoid double display
    if show and is_jupyter_environment():
        return None
    return ani


def _get_coords(cocycle, threshold, num_sampled, dists, coeff):
    """
    Reconstruct circular coordinates from cocycle information.

    Parameters:
        cocycle (ndarray): Persistent cocycle representative.
        threshold (float): Maximum allowable edge distance.
        num_sampled (int): Number of sampled points.
        dists (ndarray): Pairwise distance matrix.
        coeff (int): Finite field modulus for cohomology.

    Returns:
        f (ndarray): Circular coordinate values (in [0,1]).
        verts (ndarray): Indices of used vertices.
    """
    zint = np.where(coeff - cocycle[:, 2] < cocycle[:, 2])
    cocycle[zint, 2] = cocycle[zint, 2] - coeff
    d = np.zeros((num_sampled, num_sampled))
    d[np.tril_indices(num_sampled)] = np.nan
    d[cocycle[:, 1], cocycle[:, 0]] = cocycle[:, 2]
    d[dists > threshold] = np.nan
    d[dists == 0] = np.nan
    edges = np.where(~np.isnan(d))
    verts = np.array(np.unique(edges))
    num_edges = np.shape(edges)[1]
    num_verts = np.size(verts)
    values = d[edges]
    A = np.zeros((num_edges, num_verts), dtype=int)
    v1 = np.zeros((num_edges, 2), dtype=int)
    v2 = np.zeros((num_edges, 2), dtype=int)
    for i in range(num_edges):
        # Extract scalar indices from np.where results
        idx1 = np.where(verts == edges[0][i])[0]
        idx2 = np.where(verts == edges[1][i])[0]

        # Handle case where np.where returns multiple matches (shouldn't happen in valid data)
        if len(idx1) > 0:
            v1[i, :] = [i, idx1[0]]
        else:
            raise ValueError(f"No vertex found for edge {edges[0][i]}")

        if len(idx2) > 0:
            v2[i, :] = [i, idx2[0]]
        else:
            raise ValueError(f"No vertex found for edge {edges[1][i]}")

    A[v1[:, 0], v1[:, 1]] = -1
    A[v2[:, 0], v2[:, 1]] = 1

    L = np.ones((num_edges,))
    Aw = A * np.sqrt(L[:, np.newaxis])
    Bw = values * np.sqrt(L)
    f = lsmr(Aw, Bw)[0] % 1
    return f, verts


def _smooth_tuning_map(mtot, numangsint, sig, bClose=True):
    """
    Smooth activity map over circular topology (e.g., torus).

    Parameters:
        mtot (ndarray): Raw activity map matrix.
        numangsint (int): Grid resolution.
        sig (float): Smoothing kernel standard deviation.
        bClose (bool): Whether to assume circular boundary conditions.

    Returns:
        mtot_out (ndarray): Smoothed map matrix.
    """
    numangsint_1 = numangsint - 1
    mid = int((numangsint_1) / 2)
    indstemp1 = np.zeros((numangsint_1, numangsint_1), dtype=int)
    indstemp1[indstemp1 == 0] = np.arange((numangsint_1) ** 2)
    mid = int((numangsint_1) / 2)
    mtemp1_3 = mtot.copy()
    for i in range(numangsint_1):
        mtemp1_3[i, :] = np.roll(mtemp1_3[i, :], int(i / 2))
    mtot_out = np.zeros_like(mtot)
    mtemp1_4 = np.concatenate((mtemp1_3, mtemp1_3, mtemp1_3), 1)
    mtemp1_5 = np.zeros_like(mtemp1_4)
    mtemp1_5[:, :mid] = mtemp1_4[:, (numangsint_1) * 3 - mid :]
    mtemp1_5[:, mid:] = mtemp1_4[:, : (numangsint_1) * 3 - mid]
    if bClose:
        mtemp1_6 = _smooth_image(np.concatenate((mtemp1_5, mtemp1_4, mtemp1_5)), sigma=sig)
    else:
        mtemp1_6 = gaussian_filter(np.concatenate((mtemp1_5, mtemp1_4, mtemp1_5)), sigma=sig)
    for i in range(numangsint_1):
        mtot_out[i, :] = mtemp1_6[
            (numangsint_1) + i,
            (numangsint_1) + (int(i / 2) + 1) : (numangsint_1) * 2 + (int(i / 2) + 1),
        ]
    return mtot_out


def _smooth_image(img, sigma):
    """
    Smooth image using multivariate Gaussian kernel, handling missing (NaN) values.

    Parameters:
        img (ndarray): Input image matrix.
        sigma (float): Standard deviation of smoothing kernel.

    Returns:
        imgC (ndarray): Smoothed image with inpainting around NaNs.
    """
    filterSize = max(np.shape(img))
    grid = np.arange(-filterSize + 1, filterSize, 1)
    xx, yy = np.meshgrid(grid, grid)

    pos = np.dstack((xx, yy))

    var = multivariate_normal(mean=[0, 0], cov=[[sigma**2, 0], [0, sigma**2]])
    k = var.pdf(pos)
    k = k / np.sum(k)

    nans = np.isnan(img)
    imgA = img.copy()
    imgA[nans] = 0
    imgA = signal.convolve2d(imgA, k, mode="valid")
    imgD = img.copy()
    imgD[nans] = 0
    imgD[~nans] = 1
    radius = 1
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    dk = np.array((X**2 + Y**2) <= radius**2, dtype=bool)
    imgE = np.zeros((filterSize + 2, filterSize + 2))
    imgE[1:-1, 1:-1] = imgD
    imgE = binary_closing(imgE, iterations=1, structure=dk)
    imgD = imgE[1:-1, 1:-1]

    imgB = np.divide(
        signal.convolve2d(imgD, k, mode="valid"),
        signal.convolve2d(np.ones(np.shape(imgD)), k, mode="valid"),
    )
    imgC = np.divide(imgA, imgB)
    imgC[imgD == 0] = -np.inf
    return imgC


if __name__ == "__main__":
    from canns.data.loaders import load_grid_data

    data = load_grid_data()

    spikes, xx, yy, tt = embed_spike_trains(data)

    # import umap
    #
    # reducer = umap.UMAP(
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     n_components=3,
    #     metric='euclidean',
    #     random_state=42
    # )
    #
    # reduce_func = reducer.fit_transform
    #
    # plot_projection(reduce_func=reduce_func, embed_data=spikes, show=True)
    results = tda_vis(embed_data=spikes, maxdim=1, do_shuffle=False, show=True)
    decoding = decode_circular_coordinates(
        persistence_result=results,
        spike_data=data,
        real_ground=True,
        real_of=True,
    )

    # Visualize CohoMap
    plot_cohomap(
        decoding_result=decoding,
        position_data={"x": xx, "y": yy},
        save_path="Results/cohomap.png",
        show=True,
    )

    # results = tda_vis(embed_data=spikes, maxdim=1, do_shuffle=True, num_shuffles=10, show=True)
