import multiprocessing as mp
import numbers
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from numpy.exceptions import AxisError
from ripser import ripser
from scipy.ndimage import _ni_support, _nd_image
from scipy.ndimage._filters import _invalid_origin
from scipy.sparse import coo_matrix
from scipy.spatial.distance import squareform, pdist
from sklearn import preprocessing
from tqdm import tqdm

from canns.analyzer.experimental_data._datasets_utils import load_grid_data


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
    metric: str = 'cosine'
    nbs: int = 800
    maxdim: int = 2
    coeff: int = 47
    show: bool = True
    do_shuffle: bool = False
    num_shuffles: int = 1000


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
        "Using numba for FAST CANN1D bump fitting, now using pure numpy implementation.",
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


def embed_spike_trains(spike_trains, config: Optional[SpikeEmbeddingConfig] = None, **kwargs):
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
            res=kwargs.get('res', 100000),
            dt=kwargs.get('dt', 1000),
            sigma=kwargs.get('sigma', 5000),
            smooth=kwargs.get('smooth0', True),
            speed_filter=kwargs.get('speed0', True),
            min_speed=kwargs.get('min_speed', 2.5)
        )
    
    try:
        # Step 1: Extract and filter spike data
        spikes_filtered = _extract_spike_data(spike_trains, config)
        
        # Step 2: Create time bins
        time_bins = _create_time_bins(spike_trains['t'], config)
        
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


def _extract_spike_data(spike_trains: Dict[str, Any], config: SpikeEmbeddingConfig) -> Dict[int, np.ndarray]:
    """Extract and filter spike data within time window."""
    try:
        # Handle different spike data formats
        spike_data = spike_trains['spike']
        if hasattr(spike_data, 'item') and callable(spike_data.item):
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
            
        t = spike_trains['t']
        
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
        raise DataLoadError(f"Missing required data key: {e}")
    except Exception as e:
        raise ProcessingError(f"Error extracting spike data: {e}")


def _create_time_bins(t: np.ndarray, config: SpikeEmbeddingConfig) -> np.ndarray:
    """Create time bins for spike discretization."""
    min_time0 = np.min(t)
    max_time0 = np.max(t)
    
    min_time = min_time0 * config.res
    max_time = max_time0 * config.res
    
    return np.arange(np.floor(min_time), np.ceil(max_time) + 1, config.dt)


def _bin_spike_data(spikes: Dict[int, np.ndarray], time_bins: np.ndarray, 
                   config: SpikeEmbeddingConfig) -> np.ndarray:
    """Convert spike times to binned spike matrix."""
    t = np.arange(len(time_bins))
    min_time = time_bins[0]
    max_time = time_bins[-1]
    
    spikes_bin = np.zeros((len(time_bins), len(spikes)), dtype=int)
    
    for n in spikes:
        spike_times = np.array(spikes[n] * config.res - min_time, dtype=int)
        # Filter valid spike times
        spike_times = spike_times[
            (spike_times < (max_time - min_time)) & (spike_times > 0)
        ]
        spike_times = np.array(spike_times / config.dt, int)
        
        # Bin spikes
        for j in spike_times:
            if j < len(time_bins):
                spikes_bin[j, n] += 1
    
    return spikes_bin


def _apply_temporal_smoothing(spikes_bin: np.ndarray, 
                            config: SpikeEmbeddingConfig) -> np.ndarray:
    """Apply Gaussian temporal smoothing to spike matrix."""
    # Calculate smoothing parameters
    thresh = config.sigma * 5
    num_thresh = int(thresh / config.dt)
    num2_thresh = int(2 * num_thresh)
    sig2 = 1 / (2 * (config.sigma / config.res) ** 2)
    
    # Create Gaussian kernel
    ker = np.exp(-np.power(np.arange(thresh + 1) / config.res, 2) * sig2)
    kerwhere = np.arange(-num_thresh, num_thresh) * config.dt
    
    # Apply smoothing (simplified version - could be further optimized)
    smoothed = np.zeros((spikes_bin.shape[0], spikes_bin.shape[1]))
    
    # Use scipy's gaussian_filter1d for better performance
    from scipy.ndimage import gaussian_filter1d
    sigma_bins = config.sigma / config.dt
    
    for n in range(spikes_bin.shape[1]):
        smoothed[:, n] = gaussian_filter1d(
            spikes_bin[:, n].astype(float), 
            sigma=sigma_bins, 
            mode='constant'
        )
    
    # Normalize
    normalization_factor = 1 / np.sqrt(2 * np.pi * (config.sigma / config.res) ** 2)
    return smoothed * normalization_factor


def _apply_speed_filtering(spikes_bin: np.ndarray, spike_trains: Dict[str, Any], 
                         config: SpikeEmbeddingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply speed-based filtering to spike data."""
    try:
        xx, yy, tt_pos, speed = _load_pos(
            spike_trains['t'], 
            spike_trains['x'], 
            spike_trains['y'],
            res=config.res,
            dt=config.dt
        )
        
        valid = speed > config.min_speed
        
        return (
            spikes_bin[valid, :],
            xx[valid],
            yy[valid], 
            tt_pos[valid]
        )
        
    except KeyError as e:
        raise DataLoadError(f"Missing position data for speed filtering: {e}")
    except Exception as e:
        raise ProcessingError(f"Error in speed filtering: {e}")


def plot_projection(
    reduce_func,
    embed_data,
    title="Projection (3D)",
    xlabel="Component 1",
    ylabel="Component 2",
    zlabel="Component 3",
    save_path=None,
    show=True,
    dpi=300,
    figsize=(10, 8),
):
    """
    Plot a 3D projection of the embedded data.

    Parameters:
        reduce_func (callable): Function to reduce the dimensionality of the data.
        embed_data (ndarray): Data to be projected.
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

    reduced_data = reduce_func(embed_data[::5])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        reduced_data[:, 2],
        s=1,
        alpha=0.5
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    if save_path is None and show is None:
        raise ValueError("Either save path or show must be provided.")
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()

    plt.close(fig)

    return fig


def tda_vis(embed_data: np.ndarray, config: Optional[TDAConfig] = None, **kwargs) -> Tuple[Dict[str, Any], Optional[Dict[int, Any]]]:
    """
    Topological Data Analysis visualization with optional shuffle testing.
    
    Parameters:
        embed_data : ndarray
            Embedded spike train data.
        config : TDAConfig, optional
            Configuration object with all TDA parameters
        **kwargs : backward compatibility parameters
    
    Returns:
        real_persistence : dict
            Persistent homology results for real data.
        shuffle_max : dict or None
            Maximum persistence values from shuffle analysis.
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = TDAConfig(
            dim=kwargs.get('dim', 6),
            num_times=kwargs.get('num_times', 5),
            active_times=kwargs.get('active_times', 15000),
            k=kwargs.get('k', 1000),
            n_points=kwargs.get('n_points', 1200),
            metric=kwargs.get('metric', 'cosine'),
            nbs=kwargs.get('nbs', 800),
            maxdim=kwargs.get('maxdim', 2),
            coeff=kwargs.get('coeff', 47),
            show=kwargs.get('show', True),
            do_shuffle=kwargs.get('do_shuffle', False),
            num_shuffles=kwargs.get('num_shuffles', 1000)
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
        _handle_visualization(real_persistence, shuffle_max, config)
        
        return real_persistence, shuffle_max
        
    except Exception as e:
        raise ProcessingError(f"TDA analysis failed: {e}") from e


def _compute_real_persistence(embed_data: np.ndarray, config: TDAConfig) -> Dict[str, Any]:
    """Compute persistent homology for real data with progress tracking."""
    
    with tqdm(total=5, desc="Processing real data") as pbar:
        # Step 1: Time point downsampling
        pbar.set_description("Time point downsampling")
        times_cube = _downsample_timepoints(embed_data, config.num_times)
        pbar.update(1)
        
        # Step 2: Select most active time points  
        pbar.set_description("Selecting active time points")
        movetimes = _select_active_timepoints(embed_data, times_cube, config.active_times)
        pbar.update(1)
        
        # Step 3: PCA dimensionality reduction
        pbar.set_description("PCA dimensionality reduction")
        dimred = _apply_pca_reduction(embed_data, movetimes, config.dim)
        pbar.update(1)
        
        # Step 4: Point cloud sampling (denoising)
        pbar.set_description("Point cloud denoising")
        indstemp = _apply_denoising(dimred, config)
        pbar.update(1)
        
        # Step 5: Compute persistent homology
        pbar.set_description("Computing persistent homology")
        real_persistence = _compute_persistence_homology(dimred, indstemp, config)
        pbar.update(1)
    
    return real_persistence


def _downsample_timepoints(embed_data: np.ndarray, num_times: int) -> np.ndarray:
    """Downsample timepoints for computational efficiency."""
    return np.arange(0, embed_data.shape[0], num_times)


def _select_active_timepoints(embed_data: np.ndarray, times_cube: np.ndarray, active_times: int) -> np.ndarray:
    """Select most active timepoints based on total activity."""
    activity_scores = np.sum(embed_data[times_cube, :], 1)
    most_active_indices = np.argsort(activity_scores)[-active_times:]
    return times_cube[np.sort(most_active_indices)]


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
        omega=0.2,  # This could be made configurable
        metric=config.metric
    )
    return indstemp


def _compute_persistence_homology(dimred: np.ndarray, indstemp: np.ndarray, config: TDAConfig) -> Dict[str, Any]:
    """Compute persistent homology using ripser."""
    d = _second_build(dimred, indstemp, metric=config.metric, nbs=config.nbs)
    np.fill_diagonal(d, 0)
    
    return ripser(
        d, 
        maxdim=config.maxdim, 
        coeff=config.coeff, 
        do_cocycles=True, 
        distance_matrix=True
    )


def _perform_shuffle_analysis(embed_data: np.ndarray, config: TDAConfig) -> Dict[int, Any]:
    """Perform shuffle analysis with progress tracking."""
    print(f"\nStarting shuffle analysis with {config.num_shuffles} iterations...")
    
    # Create parameters dict for shuffle analysis
    shuffle_params = {
        'dim': config.dim,
        'num_times': config.num_times,
        'active_times': config.active_times,
        'k': config.k,
        'n_points': config.n_points,
        'metric': config.metric,
        'nbs': config.nbs,
        'maxdim': config.maxdim,
        'coeff': config.coeff
    }
    
    shuffle_max = _run_shuffle_analysis(
        embed_data, 
        num_shuffles=config.num_shuffles,
        num_cores=Constants.MULTIPROCESSING_CORES,
        **shuffle_params
    )
    
    # Print shuffle analysis summary
    _print_shuffle_summary(shuffle_max)
    
    return shuffle_max


def _print_shuffle_summary(shuffle_max: Dict[int, Any]) -> None:
    """Print summary of shuffle analysis results."""
    print("\nSummary of shuffle-based analysis:")
    for dim_idx in [0, 1, 2]:
        if shuffle_max and dim_idx in shuffle_max and shuffle_max[dim_idx]:
            values = shuffle_max[dim_idx]
            print(f"H{dim_idx}: {len(values)} valid iterations | "
                  f"Mean maximum persistence: {np.mean(values):.4f} | "
                  f"99.9th percentile: {np.percentile(values, 99.9):.4f}")


def _handle_visualization(real_persistence: Dict[str, Any], shuffle_max: Optional[Dict[int, Any]], config: TDAConfig) -> None:
    """Handle visualization based on configuration."""
    if config.show:
        if config.do_shuffle and shuffle_max is not None:
            _plot_barcode_with_shuffle(real_persistence, shuffle_max)
            plt.show()
        else:
            # TODO: Implement non-shuffle visualization
            print("Non-shuffle visualization not yet implemented")
    else:
        plt.close()


def _load_pos(
    t,
    x,
    y,
    res=100000,
    dt=1000
):
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
    steps = (1.0 / divisor[divisor > 0])
    N = np.max(divisor)
    ranges = np.multiply(np.arange(N)[np.newaxis, :], steps[:, np.newaxis])
    ranges[ranges >= 1] = np.nan

    rangesx = x[idx[:-1], np.newaxis] + np.multiply(ranges, (x[idx[1:]] - x[idx[:-1]])[:, np.newaxis])
    xx = rangesx[~np.isnan(ranges)]

    rangesy = y[idx[:-1], np.newaxis] + np.multiply(ranges, (y[idx[1:]] - y[idx[:-1]])[:, np.newaxis])
    yy = rangesy[~np.isnan(ranges)]

    xxs = _gaussian_filter1d(xx - np.min(xx), sigma=100)
    yys = _gaussian_filter1d(yy - np.min(yy), sigma=100)
    dx = (xxs[1:] - xxs[:-1]) * 100
    dy = (yys[1:] - yys[:-1]) * 100
    speed = np.sqrt(dx ** 2 + dy ** 2) / 0.01
    speed = np.concatenate(([speed[0]], speed))
    return xx, yy, tt, speed


def _gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                       mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
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
        raise ValueError('Radius must be a nonnegative integer.')
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return _correlate1d(input, weights, axis, output, mode, cval, 0)


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
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


def _correlate1d(input, weights, axis=-1, output=None, mode="reflect",
                 cval=0.0, origin=0):
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
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input or complex_weights:
        if complex_weights:
            weights = weights.conj()
            weights = weights.astype(np.complex128, copy=False)
        kwargs = dict(axis=axis, mode=mode, origin=origin)
        output = _ni_support._get_output(output, input, complex_output=True)
        return _complex_via_real_components(_correlate1d, input, weights,
                                            output, cval, **kwargs)

    output = _ni_support._get_output(output, input)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = _normalize_axis_index(axis, input.ndim)
    if _invalid_origin(origin, len(weights)):
        raise ValueError('Invalid origin; origin must satisfy '
                         '-(len(weights) // 2) <= origin <= '
                         '(len(weights)-1) // 2')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate1d(input, weights, axis, output, mode, cval,
                          origin)
    return output


def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    """Complex convolution via a linear combination of real convolutions."""
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input and complex_weights:
        # real component of the output
        func(input.real, weights.real, output=output.real,
             cval=np.real(cval), **kwargs)
        output.real -= func(input.imag, weights.imag, output=None,
                            cval=np.imag(cval), **kwargs)
        # imaginary component of the output
        func(input.real, weights.imag, output=output.imag,
             cval=np.real(cval), **kwargs)
        output.imag += func(input.imag, weights.real, output=None,
                            cval=np.imag(cval), **kwargs)
    elif complex_input:
        func(input.real, weights, output=output.real, cval=np.real(cval),
             **kwargs)
        func(input.imag, weights, output=output.imag, cval=np.imag(cval),
             **kwargs)
    else:
        if np.iscomplexobj(cval):
            raise ValueError("Cannot provide a complex-valued cval when the "
                             "input is real.")
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


def _compute_persistence(sspikes, dim=6, num_times=5, active_times=15000,
                         k=1000, n_points=1200, metric='cosine',
                         nbs=800, maxdim=1, coeff=47):
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
    persistence = ripser(d, maxdim=maxdim, coeff=coeff, do_cocycles=True, distance_matrix=True)

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
    cum_var_exp = np.cumsum(var_exp)
    components = np.dot(evecs.T, data.T).T
    return components, var_exp, evals[:dim]


def _sample_denoising(data, k=10, num_sample=500, omega=0.2, metric='euclidean'):
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
    n = data.shape[0]
    leftinds = np.arange(n)
    F_D = np.zeros(n)
    if metric in ("cosine", "correlation", "dice", "jaccard"):
        angular = True
    else:
        angular = False

    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = _smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
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
        Fs[j] = F[Fmax]
        i = inds_all[inds_left][Fmax]

        inds_left[i] = False
        inds[j] = i
    d = np.zeros((num_sample, num_sample))

    for j, i in enumerate(inds):
        d[j, :] = X[i, inds]
    return inds, d, Fs


@njit(
    fastmath=True
)
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

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > 1e-5:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                #                    psum += d / mid

                else:
                    psum += 1.0
            #                    psum += 0

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
        # TODO: This is very inefficient, but will do for now. FIXME
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


def _second_build(data, indstemp, nbs=800, metric='cosine'):
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
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()

    # Build the final distance matrix
    d = result.toarray()
    # Handle zero values to avoid log(0) warning
    # Add small epsilon to avoid divide by zero, then take negative log
    epsilon = np.finfo(float).eps
    d = np.where(d > 0, d, epsilon)
    d = -np.log(d)
    np.fill_diagonal(d, 0)

    return d

def _run_shuffle_analysis(sspikes, num_shuffles=1000, num_cores=4, **kwargs):
    """Perform shuffle analysis using multiprocessing for parallel computation."""
    max_lifetimes = {0: [], 1: [], 2: []}

    # Estimate runtime with a test iteration
    print("Running test iteration to estimate runtime...")

    _ = _process_single_shuffle((0, sspikes, kwargs))

    # Prepare task list
    tasks = [(i, sspikes, kwargs) for i in range(num_shuffles)]

    # Use multiprocessing pool for parallel processing
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(_process_single_shuffle, tasks),
                            total=num_shuffles,
                            desc="Running shuffle analysis"))

    # Collect results
    for res in results:
        for dim, lifetime in res.items():
            max_lifetimes[dim].append(lifetime)

    return max_lifetimes


def _process_single_shuffle(args):
    """Process a single shuffle task."""
    i, sspikes, kwargs = args
    try:
        shuffled_data = _shuffle_spike_trains(sspikes)
        persistence = _compute_persistence(shuffled_data, **kwargs)

        dim_max_lifetimes = {}
        for dim in [0, 1, 2]:
            if dim < len(persistence['dgms']):
                # Filter out infinite values
                valid_bars = [bar for bar in persistence['dgms'][dim] if not np.isinf(bar[1])]
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
    maxdim = len(persistence['dgms']) - 1
    dims = np.arange(maxdim + 1)

    min_birth, max_death = 0, 0
    for dim in dims:
        # Filter out infinite values
        valid_bars = [bar for bar in persistence['dgms'][dim] if not np.isinf(bar[1])]
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

    for dit, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes.axis('off')

        # Add gray background to represent shuffle region
        if dim in thresholds:
            axes.axvspan(0, thresholds[dim], alpha=0.2, color='gray', zorder=-3)
            axes.axvline(x=thresholds[dim], color='gray', linestyle='--', alpha=0.7)

        # Filter out infinite values
        d = np.array([bar for bar in persistence['dgms'][dim] if not np.isinf(bar[1])])
        if len(d) == 0:
            d = np.zeros((0, 2))

        d = np.copy(d)
        d[np.isinf(d[:, 1]), 1] = infinity
        dlife = (d[:, 1] - d[:, 0])

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
                color = 'red' if idx in significant_bars else colormap[dim]
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

        axes.plot([0, 0], [0, indsall], c='k', linestyle='-', lw=1)
        axes.plot([0, indsall], [0, 0], c='k', linestyle='-', lw=1)
        axes.set_xlim([0, infinity])
        axes.set_title(f"$H_{dim}$", loc='left')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    data = load_grid_data()

    spikes, *_ = embed_spike_trains(data)

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

    real_persistence, shuffle_max = tda_vis(
        embed_data=spikes,
        maxdim=2,
        do_shuffle=True,
        num_shuffles=3,
        show=True
    )
