import os
import random
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import linear_sum_assignment
from scipy.special import i0
from tqdm import tqdm

from ..visualization.core.jupyter_utils import (
    display_animation_in_jupyter,
    is_jupyter_environment,
)

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


from canns.data.loaders import load_roi_data

# Import PlotConfig for unified plotting
from ..visualization import PlotConfig


# ==================== Configuration Classes ====================
@dataclass
class BumpFitsConfig:
    """Configuration for CANN1D bump fitting."""

    n_steps: int = 20000
    n_roi: int = 16
    n_bump_max: int = 4
    sigma_diff: float = 0.5
    ampli_min: float = 2.0
    kappa_mean: float = 2.5
    sig2: float = 1.0
    penbump: float = 0.4
    jc: float = 1.8
    beta: float = 5.0
    random_seed: int | None = None


@dataclass
class AnimationConfig:
    """Configuration for 1D CANN bump animation."""

    show: bool = False
    max_height_value: float = 0.5
    max_width_range: int = 40
    npoints: int = 300
    nframes: int | None = None
    fps: int = 5
    bump_selection: str = "strongest"
    show_progress_bar: bool = True
    repeat: bool = False
    title: str = "1D CANN Bump Animation"


@dataclass
class CANN1DPlotConfig(PlotConfig):
    """Specialized PlotConfig for CANN1D visualizations."""

    # CANN1D specific animation parameters
    max_height_value: float = 0.5
    max_width_range: int = 40
    npoints: int = 300
    nframes: int | None = None
    bump_selection: str = "strongest"

    @classmethod
    def for_bump_animation(cls, **kwargs) -> "CANN1DPlotConfig":
        """Create configuration for 1D CANN bump animation."""
        defaults = {
            "title": "1D CANN Bump Animation",
            "figsize": Constants.DEFAULT_FIGSIZE,
            "fps": 5,
            "repeat": False,
            "show_progress_bar": True,
            "max_height_value": 0.5,
            "max_width_range": 40,
            "npoints": 300,
            "bump_selection": "strongest",
        }
        defaults.update(kwargs)
        return cls(**defaults)


# ==================== Constants ====================
class Constants:
    """Constants used throughout CANN1D analysis."""

    DEFAULT_FIGSIZE = (4, 4)
    DEFAULT_DPI = 100
    BASE_RADIUS = 1.0
    MAX_KERNEL_SIZE = 60
    NUMBA_THRESHOLD = 64  # ROI count threshold for parallel processing


# ==================== Custom Exceptions ====================
class CANN1DError(Exception):
    """Base exception for CANN1D analysis errors."""

    pass


class FittingError(CANN1DError):
    """Raised when bump fitting fails."""

    pass


class AnimationError(CANN1DError):
    """Raised when animation creation fails."""

    pass


def bump_fits(data, config: BumpFitsConfig | None = None, save_path=None, **kwargs):
    """
    Fit CANN1D bumps to data using MCMC optimization.

    Parameters:
        data : numpy.ndarray
            Input data for bump fitting
        config : BumpFitsConfig, optional
            Configuration object with all fitting parameters
        save_path : str, optional
            Path to save the results
        **kwargs : backward compatibility parameters

    Returns:
        bumps : list
            List of fitted bump objects
        fits_array : numpy.ndarray
            Array of fitted bump parameters
        nbump_array : numpy.ndarray
            Array of bump counts and reconstructed signals
        centrbump_array : numpy.ndarray
            Array of centered bump data
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = BumpFitsConfig(
            n_steps=kwargs.get("n_steps", 20000),
            n_roi=kwargs.get("n_roi", 16),
            n_bump_max=kwargs.get("n_bump_max", 4),
            sigma_diff=kwargs.get("sigma_diff", 0.5),
            ampli_min=kwargs.get("ampli_min", 2.0),
            kappa_mean=kwargs.get("kappa_mean", 2.5),
            sig2=kwargs.get("sig2", 1.0),
            penbump=kwargs.get("penbump", 0.4),
            jc=kwargs.get("jc", 1.8),
            beta=kwargs.get("beta", 5.0),
            random_seed=kwargs.get("random_seed", None),
        )

    try:
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)
            if HAS_NUMBA:
                _set_seed(config.random_seed)

        # MCMC parameters
        sigcoup = 2 * np.pi / config.n_roi
        sigcoup2 = sigcoup**2

        nbt = data.shape[0]
        flat_data = data.flatten()
        normed_data = (flat_data / np.median(flat_data)) - 1.0
        bumps = _mcmc(
            nbt=nbt,
            data=normed_data,
            n_steps=config.n_steps,
            n_roi=config.n_roi,
            n_bump_max=config.n_bump_max,
            sigma_diff=config.sigma_diff,
            ampli_min=config.ampli_min,
            kappa_mean=config.kappa_mean,
            sig2=config.sig2,
            sigcoup2=sigcoup2,
            penbump=config.penbump,
            jc=config.jc,
            beta=config.beta,
        )

        # compute total bumps and central bump points
        total_bumps = sum(bump.nbump for bump in bumps)
        total_centrbump_points = sum(bump.nbump * config.n_roi for bump in bumps)

        # preallocate arrays
        fits_array = np.zeros((total_bumps, 4))  # [time, pos, amplitude, kappa]
        nbump_array = np.zeros((nbt, config.n_roi + 2))  # [time, n_bumps, reconstructed_signal...]
        centrbump_array = np.zeros((total_centrbump_points, 2))  # [dist, normalized_amplitude]

        # create x_grid for von Mises distribution
        x_grid = np.arange(config.n_roi) * 2 * np.pi / config.n_roi
        roi_positions = np.arange(config.n_roi) * 2 * np.pi / config.n_roi

        fits_idx = 0
        centrbump_idx = 0

        for t, bump in enumerate(bumps):
            # 1. fills fits_array
            if bump.nbump > 0:
                fits_array[fits_idx : fits_idx + bump.nbump, 0] = t
                fits_array[fits_idx : fits_idx + bump.nbump, 1] = bump.pos[: bump.nbump]
                fits_array[fits_idx : fits_idx + bump.nbump, 2] = bump.ampli[: bump.nbump]
                fits_array[fits_idx : fits_idx + bump.nbump, 3] = bump.kappa[: bump.nbump]
                fits_idx += bump.nbump

            # 2. fills nbump_array
            nbump_array[t, 0] = t
            nbump_array[t, 1] = bump.nbump

            # compute von Mises distribution for bumps
            if bump.nbump > 0:
                # get bump parameters
                pos_array = np.array(bump.pos[: bump.nbump])
                ampli_array = np.array(bump.ampli[: bump.nbump])
                kappa_array = np.array(bump.kappa[: bump.nbump])

                # Use optimized von Mises computation if available
                if HAS_NUMBA:
                    if config.n_roi >= Constants.NUMBA_THRESHOLD:
                        von_mises_vals = _compute_predicted_intensity_parallel(
                            pos_array, kappa_array, ampli_array, bump.nbump, config.n_roi
                        )
                    else:
                        von_mises_vals = _compute_predicted_intensity(
                            pos_array, kappa_array, ampli_array, bump.nbump, config.n_roi
                        )
                    nbump_array[t, 2:] = von_mises_vals
                else:
                    # Fallback to broadcasting computation
                    diff = x_grid[:, None] - pos_array[None, :]
                    von_mises_vals = (
                        ampli_array[None, :]
                        * np.exp(kappa_array[None, :] * np.cos(diff))
                        / (2 * np.pi * i0(kappa_array[None, :]))
                    )
                    nbump_array[t, 2:] = np.sum(von_mises_vals, axis=1)

            # 3. fills centrbump_array
            if bump.nbump > 0:
                data_segment = flat_data[t * config.n_roi : (t + 1) * config.n_roi]

                # get distances and normalized amplitudes
                for i in range(bump.nbump):
                    start_idx = centrbump_idx + i * config.n_roi
                    end_idx = start_idx + config.n_roi

                    # distance from bump position to ROI positions
                    dist = bump.pos[i] - roi_positions
                    # adjust distances to be within [-pi, pi]
                    dist = np.where(dist > np.pi, dist - 2 * np.pi, dist)
                    dist = np.where(dist < -np.pi, dist + 2 * np.pi, dist)

                    # normalize amplitude
                    norm_amp = data_segment / bump.ampli[i]

                    centrbump_array[start_idx:end_idx, 0] = dist
                    centrbump_array[start_idx:end_idx, 1] = norm_amp

                centrbump_idx += bump.nbump * config.n_roi

        if save_path is not None:
            os.makedirs(
                os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
            )

            np.savez(
                save_path,
                fits=fits_array,  # shape: (n_fits, 4) - [time, pos, amplitude, kappa]
                nbump=nbump_array,  # shape: (n_timepoints, n_roi+2) - [time, n_bumps, reconstructed_signal...]
                centrbump=centrbump_array,  # shape: (n_points, 2) - [dist, normalized_amplitude]
            )

        return bumps, fits_array, nbump_array, centrbump_array

    except Exception as e:
        raise FittingError(f"Failed to fit bumps: {e}") from e


def create_1d_bump_animation(
    fits_data, config: CANN1DPlotConfig | None = None, save_path=None, **kwargs
):
    """
    Create 1D CANN bump animation using vectorized operations.

    Parameters:
        fits_data : numpy.ndarray
            Shape (n_fits, 4) array with columns [time, position, amplitude, kappa]
        config : AnimationConfig, optional
            Configuration object with all animation parameters
        save_path : str, optional
            Output path for the generated GIF
        **kwargs : backward compatibility parameters

    Returns:
        matplotlib.animation.FuncAnimation
            The animation object
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = CANN1DPlotConfig.for_bump_animation(**kwargs)

    # Override config with any explicitly passed parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    try:
        # ==== Smoothing functions ====
        def smooth(x, window=3):
            """Apply simple moving average smoothing"""
            return np.convolve(x, np.ones(window) / window, mode="same")

        def smooth_circle(values, window=5):
            """Apply circular smoothing for periodic data"""
            pad = window // 2
            values_ext = np.concatenate([values[-pad:], values, values[:pad]])
            kernel = np.ones(window) / window
            smoothed = np.convolve(values_ext, kernel, mode="valid")
            return smoothed

        # ==== Data validation ====
        if fits_data is None or len(fits_data) == 0:
            raise ValueError("No bump data provided")

        if fits_data.ndim != 2 or fits_data.shape[1] != 4:
            raise ValueError(
                f"fits_data must be a 2D array with 4 columns, got shape {fits_data.shape}"
            )

        # ==== Extract time information ====
        times = fits_data[:, 0].astype(int)
        unique_times = np.unique(times)

        nframes = config.nframes
        if nframes is None or nframes > len(unique_times):
            nframes = len(unique_times)

        selected_times = unique_times[:nframes]

        # ==== Vectorized data extraction ====
        # Pre-allocate arrays for better performance
        positions_raw = np.zeros(nframes)
        heights_raw = np.zeros(nframes)
        widths_raw = np.zeros(nframes)

        # Process each timepoint vectorized way
        for i, t in enumerate(selected_times):
            # Get all bumps at current timepoint
            time_mask = times == t
            if np.any(time_mask):
                time_data = fits_data[time_mask]

                # Select bump based on strategy
                if config.bump_selection == "strongest":
                    best_idx = np.argmax(time_data[:, 2])  # Maximum amplitude
                else:  # 'first'
                    best_idx = 0

                # Extract bump parameters
                positions_raw[i] = time_data[best_idx, 1]  # position
                heights_raw[i] = time_data[best_idx, 2]  # amplitude
                widths_raw[i] = time_data[best_idx, 3]  # kappa
            # Note: zeros remain for timepoints without bumps

        # ==== Apply smoothing ====
        positions = smooth(positions_raw, window=3)
        heights_raw_smooth = smooth(heights_raw, window=3)
        widths_raw_smooth = smooth(widths_raw, window=3)

        # ==== Setup parameters ====
        theta = np.linspace(0, 2 * np.pi, config.npoints, endpoint=False)
        base_radius = Constants.BASE_RADIUS

        # ==== Precompute offsets array for Gaussian kernel (avoid recomputation per frame) ====
        offsets_array = np.arange(-Constants.MAX_KERNEL_SIZE, Constants.MAX_KERNEL_SIZE + 1)

        # ==== Normalize data ranges ====
        # Height normalization
        height_range = np.max(heights_raw_smooth) - np.min(heights_raw_smooth)
        if height_range > 0:
            min_height, max_height = np.min(heights_raw_smooth), np.max(heights_raw_smooth)
            heights = np.interp(
                heights_raw_smooth, (min_height, max_height), (0.1, config.max_height_value)
            )
        else:
            heights = np.full_like(heights_raw_smooth, 0.1)

        # Width normalization
        width_range = np.max(widths_raw_smooth) - np.min(widths_raw_smooth)
        if width_range > 0:
            min_width, max_width = np.min(widths_raw_smooth), np.max(widths_raw_smooth)
            width_ranges = np.interp(
                widths_raw_smooth, (min_width, max_width), (2, config.max_width_range)
            ).astype(int)
        else:
            width_ranges = np.full_like(widths_raw_smooth, config.max_width_range // 2, dtype=int)

        # ==== Initialize matplotlib animation with blitting optimization ====
        fig, ax = plt.subplots(figsize=Constants.DEFAULT_FIGSIZE, dpi=Constants.DEFAULT_DPI)

        # Set up axes properties (done once, not per frame)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.axis("off")
        ax.set_title(config.title, fontsize=14, fontweight="bold", pad=20)

        # Pre-create all artist objects with animated=True for blitting
        # Base circle (reference) - static geometry
        inner_x = base_radius * np.cos(theta)
        inner_y = base_radius * np.sin(theta)
        (base_circle,) = ax.plot(
            inner_x, inner_y, color="gray", linestyle="--", linewidth=1, animated=True
        )

        # Bump curve - will be updated each frame
        (bump_line,) = ax.plot([], [], color="red", linewidth=2, animated=True)

        # Center marker - will be updated each frame
        (center_marker,) = ax.plot([], [], "o", color="black", markersize=6, animated=True)

        def init():
            """Initialize animation - set empty data for dynamic artists"""
            bump_line.set_data([], [])
            center_marker.set_data([], [])
            return base_circle, bump_line, center_marker

        def update(frame):
            """Update function - only modify artist data, never clear or rebuild"""
            # Get parameters for current frame
            pos_angle = positions[frame]
            height = heights[frame]
            width_range = width_ranges[frame]

            # Find center position in theta array
            center_idx = np.argmin(np.abs(theta - pos_angle))

            # Initialize radius array with base radius
            r = np.ones(config.npoints) * base_radius

            # Apply Gaussian kernel around bump center
            sigma = width_range / 2

            # Vectorized kernel application using precomputed offsets
            gauss_weights = np.exp(-(offsets_array**2) / (2 * sigma**2))
            # Filter out negligible contributions
            significant_mask = gauss_weights >= 0.01
            significant_offsets = offsets_array[significant_mask]
            significant_weights = gauss_weights[significant_mask]
            # Apply weights to corresponding indices
            indices = (center_idx + significant_offsets) % config.npoints
            r[indices] += height * significant_weights

            # Apply circular smoothing
            r = smooth_circle(r, window=5)

            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Update bump curve data (no ax.clear()!)
            bump_line.set_data(x, y)

            # Update center marker position
            dot_radius = base_radius * 0.96
            center_x = dot_radius * np.cos(pos_angle)
            center_y = dot_radius * np.sin(pos_angle)
            center_marker.set_data([center_x], [center_y])

            # Return all modified artists for blitting
            return base_circle, bump_line, center_marker

        # ==== Create and save animation ====
        if config.save_path is None and not config.show:
            raise ValueError("Either save_path or show must be specified")

        # Check if backend supports blitting
        use_blitting = True
        try:
            if not fig.canvas.supports_blit:
                use_blitting = False
                print("Warning: Backend does not support blitting. Using slower mode.")
        except AttributeError:
            use_blitting = False

        # Create animation with blitting enabled for 15-20x speedup
        ani = FuncAnimation(
            fig, update, frames=nframes, init_func=init, blit=use_blitting, repeat=config.repeat
        )

        # Save animation with progress tracking
        if config.save_path:
            # Warn if both saving and showing (causes double rendering)
            if config.show and nframes > 50:
                from ...visualization.core import warn_double_rendering

                warn_double_rendering(nframes, config.save_path, stacklevel=2)

            if config.show_progress_bar:
                pbar = tqdm(total=nframes, desc=f"Saving animation to {config.save_path}")

                def progress_callback(current_frame, total_frames):
                    pbar.update(1)

                try:
                    ani.save(
                        config.save_path,
                        writer=PillowWriter(fps=config.fps),
                        progress_callback=progress_callback,
                    )
                    pbar.close()
                    print(f"\nAnimation successfully saved to: {config.save_path}")
                except Exception as e:
                    pbar.close()
                    print(f"\nError saving animation: {e}")
                    raise
            else:
                try:
                    ani.save(config.save_path, writer=PillowWriter(fps=config.fps))
                    print(f"Animation saved to: {config.save_path}")
                except Exception as e:
                    print(f"Error saving animation: {e}")
                    raise

        if config.show:
            # Automatically detect Jupyter and display as HTML/JS
            if is_jupyter_environment():
                display_animation_in_jupyter(ani)
                plt.close(fig)  # Close after HTML conversion to prevent auto-display
            else:
                plt.show()

        # Return None in Jupyter when showing to avoid double display
        if config.show and is_jupyter_environment():
            return None
        return ani

    except Exception as e:
        raise AnimationError(f"Failed to create animation: {e}") from e
    finally:
        if not config.show:
            # Ensure we clean up the figure to avoid memory leaks
            plt.close(fig)


class SiteBump:
    def __init__(self):
        self.nbump = 0  # number of bumps
        self.pos = []  # positions of bumps
        self.ampli = []  # amplitudes of bumps
        self.kappa = []  # widths of bumps
        self.logl = 0.0  # log-likelihood

    def clone(self):
        new = SiteBump()
        new.nbump = self.nbump
        new.pos = self.pos.copy()
        new.ampli = self.ampli.copy()
        new.kappa = self.kappa.copy()
        new.logl = self.logl
        return new


@njit
def _von_mises_kernel(pos, kappa, ampli, x_roi):
    """Numba-optimized von Mises kernel calculation"""
    result = 0.0
    angle = x_roi - pos
    # Use simplified approximation for i0 for speed
    # i0(kappa) ≈ exp(kappa) / sqrt(2*pi*kappa) for large kappa
    if kappa > 3.75:
        i0_approx = np.exp(kappa) / np.sqrt(2 * np.pi * kappa)
    else:
        # For small kappa, use Taylor expansion
        i0_approx = 1.0 + (kappa * kappa / 4.0) + (kappa**4 / 64.0)

    result = ampli * np.exp(kappa * np.cos(angle)) / (2 * np.pi * i0_approx)
    return result


@njit
def _compute_predicted_intensity(positions, kappas, amplis, n_bumps, n_roi):
    """Numba-optimized predicted intensity calculation (serial for small n_roi)"""
    predicted = np.zeros(n_roi)
    x_grid = np.arange(n_roi) * 2 * np.pi / n_roi

    # Use serial execution for small n_roi (< 64) to avoid parallel overhead
    for j in range(n_roi):
        x_roi = x_grid[j]
        for i in range(n_bumps):
            # Inline von Mises calculation for better performance
            angle = x_roi - positions[i]
            kappa = kappas[i]
            ampli = amplis[i]

            # Fast i0 approximation
            if kappa > 3.75:
                i0_approx = np.exp(kappa) / np.sqrt(2 * np.pi * kappa)
            else:
                i0_approx = 1.0 + (kappa * kappa / 4.0) + (kappa**4 / 64.0)

            predicted[j] += ampli * np.exp(kappa * np.cos(angle)) / (2 * np.pi * i0_approx)

    return predicted


@njit(parallel=True)
def _compute_predicted_intensity_parallel(positions, kappas, amplis, n_bumps, n_roi):
    """Parallel version for large n_roi (>= 64)"""
    predicted = np.zeros(n_roi)
    x_grid = np.arange(n_roi) * 2 * np.pi / n_roi

    # Parallelize over ROI positions for large datasets
    for j in prange(n_roi):
        x_roi = x_grid[j]
        for i in range(n_bumps):
            angle = x_roi - positions[i]
            kappa = kappas[i]
            ampli = amplis[i]

            if kappa > 3.75:
                i0_approx = np.exp(kappa) / np.sqrt(2 * np.pi * kappa)
            else:
                i0_approx = 1.0 + (kappa * kappa / 4.0) + (kappa**4 / 64.0)

            predicted[j] += ampli * np.exp(kappa * np.cos(angle)) / (2 * np.pi * i0_approx)

    return predicted


def _site_logl(
    intens,
    bump,
    n_roi,
    penbump,
    sig2,
    beta,
):
    """
    Optimized likelihood calculation with numba acceleration
    """
    # Penalty term
    logl = -bump.nbump * penbump

    # Predicted intensity
    if bump.nbump > 0 and HAS_NUMBA:
        # Use numba-optimized version with smart parallel/serial selection
        pos_arr = np.array(bump.pos[: bump.nbump])
        kappa_arr = np.array(bump.kappa[: bump.nbump])
        ampli_arr = np.array(bump.ampli[: bump.nbump])

        # Use parallel version only for large n_roi to avoid overhead
        if n_roi >= 64:
            predicted = _compute_predicted_intensity_parallel(
                pos_arr, kappa_arr, ampli_arr, bump.nbump, n_roi
            )
        else:
            predicted = _compute_predicted_intensity(
                pos_arr, kappa_arr, ampli_arr, bump.nbump, n_roi
            )
    elif bump.nbump > 0:
        # Fallback to original vectorized version
        x = np.arange(n_roi) * 2 * np.pi / n_roi
        angles = x[:, None] - np.array(bump.pos[: bump.nbump])
        kappa_arr = np.array(bump.kappa[: bump.nbump])
        ampli_arr = np.array(bump.ampli[: bump.nbump])
        vonmises_matrix = np.exp(kappa_arr * np.cos(angles)) / (2 * np.pi * i0(kappa_arr))
        predicted = np.sum(ampli_arr * vonmises_matrix, axis=1)
    else:
        predicted = np.zeros(n_roi)

    # Likelihood from residuals
    residuals = intens - predicted
    logl -= 0.5 * np.sum(residuals**2) / sig2

    return beta * logl


@njit
def _compute_circular_distance(pos1, pos2):
    """Numba-optimized circular distance calculation"""
    dist = abs(pos1 - pos2)
    if dist > np.pi:
        dist = 2 * np.pi - dist
    return dist


@njit
def _compute_coupling_fast(pos1_arr, pos2_arr, n1, n2, sigcoup2):
    """Numba-optimized coupling calculation for small numbers of bumps"""
    if n1 == 0 or n2 == 0:
        return 0.0

    # For small numbers of bumps, use greedy matching (faster than Hungarian)
    total_likelihood = 0.0
    used_j = np.zeros(n2, dtype=np.bool_)

    for i in range(n1):
        best_likelihood = -np.inf
        best_j = -1

        for j in range(n2):
            if not used_j[j]:
                # Inline circular distance calculation (faster than function call)
                dist = abs(pos1_arr[i] - pos2_arr[j])
                if dist > np.pi:
                    dist = 2 * np.pi - dist
                likelihood = np.exp(-0.5 * dist * dist / sigcoup2)
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_j = j

        if best_j >= 0:
            used_j[best_j] = True
            total_likelihood += best_likelihood

    return total_likelihood


@njit
def _parallel_distance_matrix(pos1_array, pos2_array):
    """Optimized computation of circular distance matrix (serial for small arrays)"""
    n1, n2 = len(pos1_array), len(pos2_array)
    dist_matrix = np.zeros((n1, n2))

    # Use serial execution for small arrays to avoid parallel overhead
    for i in range(n1):
        for j in range(n2):
            dist = abs(pos1_array[i] - pos2_array[j])
            if dist > np.pi:
                dist = 2 * np.pi - dist
            dist_matrix[i, j] = dist

    return dist_matrix


def _interf_logl(
    b1,
    b2,
    n_bump_max,
    sigcoup2,
    beta,
    jc,
):
    """
    Optimized coupling likelihood with numba acceleration
    """
    if b1.nbump == 0 or b2.nbump == 0:
        return 0.0

    # For small numbers of bumps, use numba-optimized greedy matching
    if HAS_NUMBA and b1.nbump <= 4 and b2.nbump <= 4:
        pos1_arr = np.array(b1.pos[: b1.nbump])
        pos2_arr = np.array(b2.pos[: b2.nbump])
        logli = _compute_coupling_fast(pos1_arr, pos2_arr, b1.nbump, b2.nbump, sigcoup2)
    else:
        # Use parallel distance matrix computation for larger numbers
        pos1 = np.array(b1.pos[: b1.nbump])
        pos2 = np.array(b2.pos[: b2.nbump])

        if HAS_NUMBA:
            # Use parallel distance matrix calculation
            circular_dist = _parallel_distance_matrix(pos1, pos2)
        else:
            # Fallback to vectorized version
            dist_matrix = np.abs(pos1[:, None] - pos2)
            circular_dist = np.minimum(dist_matrix, 2 * np.pi - dist_matrix)
        cost_matrix = -np.exp(-0.5 * circular_dist**2 / sigcoup2)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        max_links = min(b1.nbump, b2.nbump, n_bump_max)
        n_matches = min(len(row_indices), max_links)

        if n_matches > 0:
            matched_costs = cost_matrix[row_indices[:n_matches], col_indices[:n_matches]]
            logli = -np.sum(matched_costs)
        else:
            logli = 0.0

    return beta * jc * logli


@njit
def _set_seed(value):
    np.random.seed(value)


@njit
def _uniform_random():
    """Numba-compatible random number generation"""
    return np.random.random()


@njit
def _gaussian_random(mu, sigma):
    """Numba-compatible Gaussian random number generation"""
    return np.random.normal(mu, sigma)


@njit
def _randint(n):
    """Numba-compatible random integer generation"""
    return np.random.randint(0, n)


def _create_bump(
    bump,
    n_bump_max,
    ampli_max,
    kappa_mean,
):
    if bump.nbump >= n_bump_max:
        return True
    if HAS_NUMBA:
        new_pos = _uniform_random() * 2 * np.pi
    else:
        new_pos = random.uniform(0, 2 * np.pi)
    bump.pos.append(new_pos)
    bump.ampli.append(ampli_max)
    bump.kappa.append(kappa_mean)
    bump.nbump += 1
    return False


def _del_bump(bump):
    if bump.nbump == 0:
        return True
    if HAS_NUMBA:
        i = _randint(bump.nbump)
    else:
        i = random.randrange(bump.nbump)
    if i < bump.nbump - 1:
        bump.pos[i] = bump.pos[-1]
        bump.ampli[i] = bump.ampli[-1]
        bump.kappa[i] = bump.kappa[-1]
    bump.pos.pop()
    bump.ampli.pop()
    bump.kappa.pop()
    bump.nbump -= 1
    return False


def _diffuse(
    bump,
    sigma_diff,
):
    """扩散峰位置"""
    if bump.nbump == 0:
        return True
    if HAS_NUMBA:
        i = _randint(bump.nbump)
        new_pos = bump.pos[i] + _gaussian_random(0, sigma_diff)
    else:
        i = random.randrange(bump.nbump)
        new_pos = bump.pos[i] + random.gauss(0, sigma_diff)
    new_pos %= 2 * np.pi
    if new_pos < 0:
        new_pos += 2 * np.pi
    bump.pos[i] = new_pos
    return False


def _change_ampli(bump):
    if bump.nbump == 0:
        return True
    if HAS_NUMBA:
        i = _randint(bump.nbump)
        delta = _gaussian_random(0, 1.0)
    else:
        i = random.randrange(bump.nbump)
        delta = random.gauss(0, 1.0)
    if bump.ampli[i] + delta <= 0:
        return True
    bump.ampli[i] += delta
    return False


def _change_width(bump):
    if bump.nbump == 0:
        return True
    if HAS_NUMBA:
        i = _randint(bump.nbump)
        new_kappa = bump.kappa[i] + _gaussian_random(0, 0.5)
    else:
        i = random.randrange(bump.nbump)
        new_kappa = bump.kappa[i] + random.gauss(0, 0.5)
    if new_kappa < 2.0 or new_kappa > 6.0:
        return True
    bump.kappa[i] = new_kappa
    return False


@njit
def _metropolis_accept(delta_logl):
    """Numba-optimized Metropolis acceptance criterion"""
    if delta_logl > 0:
        return True
    return np.random.random() < np.exp(delta_logl)


def _mcmc(
    nbt,
    data,
    n_steps,
    n_roi,
    n_bump_max,
    sigma_diff,
    ampli_min,
    kappa_mean,
    sig2,
    sigcoup2,
    penbump,
    jc,
    beta,
):
    """MCMC optimization with numba acceleration"""
    ntime = nbt
    # Initialize bump states for all time points
    bumps = [SiteBump() for _ in range(ntime)]
    interfe = [0.0] * (ntime - 1)

    # Initial likelihood calculation
    total_logl = 0.0
    for i in range(ntime):
        data_seg = data[i * n_roi : (i + 1) * n_roi]
        bumps[i].logl = _site_logl(data_seg, bumps[i], n_roi, penbump, sig2, beta)
        total_logl += bumps[i].logl

    for i in range(ntime - 1):
        interfe[i] = _interf_logl(bumps[i], bumps[i + 1], n_bump_max, sigcoup2, beta, jc)
        total_logl += interfe[i]

    print(f"Initial likelihood: {total_logl:.2f}")
    if HAS_NUMBA:
        print(
            f"Using Numba acceleration (n_roi={n_roi}, parallel={'Yes' if n_roi >= 64 else 'No'})"
        )
    else:
        print("Numba not available - install with: pip install numba")

    pbar = tqdm(range(n_steps), desc="MCMC fitting")

    # MCMC iterations (MUST remain serial to preserve MCMC chain correctness)
    for step in pbar:
        # Process each timepoint serially (CANNOT be parallelized due to coupling)
        for j in range(ntime):
            current = bumps[j]
            proposal = current.clone()

            # Select operation type
            rand_val = random.random()
            operation_failed = True

            if rand_val < 0.01:
                operation_failed = _create_bump(proposal, n_bump_max, ampli_min, kappa_mean)
            elif rand_val < 0.01 * (1 + proposal.nbump):
                operation_failed = _del_bump(proposal)
            elif rand_val < 0.3:
                operation_failed = _diffuse(proposal, sigma_diff)
            elif rand_val < 0.4:
                operation_failed = _change_width(proposal)
            else:
                operation_failed = _change_ampli(proposal)

            # If operation succeeded (not failed)
            if not operation_failed:
                # Calculate local likelihood for new state
                data_seg = data[j * n_roi : (j + 1) * n_roi]
                loglt = _site_logl(data_seg, proposal, n_roi, penbump, sig2, beta)

                # Calculate coupling changes
                delta_logl = loglt - current.logl

                # Handle boundary cases
                if j == 0:  # First time point
                    loglit1 = _interf_logl(proposal, bumps[1], n_bump_max, sigcoup2, beta, jc)
                    delta_logl += loglit1 - interfe[0]
                elif j == ntime - 1:  # Last time point
                    loglit1 = _interf_logl(bumps[j - 1], proposal, n_bump_max, sigcoup2, beta, jc)
                    delta_logl += loglit1 - interfe[j - 1]
                else:  # Middle time points
                    loglit1 = _interf_logl(bumps[j - 1], proposal, n_bump_max, sigcoup2, beta, jc)
                    loglit2 = _interf_logl(proposal, bumps[j + 1], n_bump_max, sigcoup2, beta, jc)
                    delta_logl += (loglit1 - interfe[j - 1]) + (loglit2 - interfe[j])

                # Metropolis-Hastings acceptance criterion
                if HAS_NUMBA:
                    accept = _metropolis_accept(delta_logl)
                else:
                    accept = delta_logl > 0 or random.random() < np.exp(delta_logl)

                if accept:
                    proposal.logl = loglt
                    bumps[j] = proposal

                    # Update coupling terms
                    if j == 0:
                        interfe[0] = loglit1
                    elif j == ntime - 1:
                        interfe[j - 1] = loglit1
                    else:
                        interfe[j - 1] = loglit1
                        interfe[j] = loglit2

                    total_logl += delta_logl
        # Update progress bar less frequently to reduce overhead
        if step % 100 == 0:
            pbar.set_postfix({"Log-Likelihood": f"{total_logl:.2f}"})

    return bumps


if __name__ == "__main__":
    data = load_roi_data()
    bumps, fits, nbump, centrbump = bump_fits(
        data, n_steps=5000, n_roi=16, save_path=os.path.join(os.getcwd(), "test.npz")
    )

    # fits = np.load(os.path.join(os.getcwd(), 'test.npz'))['fits']
    create_1d_bump_animation(
        fits,
        show=True,
        save_path=os.path.join(os.getcwd(), "bump_animation.gif"),
        nframes=100,
        bump_selection="first",
    )

    # print(bumps)
