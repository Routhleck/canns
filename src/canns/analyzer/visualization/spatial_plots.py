"""Spatial visualization functions for neural firing field heatmaps.

This module provides plotting utilities for visualizing spatial firing patterns
of neural populations, particularly for grid cells, place cells, and band cells.
Includes specialized grid cell analysis visualizations (autocorrelation, grid score,
spacing analysis) and tracking animations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm

from .core.config import PlotConfig, PlotConfigs, finalize_figure
from .core.jupyter_utils import display_animation_in_jupyter, is_jupyter_environment

__all__ = [
    "plot_firing_field_heatmap",
    "plot_autocorrelation",
    "plot_grid_score",
    "plot_grid_spacing_analysis",
    "create_grid_cell_tracking_animation",
]


@dataclass(slots=True)
class _GridCellTrackingRenderOptions:
    """Rendering options for grid cell tracking animation."""

    figsize: tuple[int, int]
    title: str
    env_size: float
    dt: float
    dpi: int
    position: np.ndarray  # Full trajectory
    activity: np.ndarray  # Full activity
    rate_map: np.ndarray  # Rate map
    sim_indices_to_render: np.ndarray  # Frame indices
    total_sim_steps: int


def _render_single_grid_tracking_frame(
    frame_index: int,
    options: _GridCellTrackingRenderOptions,
) -> np.ndarray:
    """Render a single frame for grid cell tracking animation (module-level for pickling)."""
    from io import BytesIO

    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=options.figsize)
    sim_idx = options.sim_indices_to_render[frame_index]
    current_time_s = sim_idx * options.dt / 1000.0

    # Panel 1: Trajectory
    ax1.plot(options.position[:, 0], options.position[:, 1], color="gray", alpha=0.3, linewidth=0.5)
    ax1.plot([options.position[sim_idx, 0]], [options.position[sim_idx, 1]], "ro", markersize=8)
    ax1.set_xlim(0, options.env_size)
    ax1.set_ylim(0, options.env_size)
    ax1.set_xlabel("X Position (m)", fontsize=10)
    ax1.set_ylabel("Y Position (m)", fontsize=10)
    ax1.set_title("Trajectory", fontsize=11, fontweight="bold")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Activity time course
    time_axis = np.arange(options.total_sim_steps) * options.dt
    ax2.plot(time_axis, options.activity, color="steelblue", alpha=0.3, linewidth=0.5)
    time_slice = time_axis[: sim_idx + 1]
    activity_slice = options.activity[: sim_idx + 1]
    ax2.plot(time_slice, activity_slice, "b-", linewidth=2)
    ax2.plot([time_axis[sim_idx]], [options.activity[sim_idx]], "ro", markersize=6)
    ax2.set_xlim(0, time_axis[-1])
    ax2.set_ylim(0, np.max(options.activity) * 1.1 if np.max(options.activity) > 0 else 1)
    ax2.set_xlabel("Time (ms)", fontsize=10)
    ax2.set_ylabel("Firing Rate", fontsize=10)
    ax2.set_title("Activity", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Rate map with position
    im = ax3.imshow(
        options.rate_map.T,
        origin="lower",
        cmap="hot",
        extent=[0, options.env_size, 0, options.env_size],
        aspect="auto",
    )
    ax3.plot(
        [options.position[sim_idx, 0]],
        [options.position[sim_idx, 1]],
        "c*",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax3.set_xlabel("X Position (m)", fontsize=10)
    ax3.set_ylabel("Y Position (m)", fontsize=10)
    ax3.set_title("Firing Rate Map", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Overall title with time
    fig.suptitle(
        f"{options.title}  |  Time: {current_time_s:.2f} s", fontsize=13, fontweight="bold"
    )

    fig.tight_layout()

    # Convert to numpy array
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=options.dpi, bbox_inches="tight")
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()

    # Convert to uint8
    if img.dtype in (np.float32, np.float64):
        img = (img * 255).astype(np.uint8)

    return img


def _ensure_plot_config(
    config: PlotConfig | None,
    factory,
    *,
    kwargs: dict[str, Any] | None = None,
    **defaults: Any,
) -> PlotConfig:
    """Normalize PlotConfig creation while preserving backward arguments."""

    if config is None:
        defaults.update({"kwargs": kwargs or {}})
        return factory(**defaults)

    if kwargs:
        config_kwargs = config.kwargs or {}
        config_kwargs.update(kwargs)
        config.kwargs = config_kwargs
    return config


def plot_firing_field_heatmap(
    heatmap: np.ndarray,
    config: PlotConfig | None = None,
    # Backward compatibility parameters
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int, int] = (5, 5),
    cmap: str = "jet",
    interpolation: str = "nearest",
    origin: str = "lower",
    show: bool = True,
    save_path: str | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a single spatial firing field heatmap.

    This function creates a publication-quality heatmap visualization of neural
    spatial firing patterns. It supports both modern PlotConfig-based configuration
    and legacy keyword arguments for backward compatibility.

    Args:
        heatmap (np.ndarray): 2D array of shape (M, K) representing spatial
            firing rates in each bin.
        config (PlotConfig | None): Unified configuration object. If None,
            uses backward compatibility parameters.
        title (str | None): Plot title. If None, no title is displayed.
        xlabel (str | None): X-axis label. If None, no label is displayed.
        ylabel (str | None): Y-axis label. If None, no label is displayed.
        figsize (tuple[int, int]): Figure size (width, height) in inches.
            Defaults to (5, 5).
        cmap (str): Colormap name for the heatmap. Defaults to 'jet'.
        interpolation (str): Interpolation method for imshow. Defaults to 'nearest'.
        origin (str): Origin position for imshow ('lower' or 'upper').
            Defaults to 'lower'.
        show (bool): Whether to display the plot. Defaults to True.
        save_path (str | None): Path to save the figure. If None, figure is not saved.
        **kwargs: Additional keyword arguments passed to plt.imshow().

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axis objects for further customization.

    Example:
        >>> import numpy as np
        >>> from canns.analyzer.visualization import plot_firing_field_heatmap, PlotConfig
        >>>
        >>> # Dummy input heatmap (M x K)
        >>> heatmap = np.random.rand(6, 6)
        >>> config = PlotConfig(title="Neuron 0", show=False)
        >>> fig, ax = plot_firing_field_heatmap(heatmap, config=config)
        >>> print(fig is not None)
        True
    """
    # Handle configuration
    if config is None:
        config = PlotConfig(
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            show=show,
            save_path=save_path,
            kwargs={"cmap": cmap, "interpolation": interpolation, "origin": origin, **kwargs},
        )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=config.figsize)

    # Extract plotting parameters
    plot_kwargs = config.to_matplotlib_kwargs()
    if "cmap" not in plot_kwargs:
        plot_kwargs["cmap"] = cmap
    if "interpolation" not in plot_kwargs:
        plot_kwargs["interpolation"] = interpolation
    if "origin" not in plot_kwargs:
        plot_kwargs["origin"] = origin

    # Plot heatmap
    im = ax.imshow(heatmap.T, **plot_kwargs)

    # Set labels and title if provided
    if config.title:
        ax.set_title(config.title, fontsize=14, fontweight="bold")
    if config.xlabel:
        ax.set_xlabel(config.xlabel, fontsize=11)
    if config.ylabel:
        ax.set_ylabel(config.ylabel, fontsize=11)

    # Remove ticks for cleaner appearance (only if no labels)
    if not config.xlabel:
        ax.set_xticks([])
    if not config.ylabel:
        ax.set_yticks([])

    # Tight layout for better appearance
    fig.tight_layout()

    finalize_figure(fig, config, rasterize_artists=[im] if config.rasterized else None)
    return fig, ax


def plot_autocorrelation(
    autocorr: np.ndarray,
    config: PlotConfig | None = None,
    *,
    title: str = "Spatial Autocorrelation",
    xlabel: str = "X Lag (bins)",
    ylabel: str = "Y Lag (bins)",
    figsize: tuple[int, int] = (6, 6),
    save_path: str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 2D spatial autocorrelation heatmap.

    Visualizes the spatial autocorrelation map which reveals periodic patterns
    in grid cell firing fields. For grid cells, this will show a characteristic
    hexagonal pattern of peaks indicating 60-degree rotational symmetry.

    Args:
        autocorr (np.ndarray): 2D spatial autocorrelation map, normalized to [-1, 1].
        config (PlotConfig | None): Unified configuration object. If None,
            uses backward compatibility parameters.
        title (str): Plot title. Defaults to "Spatial Autocorrelation".
        xlabel (str): X-axis label. Defaults to "X Lag (bins)".
        ylabel (str): Y-axis label. Defaults to "Y Lag (bins)".
        figsize (tuple[int, int]): Figure size (width, height) in inches.
            Defaults to (6, 6).
        save_path (str | None): Path to save the figure. If None, not saved.
        show (bool): Whether to display the plot. Defaults to True.
        **kwargs: Additional keyword arguments passed to plt.imshow().

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes objects.

    Example:
        >>> import numpy as np
        >>> from canns.analyzer.metrics.spatial_metrics import compute_spatial_autocorrelation
        >>> from canns.analyzer.visualization import plot_autocorrelation, PlotConfigs
        >>>
        >>> rate_map = np.random.rand(10, 10)
        >>> autocorr = compute_spatial_autocorrelation(rate_map)
        >>> config = PlotConfigs.grid_autocorrelation(show=False)
        >>> fig, ax = plot_autocorrelation(autocorr, config=config)
        >>> print(fig is not None)
        True

    References:
        Sargolini et al. (2006). Conjunctive representation of position, direction,
        and velocity in entorhinal cortex. Science, 312(5774), 758-762.
    """
    config = _ensure_plot_config(
        config,
        PlotConfigs.grid_autocorrelation,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        save_path=save_path,
        show=show,
        kwargs=kwargs,
    )

    fig, ax = plt.subplots(figsize=config.figsize)

    # Get plot kwargs with defaults
    plot_kwargs = config.to_matplotlib_kwargs()
    if "cmap" not in plot_kwargs:
        plot_kwargs["cmap"] = "RdBu_r"
    if "vmin" not in plot_kwargs:
        plot_kwargs["vmin"] = -1
    if "vmax" not in plot_kwargs:
        plot_kwargs["vmax"] = 1

    # Plot autocorrelation
    im = ax.imshow(autocorr, origin="lower", aspect="equal", **plot_kwargs)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", fontsize=10)

    # Set labels and title
    ax.set_title(config.title, fontsize=14, fontweight="bold")
    ax.set_xlabel(config.xlabel, fontsize=11)
    ax.set_ylabel(config.ylabel, fontsize=11)

    # Center at (0, 0)
    center = np.array(autocorr.shape) // 2
    ax.axhline(center[0], color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(center[1], color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()

    finalize_figure(fig, config, rasterize_artists=[im] if config.rasterized else None)
    return fig, ax


def plot_grid_score(
    rotated_corrs: dict[int, float],
    grid_score: float,
    config: PlotConfig | None = None,
    *,
    title: str = "Grid Score Analysis",
    xlabel: str = "Rotation Angle (°)",
    ylabel: str = "Correlation",
    figsize: tuple[int, int] = (8, 5),
    grid: bool = True,
    save_path: str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot bar chart of rotational correlations with grid score.

    Visualizes the correlations at different rotation angles used to compute
    the grid score. Highlights 60° and 120° (hexagonal angles) which should
    be high for grid cells, versus 30°, 90°, and 150° which should be lower.

    Args:
        rotated_corrs (dict[int, float]): Dictionary mapping rotation angles
            to correlation values. Keys: 30, 60, 90, 120, 150.
        grid_score (float): Computed grid score value.
        config (PlotConfig | None): Unified configuration object.
        title (str): Plot title. Defaults to "Grid Score Analysis".
        xlabel (str): X-axis label. Defaults to "Rotation Angle (°)".
        ylabel (str): Y-axis label. Defaults to "Correlation".
        figsize (tuple[int, int]): Figure size. Defaults to (8, 5).
        grid (bool): Whether to show grid lines. Defaults to True.
        save_path (str | None): Path to save the figure.
        show (bool): Whether to display the plot. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes objects.

    Example:
        >>> import numpy as np
        >>> from canns.analyzer.metrics.spatial_metrics import compute_grid_score
        >>> from canns.analyzer.visualization import plot_grid_score
        >>>
        >>> autocorr = np.random.rand(10, 10)
        >>> grid_score, rotated_corrs = compute_grid_score(autocorr)
        >>> fig, ax = plot_grid_score(rotated_corrs, grid_score, show=False)
        >>> print(isinstance(grid_score, float))
        True
    """
    config = _ensure_plot_config(
        config,
        PlotConfigs.grid_score_plot,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        grid=grid,
        save_path=save_path,
        show=show,
        kwargs=kwargs,
    )

    fig, ax = plt.subplots(figsize=config.figsize)

    angles = [30, 60, 90, 120, 150]
    correlations = [rotated_corrs[angle] for angle in angles]

    # Color bars: red for hexagonal angles (60, 120), blue for others
    colors = ["steelblue" if angle not in [60, 120] else "crimson" for angle in angles]

    # Create bar chart
    ax.bar(angles, correlations, color=colors, alpha=0.7, edgecolor="black", linewidth=1.2)

    # Add horizontal line at y=0
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)

    # Set labels and title with grid score
    ax.set_title(f"{config.title}\nGrid Score = {grid_score:.3f}", fontsize=14, fontweight="bold")
    ax.set_xlabel(config.xlabel, fontsize=12)
    ax.set_ylabel(config.ylabel, fontsize=12)
    ax.set_xticks(angles)

    # Add grid if requested
    if config.grid:
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Add text annotation if grid cell confirmed
    if grid_score > 0.3:
        ax.text(
            0.5,
            0.95,
            "Grid Cell Confirmed (score > 0.3)",
            transform=ax.transAxes,
            fontsize=11,
            color="green",
            fontweight="bold",
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3),
        )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="crimson", alpha=0.7, label="Hexagonal (60°, 120°)"),
        Patch(facecolor="steelblue", alpha=0.7, label="Non-hexagonal (30°, 90°, 150°)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()

    finalize_figure(fig, config)
    return fig, ax


def plot_grid_spacing_analysis(
    autocorr: np.ndarray,
    spacing_bins: float,
    bin_size: float | None = None,
    config: PlotConfig | None = None,
    *,
    title: str = "Grid Spacing Analysis",
    xlabel: str = "Distance (bins)",
    ylabel: str = "Autocorrelation",
    figsize: tuple[int, int] = (8, 5),
    grid: bool = True,
    save_path: str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot radial profile of autocorrelation with spacing markers.

    Visualizes how autocorrelation changes with distance from center,
    revealing the periodic spacing of grid fields. The detected spacing
    is marked with a vertical line.

    Args:
        autocorr (np.ndarray): 2D autocorrelation map.
        spacing_bins (float): Detected grid spacing in bins.
        bin_size (float | None): Size of spatial bins in real units (e.g., meters).
            If provided, shows dual x-axis with real distance.
        config (PlotConfig | None): Unified configuration object.
        title (str): Plot title. Defaults to "Grid Spacing Analysis".
        xlabel (str): X-axis label. Defaults to "Distance (bins)".
        ylabel (str): Y-axis label. Defaults to "Autocorrelation".
        figsize (tuple[int, int]): Figure size. Defaults to (8, 5).
        grid (bool): Whether to show grid lines. Defaults to True.
        save_path (str | None): Path to save the figure.
        show (bool): Whether to display the plot. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes objects.

    Example:
        >>> import numpy as np
        >>> from canns.analyzer.metrics.spatial_metrics import find_grid_spacing
        >>> from canns.analyzer.visualization import plot_grid_spacing_analysis
        >>>
        >>> autocorr = np.random.rand(12, 12)
        >>> spacing_bins, spacing_m = find_grid_spacing(autocorr, bin_size=0.05)
        >>> fig, ax = plot_grid_spacing_analysis(
        ...     autocorr,
        ...     spacing_bins,
        ...     bin_size=0.05,
        ...     show=False,
        ... )
        >>> print(spacing_m is not None)
        True
    """
    config = _ensure_plot_config(
        config,
        PlotConfigs.grid_spacing_plot,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        grid=grid,
        save_path=save_path,
        show=show,
        kwargs=kwargs,
    )

    fig, ax = plt.subplots(figsize=config.figsize)

    # Compute radial average
    center = np.array(autocorr.shape) // 2
    y, x = np.ogrid[: autocorr.shape[0], : autocorr.shape[1]]
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    # Bin by distance
    max_dist = int(min(center))
    radial_profile = []
    distances = []

    for dist in range(max_dist):
        mask = (r >= dist) & (r < dist + 1)
        if np.any(mask):
            radial_profile.append(np.mean(autocorr[mask]))
            distances.append(dist)

    # Plot radial profile
    ax.plot(distances, radial_profile, linewidth=2, color="steelblue", label="Radial Average")

    # Mark detected spacing
    ax.axvline(
        spacing_bins,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Detected Spacing: {spacing_bins:.1f} bins",
    )

    # If bin_size provided, add secondary x-axis
    if bin_size is not None:
        spacing_real = spacing_bins * bin_size
        ax2 = ax.twiny()
        ax2.set_xlim(np.array(ax.get_xlim()) * bin_size)
        ax2.set_xlabel("Distance (m)", fontsize=11, color="gray")
        ax2.tick_params(axis="x", labelcolor="gray")

        # Update legend with real units
        ax.axvline(
            spacing_bins,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"Detected Spacing: {spacing_bins:.1f} bins = {spacing_real:.3f}m",
        )

    # Set labels and title
    ax.set_title(config.title, fontsize=14, fontweight="bold")
    ax.set_xlabel(config.xlabel, fontsize=12)
    ax.set_ylabel(config.ylabel, fontsize=12)

    # Add grid and legend
    if config.grid:
        ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()

    finalize_figure(fig, config)
    return fig, ax


def create_grid_cell_tracking_animation(
    position: np.ndarray,
    activity: np.ndarray,
    rate_map: np.ndarray,
    config: PlotConfig | None = None,
    *,
    time_steps_per_second: int | None = None,
    fps: int = 20,
    title: str = "Grid Cell Tracking",
    figsize: tuple[int, int] = (15, 5),
    env_size: float = 1.0,
    dt: float = 1.0,
    repeat: bool = True,
    save_path: str | None = None,
    show: bool = True,
    show_progress_bar: bool = True,
    render_backend: str | None = "auto",
    output_dpi: int = 150,
    render_workers: int | None = None,
    render_start_method: str | None = None,
    **kwargs: Any,
) -> animation.FuncAnimation | None:
    """Create 3-panel animation showing grid cell tracking behavior.

    Creates a synchronized animation with three panels:
    1. Left: Trajectory with current position marker
    2. Center: Firing rate time course
    3. Right: Rate map with position overlay

    Args:
        position (np.ndarray): Trajectory array of shape (T, 2) with (x, y) coordinates.
        activity (np.ndarray): Neural activity time series of shape (T,).
        rate_map (np.ndarray): Spatial firing field of shape (M, K).
        config (PlotConfig | None): Unified configuration object.
        time_steps_per_second (int | None): Number of simulation steps per second
            (e.g., 1000 for dt=1ms). Required unless in config.
        fps (int): Frames per second for the animation. Defaults to 20.
        title (str): Overall plot title. Defaults to "Grid Cell Tracking".
        figsize (tuple[int, int]): Figure size. Defaults to (15, 5).
        env_size (float): Environment size for trajectory plot. Defaults to 1.0.
        dt (float): Time step size in milliseconds. Defaults to 1.0.
        repeat (bool): Whether animation should loop. Defaults to True.
        save_path (str | None): Path to save animation (e.g., 'tracking.gif').
        show (bool): Whether to display the animation. Defaults to True.
        show_progress_bar (bool): Whether to show progress bar during save. Defaults to True.
        render_backend (str | None): Rendering backend ('imageio', 'matplotlib', or 'auto')
        output_dpi (int): DPI for rendered frames (affects file size and quality)
        render_workers (int | None): Number of parallel workers (None = auto-detect)
        render_start_method (str | None): Multiprocessing start method ('fork', 'spawn', or None)
        **kwargs: Additional keyword arguments.

    Returns:
        FuncAnimation | None: Animation object, or None if displayed in Jupyter.

    Example:
        >>> import numpy as np
        >>> from canns.analyzer.visualization import (
        ...     create_grid_cell_tracking_animation,
        ...     PlotConfigs,
        ... )
        >>>
        >>> position = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        >>> activity = np.array([0.0, 0.5, 1.0])
        >>> rate_map = np.random.rand(5, 5)
        >>> config = PlotConfigs.grid_cell_tracking_animation(
        ...     time_steps_per_second=10,
        ...     fps=2,
        ...     show=False,
        ... )
        >>> anim = create_grid_cell_tracking_animation(
        ...     position,
        ...     activity,
        ...     rate_map,
        ...     config=config,
        ...     env_size=1.0,
        ... )
        >>> print(anim is not None)
        True
    """
    config = _ensure_plot_config(
        config,
        PlotConfigs.grid_cell_tracking_animation,
        time_steps_per_second=time_steps_per_second,
        fps=fps,
        title=title,
        figsize=figsize,
        repeat=repeat,
        save_path=save_path,
        show=show,
        show_progress_bar=show_progress_bar,
        kwargs=kwargs,
    )

    if config.time_steps_per_second is None:
        config.time_steps_per_second = time_steps_per_second

    if config.time_steps_per_second is None:
        raise ValueError("time_steps_per_second must be provided via argument or config.")

    # Validate inputs
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError(f"position must be (T, 2), got shape {position.shape}")
    if activity.ndim != 1:
        raise ValueError(f"activity must be 1D array, got shape {activity.shape}")
    if position.shape[0] != activity.shape[0]:
        raise ValueError(
            f"position and activity must have same length: {position.shape[0]} != {activity.shape[0]}"
        )
    if rate_map.ndim != 2:
        raise ValueError(f"rate_map must be 2D array, got shape {rate_map.shape}")

    # Calculate frame indices
    total_sim_steps = len(position)
    total_duration_s = total_sim_steps / config.time_steps_per_second
    num_video_frames = int(total_duration_s * config.fps)
    sim_indices_to_render = np.linspace(0, total_sim_steps - 1, num_video_frames, dtype=int)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=config.figsize)

    try:
        # Panel 1: Trajectory
        ax1.plot(position[:, 0], position[:, 1], color="gray", alpha=0.3, linewidth=0.5)
        (scatter,) = ax1.plot([], [], "ro", markersize=8, label="Current Position")
        ax1.set_xlim(0, env_size)
        ax1.set_ylim(0, env_size)
        ax1.set_xlabel("X Position (m)", fontsize=10)
        ax1.set_ylabel("Y Position (m)", fontsize=10)
        ax1.set_title("Trajectory", fontsize=11, fontweight="bold")
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Activity time course
        time_axis = np.arange(total_sim_steps) * dt
        ax2.plot(time_axis, activity, color="steelblue", alpha=0.3, linewidth=0.5)
        (activity_line,) = ax2.plot([], [], "b-", linewidth=2, label="Activity")
        (activity_marker,) = ax2.plot([], [], "ro", markersize=6)
        ax2.set_xlim(0, time_axis[-1])
        ax2.set_ylim(0, np.max(activity) * 1.1 if np.max(activity) > 0 else 1)
        ax2.set_xlabel("Time (ms)", fontsize=10)
        ax2.set_ylabel("Firing Rate", fontsize=10)
        ax2.set_title("Activity", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Rate map with position
        im = ax3.imshow(
            rate_map.T, origin="lower", cmap="hot", extent=[0, env_size, 0, env_size], aspect="auto"
        )
        (pos_marker,) = ax3.plot(
            [], [], "c*", markersize=15, markeredgecolor="white", markeredgewidth=1.5
        )
        ax3.set_xlabel("X Position (m)", fontsize=10)
        ax3.set_ylabel("Y Position (m)", fontsize=10)
        ax3.set_title("Firing Rate Map", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # Overall title with time
        time_text = fig.suptitle("", fontsize=13, fontweight="bold")

        fig.tight_layout()

        def animate(frame_index: int):
            sim_idx = sim_indices_to_render[frame_index]
            current_time_s = sim_idx * dt / 1000.0  # Convert ms to s

            # Update trajectory marker
            scatter.set_data([position[sim_idx, 0]], [position[sim_idx, 1]])

            # Update activity line (show up to current time)
            time_slice = time_axis[: sim_idx + 1]
            activity_slice = activity[: sim_idx + 1]
            activity_line.set_data(time_slice, activity_slice)
            activity_marker.set_data([time_axis[sim_idx]], [activity[sim_idx]])

            # Update position marker on rate map
            pos_marker.set_data([position[sim_idx, 0]], [position[sim_idx, 1]])

            # Update time display
            time_text.set_text(f"{title}  |  Time: {current_time_s:.2f} s")

            return scatter, activity_line, activity_marker, pos_marker, time_text

        ani = None
        progress_bar_enabled = getattr(config, "show_progress_bar", show_progress_bar)

        # Save animation using unified backend system
        if config.save_path:
            # Warn if both saving and showing (causes double rendering)
            if config.show and num_video_frames > 50:
                from canns.analyzer.visualization.core import warn_double_rendering

                warn_double_rendering(num_video_frames, config.save_path, stacklevel=2)

            # Use unified backend selection system
            from canns.analyzer.visualization.core import (
                emit_backend_warnings,
                get_imageio_writer_kwargs,
                get_multiprocessing_context,
                get_optimal_worker_count,
                select_animation_backend,
            )

            backend_selection = select_animation_backend(
                save_path=config.save_path,
                requested_backend=render_backend,
                check_imageio_plugins=True,
            )

            emit_backend_warnings(backend_selection.warnings, stacklevel=2)
            backend = backend_selection.backend

            if backend == "imageio":
                # Use imageio backend with parallel rendering
                workers = (
                    render_workers if render_workers is not None else get_optimal_worker_count()
                )
                ctx, start_method = get_multiprocessing_context(
                    prefer_fork=(render_start_method == "fork")
                )

                # Create render options
                render_options = _GridCellTrackingRenderOptions(
                    figsize=config.figsize,
                    title=title,
                    env_size=env_size,
                    dt=dt,
                    dpi=output_dpi,
                    position=position,
                    activity=activity,
                    rate_map=rate_map,
                    sim_indices_to_render=sim_indices_to_render,
                    total_sim_steps=total_sim_steps,
                )

                # Get format-specific kwargs
                writer_kwargs, mode = get_imageio_writer_kwargs(config.save_path, config.fps)

                try:
                    from functools import partial

                    import imageio

                    # Create partial function with options
                    render_func = partial(
                        _render_single_grid_tracking_frame, options=render_options
                    )

                    with imageio.get_writer(config.save_path, mode=mode, **writer_kwargs) as writer:
                        if workers > 1 and ctx is not None:
                            # Parallel rendering
                            with ctx.Pool(processes=workers) as pool:
                                frames_iter = pool.imap(render_func, range(num_video_frames))

                                if progress_bar_enabled:
                                    frames_iter = tqdm(
                                        frames_iter,
                                        total=num_video_frames,
                                        desc=f"Rendering {config.save_path}",
                                    )

                                for frame_img in frames_iter:
                                    writer.append_data(frame_img)
                        else:
                            # Serial rendering
                            frames_range = range(num_video_frames)
                            if progress_bar_enabled:
                                frames_range = tqdm(
                                    frames_range,
                                    desc=f"Rendering {config.save_path}",
                                )

                            for frame_idx in frames_range:
                                frame_img = render_func(frame_idx)
                                writer.append_data(frame_img)

                    print(f"Animation saved to: {config.save_path}")

                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"imageio rendering failed: {e}. Falling back to matplotlib.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    backend = "matplotlib"

            if backend == "matplotlib":
                # Use matplotlib backend (traditional FuncAnimation)
                ani = animation.FuncAnimation(
                    fig,
                    animate,
                    frames=num_video_frames,
                    interval=1000 / config.fps,
                    blit=True,
                    repeat=config.repeat,
                )

                from canns.analyzer.visualization.core import get_matplotlib_writer

                writer = get_matplotlib_writer(config.save_path, fps=config.fps)

                if progress_bar_enabled:
                    pbar = tqdm(total=num_video_frames, desc=f"Saving to {config.save_path}")

                    def progress_callback(current_frame: int, total_frames: int) -> None:
                        pbar.update(1)

                    try:
                        ani.save(
                            config.save_path, writer=writer, progress_callback=progress_callback
                        )
                        print(f"Animation saved to: {config.save_path}")
                    finally:
                        pbar.close()
                else:
                    ani.save(config.save_path, writer=writer)
                    print(f"Animation saved to: {config.save_path}")

        # Create animation object for showing (if not already created)
        if config.show and ani is None:
            ani = animation.FuncAnimation(
                fig,
                animate,
                frames=num_video_frames,
                interval=1000 / config.fps,
                blit=True,
                repeat=config.repeat,
            )

        if config.show:
            if is_jupyter_environment():
                display_animation_in_jupyter(ani)
                plt.close(fig)
            else:
                plt.show()
    finally:
        if not config.show:
            plt.close(fig)

    # Return None in Jupyter when showing to avoid double display
    if config.show and is_jupyter_environment():
        return None
    return ani
