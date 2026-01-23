from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm
from scipy import signal
from scipy.ndimage import binary_closing, gaussian_filter
from scipy.stats import binned_statistic_2d, multivariate_normal
from tqdm import tqdm

from ...visualization.core import (
    PlotConfig,
    emit_backend_warnings,
    finalize_figure,
    get_matplotlib_writer,
    get_optimal_worker_count,
    render_animation_parallel,
    select_animation_backend,
    warn_double_rendering,
)
from ...visualization.core.jupyter_utils import display_animation_in_jupyter, is_jupyter_environment
from .config import CANN2DPlotConfig, ProcessingError, SpikeEmbeddingConfig
from .embedding import embed_spike_trains


def _ensure_plot_config(
    config: PlotConfig | None,
    factory,
    *,
    kwargs: dict[str, Any] | None = None,
    **defaults: Any,
) -> PlotConfig:
    if config is None:
        defaults.update({"kwargs": kwargs or {}})
        return factory(**defaults)

    if kwargs:
        config_kwargs = config.kwargs or {}
        config_kwargs.update(kwargs)
        config.kwargs = config_kwargs
    return config


def _ensure_parent_dir(save_path: str | None) -> None:
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def _render_torus_frame(frame_index: int, frame_data: dict[str, Any]) -> np.ndarray:
    from io import BytesIO

    import numpy as np

    fig = plt.figure(figsize=frame_data["figsize"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_zlim(*frame_data["zlim"])
    ax.view_init(frame_data["elev"], frame_data["azim"])
    ax.axis("off")

    frame = frame_data["frames"][frame_index]
    m = frame["m"]

    ax.plot_surface(
        frame_data["torus_x"],
        frame_data["torus_y"],
        frame_data["torus_z"],
        facecolors=cm.viridis(m / (np.max(m) + 1e-9)),
        alpha=1,
        linewidth=0.1,
        antialiased=True,
        rstride=1,
        cstride=1,
        shade=False,
    )

    time_label = frame.get("time")
    label_text = f"Frame: {frame_index + 1}/{len(frame_data['frames'])}"
    if time_label is not None:
        label_text = f"{label_text} | Time: {time_label}"
    ax.text2D(
        0.05,
        0.95,
        label_text,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=frame_data["dpi"], bbox_inches="tight")
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()

    if img.dtype in (np.float32, np.float64):
        img = (img * 255).astype(np.uint8)

    return img


def _render_2d_bump_frame(frame_index: int, frame_data: dict[str, Any]) -> np.ndarray:
    from io import BytesIO

    fig, ax = plt.subplots(figsize=frame_data["figsize"])
    ax.set_xlabel("Manifold Dimension 1 (rad)", fontsize=12)
    ax.set_ylabel("Manifold Dimension 2 (rad)", fontsize=12)
    ax.set_title("CANN2D Bump Activity (2D Projection)", fontsize=14, fontweight="bold")

    im = ax.imshow(
        frame_data["maps"][frame_index].T,
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    fig.colorbar(im, ax=ax).set_label("Activity", fontsize=11)
    ax.text(
        0.02,
        0.98,
        f"Frame: {frame_index + 1}/{len(frame_data['maps'])}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=frame_data["dpi"], bbox_inches="tight")
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    buf.close()

    if img.dtype in (np.float32, np.float64):
        img = (img * 255).astype(np.uint8)

    return img


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
    else:
        if save_path is not None:
            config.save_path = save_path
        if show is not None:
            config.show = show
        if not config.title:
            config.title = title
        if not config.xlabel:
            config.xlabel = xlabel
        if not config.ylabel:
            config.ylabel = ylabel
        if not config.zlabel:
            config.zlabel = zlabel
        if config.figsize == PlotConfig().figsize:
            config.figsize = figsize
        if dpi is not None:
            config.dpi = dpi

    reduced_data = reduce_func(embed_data[::5])

    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], s=1, alpha=0.5)

    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_zlabel(config.zlabel)

    config.save_dpi = getattr(config, "dpi", config.save_dpi)
    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)

    return fig


def plot_path_compare(
    x: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    config: PlotConfig | None = None,
    *,
    title: str = "Path Compare",
    figsize: tuple[int, int] = (12, 5),
    show: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot physical path vs decoded coho-space path side-by-side."""
    from .path import draw_base_parallelogram, skew_transform, snake_wrap_trail_in_parallelogram

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    coords = np.asarray(coords)

    if coords.ndim != 2 or coords.shape[1] < 1:
        raise ValueError(f"coords must be 2D with at least 1 column, got {coords.shape}")

    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title=title,
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    fig, axes = plt.subplots(1, 2, figsize=config.figsize)
    if config.title:
        fig.suptitle(config.title)

    ax0 = axes[0]
    ax0.set_title("Physical path (x,y)")
    ax0.set_aspect("equal", "box")
    ax0.axis("off")
    ax0.plot(x, y, lw=0.9, alpha=0.8)

    ax1 = axes[1]
    ax1.set_title("Decoded coho path")
    ax1.set_aspect("equal", "box")
    ax1.axis("off")

    if coords.shape[1] >= 2:
        theta2 = coords[:, :2] % (2 * np.pi)
        xy = skew_transform(theta2)
        draw_base_parallelogram(ax1)
        trail = snake_wrap_trail_in_parallelogram(
            xy, np.array([2 * np.pi, 0.0]), np.array([np.pi, np.sqrt(3) * np.pi])
        )
        ax1.plot(trail[:, 0], trail[:, 1], lw=0.9, alpha=0.9)
    else:
        th = coords[:, 0] % (2 * np.pi)
        ax1.plot(np.cos(th), np.sin(th), lw=0.9, alpha=0.9)
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)

    fig.tight_layout()
    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return fig, axes


def plot_cohomap(
    decoding_result: dict[str, Any],
    position_data: dict[str, Any],
    config: PlotConfig | None = None,
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
    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title="CohoMap",
        xlabel="",
        ylabel="",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )
    config.save_dpi = dpi

    # Extract data
    coordsbox = decoding_result["coordsbox"]
    times_box = decoding_result["times_box"]
    xx = position_data["x"]
    yy = position_data["y"]

    # Subsample time indices for plotting
    plot_times = np.arange(0, len(coordsbox), subsample)

    # Create a two-panel figure (one per cohomology dimension)
    plt.set_cmap("viridis")
    fig, ax = plt.subplots(1, 2, figsize=config.figsize)

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

    fig.tight_layout()

    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return fig


def plot_cohomap1(
    decoding_result: dict,
    position_data: dict,
    config: PlotConfig | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (10, 4),
    dpi: int = 300,
    subsample: int = 10,
) -> plt.Figure:
    """
    Visualize CohoMap: decoded circular coordinates mapped onto spatial trajectory.
    Automatically handles N-dimensional coordsbox.

    Each subplot shows the spatial trajectory colored by cos(coord_i)
    """
    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title="CohoMap",
        xlabel="",
        ylabel="",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )
    config.save_dpi = dpi

    coordsbox = decoding_result["coordsbox"]
    times_box = decoding_result["times_box"]
    xx = position_data["x"]
    yy = position_data["y"]

    plot_times = np.arange(0, len(coordsbox), subsample)
    num_dims = coordsbox.shape[1]

    fig, axes = plt.subplots(1, num_dims, figsize=(5 * num_dims, 4))
    if num_dims == 1:
        axes = [axes]

    for i in range(num_dims):
        axes[i].axis("off")
        axes[i].set_aspect("equal", "box")
        im = axes[i].scatter(
            xx[times_box][plot_times],
            yy[times_box][plot_times],
            c=np.cos(coordsbox[plot_times, i]),
            s=8,
            cmap="viridis",
        )
        plt.colorbar(im, ax=axes[i], label=f"cos(coord {i + 1})")
        axes[i].set_title(f"CohoMap Dim {i + 1}")

    fig.tight_layout()
    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return fig


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
    render_backend: str | None = "auto",
    output_dpi: int = 150,
    render_workers: int | None = None,
    **kwargs,
) -> animation.FuncAnimation | None:
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
        matplotlib.animation.FuncAnimation | None: The animation object, or None when shown in Jupyter.
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = CANN2DPlotConfig.for_torus_animation(
            title=kwargs.get("title", "3D Bump on Torus"),
            figsize=figsize,
            fps=fps,
            repeat=True,
            show_progress_bar=show_progress,
            save_path=save_path,
            show=show,
            numangsint=numangsint,
            r1=r1,
            r2=r2,
            window_size=window_size,
            frame_step=frame_step,
            n_frames=n_frames,
            **kwargs,
        )
    else:
        if save_path is not None:
            config.save_path = save_path
        if show is not None:
            config.show = show
        if figsize is not None:
            config.figsize = figsize
        if fps is not None:
            config.fps = fps
        if show_progress is not None:
            config.show_progress_bar = show_progress
        config.numangsint = numangsint
        config.r1 = r1
        config.r2 = r2
        config.window_size = window_size
        config.frame_step = frame_step
        config.n_frames = n_frames

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Extract configuration values
    save_path = config.save_path
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
        ani = None
        progress_bar_enabled = show_progress

        if save_path:
            _ensure_parent_dir(save_path)
            if show and len(frame_data) > 50:
                warn_double_rendering(len(frame_data), save_path, stacklevel=2)

            backend_selection = select_animation_backend(
                save_path=save_path,
                requested_backend=render_backend,
                check_imageio_plugins=True,
            )
            emit_backend_warnings(backend_selection.warnings, stacklevel=2)
            backend = backend_selection.backend

            if backend == "imageio":
                render_data = {
                    "frames": frame_data,
                    "torus_x": torus_x,
                    "torus_y": torus_y,
                    "torus_z": torus_z,
                    "figsize": figsize,
                    "dpi": output_dpi,
                    "elev": -125,
                    "azim": 135,
                    "zlim": (-2, 2),
                }
                workers = render_workers
                if workers is None:
                    workers = config.render_workers
                if workers is None:
                    workers = get_optimal_worker_count()
                try:
                    render_animation_parallel(
                        _render_torus_frame,
                        render_data,
                        num_frames=len(frame_data),
                        save_path=save_path,
                        fps=fps,
                        num_workers=workers,
                        show_progress=progress_bar_enabled,
                    )
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"imageio rendering failed: {e}. Falling back to matplotlib.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    backend = "matplotlib"

            if backend == "matplotlib":
                ani = animation.FuncAnimation(
                    fig,
                    animate,
                    frames=len(frame_data),
                    interval=interval_ms,
                    blit=False,
                    repeat=config.repeat,
                )

                writer = get_matplotlib_writer(save_path, fps=fps)
                if progress_bar_enabled:
                    pbar = tqdm(total=len(frame_data), desc=f"Saving to {save_path}")

                    def progress_callback(current_frame: int, total_frames: int) -> None:
                        pbar.update(1)

                    try:
                        ani.save(save_path, writer=writer, progress_callback=progress_callback)
                    finally:
                        pbar.close()
                else:
                    ani.save(save_path, writer=writer)

        if show:
            if ani is None:
                ani = animation.FuncAnimation(
                    fig,
                    animate,
                    frames=len(frame_data),
                    interval=interval_ms,
                    blit=False,
                    repeat=config.repeat,
                )
            if is_jupyter_environment():
                display_animation_in_jupyter(ani)
                plt.close(fig)
            else:
                plt.show()
        else:
            plt.close(fig)

        if show and is_jupyter_environment():
            return None
        return ani

    except Exception as e:
        plt.close(fig)
        raise ProcessingError(f"Failed to create torus animation: {e}") from e


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
    config: PlotConfig | None = None,
    render_backend: str | None = "auto",
    output_dpi: int = 150,
    render_workers: int | None = None,
) -> animation.FuncAnimation | None:
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
            render_backend=render_backend,
            output_dpi=output_dpi,
            render_workers=render_workers,
        )

    if mode != "fast":
        raise ProcessingError(f"Invalid mode '{mode}'. Must be 'fast' or '3d'.")

    if config is None:
        config = PlotConfig.for_animation(
            time_steps_per_second=1000,
            title="CANN2D Bump Activity (2D Projection)",
            figsize=figsize,
            fps=fps,
            show=show,
            save_path=save_path,
            show_progress_bar=show_progress,
        )
    else:
        if save_path is not None:
            config.save_path = save_path
        if show is not None:
            config.show = show
        if figsize is not None:
            config.figsize = figsize
        if fps is not None:
            config.fps = fps
        if show_progress is not None:
            config.show_progress_bar = show_progress

    save_path = config.save_path
    show = config.show
    fps = config.fps
    figsize = config.figsize
    show_progress = config.show_progress_bar

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
    except AttributeError:
        use_blitting = False

    interval_ms = 1000 / fps

    def _build_animation():
        return animation.FuncAnimation(
            fig,
            update,
            frames=len(frame_activity_maps),
            init_func=init,
            interval=interval_ms,
            blit=use_blitting,
            repeat=config.repeat,
        )

    ani = None
    progress_bar_enabled = show_progress

    if save_path:
        _ensure_parent_dir(save_path)
        if show and len(frame_activity_maps) > 50:
            warn_double_rendering(len(frame_activity_maps), save_path, stacklevel=2)

        backend_selection = select_animation_backend(
            save_path=save_path,
            requested_backend=render_backend,
            check_imageio_plugins=True,
        )
        emit_backend_warnings(backend_selection.warnings, stacklevel=2)
        backend = backend_selection.backend

        if backend == "imageio":
            render_data = {
                "maps": frame_activity_maps,
                "figsize": figsize,
                "dpi": output_dpi,
            }
            workers = render_workers
            if workers is None:
                workers = config.render_workers
            if workers is None:
                workers = get_optimal_worker_count()
            try:
                render_animation_parallel(
                    _render_2d_bump_frame,
                    render_data,
                    num_frames=len(frame_activity_maps),
                    save_path=save_path,
                    fps=fps,
                    num_workers=workers,
                    show_progress=progress_bar_enabled,
                )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"imageio rendering failed: {e}. Falling back to matplotlib.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                backend = "matplotlib"

        if backend == "matplotlib":
            ani = _build_animation()
            writer = get_matplotlib_writer(save_path, fps=fps)
            if progress_bar_enabled:
                pbar = tqdm(total=len(frame_activity_maps), desc=f"Saving to {save_path}")

                def progress_callback(current_frame: int, total_frames: int) -> None:
                    pbar.update(1)

                try:
                    ani.save(save_path, writer=writer, progress_callback=progress_callback)
                finally:
                    pbar.close()
            else:
                ani.save(save_path, writer=writer)

    if show:
        if ani is None:
            ani = _build_animation()
        if is_jupyter_environment():
            display_animation_in_jupyter(ani)
            plt.close(fig)
        else:
            plt.show()
    else:
        plt.close(fig)

    if show and is_jupyter_environment():
        return None
    return ani
