"""
Theta sweep specific visualization functions for CANNs models.

This module contains specialized plotting functions for analyzing theta-modulated
neural activity, particularly for direction cell and grid cell networks.
"""

import logging
import multiprocessing as mp
import platform
import sys
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import matplotlib.patheffects as mpatheffects
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from .core.config import PlotConfig
from .core.jupyter_utils import display_animation_in_jupyter, is_jupyter_environment


@dataclass(slots=True)
class _ThetaSweepPreparedData:
    """Immutable container for precomputed animation arrays."""

    frames: int
    dt: float
    n_step: int
    fps: int
    env_size: float
    position4ani: np.ndarray
    direction4ani: np.ndarray
    direction_bins: np.ndarray
    dc_activity4ani: np.ndarray
    max_dc: float
    gc_bump4ani: np.ndarray
    gc_manifold_flat: np.ndarray
    gc_real_flat: np.ndarray
    pos2phase_twisted: np.ndarray
    manifold_traj_x: np.ndarray
    manifold_traj_y: np.ndarray
    value_grid_twisted: np.ndarray
    points_all: np.ndarray
    vmin: float
    vmax: float


@dataclass(slots=True)
class _ThetaSweepRenderOptions:
    """Configuration for offline rendering backends."""

    figsize: tuple[int, int]
    width_ratios: tuple[float, float, float, float]
    cmap_name: str
    alpha: float
    dpi: int
    show_progress_bar: bool
    workers: int
    start_method: str | None


_FRAME_DATA_CACHE: _ThetaSweepPreparedData | None = None
_FRAME_RENDER_OPTIONS_CACHE: _ThetaSweepRenderOptions | None = None


def _emit_info(message: str) -> None:
    """Emit a lightweight informational message for theta sweep utilities."""

    logger = logging.getLogger(__name__)
    if logger.handlers:
        logger.info(message)
    else:
        print(f"[theta_sweep] {message}")


def _init_theta_sweep_worker(
    data: _ThetaSweepPreparedData,
    render_options: _ThetaSweepRenderOptions,
) -> None:
    """Initializer for multiprocessing workers."""

    global _FRAME_DATA_CACHE, _FRAME_RENDER_OPTIONS_CACHE
    _FRAME_DATA_CACHE = data
    _FRAME_RENDER_OPTIONS_CACHE = render_options


def _render_theta_sweep_frame_from_cache(frame_idx: int) -> np.ndarray:
    """Render frame using data cached by worker initializer."""

    if _FRAME_DATA_CACHE is None or _FRAME_RENDER_OPTIONS_CACHE is None:
        raise RuntimeError("Theta sweep worker has not been initialized with shared data.")
    return _theta_sweep_frame_to_image(frame_idx, _FRAME_DATA_CACHE, _FRAME_RENDER_OPTIONS_CACHE)


def _prepare_theta_sweep_animation_data(
    position_data: np.ndarray,
    direction_data: np.ndarray,
    dc_activity_data: np.ndarray,
    gc_activity_data: np.ndarray,
    gc_network,
    env_size: float,
    mapping_ratio: float,
    n_step: int,
    dt: float,
    fps: int,
) -> _ThetaSweepPreparedData:
    """Compute reusable arrays required for theta sweep animations."""

    position_np = np.asarray(position_data)
    direction_np = np.asarray(direction_data)
    dc_activity_np = np.asarray(dc_activity_data)
    gc_activity_np = np.asarray(gc_activity_data)

    position4ani = position_np[::n_step, :]
    direction4ani = direction_np[::n_step]
    dc_activity4ani = dc_activity_np[::n_step, :].astype(np.float32, copy=False)

    grid_cell_activity = gc_activity_np.reshape(
        -1, gc_network.num_gc_1side, gc_network.num_gc_1side
    )
    gc_bump4ani = grid_cell_activity[::n_step, :, :].astype(np.float32, copy=True)

    edge_margin = 3
    if edge_margin > 0 and min(gc_bump4ani.shape[1:]) > edge_margin * 2:
        gc_bump4ani[:, :edge_margin, :] = np.nan
        gc_bump4ani[:, -edge_margin:, :] = np.nan
        gc_bump4ani[:, :, :edge_margin] = np.nan
        gc_bump4ani[:, :, -edge_margin:] = np.nan

    frames = gc_bump4ani.shape[0]

    direction_bins = np.linspace(-np.pi, np.pi, dc_activity_np.shape[1], endpoint=False)
    max_dc = float(np.max(dc_activity4ani)) if dc_activity4ani.size else 1.0

    vmax_raw = np.nanmax(gc_bump4ani)
    vmax = float(vmax_raw if np.isfinite(vmax_raw) else 1.0)
    vmin = 0.0

    gc_manifold_flat = gc_bump4ani.reshape(frames, -1)

    pos2phase = np.asarray(gc_network.position2phase(position_np.T))[:, ::n_step]
    pos2phase4ani_twisted = np.dot(gc_network.coor_transform_inv, pos2phase)
    x, y = pos2phase4ani_twisted[0, :], pos2phase4ani_twisted[1, :]
    jumps_x = np.where(np.abs(np.diff(x)) > np.pi)[0]
    jumps_y = np.where(np.abs(np.diff(y)) > np.pi)[0]
    jumps = np.unique(np.concatenate([jumps_x, jumps_y])) if jumps_x.size + jumps_y.size > 0 else []
    x_plot, y_plot = x.copy(), y.copy()
    if len(jumps) > 0:
        x_plot[jumps + 1] = np.nan
        y_plot[jumps + 1] = np.nan

    value_grid_twisted = np.dot(gc_network.coor_transform_inv, gc_network.value_grid.T).T
    points_base = value_grid_twisted / mapping_ratio
    nx = int(np.sqrt(gc_network.candidate_centers.shape[0]))
    ny = nx
    candidate_centers = gc_network.candidate_centers.reshape(nx, ny, 2)
    tile_is = range(max(0, nx // 2 - 1), nx)
    tile_js = range(max(0, ny // 2 - 1), ny)
    tile_centers = np.array([candidate_centers[i, j] for i in tile_is for j in tile_js])
    points_all = (points_base[None, :, :] + tile_centers[:, None, :]).reshape(-1, 2)

    K = tile_centers.shape[0] if tile_centers.size else 1
    if K > 1:
        gc_real_flat = np.tile(gc_manifold_flat, (1, K))
    else:
        gc_real_flat = gc_manifold_flat.copy()

    return _ThetaSweepPreparedData(
        frames=frames,
        dt=dt,
        n_step=n_step,
        fps=fps,
        env_size=env_size,
        position4ani=position4ani,
        direction4ani=direction4ani,
        direction_bins=direction_bins,
        dc_activity4ani=dc_activity4ani,
        max_dc=max_dc,
        gc_bump4ani=gc_bump4ani,
        gc_manifold_flat=gc_manifold_flat,
        gc_real_flat=gc_real_flat,
        pos2phase_twisted=pos2phase4ani_twisted,
        manifold_traj_x=x_plot,
        manifold_traj_y=y_plot,
        value_grid_twisted=value_grid_twisted,
        points_all=points_all,
        vmin=vmin,
        vmax=vmax,
    )


def _theta_sweep_frame_to_image(
    frame_idx: int,
    data: _ThetaSweepPreparedData,
    render_options: _ThetaSweepRenderOptions,
) -> np.ndarray:
    """Render a single frame using matplotlib's Agg backend."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=render_options.figsize, dpi=render_options.dpi)
    canvas = FigureCanvasAgg(fig)
    grid_spec = fig.add_gridspec(1, 4, width_ratios=render_options.width_ratios, wspace=0.3)

    ax_traj = fig.add_subplot(grid_spec[0, 0])
    ax_dc = fig.add_subplot(grid_spec[0, 1], projection="polar")
    ax_manifold = fig.add_subplot(grid_spec[0, 2])
    ax_realgc = fig.add_subplot(grid_spec[0, 3])

    # Panel 1: Trajectory
    ax_traj.plot(data.position4ani[:, 0], data.position4ani[:, 1], color="#F18D00", lw=1)
    ax_traj.plot(
        data.position4ani[frame_idx, 0],
        data.position4ani[frame_idx, 1],
        "ro",
        markersize=5,
    )
    ax_traj.set_xlim(0, data.env_size)
    ax_traj.set_ylim(0, data.env_size)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_xticks([0, data.env_size])
    ax_traj.set_yticks([0, data.env_size])
    sns.despine(ax=ax_traj)

    # Panel 2: Direction cells
    ax_dc.plot(data.direction_bins, data.dc_activity4ani[frame_idx], color="#009FB9")
    ax_dc.plot(
        [data.direction4ani[frame_idx], data.direction4ani[frame_idx]],
        [0.0, data.max_dc],
        color="black",
        lw=2,
    )
    ax_dc.set_ylim(0.0, data.max_dc * 1.2 if data.max_dc > 0 else 1.0)
    ax_dc.set_yticks([])
    ax_dc.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax_dc.set_xticklabels(["0°", "90°", "180°", "270°"])

    # Panel 3: Manifold scatter
    ax_manifold.plot(data.manifold_traj_x, data.manifold_traj_y, color="#888888", lw=1, alpha=0.6)
    heatmap = ax_manifold.scatter(
        data.value_grid_twisted[:, 0],
        data.value_grid_twisted[:, 1],
        c=data.gc_manifold_flat[frame_idx],
        s=4,
        cmap=render_options.cmap_name,
        vmin=data.vmin,
        vmax=data.vmax,
        alpha=render_options.alpha,
    )
    ax_manifold.plot(
        data.pos2phase_twisted[0, frame_idx],
        data.pos2phase_twisted[1, frame_idx],
        "ro",
        markersize=5,
    )
    ax_manifold.set_aspect("equal")
    ax_manifold.axis("off")
    fig.colorbar(heatmap, ax=ax_manifold, fraction=0.046, pad=0.04)

    # Panel 4: Real space scatter
    ax_realgc.scatter(
        data.points_all[:, 0],
        data.points_all[:, 1],
        c=data.gc_real_flat[frame_idx],
        s=8,
        cmap=render_options.cmap_name,
        vmin=data.vmin,
        vmax=data.vmax,
        alpha=render_options.alpha,
    )
    ax_realgc.plot(
        data.position4ani[:, 0], data.position4ani[:, 1], color="#F18D00", lw=1, alpha=0.8
    )
    ax_realgc.plot(
        data.position4ani[frame_idx, 0],
        data.position4ani[frame_idx, 1],
        "ro",
        markersize=6,
    )
    ax_realgc.set_xlim(0, data.env_size)
    ax_realgc.set_ylim(0, data.env_size)
    ax_realgc.set_aspect("equal", adjustable="box")
    ax_realgc.set_xticks([])
    ax_realgc.set_yticks([])

    fig.subplots_adjust(left=0.05, right=0.98, top=0.75, bottom=0.12, wspace=0.35)

    # Draw titles in a shared row so they keep a common height and avoid clipping in exports.
    titles_row_y = fig.subplotpars.top + (1.0 - fig.subplotpars.top) * 0.55

    def _axis_midpoint(axis: plt.Axes) -> float:
        bbox = axis.get_position()
        return bbox.x0 + bbox.width / 2.0

    title_fontsize = plt.rcParams.get("axes.titlesize")
    time_seconds = frame_idx * data.n_step * data.dt

    fig.text(
        _axis_midpoint(ax_traj),
        titles_row_y,
        f"Animal Trajectory (t={time_seconds:.2f}s)",
        ha="center",
        va="center",
        fontsize=title_fontsize,
    )
    fig.text(
        _axis_midpoint(ax_dc),
        titles_row_y,
        "Direction Sweep",
        ha="center",
        va="center",
        fontsize=title_fontsize,
    )
    fig.text(
        _axis_midpoint(ax_manifold),
        titles_row_y,
        "GC Sweep on Manifold",
        ha="center",
        va="center",
        fontsize=title_fontsize,
    )
    fig.text(
        _axis_midpoint(ax_realgc),
        titles_row_y,
        "GC Sweep in Real Space",
        ha="center",
        va="center",
        fontsize=title_fontsize,
    )

    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(fig)
    return image


def _render_theta_sweep_animation_with_imageio(
    data: _ThetaSweepPreparedData,
    save_path: str,
    render_options: _ThetaSweepRenderOptions,
) -> None:
    """Render frames to a GIF via imageio (requires optional dependency).

    When ``render_options.workers`` > 1 and the platform provides a ``fork`` start
    method, frames are rendered in parallel across processes. On platforms that
    only support ``spawn`` (e.g., Windows), the function falls back to sequential
    rendering to avoid recursive process spawning.
    """

    try:
        import imageio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "render_backend='imageio' requires the optional dependency 'imageio'. "
            "Install it via `uv pip install imageio` or `pip install imageio`."
        ) from exc

    path_str = str(save_path)
    if not path_str.lower().endswith(".gif"):
        raise ValueError("imageio backend currently supports only GIF outputs (use .gif extension)")

    writer_kwargs = {"duration": 1.0 / data.fps, "loop": 0}

    frame_indices: Iterable[int] = range(data.frames)
    progress_bar = None
    if render_options.show_progress_bar:
        progress_bar = tqdm(total=data.frames, desc="<theta_sweep> Rendering frames")

    use_parallel = render_options.workers > 1
    ctx: mp.context.BaseContext | None = None
    start_method = render_options.start_method
    auto_start_method = start_method is None
    if use_parallel:
        if auto_start_method:
            start_method = "fork" if platform.system() == "Linux" else "spawn"
        if start_method == "fork" and any(
            module_name.startswith("jax") for module_name in sys.modules
        ):
            warnings.warn(
                "Detected JAX in current process; switching theta sweep rendering to 'spawn' "
                "start method to avoid fork deadlocks.",
                RuntimeWarning,
                stacklevel=2,
            )
            start_method = "spawn"
        try:
            ctx = mp.get_context(start_method)
        except (RuntimeError, ValueError):
            use_parallel = False
            warnings.warn(
                f"Multiprocessing start method '{start_method}' is unavailable; "
                "falling back to sequential rendering.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if start_method == "spawn" and auto_start_method:
                warnings.warn(
                    "Theta sweep frames will be rendered using 'spawn'. Large arrays will be pickled "
                    "per worker; monitor memory usage.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    try:
        with imageio.get_writer(path_str, mode="I", **writer_kwargs) as writer:
            if use_parallel and ctx is not None:
                with ProcessPoolExecutor(
                    max_workers=render_options.workers,
                    mp_context=ctx,
                    initializer=_init_theta_sweep_worker,
                    initargs=(data, render_options),
                ) as executor:
                    for frame_image in executor.map(
                        _render_theta_sweep_frame_from_cache,
                        frame_indices,
                    ):
                        writer.append_data(frame_image)
                        if progress_bar is not None:
                            progress_bar.update(1)
            else:
                for frame_idx in frame_indices:
                    frame_image = _theta_sweep_frame_to_image(frame_idx, data, render_options)
                    writer.append_data(frame_image)
                    if progress_bar is not None:
                        progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()


def plot_population_activity_with_theta(
    time_steps: np.ndarray,
    theta_phase: np.ndarray,
    net_activity: np.ndarray,
    direction: np.ndarray,
    config: PlotConfig | None = None,
    add_lines: bool = True,
    atol: float = 1e-2,
    # Backward compatibility parameters
    title: str = "Population Activity with Theta",
    xlabel: str = "Time (s)",
    ylabel: str = "Direction (°)",
    figsize: tuple[int, int] = (12, 4),
    cmap: str = "jet",
    show: bool = True,
    save_path: str | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot neural population activity with theta oscillation markers and direction trace.

    Args:
        time_steps: Array of time points
        theta_phase: Array of theta phase values [-π, π]
        net_activity: 2D array of network activity (time, neurons)
        direction: Array of direction values
        config: PlotConfig object for unified configuration
        add_lines: Whether to add vertical lines at theta phase zeros
        atol: Tolerance for detecting theta phase zeros
        **kwargs: Additional parameters for backward compatibility

    Returns:
        tuple: (figure, axis) objects
    """
    # Handle configuration
    if config is None:
        config = PlotConfig(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            show=show,
            save_path=save_path,
            kwargs={"cmap": cmap, **kwargs},
        )

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        # Plot population activity as heatmap
        plot_kwargs = config.to_matplotlib_kwargs()
        if "cmap" not in plot_kwargs:
            plot_kwargs["cmap"] = "jet"  # Use original default colormap
        im = ax.imshow(
            net_activity.T * 100,
            aspect="auto",
            extent=[time_steps[0], time_steps[-1], -np.pi, np.pi],
            origin="lower",
            **plot_kwargs,
        )

        # Handle direction wrapping for plotting
        jumps = np.where(np.abs(np.diff(direction)) > np.pi)[0]
        direction_plot = direction.copy()
        direction_plot[jumps + 1] = np.nan
        ax.plot(time_steps, direction_plot, color="white", lw=3)

        # Add theta phase markers
        if add_lines:
            zero_phase_index = np.where(np.isclose(theta_phase, 0, atol=atol))[0]
            for i in zero_phase_index:
                ax.axvline(x=time_steps[i], color="grey", linestyle="--", linewidth=1, alpha=0.5)

        # Configure axes
        ax.set_yticks([-np.pi, np.pi])
        ax.set_yticklabels([0, 360])
        ax.set_title(config.title, fontsize=16, fontweight="bold")
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)

        sns.despine(ax=ax)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Activity (%)", fontsize=12)

        # Save and show
        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show:
            plt.show()

        return fig, ax

    except Exception as e:
        plt.close(fig)
        raise e


def plot_direction_cell_polar(
    direction_bins: np.ndarray,
    direction_activity: np.ndarray,
    true_direction: float,
    config: PlotConfig | None = None,
    # Backward compatibility parameters
    title: str = "Direction Cell Activity",
    figsize: tuple[int, int] = (6, 6),
    show: bool = True,
    save_path: str | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot direction cell activity in polar coordinates.

    Args:
        direction_bins: Array of direction bins (radians)
        direction_activity: Array of activity values for each direction
        true_direction: True direction value (radians)
        config: PlotConfig object for unified configuration
        **kwargs: Additional parameters for backward compatibility

    Returns:
        tuple: (figure, axis) objects
    """
    # Handle configuration
    if config is None:
        config = PlotConfig(
            title=title, figsize=figsize, show=show, save_path=save_path, kwargs=kwargs
        )

    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection="polar")

    try:
        # Plot activity
        ax.plot(direction_bins, direction_activity, "k-", linewidth=2)

        # Mark true direction
        max_activity = np.max(direction_activity)
        ax.plot(
            [true_direction, true_direction], [0, max_activity * 1.2], color="#009FB9", linewidth=3
        )

        # Configure polar plot
        ax.set_ylim(0, max_activity * 1.2)
        ax.set_yticks([])
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["0°", "90°", "180°", "270°"])
        ax.set_title(config.title, fontsize=14, fontweight="bold", pad=20)

        # Save and show
        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show:
            plt.show()

        return fig, ax

    except Exception as e:
        plt.close(fig)
        raise e


def plot_grid_cell_manifold(
    value_grid_twisted: np.ndarray,
    grid_cell_activity: np.ndarray,
    config: PlotConfig | None = None,
    ax: plt.Axes | None = None,
    # Backward compatibility parameters
    title: str = "Grid Cell Activity on Manifold",
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "jet",
    show: bool = True,
    save_path: str | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot grid cell activity on the twisted torus manifold.

    Args:
        value_grid_twisted: Coordinates on twisted manifold
        grid_cell_activity: 2D array of grid cell activities
        config: PlotConfig object for unified configuration
        ax: Optional axis to draw on instead of creating a new figure
        **kwargs: Additional parameters for backward compatibility

    Returns:
        tuple: (figure, axis) objects
    """
    # Handle configuration
    if config is None:
        config = PlotConfig(
            title=title,
            figsize=figsize,
            show=show,
            save_path=save_path,
            kwargs={"cmap": cmap, **kwargs},
        )

    plot_kwargs = config.to_matplotlib_kwargs()
    add_colorbar = bool(plot_kwargs.pop("add_colorbar", True))
    colorbar_options = plot_kwargs.pop("colorbar", {}) if add_colorbar else {}
    if "cmap" not in plot_kwargs:
        plot_kwargs["cmap"] = "jet"  # Use original default colormap

    axis_provided = ax is not None
    if not axis_provided:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure

    try:
        # Plot grid cell activity on manifold
        scatter = ax.scatter(
            value_grid_twisted[:, 0],
            value_grid_twisted[:, 1],
            c=grid_cell_activity.flatten(),
            s=8,
            alpha=0.8,
            **plot_kwargs,
        )

        # Configure plot
        ax.set_aspect("equal", adjustable="box")
        if config.title:
            ax.set_title(config.title, fontsize=16, fontweight="bold")
        sns.despine(ax=ax)

        # Add colorbar with a divider-based layout for better spacing
        if add_colorbar:
            default_cbar_opts = {"pad": 0.15, "size": "5%", "label": "Activity"}
            if isinstance(colorbar_options, dict):
                extra_cbar_kwargs = colorbar_options.get("kwargs", {})
                for key in ("pad", "size", "label"):
                    if key in colorbar_options:
                        default_cbar_opts[key] = colorbar_options[key]
            else:
                extra_cbar_kwargs = {}

            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "right", size=default_cbar_opts["size"], pad=default_cbar_opts["pad"]
            )
            cbar = fig.colorbar(scatter, cax=cax, **extra_cbar_kwargs)
            if default_cbar_opts["label"]:
                cbar.set_label(default_cbar_opts["label"], fontsize=12)

        if not axis_provided:
            fig.tight_layout()

        # Save and show
        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show and not axis_provided:
            plt.show()

        return fig, ax

    except Exception as e:
        plt.close(fig)
        raise e


@dataclass(slots=True)
class _PlaceCellAnimationData:
    """Immutable container for place cell animation arrays."""

    frames: int
    dt: float
    n_step: int
    fps: int
    position4ani: np.ndarray
    activity_grid: np.ndarray  # Grid-based activity (frames × rows × cols)
    x_edges: np.ndarray  # Grid cell edges for pcolormesh
    y_edges: np.ndarray
    vmin: float
    vmax: float


def _prepare_place_cell_animation_data(
    position_data: np.ndarray,
    pc_activity_data: np.ndarray,
    pc_network,
    n_step: int,
    dt: float,
    fps: int,
) -> _PlaceCellAnimationData:
    """Compute reusable arrays required for place cell animations."""

    position_np = np.asarray(position_data)
    pc_activity_np = np.asarray(pc_activity_data)

    position4ani = position_np[::n_step, :]
    pc_activity4ani = pc_activity_np[::n_step, :].astype(np.float32, copy=False)

    frames = pc_activity4ani.shape[0]

    # Extract grid structure
    accessible_indices = pc_network.geodesic_result.accessible_indices
    cost_grid = pc_network.geodesic_result.cost_grid

    # Create activity grid: map place cell activity back to 2D grid
    n_rows, n_cols = cost_grid.shape
    activity_grid = np.full((frames, n_rows, n_cols), np.nan, dtype=np.float32)

    # Map activity back to grid positions
    for cell_idx, (row, col) in enumerate(accessible_indices):
        activity_grid[:, row, col] = pc_activity4ani[:, cell_idx]

    # Compute activity range for consistent color scaling
    vmax_raw = np.nanmax(activity_grid)
    vmax = float(vmax_raw if np.isfinite(vmax_raw) else 1.0)
    vmin = 0.0

    return _PlaceCellAnimationData(
        frames=frames,
        dt=dt,
        n_step=n_step,
        fps=fps,
        position4ani=position4ani,
        activity_grid=activity_grid,
        x_edges=cost_grid.x_edges,
        y_edges=cost_grid.y_edges,
        vmin=vmin,
        vmax=vmax,
    )


def create_theta_sweep_place_cell_animation(
    position_data: np.ndarray,
    pc_activity_data: np.ndarray,
    pc_network,  # PlaceCellNetwork instance
    navigation_task,  # BaseNavigationTask instance
    dt: float = 0.001,
    config: PlotConfig | None = None,
    # Animation control parameters
    n_step: int = 10,  # Subsample every n_step frames
    fps: int = 10,
    figsize: tuple[int, int] = (12, 4),
    save_path: str | None = None,
    show: bool = True,
    show_progress_bar: bool = True,
    **kwargs,
) -> FuncAnimation | None:
    """
    Create theta sweep animation for place cell network with 2 panels:
    1. Environment trajectory with place cell bump overlay
    2. Population activity heatmap over time

    Args:
        position_data: Animal position data (time, 2)
        pc_activity_data: Place cell activity (time, num_cells)
        pc_network: PlaceCellNetwork instance
        navigation_task: BaseNavigationTask instance for environment visualization
        dt: Time step size
        config: PlotConfig object for unified configuration
        n_step: Subsample every n_step frames for animation
        fps: Frames per second for animation
        figsize: Figure size (width, height)
        save_path: Path to save animation (GIF or MP4)
        show: Whether to display animation
        show_progress_bar: Whether to show progress bar during saving
        **kwargs: Additional parameters (cmap, alpha, etc.)

    Returns:
        FuncAnimation: Matplotlib animation object
    """
    # Handle configuration
    if config is None:
        config = PlotConfig(
            figsize=figsize,
            fps=fps,
            save_path=save_path,
            show=show,
            show_progress_bar=show_progress_bar,
            kwargs=kwargs,
        )
    else:
        config.show_progress_bar = show_progress_bar
        if save_path is not None:
            config.save_path = save_path
        if fps is not None:
            config.fps = fps
        if kwargs:
            merged_kwargs = dict(config.kwargs or {})
            merged_kwargs.update(kwargs)
            config.kwargs = merged_kwargs

    # Prepare animation data
    data = _prepare_place_cell_animation_data(
        position_data=position_data,
        pc_activity_data=pc_activity_data,
        pc_network=pc_network,
        n_step=n_step,
        dt=dt,
        fps=config.fps,
    )

    cmap_name = config.kwargs.get("cmap", "viridis")
    alpha = config.kwargs.get("alpha", 0.8)
    trajectory_color = config.kwargs.get("trajectory_color", "#F18D00")

    # Setup figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=config.figsize, width_ratios=[1, 1])
    ax_env, ax_activity = axes

    # Panel 1: Environment with place cell bump overlay
    # Plot environment boundaries
    env = navigation_task.env
    if env.boundary is not None:
        boundary_array = np.array(env.boundary)
        ax_env.fill(boundary_array[:, 0], boundary_array[:, 1], color="lightgrey", alpha=0.3)
        ax_env.plot(
            np.append(boundary_array[:, 0], boundary_array[0, 0]),
            np.append(boundary_array[:, 1], boundary_array[0, 1]),
            color="black",
            linewidth=2,
        )

    # Plot walls if they exist
    if env.walls is not None and len(env.walls) > 0:
        for wall in env.walls:
            wall_array = np.array(wall)
            ax_env.plot(wall_array[:, 0], wall_array[:, 1], color="black", linewidth=2)

    # Create pcolormesh for place cell activity (grid-based visualization like grid cells)
    # Flip y_edges for proper display (top-down to bottom-up)
    y_edges_plot = data.y_edges[::-1]
    initial_grid = np.flipud(data.activity_grid[0])

    pc_mesh = ax_env.pcolormesh(
        data.x_edges,
        y_edges_plot,
        initial_grid,
        cmap=cmap_name,
        vmin=data.vmin,
        vmax=data.vmax,
        alpha=alpha,
        shading="auto",
        zorder=1,
    )

    # Plot trajectory on top
    ax_env.plot(
        data.position4ani[:, 0],
        data.position4ani[:, 1],
        color=trajectory_color,
        lw=1.5,
        alpha=0.9,
        zorder=15,
    )

    # Current position marker
    (pos_marker,) = ax_env.plot([], [], "ro", markersize=8, zorder=20)

    ax_env.set_aspect("equal", adjustable="box")
    ax_env.set_title("Place Cell Activity in Environment")

    # Panel 2: Population activity heatmap over time
    time_axis = np.arange(data.frames) * n_step * dt

    # Only show accessible cells (filter out NaN cells)
    accessible_indices = pc_network.geodesic_result.accessible_indices
    n_cells = len(accessible_indices)

    # Extract activity only for accessible cells (cells × time)
    activity_flat = np.zeros((n_cells, data.frames), dtype=np.float32)
    for cell_idx, (row, col) in enumerate(accessible_indices):
        activity_flat[cell_idx, :] = data.activity_grid[:, row, col]

    # Create heatmap
    heatmap = ax_activity.imshow(
        activity_flat,
        aspect="auto",
        extent=[time_axis[0], time_axis[-1], 0, n_cells],
        origin="lower",
        cmap=cmap_name,
        vmin=data.vmin,
        vmax=data.vmax,
    )

    # Current time marker (vertical line)
    (time_marker,) = ax_activity.plot([], [], "w-", lw=2, alpha=0.8)

    ax_activity.set_xlabel("Time (s)")
    ax_activity.set_ylabel("Place Cell Index")
    ax_activity.set_title("Population Activity Over Time")

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax_activity)
    cbar.set_label("Activity", fontsize=10)

    fig.tight_layout()

    # Animation title with time
    time_text = fig.text(
        0.5,
        0.98,
        "",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    time_text.set_animated(True)

    def update(frame):
        """Update function for animation."""
        t = frame * n_step * dt

        # Update panel 1: place cell activity pcolormesh
        # Flip the grid for proper display
        grid_flipped = np.flipud(data.activity_grid[frame])
        pc_mesh.set_array(grid_flipped.ravel())

        # Update current position marker
        pos_marker.set_data([data.position4ani[frame, 0]], [data.position4ani[frame, 1]])

        # Update panel 2: time marker
        time_marker.set_data([t, t], [0, n_cells])

        # Update title with current time
        time_text.set_text(f"t = {t:.2f}s")

        return pc_mesh, pos_marker, time_marker, time_text

    # Create animation
    interval_ms = 1000 // config.fps
    ani = FuncAnimation(
        fig, update, frames=data.frames, interval=interval_ms, blit=True, repeat=True
    )

    # Save and/or show animation
    if config.save_path:
        # Warn if both saving and showing (causes double rendering)
        if config.show and data.frames > 50:
            from .core import warn_double_rendering

            warn_double_rendering(data.frames, config.save_path, stacklevel=2)

        from canns.analyzer.visualization.core import get_matplotlib_writer

        writer = get_matplotlib_writer(config.save_path, fps=config.fps)

        if config.show_progress_bar:
            progress_bar = tqdm(
                total=data.frames,
                desc=f"<{create_theta_sweep_place_cell_animation.__name__}> Saving to {config.save_path}",
            )

            last_frame = {"value": -1}

            def progress_callback(current_frame, _total_frames):
                update = current_frame - last_frame["value"]
                if update > 0:
                    progress_bar.update(update)
                    last_frame["value"] = current_frame

            ani.save(config.save_path, writer=writer, progress_callback=progress_callback)
            progress_bar.close()
        else:
            ani.save(config.save_path, writer=writer)
        print(f"Animation saved to: {config.save_path}")

    if config.show:
        # Automatically detect Jupyter and display as HTML/JS
        if is_jupyter_environment():
            display_animation_in_jupyter(ani)
            plt.close(fig)  # Close after HTML conversion to prevent auto-display
        else:
            plt.show()
    else:
        plt.close(fig)

    # Return None in Jupyter when showing to avoid double display
    if config.show and is_jupyter_environment():
        return None
    return ani


def create_theta_sweep_grid_cell_animation(
    position_data: np.ndarray,
    direction_data: np.ndarray,
    dc_activity_data: np.ndarray,
    gc_activity_data: np.ndarray,
    gc_network,  # GridCellNetwork instance
    env_size: float,
    mapping_ratio: float,
    dt: float = 0.001,
    config: PlotConfig | None = None,
    # Animation control parameters
    n_step: int = 10,  # Subsample every n_step frames
    fps: int = 10,
    figsize: tuple[int, int] = (12, 3),
    save_path: str | None = None,
    show: bool = True,
    show_progress_bar: bool = True,
    render_backend: str | None = "auto",
    output_dpi: int = 150,
    render_workers: int | None = None,
    render_start_method: str | None = None,
    **kwargs,
) -> FuncAnimation | None:
    """
    Create comprehensive theta sweep animation with 4 panels (optimized for speed):
    1. Animal trajectory
    2. Direction cell polar plot
    3. Grid cell activity on manifold
    4. Grid cell activity in real space

    Args:
        position_data: Animal position data (time, 2)
        direction_data: Direction data (time,)
        dc_activity_data: Direction cell activity (time, neurons)
        gc_activity_data: Grid cell activity (time, neurons)
        gc_network: GridCellNetwork instance for coordinate transformations
        env_size: Environment size
        mapping_ratio: Mapping ratio for grid cells
        dt: Time step size
        config: PlotConfig object for unified configuration
        n_step: Subsample every n_step frames for animation
        render_backend: Rendering backend. Use 'matplotlib', 'imageio', or 'auto'/'None' for auto-detect.
        output_dpi: Target DPI when rendering frames with non-interactive backends
        render_workers: Worker processes for imageio backend. ``None`` auto-selects, 0 disables.
        render_start_method: Multiprocessing start method ('fork', 'spawn', 'forkserver') or None for auto
        **kwargs: Additional parameters for backward compatibility

    Returns:
        FuncAnimation | None: Matplotlib animation object for interactive backend, otherwise None
    """
    # Handle configuration
    if config is None:
        config = PlotConfig(
            figsize=figsize,
            fps=fps,
            save_path=save_path,
            show=show,
            show_progress_bar=show_progress_bar,
            kwargs=kwargs,
        )
    else:
        # Keep backward compatibility while respecting explicit function arguments.
        config.show_progress_bar = show_progress_bar
        if save_path is not None:
            config.save_path = save_path
        if fps is not None:
            config.fps = fps
        if kwargs:
            merged_kwargs = dict(config.kwargs or {})
            merged_kwargs.update(kwargs)
            config.kwargs = merged_kwargs

    data = _prepare_theta_sweep_animation_data(
        position_data=position_data,
        direction_data=direction_data,
        dc_activity_data=dc_activity_data,
        gc_activity_data=gc_activity_data,
        gc_network=gc_network,
        env_size=env_size,
        mapping_ratio=mapping_ratio,
        n_step=n_step,
        dt=dt,
        fps=config.fps,
    )

    cmap_name = config.kwargs.get("cmap", "jet")
    alpha = config.kwargs.get("alpha", 0.8)
    trajectory_color = config.kwargs.get("trajectory_color", "#FFFFFF")
    trajectory_outline = config.kwargs.get("trajectory_outline", "#1A1A1A")
    current_marker_color = config.kwargs.get("current_marker_color", "#FF2D00")

    backend_requested = (render_backend or "auto").lower()
    auto_backend = backend_requested in {"auto", "none", ""}
    backend = backend_requested
    if auto_backend:
        try:
            import imageio  # noqa: F401
        except ImportError:
            backend = "matplotlib"
            warnings.warn(
                "Falling back to Matplotlib backend because imageio is not installed.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            backend = "imageio"
            _emit_info("Using imageio backend for theta sweep animation (auto-detected).")

    if backend == "imageio" and not config.save_path:
        if auto_backend:
            _emit_info(
                "Auto-selected imageio backend requires save_path; falling back to Matplotlib."
            )
            backend = "matplotlib"
        else:
            raise ValueError(
                "render_backend='imageio' requires a valid save_path (e.g., path/to/output.gif)"
            )

    if backend == "imageio":
        try:
            import imageio  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "render_backend='imageio' requires the optional dependency 'imageio'. "
                "Install it via `uv pip install imageio` or `pip install imageio`."
            ) from exc

        if render_workers is not None and render_workers < 0:
            raise ValueError("render_workers must be >= 0")

        available_methods: set[str]
        try:
            available_methods = set(mp.get_all_start_methods())
        except (AttributeError, NotImplementedError):
            available_methods = {"spawn"}

        start_method = render_start_method
        if start_method is not None and start_method not in available_methods:
            warnings.warn(
                f"Requested start method '{start_method}' is unavailable; choosing automatically instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            start_method = None

        jax_loaded = any(name.startswith("jax") for name in sys.modules)
        if start_method is None:
            if platform.system() == "Linux" and "fork" in available_methods and not jax_loaded:
                start_method = "fork"
            elif "spawn" in available_methods:
                start_method = "spawn"
            elif available_methods:
                start_method = sorted(available_methods)[0]
            else:
                start_method = None

        workers: int
        if render_workers is None:
            if start_method is None:
                workers = 0
            else:
                workers = max(mp.cpu_count() - 1, 1)
        else:
            workers = render_workers

        if workers > 0 and start_method is None:
            warnings.warn(
                "Parallel rendering requested but no multiprocessing start method is available; "
                "falling back to sequential rendering.",
                RuntimeWarning,
                stacklevel=2,
            )
            workers = 0

        if start_method == "spawn" and jax_loaded:
            _emit_info("Detected JAX; using 'spawn' start method to avoid fork-related deadlocks.")

        render_options = _ThetaSweepRenderOptions(
            figsize=config.figsize,
            width_ratios=(1.0, 1.0, 1.0, 1.0),
            cmap_name=cmap_name,
            alpha=alpha,
            dpi=output_dpi,
            show_progress_bar=config.show_progress_bar,
            workers=workers,
            start_method=start_method,
        )
        _render_theta_sweep_animation_with_imageio(data, config.save_path, render_options)
        return None

    if backend != "matplotlib":
        raise ValueError("Unsupported render_backend. Use 'matplotlib' or 'imageio'.")

    if render_workers not in (None, 0):
        _emit_info("render_workers is ignored when render_backend='matplotlib'.")
    if render_start_method is not None:
        _emit_info("render_start_method is ignored when render_backend='matplotlib'.")

    # Setup figure with 4 panels
    fig, axes = plt.subplots(1, 4, figsize=config.figsize, width_ratios=[1, 1, 1, 1])
    ax_traj, ax_dc_placeholder, ax_manifold, ax_realgc = axes

    # Panel 1: Animal Trajectory (static trajectory + dynamic dot, time in title)
    ax_traj.plot(data.position4ani[:, 0], data.position4ani[:, 1], color="#F18D00", lw=1)
    (traj_dot,) = ax_traj.plot([], [], "ro", markersize=5)
    ax_traj.set_xlim(0, env_size)
    ax_traj.set_ylim(0, env_size)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_xticks([0, env_size])
    ax_traj.set_yticks([0, env_size])

    # Panel 2: Direction Cells (Polar plot) - replace placeholder
    ax_dc_placeholder.remove()
    ax_dc = fig.add_subplot(1, 4, 2, projection="polar")
    max_dc = data.max_dc
    (hd_line,) = ax_dc.plot([], [], "k-", lw=2)
    (id_line,) = ax_dc.plot(data.direction_bins, data.dc_activity4ani[0], color="#009FB9")
    ax_dc.set_ylim(0.0, max_dc * 1.2)
    ax_dc.set_yticks([])
    ax_dc.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax_dc.set_xticklabels(["0°", "90°", "180°", "270°"])

    # Panel 3: Grid Cells on Manifold
    # Calculate manifold coordinates and static trajectory
    # Static manifold trajectory
    ax_manifold.plot(
        data.manifold_traj_x,
        data.manifold_traj_y,
        color="#888888",
        lw=1,
        alpha=0.6,
    )

    # Grid cell activity heatmap on manifold
    heatmap = ax_manifold.scatter(
        data.value_grid_twisted[:, 0],
        data.value_grid_twisted[:, 1],
        c=np.zeros(len(data.value_grid_twisted)),
        s=4,
        cmap=cmap_name,
        vmin=data.vmin,
        vmax=data.vmax,
        alpha=alpha,
        zorder=1,
    )
    (manifold_traj_line,) = ax_manifold.plot(
        [],
        [],
        color=trajectory_color,
        lw=2.4,
        alpha=1.0,
        zorder=3,
    )
    manifold_traj_line.set_animated(True)
    manifold_traj_line.set_data([], [])
    manifold_traj_line.set_path_effects(
        [
            mpatheffects.Stroke(
                linewidth=manifold_traj_line.get_linewidth() + 1.6,
                foreground=trajectory_outline,
                alpha=0.9,
            ),
            mpatheffects.Normal(),
        ]
    )
    (manifold_dot,) = ax_manifold.plot(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=8,
        markerfacecolor=current_marker_color,
        markeredgecolor=trajectory_outline,
        markeredgewidth=1.2,
        zorder=4,
    )
    manifold_dot.set_animated(True)

    ax_manifold.set_aspect("equal")
    ax_manifold.axis("off")

    # Panel 4: Grid Cells in Real Space (Pre-compute all tile positions)
    points_all = data.points_all
    M = data.gc_bump4ani.shape[1] * data.gc_bump4ani.shape[2]
    total_points = len(points_all)
    K = total_points // M if M > 0 else 1

    # Create single scatter object for all tiles
    colors_buffer = np.zeros((K, M), dtype=np.float32)
    colors_buffer_flat = colors_buffer.reshape(-1)
    scatter_real = ax_realgc.scatter(
        points_all[:, 0],
        points_all[:, 1],
        c=colors_buffer_flat,
        cmap=cmap_name,
        vmin=data.vmin,
        vmax=data.vmax,
        s=8,
        alpha=alpha,
        zorder=1,
    )

    # Static trajectory and dynamic dot
    (traj_line_real,) = ax_realgc.plot(
        [],
        [],
        color=trajectory_color,
        lw=2.6,
        alpha=1.0,
        zorder=3,
    )
    traj_line_real.set_animated(True)
    traj_line_real.set_data([], [])
    traj_line_real.set_path_effects(
        [
            mpatheffects.Stroke(
                linewidth=traj_line_real.get_linewidth() + 1.6,
                foreground=trajectory_outline,
                alpha=0.9,
            ),
            mpatheffects.Normal(),
        ]
    )
    (red_dot_real,) = ax_realgc.plot(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=8,
        markerfacecolor=current_marker_color,
        markeredgecolor=trajectory_outline,
        markeredgewidth=1.2,
        zorder=4,
    )
    red_dot_real.set_animated(True)

    ax_realgc.set_xlim(0, env_size)
    ax_realgc.set_ylim(0, env_size)
    ax_realgc.set_aspect("equal", adjustable="box")
    ax_realgc.set_xticks([])
    ax_realgc.set_yticks([])

    fig.subplots_adjust(left=0.05, right=0.98, top=0.75, bottom=0.12, wspace=0.35)

    # Draw titles in a shared row so they keep a common height and avoid clipping in exports.
    titles_row_y = fig.subplotpars.top + (1.0 - fig.subplotpars.top) * 0.55

    def _axis_midpoint(axis: plt.Axes) -> float:
        bbox = axis.get_position()
        return bbox.x0 + bbox.width / 2.0

    title_fontsize = plt.rcParams.get("axes.titlesize")

    traj_title = fig.text(
        _axis_midpoint(ax_traj),
        titles_row_y,
        "Animal Trajectory",
        ha="center",
        va="center",
        fontsize=title_fontsize,
    )
    traj_title.set_animated(True)
    fig.text(
        _axis_midpoint(ax_dc),
        titles_row_y,
        "Direction Sweep",
        ha="center",
        va="center",
        fontsize=title_fontsize,
    )
    fig.text(
        _axis_midpoint(ax_manifold),
        titles_row_y,
        "GC Sweep on Manifold",
        ha="center",
        va="center",
        fontsize=title_fontsize,
    )
    fig.text(
        _axis_midpoint(ax_realgc),
        titles_row_y,
        "GC Sweep in Real Space",
        ha="center",
        va="center",
        fontsize=title_fontsize,
    )

    def update(frame):
        """Optimized animation update function"""
        t = frame * n_step * dt

        # Panel 1: Update trajectory dot and title with time
        traj_dot.set_data([data.position4ani[frame, 0]], [data.position4ani[frame, 1]])
        traj_title.set_text(f"Animal Trajectory (t={t:.2f}s)")

        # Panel 2: Update direction cells
        hd_line.set_data([data.direction4ani[frame], data.direction4ani[frame]], [0, max_dc])
        id_line.set_ydata(data.dc_activity4ani[frame])

        # Panel 3: Update manifold heatmap and dot
        heatmap.set_array(data.gc_bump4ani[frame].flatten())
        # Show trajectory up to current frame for clarity
        manifold_traj_line.set_data(
            data.manifold_traj_x[: frame + 1],
            data.manifold_traj_y[: frame + 1],
        )
        manifold_dot.set_data(
            [data.pos2phase_twisted[0, frame]],
            [data.pos2phase_twisted[1, frame]],
        )

        # Panel 4: Update real space scatter (OPTIMIZED - no cla()!)
        frame_mat = data.gc_bump4ani[frame]
        np.copyto(colors_buffer, frame_mat.reshape(1, M))
        scatter_real.set_array(colors_buffer_flat)

        # Update current position dot
        traj_line_real.set_data(
            data.position4ani[: frame + 1, 0],
            data.position4ani[: frame + 1, 1],
        )
        red_dot_real.set_data([data.position4ani[frame, 0]], [data.position4ani[frame, 1]])

        return (
            traj_dot,
            hd_line,
            id_line,
            heatmap,
            manifold_traj_line,
            manifold_dot,
            scatter_real,
            traj_line_real,
            red_dot_real,
            traj_title,
        )

    # Create animation with blit for maximum speed
    interval_ms = 1000 // config.fps
    ani = FuncAnimation(
        fig, update, frames=data.frames, interval=interval_ms, blit=True, repeat=True
    )

    # Save and/or show animation with progress bar (following plotting utilities pattern)
    if config.save_path:
        # Warn if both saving and showing (causes double rendering)
        if config.show and data.frames > 50:
            from .core import warn_double_rendering

            warn_double_rendering(data.frames, config.save_path, stacklevel=2)

        # Use FFMpegWriter for better performance than PillowWriter
        from canns.analyzer.visualization.core import get_matplotlib_writer

        writer = get_matplotlib_writer(config.save_path, fps=config.fps)

        if config.show_progress_bar:
            progress_bar = tqdm(
                total=data.frames,
                desc=f"<{create_theta_sweep_grid_cell_animation.__name__}> Saving to {config.save_path}",
            )

            last_frame = {"value": -1}

            def progress_callback(current_frame, _total_frames):
                update = current_frame - last_frame["value"]
                if update > 0:
                    progress_bar.update(update)
                    last_frame["value"] = current_frame

            ani.save(config.save_path, writer=writer, progress_callback=progress_callback)
            progress_bar.close()
        else:
            ani.save(config.save_path, writer=writer)
        print(f"Animation saved to: {config.save_path}")

    if config.show:
        # Automatically detect Jupyter and display as HTML/JS
        if is_jupyter_environment():
            display_animation_in_jupyter(ani)
            plt.close(fig)  # Close after HTML conversion to prevent auto-display
        else:
            plt.show()
    else:
        plt.close(fig)

    # Return None in Jupyter when showing to avoid double display
    if config.show and is_jupyter_environment():
        return None
    return ani
