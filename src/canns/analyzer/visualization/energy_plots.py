"""Energy landscape visualization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm

from .core.config import PlotConfig, PlotConfigs
from .core.jupyter_utils import display_animation_in_jupyter, is_jupyter_environment

__all__ = [
    "energy_landscape_1d_static",
    "energy_landscape_1d_animation",
    "energy_landscape_2d_static",
    "energy_landscape_2d_animation",
]


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


def energy_landscape_1d_static(
    data_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    config: PlotConfig | None = None,
    *,
    title: str = "1D Energy Landscape",
    xlabel: str = "Collective Variable / State",
    ylabel: str = "Energy",
    show_legend: bool = True,
    figsize: tuple[int, int] = (10, 6),
    grid: bool = False,
    save_path: str | None = None,
    show: bool = True,
    **kwargs: Any,
):
    """Plot a 1D static energy landscape using Matplotlib.

    This mirrors the long-form description from the pre-reorganisation module so
    existing documentation references stay accurate. The function accepts a
    dictionary of datasets, plotting each curve on the same set of axes while
    honouring the ``PlotConfig`` defaults callers relied on previously.

    Args:
        data_sets: Mapping of series labels to ``(x, y)`` tuples representing
            the energy curve to draw.
        config: Optional :class:`PlotConfig` carrying shared styling.
        title: Plot title when no config override is supplied.
        xlabel: X-axis label when no config override is supplied.
        ylabel: Y-axis label when no config override is supplied.
        show_legend: Whether to display the legend for labelled curves.
        figsize: Figure size forwarded to Matplotlib when creating the axes.
        grid: Whether to enable a grid background.
        save_path: Optional path for persisting the plot to disk.
        show: Whether to display the generated figure.
        **kwargs: Additional keyword arguments forwarded to ``ax.plot``.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The created figure and axes handles.
    """

    config = _ensure_plot_config(
        config,
        PlotConfigs.energy_landscape_1d_static,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show_legend=show_legend,
        figsize=figsize,
        grid=grid,
        save_path=save_path,
        show=show,
        kwargs=kwargs,
    )

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        for label, (x_data, y_data) in data_sets.items():
            ax.plot(x_data, y_data, label=label, **config.to_matplotlib_kwargs())

        ax.set_title(config.title, fontsize=16, fontweight="bold")
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)

        if config.show_legend:
            ax.legend()
        if config.grid:
            ax.grid(True, linestyle="--", alpha=0.6)

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show:
            plt.show()
    finally:
        plt.close(fig)

    return fig, ax


def energy_landscape_1d_animation(
    data_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    time_steps_per_second: int | None = None,
    config: PlotConfig | None = None,
    *,
    fps: int = 30,
    title: str = "Evolving 1D Energy Landscape",
    xlabel: str = "Collective Variable / State",
    ylabel: str = "Energy",
    figsize: tuple[int, int] = (10, 6),
    grid: bool = False,
    repeat: bool = True,
    save_path: str | None = None,
    show: bool = True,
    show_progress_bar: bool = True,
    **kwargs: Any,
) -> animation.FuncAnimation:
    """Create an animation of an evolving 1D energy landscape.

    The docstring intentionally preserves the guidance from the previous
    implementation so existing callers can rely on the same parameter
    explanations.

    Args:
        data_sets: Dictionary whose keys are legend labels and values are
            ``(x_data, y_data)`` tuples where ``y_data`` is shaped as
            ``(time, state)``.
        time_steps_per_second: Number of simulation time steps per second of
            wall-clock time (e.g., ``1/dt``).
        config: Optional :class:`PlotConfig` with shared styling overrides.
        fps: Frames per second to render in the resulting animation.
        title: Title used when ``config`` is not provided.
        xlabel: X-axis label used when ``config`` is not provided.
        ylabel: Y-axis label used when ``config`` is not provided.
        figsize: Figure size passed to Matplotlib when building the canvas.
        grid: Whether to overlay a grid on the animation axes.
        repeat: Whether the animation should loop once it finishes.
        save_path: Optional path to persist the animation (``.gif`` / ``.mp4``).
        show: Whether to display the animation interactively.
        show_progress_bar: Whether to show a ``tqdm`` progress bar when saving.
        **kwargs: Further keyword arguments passed through to ``ax.plot``.

    Returns:
        ``matplotlib.animation.FuncAnimation``: The constructed animation.
    """

    config = _ensure_plot_config(
        config,
        PlotConfigs.energy_landscape_1d_animation,
        time_steps_per_second=time_steps_per_second,
        fps=fps,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        grid=grid,
        repeat=repeat,
        save_path=save_path,
        show=show,
        show_progress_bar=show_progress_bar,
        kwargs=kwargs,
    )

    if config.time_steps_per_second is None:
        config.time_steps_per_second = time_steps_per_second

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        if not data_sets:
            raise ValueError("The 'data_sets' dictionary cannot be empty.")

        first_key = next(iter(data_sets))
        total_sim_steps = data_sets[first_key][1].shape[0]
        if config.time_steps_per_second is None:
            raise ValueError("time_steps_per_second must be provided via argument or config.")
        total_duration_s = total_sim_steps / config.time_steps_per_second
        num_video_frames = int(total_duration_s * config.fps)
        sim_indices_to_render = np.linspace(0, total_sim_steps - 1, num_video_frames, dtype=int)

        lines: dict[str, Any] = {}
        global_ymin, global_ymax = float("inf"), float("-inf")
        for _, (_, ys_data) in data_sets.items():
            if ys_data.shape[0] != total_sim_steps:
                raise ValueError("All datasets must have the same number of time steps.")
            global_ymin = min(global_ymin, float(np.min(ys_data)))
            global_ymax = max(global_ymax, float(np.max(ys_data)))

        y_buffer = (global_ymax - global_ymin) * 0.1 if global_ymax > global_ymin else 1.0
        ax.set_ylim(global_ymin - y_buffer, global_ymax + y_buffer)

        initial_sim_index = sim_indices_to_render[0]
        for label, (x_data, ys_data) in data_sets.items():
            (line,) = ax.plot(
                x_data,
                ys_data[initial_sim_index, :],
                label=label,
                **config.to_matplotlib_kwargs(),
            )
            lines[label] = line

        ax.set_title(config.title, fontsize=16, fontweight="bold")
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        time_text = ax.text(
            0.05,
            0.9,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        def animate(frame_index: int):
            sim_index = sim_indices_to_render[frame_index]
            artists_to_update: list[Any] = []

            for label, line in lines.items():
                _, ys_data = data_sets[label]
                line.set_ydata(ys_data[sim_index, :])
                artists_to_update.append(line)

            current_time_s = sim_index / config.time_steps_per_second
            time_text.set_text(f"Time: {current_time_s:.2f} s")
            artists_to_update.append(time_text)
            return artists_to_update

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=num_video_frames,
            interval=1000 / config.fps,
            blit=True,
            repeat=config.repeat,
        )

        progress_bar_enabled = (
            getattr(config, "show_progress_bar", show_progress_bar)
            if config is not None
            else show_progress_bar
        )

        if config.save_path:

            def _save(write_animation):
                try:
                    from canns.analyzer.visualization.core import get_matplotlib_writer
                    writer = get_matplotlib_writer(config.save_path, fps=config.fps)
                    write_animation(writer)
                    print(f"Animation saved to: {config.save_path}")
                except Exception as exc:
                    print(f"Error saving animation: {exc}")

            if progress_bar_enabled:
                pbar = tqdm(total=num_video_frames, desc=f"Saving to {config.save_path}")

                def progress_callback(current_frame: int, total_frames: int) -> None:
                    pbar.update(1)

                def with_writer(writer):
                    ani.save(
                        config.save_path,
                        writer=writer,
                        progress_callback=progress_callback,
                    )

                try:
                    _save(with_writer)
                finally:
                    pbar.close()
            else:
                _save(lambda writer: ani.save(config.save_path, writer=writer))

        if config.show:
            # Automatically detect Jupyter and display as HTML/JS
            if is_jupyter_environment():
                display_animation_in_jupyter(ani)
                plt.close(fig)  # Close after HTML conversion to prevent auto-display
            else:
                plt.show()
    finally:
        if not config.show:
            # Close figure if not showing
            plt.close(fig)

    # Return None in Jupyter when showing to avoid double display
    if config.show and is_jupyter_environment():
        return None
    return ani


def energy_landscape_2d_static(
    z_data: np.ndarray,
    config: PlotConfig | None = None,
    *,
    title: str = "2D Static Landscape",
    xlabel: str = "X-Index",
    ylabel: str = "Y-Index",
    clabel: str = "Value",
    figsize: tuple[int, int] = (8, 7),
    grid: bool = False,
    save_path: str | None = None,
    show: bool = True,
    **kwargs: Any,
):
    """Plot a static 2D landscape from a 2D array as a heatmap.

    Args:
        z_data: 2D array ``(dim_y, dim_x)`` representing the landscape.
        config: Optional :class:`PlotConfig` with pre-set styling.
        title: Plot title when ``config`` is not provided.
        xlabel: X-axis label when ``config`` is not provided.
        ylabel: Y-axis label when ``config`` is not provided.
        clabel: Colorbar label when ``config`` is not provided.
        figsize: Figure size forwarded to Matplotlib when allocating the canvas.
        grid: Whether to draw a grid overlay.
        save_path: Optional path that triggers saving the figure to disk.
        show: Whether to display the figure interactively.
        **kwargs: Additional keyword arguments passed through to ``ax.imshow``.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The Matplotlib figure and axes objects.
    """

    config = _ensure_plot_config(
        config,
        PlotConfigs.energy_landscape_2d_static,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=clabel,
        figsize=figsize,
        grid=grid,
        save_path=save_path,
        show=show,
        kwargs=kwargs,
    )

    if z_data.ndim != 2:
        raise ValueError(f"Input z_data must be a 2D array, but got shape {z_data.shape}")
    if z_data.size == 0:
        raise ValueError("Input z_data must not be empty.")

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        im = ax.imshow(
            z_data,
            origin="lower",
            aspect="auto",
            **config.to_matplotlib_kwargs(),
        )

        ax.set_title(config.title, fontsize=16, fontweight="bold")
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(config.clabel)

        if config.grid:
            ax.grid(True, linestyle="--", alpha=0.6)

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show:
            plt.show()
    finally:
        plt.close(fig)

    return fig, ax


def energy_landscape_2d_animation(
    zs_data: np.ndarray,
    config: PlotConfig | None = None,
    *,
    time_steps_per_second: int | None = None,
    fps: int = 30,
    title: str = "Evolving 2D Landscape",
    xlabel: str = "X-Index",
    ylabel: str = "Y-Index",
    clabel: str = "Value",
    figsize: tuple[int, int] = (8, 7),
    grid: bool = False,
    repeat: bool = True,
    save_path: str | None = None,
    show: bool = True,
    show_progress_bar: bool = True,
    **kwargs: Any,
) -> animation.FuncAnimation:
    """Create an animation of an evolving 2D landscape.

    The long-form description mirrors the previous implementation to maintain
    backwards-compatible documentation for downstream users.

    Args:
        zs_data: Array of shape ``(timesteps, dim_y, dim_x)`` describing the
            landscape at each simulation step.
        config: Optional :class:`PlotConfig` carrying display preferences.
        time_steps_per_second: Number of simulation steps per second of
            simulated time; required unless encoded in ``config``.
        fps: Frames per second in the generated animation.
        title: Title used when ``config`` is not provided.
        xlabel: X-axis label used when ``config`` is not provided.
        ylabel: Y-axis label used when ``config`` is not provided.
        clabel: Colorbar label used when ``config`` is not provided.
        figsize: Figure size passed to Matplotlib.
        grid: Whether to overlay a grid on the heatmap.
        repeat: Whether the animation should loop.
        save_path: Optional output path (``.gif`` / ``.mp4``).
        show: Whether to display the animation interactively.
        show_progress_bar: Whether to render a ``tqdm`` progress bar during save.
        **kwargs: Additional keyword arguments forwarded to ``ax.imshow``.

    Returns:
        ``matplotlib.animation.FuncAnimation``: The constructed animation.
    """

    config = _ensure_plot_config(
        config,
        PlotConfigs.energy_landscape_2d_animation,
        time_steps_per_second=time_steps_per_second,
        fps=fps,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=clabel,
        figsize=figsize,
        grid=grid,
        repeat=repeat,
        save_path=save_path,
        show=show,
        show_progress_bar=show_progress_bar,
        kwargs=kwargs,
    )

    if config.time_steps_per_second is None:
        config.time_steps_per_second = time_steps_per_second

    if config.time_steps_per_second is None:
        raise ValueError("time_steps_per_second is required")

    if zs_data.ndim != 3:
        raise ValueError("Input zs_data must be a 3D array with shape (timesteps, dim_y, dim_x).")
    if zs_data.size == 0:
        raise ValueError("Input zs_data must not be empty.")

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        total_sim_steps = zs_data.shape[0]
        total_duration_s = total_sim_steps / config.time_steps_per_second
        num_video_frames = int(total_duration_s * config.fps)
        sim_indices_to_render = np.linspace(0, total_sim_steps - 1, num_video_frames, dtype=int)

        vmin = float(np.min(zs_data))
        vmax = float(np.max(zs_data))

        initial_sim_index = sim_indices_to_render[0]
        initial_z_data = zs_data[initial_sim_index, :, :]

        im = ax.imshow(
            initial_z_data,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            **config.to_matplotlib_kwargs(),
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(config.clabel, fontsize=12)

        ax.set_title(config.title, fontsize=16, fontweight="bold")
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.4, color="white")

        time_text = ax.text(
            0.05,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            color="white",
            bbox=dict(facecolor="black", alpha=0.5),
            verticalalignment="top",
        )

        def animate(frame_index: int):
            sim_index = sim_indices_to_render[frame_index]
            im.set_data(zs_data[sim_index, :, :])
            current_time_s = sim_index / config.time_steps_per_second
            time_text.set_text(f"Time: {current_time_s:.2f} s")
            return [im, time_text]

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=num_video_frames,
            interval=1000 / config.fps,
            blit=True,
            repeat=config.repeat,
        )

        progress_bar_enabled = (
            getattr(config, "show_progress_bar", show_progress_bar)
            if config is not None
            else show_progress_bar
        )

        if config.save_path:

            def _save(write_animation):
                try:
                    from canns.analyzer.visualization.core import get_matplotlib_writer
                    writer = get_matplotlib_writer(config.save_path, fps=config.fps)
                    write_animation(writer)
                    print(f"Animation saved to: {config.save_path}")
                except Exception as exc:
                    print(f"Error saving animation: {exc}")

            if progress_bar_enabled:
                pbar = tqdm(total=num_video_frames, desc=f"Saving to {config.save_path}")

                def progress_callback(current_frame: int, total_frames: int) -> None:
                    pbar.update(1)

                def with_writer(writer):
                    ani.save(
                        config.save_path,
                        writer=writer,
                        progress_callback=progress_callback,
                    )

                try:
                    _save(with_writer)
                finally:
                    pbar.close()
            else:
                _save(lambda writer: ani.save(config.save_path, writer=writer))

        if config.show:
            # Automatically detect Jupyter and display as HTML/JS
            if is_jupyter_environment():
                display_animation_in_jupyter(ani)
                plt.close(fig)  # Close after HTML conversion to prevent auto-display
            else:
                plt.show()
    finally:
        if not config.show:
            # Close figure if not showing
            plt.close(fig)

    # Return None in Jupyter when showing to avoid double display
    if config.show and is_jupyter_environment():
        return None
    return ani
