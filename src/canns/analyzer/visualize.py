import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
from tqdm import tqdm


# ==================== Unified Plot Configuration ====================
@dataclass
class PlotConfig:
    """Unified configuration class for all plotting functions in canns.analyzer module.

    This class standardizes parameters across static and dynamic plotting functions,
    providing a consistent interface while maintaining backward compatibility.
    """

    # Basic plot configuration
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: tuple[int, int] = (10, 6)
    grid: bool = False
    save_path: str | None = None
    show: bool = True

    # Animation-specific configuration
    time_steps_per_second: int | None = None
    fps: int = 30
    repeat: bool = True
    show_progress_bar: bool = True

    # Specialized plot configuration
    show_legend: bool = True
    color: str = "black"
    clabel: str = "Value"  # Color bar label for 2D plots

    # Additional matplotlib parameters
    kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

    @classmethod
    def for_static_plot(cls, **kwargs) -> "PlotConfig":
        """Create configuration optimized for static plots."""
        config = cls(**kwargs)
        # Static plots don't need animation parameters
        config.time_steps_per_second = None
        return config

    @classmethod
    def for_animation(cls, time_steps_per_second: int, **kwargs) -> "PlotConfig":
        """Create configuration optimized for animations."""
        config = cls(time_steps_per_second=time_steps_per_second, **kwargs)
        return config

    def to_matplotlib_kwargs(self) -> dict[str, Any]:
        """Extract matplotlib-compatible keyword arguments."""
        return self.kwargs.copy() if self.kwargs else {}


# ==================== Pre-configured Plot Configs ====================
class PlotConfigs:
    """Collection of commonly used plot configurations."""

    @staticmethod
    def energy_landscape_1d_static(**kwargs) -> PlotConfig:
        """Configuration for 1D energy landscape static plots."""
        defaults = {
            "title": "1D Energy Landscape",
            "xlabel": "Collective Variable / State",
            "ylabel": "Energy",
            "figsize": (10, 6),
        }
        defaults.update(kwargs)
        return PlotConfig.for_static_plot(**defaults)

    @staticmethod
    def energy_landscape_1d_animation(**kwargs) -> PlotConfig:
        """Configuration for 1D energy landscape animations."""
        defaults = {
            "title": "Evolving 1D Energy Landscape",
            "xlabel": "Collective Variable / State",
            "ylabel": "Energy",
            "figsize": (10, 6),
            "fps": 30,
        }
        time_steps = kwargs.pop(
            "time_steps_per_second", 1000
        )  # Remove from kwargs to avoid duplication
        defaults.update(kwargs)
        return PlotConfig.for_animation(time_steps, **defaults)

    @staticmethod
    def energy_landscape_2d_static(**kwargs) -> PlotConfig:
        """Configuration for 2D energy landscape static plots."""
        defaults = {
            "title": "2D Static Landscape",
            "xlabel": "X-Index",
            "ylabel": "Y-Index",
            "clabel": "Value",
            "figsize": (8, 7),
        }
        defaults.update(kwargs)
        return PlotConfig.for_static_plot(**defaults)

    @staticmethod
    def energy_landscape_2d_animation(**kwargs) -> PlotConfig:
        """Configuration for 2D energy landscape animations."""
        defaults = {
            "title": "Evolving 2D Landscape",
            "xlabel": "X-Index",
            "ylabel": "Y-Index",
            "clabel": "Value",
            "figsize": (8, 7),
            "fps": 30,
        }
        time_steps = kwargs.pop(
            "time_steps_per_second", 1000
        )  # Remove from kwargs to avoid duplication
        defaults.update(kwargs)
        return PlotConfig.for_animation(time_steps, **defaults)

    @staticmethod
    def raster_plot(mode: str = "block", **kwargs) -> PlotConfig:
        """Configuration for raster plots.

        Args:
            mode: Plot mode ('scatter' or 'block')
            **kwargs: Additional parameters
        """
        defaults = {
            "title": "Raster Plot",
            "xlabel": "Time Step",
            "ylabel": "Neuron Index",
            "figsize": (12, 6),
            "color": "black",
        }
        defaults.update(kwargs)
        config = PlotConfig.for_static_plot(**defaults)

        config.mode = mode
        return config

    @staticmethod
    def average_firing_rate_plot(mode: str = "per_neuron", **kwargs) -> PlotConfig:
        """Configuration for average firing rate plots."""
        defaults = {
            "title": "Average Firing Rate",
            "figsize": (12, 5),
        }
        defaults.update(kwargs)
        config = PlotConfig.for_static_plot(**defaults)
        config.mode = mode
        return config

    @staticmethod
    def tuning_curve(
        num_bins: int = 50, pref_stim: np.ndarray | None = None, **kwargs
    ) -> PlotConfig:
        """Configuration for tuning curve plots."""
        defaults = {
            "title": "Tuning Curve",
            "xlabel": "Stimulus Value",
            "ylabel": "Average Firing Rate",
            "figsize": (10, 6),
        }
        defaults.update(kwargs)

        config = PlotConfig.for_static_plot(**defaults)
        config.num_bins = num_bins
        config.pref_stim = pref_stim
        return config


# --- CANN Model related visualization method ---


def energy_landscape_1d_static(
    data_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    config: PlotConfig | None = None,
    # Backward compatibility parameters
    title: str = "1D Energy Landscape",
    xlabel: str = "Collective Variable / State",
    ylabel: str = "Energy",
    show_legend: bool = True,
    figsize: tuple[int, int] = (10, 6),
    grid: bool = False,
    save_path: str | None = None,
    show=True,
    **kwargs,
):
    """
    Plots a 1D static energy landscape using Matplotlib.

    This function takes a dictionary where keys are used as legend labels and
    values are the energy curve data, plotting all curves on the same figure.

    Args:
        data_sets (Dict[str, Tuple[np.ndarray, np.ndarray]]):
            A dictionary where keys (str) are the labels for the legend and
            values (Tuple) are the (x_data, y_data) pairs.
        config (Optional[PlotConfig]): Configuration object for plot parameters.
            If None, will create from backward compatibility parameters.
        title (str, optional): The title of the plot. Defaults to "1D Energy Landscape".
        xlabel (str, optional): The label for the X-axis. Defaults to "Collective Variable / State".
        ylabel (str, optional): The label for the Y-axis. Defaults to "Energy".
        show_legend (bool, optional): Whether to display the legend. Defaults to True.
        figsize (Tuple[int, int], optional): The size of the figure, as a tuple (width, height). Defaults to (10, 6).
        grid (bool, optional): Whether to display a grid. Defaults to False.
        save_path (Optional[str], optional):
            The file path to save the plot. If provided, the plot will be saved to a file.
            Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
        **kwargs:
            Any other keyword arguments supported by matplotlib.pyplot.plot,
            e.g., linewidth, linestyle, marker, color. These will be applied to all curves.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Returns the Matplotlib Figure and Axes objects
            for further modification outside the function.
    """
    # Handle configuration - use config if provided, otherwise create from parameters
    if config is None:
        config = PlotConfigs.energy_landscape_1d_static(
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

    # --- Create the figure and axes ---
    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        # --- Loop through and plot each energy curve ---
        # Use .items() to iterate over both keys (labels) and values (data) of the dictionary
        for label, (x_data, y_data) in data_sets.items():
            # Plot the curve, using the dictionary key directly as the label
            ax.plot(x_data, y_data, label=label, **config.to_matplotlib_kwargs())

        # --- Configure the plot's appearance ---
        ax.set_title(config.title, fontsize=16, fontweight="bold")
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)

        # If requested, display the legend
        if config.show_legend:
            ax.legend()

        # Set the grid
        if config.grid:
            ax.grid(True, linestyle="--", alpha=0.6)

        # --- Save and display the plot ---
        if config.save_path:
            # Using bbox_inches='tight' prevents labels from being cut off
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show:
            plt.show()

    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def energy_landscape_1d_animation(
    data_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    time_steps_per_second: int = None,
    config: PlotConfig | None = None,
    # Backward compatibility parameters
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
    **kwargs,
):
    """
    Creates an animation of an evolving 1D energy landscape with intuitive timing controls.

    Args:
        data_sets (Dict[str, Tuple[np.ndarray, np.ndarray]]):
            A dictionary of the evolving landscapes.
            - Keys (str) are the labels for the legend.
            - Values are tuples (x_data, ys_data), where ys_data is a 2D array
            of shape (total_sim_steps, num_states).
        time_steps_per_second (int):
            The number of data points (rows in ys_data) that correspond to one
            second of simulation time. (e.g., if dt=0.001s, this is 1000).
        config (Optional[PlotConfig]): Configuration object for plot parameters.
        fps (int, optional):
            Frames per second for the output animation. Defaults to 30.
        title (str, optional): The title of the plot. Defaults to "Evolving 1D Energy Landscape".
        xlabel (str, optional): The label for the X-axis. Defaults to "Collective Variable / State".
        ylabel (str, optional): The label for the Y-axis. Defaults to "Energy".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
        grid (bool, optional): Whether to display a grid. Defaults to False.
        repeat (bool, optional): Whether the animation should repeat. Defaults to True.
        save_path (Optional[str], optional):
            File path to save the animation (e.g., 'animation.gif' or 'animation.mp4').
            Defaults to None. NOTE: Requires a writer like 'Pillow' or 'ffmpeg'.
        show (bool, optional): Whether to show the animation. Defaults to True.
        show_progress_bar (bool, optional):
            Whether to show a progress bar while saving the animation. Defaults to True.
        **kwargs:
            Any other keyword arguments for matplotlib.pyplot.plot (e.g., linewidth).

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    # Handle configuration - use config if provided, otherwise create from parameters
    if config is None:
        config = PlotConfigs.energy_landscape_1d_animation(
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
    else:
        # Ensure config has time_steps_per_second set
        if config.time_steps_per_second is None:
            config.time_steps_per_second = time_steps_per_second
    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        # --- Input Validation and Timing Calculation ---
        if not data_sets:
            raise ValueError("The 'data_sets' dictionary cannot be empty.")

        first_key = list(data_sets.keys())[0]
        total_sim_steps = data_sets[first_key][1].shape[0]

        # Calculate total simulation duration in seconds from the data itself
        total_duration_s = total_sim_steps / config.time_steps_per_second

        # Calculate the total number of frames needed for the output video
        num_video_frames = int(total_duration_s * config.fps)

        # Create an array of the simulation step indices that we will actually render
        # This correctly handles up-sampling or down-sampling the data to match the desired fps
        sim_indices_to_render = np.linspace(0, total_sim_steps - 1, num_video_frames, dtype=int)

        lines = {}

        # Set stable axis limits to prevent jumping
        global_ymin, global_ymax = float("inf"), float("-inf")
        for _, (_, ys_data) in data_sets.items():
            if ys_data.shape[0] != total_sim_steps:
                raise ValueError("All datasets must have the same number of time steps.")
            global_ymin = min(global_ymin, np.min(ys_data))
            global_ymax = max(global_ymax, np.max(ys_data))

        y_buffer = (global_ymax - global_ymin) * 0.1 if global_ymax > global_ymin else 1.0
        ax.set_ylim(global_ymin - y_buffer, global_ymax + y_buffer)

        # --- Plot the Initial Frame ---
        initial_sim_index = sim_indices_to_render[0]
        for label, (x_data, ys_data) in data_sets.items():
            (line,) = ax.plot(
                x_data, ys_data[initial_sim_index, :], label=label, **config.to_matplotlib_kwargs()
            )
            lines[label] = line

        # Configure plot appearance
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

        # --- Define the Animation Update Function ---
        def animate(frame_index):
            """This function is called for each frame of the video."""
            sim_index = sim_indices_to_render[frame_index]

            artists_to_update = []
            for label, line in lines.items():
                _, ys_data = data_sets[label]
                line.set_ydata(ys_data[sim_index, :])
                artists_to_update.append(line)

            # Update time text to show actual simulation time
            current_time_s = sim_index / config.time_steps_per_second
            time_text.set_text(f"Time: {current_time_s:.2f} s")
            artists_to_update.append(time_text)

            return artists_to_update

        # --- Create and Return the Animation ---
        interval_ms = 1000 / fps
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=num_video_frames,
            interval=interval_ms,
            blit=True,
            repeat=config.repeat,
        )

        # --- Save or Show the Animation ---
        if config.save_path:
            if show_progress_bar:
                # Setup the progress bar
                pbar = tqdm(
                    total=num_video_frames,
                    desc=f"<{sys._getframe().f_code.co_name}> Saving to {config.save_path}",
                )

                # Define the callback function that updates the progress bar
                def progress_callback(current_frame, total_frames):
                    pbar.update(1)

                # Save the animation with the callback
                try:
                    writer = animation.PillowWriter(fps=fps)
                    ani.save(config.save_path, writer=writer, progress_callback=progress_callback)
                    pbar.close()  # Close the progress bar upon completion
                    print(f"\nAnimation successfully saved to: {config.save_path}")
                except Exception as e:
                    pbar.close()
                    print(f"\nError saving animation: {e}")
            else:
                # Save without a progress bar
                try:
                    writer = animation.PillowWriter(fps=fps)
                    ani.save(config.save_path, writer=writer)
                    print(f"Animation saved to: {config.save_path}")
                except Exception as e:
                    print(f"Error saving animation: {e}")
        if config.show:
            plt.show()
    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def energy_landscape_2d_static(
    z_data: np.ndarray,
    config: PlotConfig | None = None,
    title: str = "2D Static Landscape",
    xlabel: str = "X-Index",
    ylabel: str = "Y-Index",
    clabel: str = "Value",
    figsize: tuple[int, int] = (8, 7),
    grid: bool = False,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
):
    """
    Plots a static 2D landscape from a 2D array as a heatmap.

    Args:
        z_data (np.ndarray): A 2D array of shape (dim_y, dim_x) representing the values on the grid.
        config (PlotConfig, optional): Configuration object for unified plotting parameters.
        title (str, optional): The title of the plot.
        xlabel (str, optional): The label for the X-axis.
        ylabel (str, optional): The label for the Y-axis.
        clabel (str, optional): The label for the color bar.
        figsize (Tuple[int, int], optional): The size of the figure.
        grid (bool, optional): Whether to display a grid.
        save_path (Optional[str], optional): The file path to save the plot.
        show (bool, optional): Whether to show the plot.
        **kwargs: Any other keyword arguments for matplotlib.pyplot.imshow (e.g., cmap, vmin, vmax).

    Returns:
        Tuple[plt.Figure, plt.Axes]: The Matplotlib Figure and Axes objects.
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = PlotConfigs.energy_landscape_2d_static(
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
    assert z_data.size > 0, "Input z_data must not be empty."

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        # Use imshow for efficient 2D plotting. origin='lower' puts (0,0) at the bottom-left.
        im = ax.imshow(z_data, origin="lower", aspect="auto", **config.to_matplotlib_kwargs())

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(config.clabel, fontsize=12)

        ax.set_title(config.title, fontsize=16, fontweight="bold")
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)

        if config.grid:
            ax.grid(True, linestyle="--", alpha=0.4, color="white")

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show:
            plt.show()

    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def energy_landscape_2d_animation(
    zs_data: np.ndarray,
    time_steps_per_second: int | None = None,
    config: PlotConfig | None = None,
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
    **kwargs,
):
    """
    Creates an animation of an evolving 2D landscape from a 3D data cube.

    Args:
        zs_data (np.ndarray): A 3D array of shape (time_steps, dim_y, dim_x).
        time_steps_per_second (int, optional): Number of data points (frames) per second of simulation time.
        config (PlotConfig, optional): Configuration object for unified plotting parameters.
        fps (int, optional): Frames per second for the output animation.
        title (str, optional): The title of the plot.
        xlabel (str, optional): The label for the X-axis.
        ylabel (str, optional): The label for the Y-axis.
        clabel (str, optional): The label for the color bar.
        figsize (Tuple[int, int], optional): The size of the figure.
        grid (bool, optional): Whether to display a grid.
        save_path (Optional[str], optional): The file path to save the plot.
        show (bool, optional): Whether to show the plot.
        repeat (bool, optional): Whether the animation should repeat. Defaults to True.
        show_progress_bar (bool, optional): Whether to show a progress bar while saving the animation.
        **kwargs: Any other keyword arguments for matplotlib.pyplot.imshow (e.g., cmap).

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = PlotConfigs.energy_landscape_2d_animation(
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
        raise ValueError("time_steps_per_second is required")

    fig, ax = plt.subplots(figsize=figsize)

    if zs_data.ndim != 3:
        raise ValueError(f"Input zs_data must be a 3D array, but got shape {zs_data.shape}")

    try:
        # --- Timing Calculation ---
        total_sim_steps = zs_data.shape[0]
        total_duration_s = total_sim_steps / config.time_steps_per_second
        num_video_frames = int(total_duration_s * config.fps)
        sim_indices_to_render = np.linspace(0, total_sim_steps - 1, num_video_frames, dtype=int)

        # Set stable color limits by finding global min/max across all time
        vmin = np.min(zs_data)
        vmax = np.max(zs_data)

        # --- Plot the Initial Frame ---
        initial_sim_index = sim_indices_to_render[0]
        initial_z_data = zs_data[initial_sim_index, :, :]

        im = ax.imshow(
            initial_z_data,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,  # Use stable color limits
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

        # --- Define the Animation Update Function ---
        def animate(frame_index):
            sim_index = sim_indices_to_render[frame_index]
            im.set_data(zs_data[sim_index, :, :])
            current_time_s = sim_index / config.time_steps_per_second
            time_text.set_text(f"Time: {current_time_s:.2f} s")
            return im, time_text

        # --- Create and Return the Animation ---
        interval_ms = 1000 / config.fps
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=num_video_frames,
            interval=interval_ms,
            blit=True,
            repeat=config.repeat,
        )

        # --- Save or Show the Animation ---
        if config.save_path:
            if show_progress_bar:
                pbar = tqdm(total=num_video_frames, desc=f"Saving to {config.save_path}")

                def progress_callback(current_frame, total_frames):
                    pbar.update(1)

                try:
                    writer = animation.PillowWriter(fps=config.fps)
                    ani.save(config.save_path, writer=writer, progress_callback=progress_callback)
                    pbar.close()
                    print(f"\nAnimation successfully saved to: {config.save_path}")
                except Exception as e:
                    pbar.close()
                    print(f"\nError saving animation: {e}")
            else:
                try:
                    writer = animation.PillowWriter(fps=config.fps)
                    ani.save(config.save_path, writer=writer)
                    print(f"Animation saved to: {config.save_path}")
                except Exception as e:
                    print(f"Error saving animation: {e}")

        if config.show:
            plt.show()

    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def raster_plot(
    spike_train: np.ndarray,
    config: PlotConfig | None = None,
    mode: str = "block",
    title: str = "Raster Plot",
    xlabel: str = "Time Step",
    ylabel: str = "Neuron Index",
    figsize: tuple[int, int] = (12, 6),
    color: str = "black",
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
):
    """
    Generates a raster plot from a spike train matrix.

    This function can generate two styles of plots:
    - 'scatter': A traditional raster plot with markers for each spike. Best for a large number of neurons.
    - 'block': A heatmap-style plot where each spike is a colored block. Best for a small number of neurons.

    Args:
        spike_train (np.ndarray):
            A 2D boolean/integer array of shape (timesteps, num_neurons).
        config (Optional[PlotConfig]): Configuration object for unified plotting parameters.
        mode (str, optional):
            The plotting mode, either 'scatter' or 'block'. Defaults to 'scatter'.
        title (str, optional): The title of the plot.
        xlabel (str, optional): The label for the X-axis.
        ylabel (str, optional): The label for the Y-axis.
        figsize (Tuple[int, int], optional): The size of the figure.
        color (str, optional):
            The color for the spikes. For 'scatter' mode, this is the marker color.
            For 'block' mode, this is the 'on' color in the colormap.
        save_path (Optional[str], optional): The file path to save the plot.
        show (bool, optional): Whether to show the plot.
        **kwargs:
            Additional keyword arguments passed to the plotting function.
            For 'scatter' mode, passed to `ax.scatter()` (e.g., `marker_size`).
            For 'block' mode, passed to `ax.imshow()` (e.g., `cmap`).
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = PlotConfigs.raster_plot(
            mode=mode,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            color=color,
            save_path=save_path,
            show=show,
            kwargs=kwargs,
        )
    else:
        # If config is provided but doesn't have mode, add it
        if not hasattr(config, "mode"):
            config.mode = mode

    if spike_train.ndim != 2:
        raise ValueError(f"Input spike_train must be a 2D array, but got shape {spike_train.shape}")
    assert spike_train.size > 0, "Input spike_train must not be empty."
    assert config.mode in ("block", "scatter"), (
        f"Invalid mode '{config.mode}'. Choose 'scatter' or 'block'."
    )

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        ax.set_title(config.title, fontsize=16, fontweight="bold")
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)

        if config.mode == "scatter":
            # --- Traditional Scatter Plot Mode ---
            time_indices, neuron_indices = np.where(spike_train)

            # Set default marker size if not provided in kwargs
            marker_size = config.kwargs.pop("marker_size", 1.0)

            ax.scatter(
                time_indices,
                neuron_indices,
                s=marker_size,
                c=config.color,
                marker="|",
                alpha=0.8,
                **config.to_matplotlib_kwargs(),
            )
            ax.set_xlim(0, spike_train.shape[0])
            ax.set_ylim(-1, spike_train.shape[1])

        elif config.mode == "block":
            # --- Block / Image Mode ---
            # imshow expects data oriented as (row, column), which corresponds to (neuron, time).
            # So we need to transpose the spike_train.
            data_to_show = spike_train.T

            # Create a custom colormap: 0 -> transparent, 1 -> specified color
            from matplotlib.colors import ListedColormap

            cmap = config.kwargs.pop("cmap", ListedColormap(["white", config.color]))

            # Use imshow to create the block plot.
            # `interpolation='none'` ensures sharp, non-blurry blocks.
            # `aspect='auto'` allows the blocks to be non-square to fill the space.
            ax.imshow(
                data_to_show,
                aspect="auto",
                interpolation="none",
                cmap=cmap,
                **config.to_matplotlib_kwargs(),
            )

            # Set the ticks to be at the center of the neurons
            ax.set_yticks(np.arange(spike_train.shape[1]))
            ax.set_yticklabels(np.arange(spike_train.shape[1]))
            # Optional: reduce the number of y-ticks if there are too many neurons
            if spike_train.shape[1] > 20:
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show:
            plt.show()

    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def average_firing_rate_plot(
    spike_train: np.ndarray,
    dt: float,
    config: PlotConfig | None = None,
    mode: str = "population",
    weights: np.ndarray | None = None,
    title: str = "Average Firing Rate",
    figsize: tuple[int, int] = (12, 5),
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
):
    """
    Calculates and plots different types of average neural activity from a spike train.

    Args:
        spike_train (np.ndarray):
            A 2D boolean/integer array of shape (timesteps, num_neurons).
        dt (float):
            Time step of the simulation in seconds.
        config (Optional[PlotConfig], optional):
            Configuration object for unified plotting parameters.
        mode (str, optional):
            The plotting mode. Can be:
            - 'per_neuron': Average rate for each neuron over the entire duration. (X-axis: Neuron Index)
            - 'population': Average rate of all neurons at each time step. (X-axis: Time)
            - 'weighted_average': Weighted average of neural activity over time. Requires 'weights' argument. (X-axis: Time)
            Defaults to 'population'.
        weights (Optional[np.ndarray], optional):
            A 1D array of shape (num_neurons,) required for 'weighted_average' mode.
            Represents the preferred value (e.g., angle, position) for each neuron.
        title (str, optional): The title of the plot.
        figsize (Tuple[int, int], optional): The size of the figure.
        save_path (Optional[str], optional): The file path to save the plot.
        show (bool, optional): Whether to show the plot.
        **kwargs:
            Additional keyword arguments for the plot, such as line style, color, etc.

    Returns:
        Tuple[np.ndarray, Tuple[plt.Figure, plt.Axes]]:
            A tuple containing the calculated data and the plot objects.
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = PlotConfigs.average_firing_rate_plot(
            mode=mode, title=title, figsize=figsize, save_path=save_path, show=show, kwargs=kwargs
        )
    else:
        # If config is provided but doesn't have mode, add it
        if not hasattr(config, "mode"):
            config.mode = mode

    if spike_train.ndim != 2:
        raise ValueError("Input spike_train must be a 2D array.")

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        num_timesteps, num_neurons = spike_train.shape
        ax.set_title(config.title, fontsize=16, fontweight="bold")

        if config.mode == "per_neuron":
            # --- Average rate for each neuron over time ---
            duration_s = num_timesteps * dt
            total_spikes_per_neuron = np.sum(spike_train, axis=0)
            # Rate = total spikes / total duration
            calculated_data = total_spikes_per_neuron / duration_s

            ax.plot(np.arange(num_neurons), calculated_data, **config.to_matplotlib_kwargs())
            ax.set_xlabel("Neuron Index", fontsize=12)
            ax.set_ylabel("Average Firing Rate (Hz)", fontsize=12)
            ax.set_xlim(0, num_neurons - 1)

        elif config.mode == "population":
            # --- Average rate of the whole population over time ---
            spikes_per_timestep = np.sum(spike_train, axis=1)
            # Population Rate = (total spikes in bin) / (num_neurons * bin_duration)
            # This definition is debated, another is just total spikes / bin_duration.
            # We will use the simpler total spikes / bin_duration, which is the summed rate.
            calculated_data = spikes_per_timestep / dt

            time_vector = np.arange(num_timesteps) * dt
            ax.plot(time_vector, calculated_data, **config.to_matplotlib_kwargs())
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Total Population Rate (Hz)", fontsize=12)
            ax.set_xlim(0, time_vector[-1])

        elif config.mode == "weighted_average":
            # --- Weighted average of activity over time (decoding) ---
            if weights is None:
                raise ValueError("'weights' argument is required for 'weighted_average' mode.")
            if weights.shape != (num_neurons,):
                raise ValueError(
                    f"Shape of 'weights' {weights.shape} must match num_neurons ({num_neurons})."
                )

            # Calculate the sum of spikes at each time step
            total_spikes_per_timestep = np.sum(spike_train, axis=1)

            # Calculate the weighted sum of spikes at each time step
            # spike_train (T, N) * weights (N,) -> broadcasting -> (T, N) -> sum(axis=1) -> (T,)
            weighted_sum_of_spikes = np.sum(spike_train * weights, axis=1)

            calculated_data = weighted_sum_of_spikes / (total_spikes_per_timestep + 1e-9)

            # Handle time steps with no spikes: set them to NaN (Not a Number) so they don't get plotted
            calculated_data[total_spikes_per_timestep == 0] = np.nan

            time_vector = np.arange(num_timesteps) * dt
            ax.plot(time_vector, calculated_data, **config.to_matplotlib_kwargs())
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Decoded Value (Weighted Average)", fontsize=12)
            ax.set_xlim(0, time_vector[-1])

        else:
            raise ValueError(
                f"Invalid mode '{config.mode}'. Choose 'per_neuron', 'population', or 'weighted_average'."
            )

        ax.grid(True, linestyle="--", alpha=0.6)

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show:
            plt.show()

    finally:
        plt.close(fig)


def tuning_curve(
    stimulus: np.ndarray,
    firing_rates: np.ndarray,
    neuron_indices: np.ndarray | int,
    config: PlotConfig | None = None,
    pref_stim: np.ndarray | None = None,
    num_bins: int = 50,
    title: str = "Tuning Curve",
    xlabel: str = "Stimulus Value",
    ylabel: str = "Average Firing Rate",
    figsize: tuple[int, int] = (10, 6),
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
):
    """
    Computes and plots the tuning curve for one or more neurons.

    A tuning curve shows how the average firing rate of a neuron changes as a
    function of an external stimulus.

    Args:
        stimulus (np.ndarray): A 1D array representing the stimulus value at each
                               time step. Shape: (num_time_steps,).
        firing_rates (np.ndarray): A 2D array of firing rates for all neurons at
                                   each time step.
                                   Shape: (num_time_steps, num_neurons).
        neuron_indices (np.ndarray | int): The index or a list/array of indices
                                           of the neuron(s) to plot.
        config (PlotConfig, optional): Configuration object for unified plotting parameters.
        pref_stim (np.ndarray | None, optional): A 1D array containing the preferred
                                                 stimulus for each neuron. If provided,
                                                 it's used for the legend labels.
                                                 Shape: (num_neurons,). Defaults to None.
        num_bins (int, optional): The number of bins to use for grouping the
                                  stimulus space. Defaults to 50.
        title (str, optional): The title of the plot. Defaults to "Tuning Curve".
        xlabel (str, optional): The label for the x-axis. Defaults to "Stimulus Value".
        ylabel (str, optional): The label for the y-axis. Defaults to "Average Firing Rate".
        figsize (tuple[int, int], optional): The figure size if a new figure is
                                             created. Defaults to (10, 6).
        save_path (str | None, optional): The file path to save the figure.
                                          If None, the figure is not saved.
                                          Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to `ax.plot`
                  (e.g., linewidth, marker, color).
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = PlotConfigs.tuning_curve(
            pref_stim=pref_stim,
            num_bins=num_bins,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            save_path=save_path,
            show=show,
            kwargs=kwargs,
        )
    else:
        # If config is provided but doesn't have mode, add it
        if not hasattr(config, "num_bins"):
            config.num_bins = num_bins
        if not hasattr(config, "pref_stim"):
            config.pref_stim = pref_stim

    # --- 1. Input Validation and Preparation ---
    if stimulus.ndim != 1:
        raise ValueError(f"stimulus must be a 1D array, but has {stimulus.ndim} dimensions.")
    if firing_rates.ndim != 2:
        raise ValueError(
            f"firing_rates must be a 2D array, but has {firing_rates.ndim} dimensions."
        )
    if stimulus.shape[0] != firing_rates.shape[0]:
        raise ValueError(
            f"The first dimension (time steps) of stimulus and firing_rates must match: "
            f"{stimulus.shape[0]} != {firing_rates.shape[0]}"
        )

    # Ensure neuron_indices is a list for consistent processing
    if isinstance(neuron_indices, int):
        neuron_indices = [neuron_indices]
    elif not isinstance(neuron_indices, Iterable):
        raise TypeError(
            "neuron_indices must be an integer or an iterable (e.g., list, np.ndarray)."
        )

    # --- Setup Plotting Environment ---
    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        # --- Computation and Plotting Loop ---
        for neuron_idx in neuron_indices:
            # Get the time series of firing rates for the current neuron
            neuron_fr = firing_rates[:, neuron_idx]

            # Use binned_statistic for efficient binning and averaging.
            # 'statistic'='mean' calculates the average of values in each bin.
            # 'bins'=num_bins divides the stimulus range into num_bins equal intervals.
            mean_rates, bin_edges, _ = binned_statistic(
                x=stimulus, values=neuron_fr, statistic="mean", bins=config.num_bins
            )

            # Calculate the center of each bin for plotting on the x-axis
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Create a label for the legend
            label = f"Neuron {neuron_idx}"
            if config.pref_stim is not None and neuron_idx < len(config.pref_stim):
                label += f" (pref_stim={config.pref_stim[neuron_idx]:.2f})"

            # Plot the curve. Bins that were empty will have a `nan` mean,
            # which matplotlib handles gracefully (it won't plot them).
            ax.plot(bin_centers, mean_rates, label=label, **config.to_matplotlib_kwargs())

        # --- Final Touches and Output ---
        ax.set_title(config.title, fontsize=16)
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()

        if config.save_path:
            plt.savefig(config.save_path, dpi=300)
            print(f"Tuning curve saved to {config.save_path}")

        if config.show:
            plt.show()
    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


# TODO: Implement phase_plane_plot (NEED DISCUSSION)
def phase_plane_plot(): ...


# ==================== Place Cell Visualization ====================


@dataclass
class PlaceCellConfig(PlotConfig):
    """Configuration for Place Cell visualizations."""

    colormap: str = "viridis"
    show_trajectory: bool = True
    trajectory_alpha: float = 0.6
    field_contour_levels: int = 5
    center_marker_size: int = 50
    center_marker_color: str = "red"


class PlaceCellConfigs:
    """Pre-configured settings for Place Cell visualizations."""

    @staticmethod
    def place_field_heatmap(**kwargs) -> PlaceCellConfig:
        """Configuration for place field heatmap plots."""
        defaults = {
            "title": "Place Field Heatmap",
            "xlabel": "X Position",
            "ylabel": "Y Position",
            "clabel": "Firing Rate (Hz)",
            "figsize": (8, 7),
            "colormap": "hot",
        }
        defaults.update(kwargs)
        return PlaceCellConfig.for_static_plot(**defaults)

    @staticmethod
    def place_field_trajectory(**kwargs) -> PlaceCellConfig:
        """Configuration for trajectory with place field overlay."""
        defaults = {
            "title": "Animal Trajectory with Place Field",
            "xlabel": "X Position",
            "ylabel": "Y Position",
            "figsize": (10, 8),
            "show_trajectory": True,
            "trajectory_alpha": 0.7,
        }
        defaults.update(kwargs)
        return PlaceCellConfig.for_static_plot(**defaults)


def plot_place_field_heatmap(
    heatmap: np.ndarray,
    width: float = 1.0,
    height: float = 1.0,
    positions: np.ndarray | None = None,
    config: PlaceCellConfig | None = None,
    **kwargs,
):
    """
    Plot place field heatmap with optional trajectory overlay.

    Args:
        heatmap (np.ndarray): 2D firing rate heatmap
        width (float): Environment width
        height (float): Environment height
        positions (np.ndarray, optional): Animal positions for trajectory
        config (PlaceCellConfig, optional): Plot configuration
        **kwargs: Additional matplotlib parameters
    """
    if config is None:
        config = PlaceCellConfigs.place_field_heatmap(**kwargs)

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        # Create coordinate arrays
        x = np.linspace(0, width, heatmap.shape[1])
        y = np.linspace(0, height, heatmap.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot heatmap
        im = ax.contourf(X, Y, heatmap, levels=50, cmap=config.colormap, alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(config.clabel)

        # Overlay trajectory if provided
        if positions is not None and config.show_trajectory:
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                "k-",
                alpha=config.trajectory_alpha,
                linewidth=1,
                label="Trajectory",
            )
            if config.show_legend:
                ax.legend()

        # Add contour lines
        if hasattr(config, "field_contour_levels") and config.field_contour_levels > 0:
            ax.contour(
                X,
                Y,
                heatmap,
                levels=config.field_contour_levels,
                colors="white",
                alpha=0.6,
                linewidths=0.8,
            )

        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)
        ax.set_aspect("equal")

        if config.grid:
            ax.grid(True, alpha=0.3)

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Place field heatmap saved to {config.save_path}")

        if config.show:
            plt.show()

    finally:
        plt.close(fig)


def plot_multiple_place_fields(
    heatmaps: np.ndarray,
    place_indices: np.ndarray,
    centers: np.ndarray | None = None,
    width: float = 1.0,
    height: float = 1.0,
    config: PlaceCellConfig | None = None,
    **kwargs,
):
    """
    Plot multiple place fields in a grid layout.

    Args:
        heatmaps (np.ndarray): Shape (M, K, N) - heatmaps for N neurons
        place_indices (np.ndarray): Indices of place cells to plot
        centers (np.ndarray, optional): Place field centers
        width (float): Environment width
        height (float): Environment height
        config (PlaceCellConfig, optional): Plot configuration
        **kwargs: Additional parameters
    """
    if config is None:
        config = PlaceCellConfigs.place_field_heatmap(**kwargs)

    num_cells = len(place_indices)
    cols = int(np.ceil(np.sqrt(num_cells)))
    rows = int(np.ceil(num_cells / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_cells == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    try:
        x = np.linspace(0, width, heatmaps.shape[1])
        y = np.linspace(0, height, heatmaps.shape[0])
        X, Y = np.meshgrid(x, y)

        for i, neuron_idx in enumerate(place_indices):
            if i >= len(axes):
                break

            ax = axes[i]
            heatmap = heatmaps[:, :, neuron_idx]

            # Plot heatmap
            ax.contourf(X, Y, heatmap, levels=20, cmap=config.colormap)

            # Mark center if provided
            if centers is not None and i < len(centers):
                ax.scatter(
                    centers[i, 0],
                    centers[i, 1],
                    s=config.center_marker_size,
                    c=config.center_marker_color,
                    marker="x",
                    linewidths=2,
                )

            ax.set_title(f"Place Cell {neuron_idx}")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_aspect("equal")

        # Hide unused subplots
        for i in range(num_cells, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Multiple place fields saved to {config.save_path}")

        if config.show:
            plt.show()

    finally:
        plt.close(fig)


def plot_place_field_properties(analysis_results: dict, config: PlotConfig | None = None, **kwargs):
    """
    Plot place field property distributions.

    Args:
        analysis_results (dict): Results from analyze_place_field_properties
        config (PlotConfig, optional): Plot configuration
        **kwargs: Additional parameters
    """
    if config is None:
        # Extract title from kwargs if provided
        title = kwargs.pop("title", "Place Field Properties")
        config = PlotConfig.for_static_plot(title=title, figsize=(15, 10), **kwargs)

    fig, axes = plt.subplots(2, 3, figsize=config.figsize)
    axes = axes.flatten()

    try:
        # 1. Place scores distribution
        axes[0].hist(analysis_results["place_scores"], bins=20, alpha=0.7, color="blue")
        axes[0].set_xlabel("Place Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Place Score Distribution")

        # 2. Field sizes distribution
        axes[1].hist(analysis_results["field_sizes"], bins=20, alpha=0.7, color="green")
        axes[1].set_xlabel("Field Size (σ)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Place Field Size Distribution")

        # 3. Spatial information distribution
        axes[2].hist(analysis_results["spatial_info"], bins=20, alpha=0.7, color="orange")
        axes[2].set_xlabel("Spatial Information (bits/spike)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Spatial Information Distribution")

        # 4. Peak firing rates
        axes[3].hist(analysis_results["peak_rates"], bins=20, alpha=0.7, color="red")
        axes[3].set_xlabel("Peak Firing Rate (Hz)")
        axes[3].set_ylabel("Count")
        axes[3].set_title("Peak Rate Distribution")

        # 5. Field size vs spatial info scatter
        axes[4].scatter(
            analysis_results["field_sizes"],
            analysis_results["spatial_info"],
            alpha=0.6,
            color="purple",
        )
        axes[4].set_xlabel("Field Size (σ)")
        axes[4].set_ylabel("Spatial Information")
        axes[4].set_title("Field Size vs Spatial Info")

        # 6. Place centers distribution
        if "centers" in analysis_results and len(analysis_results["centers"]) > 0:
            centers = analysis_results["centers"]
            axes[5].scatter(centers[:, 0], centers[:, 1], alpha=0.7, color="brown")
            axes[5].set_xlabel("X Position")
            axes[5].set_ylabel("Y Position")
            axes[5].set_title("Place Field Centers")
            axes[5].set_aspect("equal")

        plt.tight_layout()

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Place field properties plot saved to {config.save_path}")

        if config.show:
            plt.show()

    finally:
        plt.close(fig)


# ==================== Grid Cell Visualization ====================


@dataclass
class GridCellConfig(PlotConfig):
    """Configuration for Grid Cell visualizations."""

    colormap: str = "viridis"
    show_field_centers: bool = True
    center_marker_size: int = 30
    center_marker_color: str = "white"
    grid_line_color: str = "white"
    grid_line_alpha: float = 0.8


class GridCellConfigs:
    """Pre-configured settings for Grid Cell visualizations."""

    @staticmethod
    def grid_pattern(**kwargs) -> GridCellConfig:
        """Configuration for grid pattern visualization."""
        defaults = {
            "title": "Grid Cell Firing Pattern",
            "xlabel": "X Position",
            "ylabel": "Y Position",
            "clabel": "Firing Rate (Hz)",
            "figsize": (8, 8),
            "colormap": "plasma",
            "show_field_centers": True,
        }
        defaults.update(kwargs)
        return GridCellConfig.for_static_plot(**defaults)

    @staticmethod
    def grid_score_analysis(**kwargs) -> GridCellConfig:
        """Configuration for grid score analysis plots."""
        defaults = {
            "title": "Grid Cell Analysis",
            "figsize": (12, 8),
        }
        defaults.update(kwargs)
        return GridCellConfig.for_static_plot(**defaults)


def plot_grid_pattern(
    heatmap: np.ndarray,
    field_centers: np.ndarray | None = None,
    width: float = 1.0,
    height: float = 1.0,
    config: GridCellConfig | None = None,
    **kwargs,
):
    """
    Plot grid cell firing pattern with hexagonal field structure.

    Args:
        heatmap (np.ndarray): 2D firing rate heatmap
        field_centers (np.ndarray, optional): Grid field centers
        width (float): Environment width
        height (float): Environment height
        config (GridCellConfig, optional): Plot configuration
        **kwargs: Additional parameters
    """
    if config is None:
        config = GridCellConfigs.grid_pattern(**kwargs)

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        # Create coordinate arrays
        x = np.linspace(0, width, heatmap.shape[1])
        y = np.linspace(0, height, heatmap.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot heatmap
        im = ax.contourf(X, Y, heatmap, levels=50, cmap=config.colormap)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(config.clabel)

        # Mark field centers if provided
        if field_centers is not None and config.show_field_centers and len(field_centers) > 0:
            ax.scatter(
                field_centers[:, 0],
                field_centers[:, 1],
                s=config.center_marker_size,
                c=config.center_marker_color,
                marker="o",
                edgecolors="black",
                linewidths=1,
                label=f"{len(field_centers)} fields",
            )

            if config.show_legend:
                ax.legend()

        ax.set_xlabel(config.xlabel)
        ax.set_ylabel(config.ylabel)
        ax.set_title(config.title)
        ax.set_aspect("equal")

        if config.grid:
            ax.grid(True, alpha=0.3)

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Grid pattern plot saved to {config.save_path}")

        if config.show:
            plt.show()

    finally:
        plt.close(fig)


def plot_multiple_grid_patterns(
    heatmaps: np.ndarray,
    grid_indices: np.ndarray,
    grid_parameters: dict,
    width: float = 1.0,
    height: float = 1.0,
    config: GridCellConfig | None = None,
    **kwargs,
):
    """
    Plot multiple grid patterns with their parameters.

    Args:
        heatmaps (np.ndarray): Shape (M, K, N) - heatmaps for N neurons
        grid_indices (np.ndarray): Indices of grid cells to plot
        grid_parameters (dict): Grid analysis parameters
        width (float): Environment width
        height (float): Environment height
        config (GridCellConfig, optional): Plot configuration
        **kwargs: Additional parameters
    """
    if config is None:
        config = GridCellConfigs.grid_pattern(**kwargs)

    num_cells = len(grid_indices)
    cols = int(np.ceil(np.sqrt(num_cells)))
    rows = int(np.ceil(num_cells / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_cells == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    try:
        x = np.linspace(0, width, heatmaps.shape[1])
        y = np.linspace(0, height, heatmaps.shape[0])
        X, Y = np.meshgrid(x, y)

        for i, neuron_idx in enumerate(grid_indices):
            if i >= len(axes):
                break

            ax = axes[i]
            heatmap = heatmaps[:, :, neuron_idx]

            # Plot heatmap
            ax.contourf(X, Y, heatmap, levels=20, cmap=config.colormap)

            # Get parameters for this cell
            grid_score = (
                grid_parameters["grid_scores"][i] if i < len(grid_parameters["grid_scores"]) else 0
            )
            spacing = grid_parameters["spacings"][i] if i < len(grid_parameters["spacings"]) else 0
            num_fields = (
                grid_parameters["num_fields"][i] if i < len(grid_parameters["num_fields"]) else 0
            )

            # Mark field centers if available
            if "field_centers" in grid_parameters and i < len(grid_parameters["field_centers"]):
                centers = grid_parameters["field_centers"][i]
                if len(centers) > 0:
                    ax.scatter(
                        centers[:, 0],
                        centers[:, 1],
                        s=config.center_marker_size,
                        c=config.center_marker_color,
                        marker="o",
                        edgecolors="black",
                        linewidths=1,
                    )

            ax.set_title(
                f"Grid Cell {neuron_idx}\nScore: {grid_score:.3f}, "
                f"Spacing: {spacing:.3f}, Fields: {num_fields}"
            )
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_aspect("equal")

        # Hide unused subplots
        for i in range(num_cells, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Multiple grid patterns saved to {config.save_path}")

        if config.show:
            plt.show()

    finally:
        plt.close(fig)


def plot_grid_score_analysis(
    grid_scores: np.ndarray,
    grid_parameters: dict | None = None,
    config: GridCellConfig | None = None,
    **kwargs,
):
    """
    Plot grid score analysis and parameter distributions.

    Args:
        grid_scores (np.ndarray): Grid scores for all neurons
        grid_parameters (dict, optional): Additional grid parameters
        config (GridCellConfig, optional): Plot configuration
        **kwargs: Additional parameters
    """
    if config is None:
        config = GridCellConfigs.grid_score_analysis(**kwargs)

    if grid_parameters is not None:
        fig, axes = plt.subplots(2, 3, figsize=config.figsize)
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]

    try:
        # 1. Grid scores distribution
        axes[0].hist(grid_scores, bins=50, alpha=0.7, color="blue", edgecolor="black")
        axes[0].axvline(0.3, color="red", linestyle="--", label="Threshold (0.3)")
        axes[0].set_xlabel("Grid Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Grid Score Distribution")
        axes[0].legend()

        if grid_parameters is not None and len(axes) > 1:
            # 2. Grid spacing distribution
            if "spacings" in grid_parameters:
                spacings = grid_parameters["spacings"]
                spacings = spacings[spacings > 0]  # Remove zeros
                if len(spacings) > 0:
                    axes[1].hist(spacings, bins=20, alpha=0.7, color="green")
                    axes[1].set_xlabel("Grid Spacing")
                    axes[1].set_ylabel("Count")
                    axes[1].set_title("Grid Spacing Distribution")

            # 3. Number of fields distribution
            if "num_fields" in grid_parameters:
                axes[2].hist(
                    grid_parameters["num_fields"],
                    bins=range(0, 20),
                    alpha=0.7,
                    color="orange",
                    align="left",
                )
                axes[2].set_xlabel("Number of Fields")
                axes[2].set_ylabel("Count")
                axes[2].set_title("Number of Fields Distribution")

            # 4. Grid regularity
            if "regularities" in grid_parameters:
                axes[3].hist(grid_parameters["regularities"], bins=20, alpha=0.7, color="purple")
                axes[3].set_xlabel("Grid Regularity")
                axes[3].set_ylabel("Count")
                axes[3].set_title("Grid Regularity Distribution")

            # 5. Score vs spacing scatter
            if "spacings" in grid_parameters:
                valid_idx = grid_parameters["spacings"] > 0
                if np.any(valid_idx):
                    valid_scores = grid_scores[valid_idx]
                    valid_spacings = grid_parameters["spacings"][valid_idx]
                    axes[4].scatter(valid_spacings, valid_scores, alpha=0.6, color="red")
                    axes[4].set_xlabel("Grid Spacing")
                    axes[4].set_ylabel("Grid Score")
                    axes[4].set_title("Grid Score vs Spacing")

            # 6. Score vs number of fields
            if "num_fields" in grid_parameters:
                axes[5].scatter(
                    grid_parameters["num_fields"], grid_scores, alpha=0.6, color="brown"
                )
                axes[5].set_xlabel("Number of Fields")
                axes[5].set_ylabel("Grid Score")
                axes[5].set_title("Grid Score vs Number of Fields")

        plt.tight_layout()

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Grid score analysis saved to {config.save_path}")

        if config.show:
            plt.show()

    finally:
        plt.close(fig)


# ==================== Combined Spatial Analysis ====================


def plot_spatial_tuning_curves(
    activity: np.ndarray,
    positions: np.ndarray,
    neuron_indices: np.ndarray,
    width: float = 1.0,
    height: float = 1.0,
    config: PlotConfig | None = None,
    **kwargs,
):
    """
    Plot 1D spatial tuning curves for selected neurons.

    Args:
        activity (np.ndarray): Shape (T, N) - neural activity
        positions (np.ndarray): Shape (T, 2) - positions
        neuron_indices (np.ndarray): Indices of neurons to plot
        width (float): Environment width
        height (float): Environment height
        config (PlotConfig, optional): Plot configuration
        **kwargs: Additional parameters
    """
    if config is None:
        # Extract title from kwargs if provided
        title = kwargs.pop("title", "Spatial Tuning Curves")
        config = PlotConfig.for_static_plot(
            title=title,
            xlabel="Position",
            ylabel="Firing Rate (Hz)",
            figsize=(12, 8),
            **kwargs,
        )

    num_neurons = len(neuron_indices)
    cols = 2
    rows = int(np.ceil(num_neurons / cols))

    fig, axes = plt.subplots(rows, cols, figsize=config.figsize)
    if num_neurons == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    try:
        # Create position bins
        n_bins = 50
        x_bins = np.linspace(0, width, n_bins)
        y_bins = np.linspace(0, height, n_bins)

        for i, neuron_idx in enumerate(neuron_indices):
            if i >= len(axes):
                break

            ax = axes[i]

            # X-direction tuning curve
            x_rates, _, _ = binned_statistic(
                positions[:, 0], activity[:, neuron_idx], "mean", bins=x_bins
            )
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2

            # Y-direction tuning curve
            y_rates, _, _ = binned_statistic(
                positions[:, 1], activity[:, neuron_idx], "mean", bins=y_bins
            )
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2

            ax.plot(x_centers, x_rates, "b-", label="X-direction", linewidth=2)
            ax.plot(y_centers, y_rates, "r-", label="Y-direction", linewidth=2)

            ax.set_title(f"Neuron {neuron_idx}")
            ax.set_xlabel("Position")
            ax.set_ylabel("Firing Rate (Hz)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(num_neurons, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Spatial tuning curves saved to {config.save_path}")

        if config.show:
            plt.show()

    finally:
        plt.close(fig)
