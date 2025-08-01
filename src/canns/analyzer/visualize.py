import sys
from collections.abc import Iterable

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic
from tqdm import tqdm

# --- CANN Model related visualization method ---


def energy_landscape_1d_static(
    data_sets: dict[str, tuple[np.ndarray, np.ndarray]],
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
    # --- Create the figure and axes ---
    fig, ax = plt.subplots(figsize=figsize)

    try:
        # --- Loop through and plot each energy curve ---
        # Use .items() to iterate over both keys (labels) and values (data) of the dictionary
        for label, (x_data, y_data) in data_sets.items():
            # Plot the curve, using the dictionary key directly as the label
            ax.plot(x_data, y_data, label=label, **kwargs)

        # --- Configure the plot's appearance ---
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # If requested, display the legend
        if show_legend:
            ax.legend()

        # Set the grid
        if grid:
            ax.grid(True, linestyle="--", alpha=0.6)

        # --- Save and display the plot ---
        if save_path:
            # Using bbox_inches='tight' prevents labels from being cut off
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()

    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def energy_landscape_1d_animation(
    data_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    time_steps_per_second: int,
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
        **kwargs:
            Any other keyword arguments for matplotlib.pyplot.plot (e.g., linewidth).

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    try:
        # --- Input Validation and Timing Calculation ---
        if not data_sets:
            raise ValueError("The 'data_sets' dictionary cannot be empty.")

        first_key = list(data_sets.keys())[0]
        total_sim_steps = data_sets[first_key][1].shape[0]

        # Calculate total simulation duration in seconds from the data itself
        total_duration_s = total_sim_steps / time_steps_per_second

        # Calculate the total number of frames needed for the output video
        num_video_frames = int(total_duration_s * fps)

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
            (line,) = ax.plot(x_data, ys_data[initial_sim_index, :], label=label, **kwargs)
            lines[label] = line

        # Configure plot appearance
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
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
            current_time_s = sim_index / time_steps_per_second
            time_text.set_text(f"Time: {current_time_s:.2f} s")
            artists_to_update.append(time_text)

            return artists_to_update

        # --- Create and Return the Animation ---
        interval_ms = 1000 / fps
        ani = animation.FuncAnimation(
            fig, animate, frames=num_video_frames, interval=interval_ms, blit=True, repeat=repeat
        )

        # --- Save or Show the Animation ---
        if save_path:
            if show_progress_bar:
                # Setup the progress bar
                pbar = tqdm(
                    total=num_video_frames,
                    desc=f"<{sys._getframe().f_code.co_name}> Saving to {save_path}",
                )

                # Define the callback function that updates the progress bar
                def progress_callback(current_frame, total_frames):
                    pbar.update(1)

                # Save the animation with the callback
                try:
                    writer = animation.PillowWriter(fps=fps)
                    ani.save(save_path, writer=writer, progress_callback=progress_callback)
                    pbar.close()  # Close the progress bar upon completion
                    print(f"\nAnimation successfully saved to: {save_path}")
                except Exception as e:
                    pbar.close()
                    print(f"\nError saving animation: {e}")
            else:
                # Save without a progress bar
                try:
                    writer = animation.PillowWriter(fps=fps)
                    ani.save(save_path, writer=writer)
                    print(f"Animation saved to: {save_path}")
                except Exception as e:
                    print(f"Error saving animation: {e}")
        if show:
            plt.show()
    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def energy_landscape_2d_static(
    z_data: np.ndarray,
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
    if z_data.ndim != 2:
        raise ValueError(f"Input z_data must be a 2D array, but got shape {z_data.shape}")
    assert z_data.size > 0, "Input z_data must not be empty."

    fig, ax = plt.subplots(figsize=figsize)

    try:
        # Use imshow for efficient 2D plotting. origin='lower' puts (0,0) at the bottom-left.
        im = ax.imshow(z_data, origin="lower", aspect="auto", **kwargs)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(clabel, fontsize=12)

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        if grid:
            ax.grid(True, linestyle="--", alpha=0.4, color="white")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()

    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def energy_landscape_2d_animation(
    zs_data: np.ndarray,
    time_steps_per_second: int,
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
        time_steps_per_second (int): Number of data points (frames) per second of simulation time.
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
    fig, ax = plt.subplots(figsize=figsize)

    if zs_data.ndim != 3:
        raise ValueError(f"Input zs_data must be a 3D array, but got shape {zs_data.shape}")

    try:
        # --- Timing Calculation ---
        total_sim_steps = zs_data.shape[0]
        total_duration_s = total_sim_steps / time_steps_per_second
        num_video_frames = int(total_duration_s * fps)
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
            **kwargs,
        )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(clabel, fontsize=12)

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
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
            current_time_s = sim_index / time_steps_per_second
            time_text.set_text(f"Time: {current_time_s:.2f} s")
            return im, time_text

        # --- Create and Return the Animation ---
        interval_ms = 1000 / fps
        ani = animation.FuncAnimation(
            fig, animate, frames=num_video_frames, interval=interval_ms, blit=True, repeat=repeat
        )

        # --- Save or Show the Animation ---
        if save_path:
            if show_progress_bar:
                pbar = tqdm(total=num_video_frames, desc=f"Saving to {save_path}")

                def progress_callback(current_frame, total_frames):
                    pbar.update(1)

                try:
                    writer = animation.PillowWriter(fps=fps)
                    ani.save(save_path, writer=writer, progress_callback=progress_callback)
                    pbar.close()
                    print(f"\nAnimation successfully saved to: {save_path}")
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
            plt.show()

    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def raster_plot(
    spike_train: np.ndarray,
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
    if spike_train.ndim != 2:
        raise ValueError(f"Input spike_train must be a 2D array, but got shape {spike_train.shape}")
    assert spike_train.size > 0, "Input spike_train must not be empty."
    assert mode in ("block", "scatter"), f"Invalid mode '{mode}'. Choose 'scatter' or 'block'."

    fig, ax = plt.subplots(figsize=figsize)

    try:
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        if mode == "scatter":
            # --- Traditional Scatter Plot Mode ---
            time_indices, neuron_indices = np.where(spike_train)

            # Set default marker size if not provided in kwargs
            marker_size = kwargs.pop("marker_size", 1.0)

            ax.scatter(
                time_indices,
                neuron_indices,
                s=marker_size,
                c=color,
                marker="|",
                alpha=0.8,
                **kwargs,
            )
            ax.set_xlim(0, spike_train.shape[0])
            ax.set_ylim(-1, spike_train.shape[1])

        elif mode == "block":
            # --- Block / Image Mode ---
            # imshow expects data oriented as (row, column), which corresponds to (neuron, time).
            # So we need to transpose the spike_train.
            data_to_show = spike_train.T

            # Create a custom colormap: 0 -> transparent, 1 -> specified color
            from matplotlib.colors import ListedColormap

            cmap = kwargs.pop("cmap", ListedColormap(["white", color]))

            # Use imshow to create the block plot.
            # `interpolation='none'` ensures sharp, non-blurry blocks.
            # `aspect='auto'` allows the blocks to be non-square to fill the space.
            ax.imshow(data_to_show, aspect="auto", interpolation="none", cmap=cmap, **kwargs)

            # Set the ticks to be at the center of the neurons
            ax.set_yticks(np.arange(spike_train.shape[1]))
            ax.set_yticklabels(np.arange(spike_train.shape[1]))
            # Optional: reduce the number of y-ticks if there are too many neurons
            if spike_train.shape[1] > 20:
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))

        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'scatter' or 'block'.")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()

    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


def average_firing_rate_plot(
    spike_train: np.ndarray,
    dt: float,
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
        mode (str, optional):
            The plotting mode. Can be:
            - 'per_neuron': Average rate for each neuron over the entire duration. (X-axis: Neuron Index)
            - 'population': Average rate of all neurons at each time step. (X-axis: Time)
            - 'weighted_average': Weighted average of neural activity over time. Requires 'weights' argument. (X-axis: Time)
            Defaults to 'population'.
        weights (Optional[np.ndarray], optional):
            A 1D array of shape (num_neurons,) required for 'weighted_average' mode.
            Represents the preferred value (e.g., angle, position) for each neuron.
        ... (other args are the same) ...

    Returns:
        Tuple[np.ndarray, Tuple[plt.Figure, plt.Axes]]:
            A tuple containing the calculated data and the plot objects.
    """
    if spike_train.ndim != 2:
        raise ValueError("Input spike_train must be a 2D array.")

    fig, ax = plt.subplots(figsize=figsize)

    try:
        num_timesteps, num_neurons = spike_train.shape
        ax.set_title(title, fontsize=16, fontweight="bold")

        if mode == "per_neuron":
            # --- Average rate for each neuron over time ---
            duration_s = num_timesteps * dt
            total_spikes_per_neuron = np.sum(spike_train, axis=0)
            # Rate = total spikes / total duration
            calculated_data = total_spikes_per_neuron / duration_s

            ax.plot(np.arange(num_neurons), calculated_data, **kwargs)
            ax.set_xlabel("Neuron Index", fontsize=12)
            ax.set_ylabel("Average Firing Rate (Hz)", fontsize=12)
            ax.set_xlim(0, num_neurons - 1)

        elif mode == "population":
            # --- Average rate of the whole population over time ---
            spikes_per_timestep = np.sum(spike_train, axis=1)
            # Population Rate = (total spikes in bin) / (num_neurons * bin_duration)
            # This definition is debated, another is just total spikes / bin_duration.
            # We will use the simpler total spikes / bin_duration, which is the summed rate.
            calculated_data = spikes_per_timestep / dt

            time_vector = np.arange(num_timesteps) * dt
            ax.plot(time_vector, calculated_data, **kwargs)
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Total Population Rate (Hz)", fontsize=12)
            ax.set_xlim(0, time_vector[-1])

        elif mode == "weighted_average":
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
            ax.plot(time_vector, calculated_data, **kwargs)
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Decoded Value (Weighted Average)", fontsize=12)
            ax.set_xlim(0, time_vector[-1])

        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose 'per_neuron', 'population', or 'weighted_average'."
            )

        ax.grid(True, linestyle="--", alpha=0.6)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()

    finally:
        plt.close(fig)


def tuning_curve(
    stimulus: np.ndarray,
    firing_rates: np.ndarray,
    neuron_indices: np.ndarray | int,
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
    fig, ax = plt.subplots(figsize=figsize)

    try:
        # --- Computation and Plotting Loop ---
        for neuron_idx in neuron_indices:
            # Get the time series of firing rates for the current neuron
            neuron_fr = firing_rates[:, neuron_idx]

            # Use binned_statistic for efficient binning and averaging.
            # 'statistic'='mean' calculates the average of values in each bin.
            # 'bins'=num_bins divides the stimulus range into num_bins equal intervals.
            mean_rates, bin_edges, _ = binned_statistic(
                x=stimulus, values=neuron_fr, statistic="mean", bins=num_bins
            )

            # Calculate the center of each bin for plotting on the x-axis
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Create a label for the legend
            label = f"Neuron {neuron_idx}"
            if pref_stim is not None and neuron_idx < len(pref_stim):
                label += f" (pref_stim={pref_stim[neuron_idx]:.2f})"

            # Plot the curve. Bins that were empty will have a `nan` mean,
            # which matplotlib handles gracefully (it won't plot them).
            ax.plot(bin_centers, mean_rates, label=label, **kwargs)

        # --- Final Touches and Output ---
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Tuning curve saved to {save_path}")

        if show:
            plt.show()
    finally:
        # Ensure we clean up the figure to avoid memory leaks
        plt.close(fig)


# TODO: Implement phase_plane_plot (NEED DISCUSSION)
def phase_plane_plot(): ...
