from typing import Dict, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt, animation


# --- CANN Model related visualization method ---

def energy_landscape_1d_static(
    data_sets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "1D Energy Landscape",
    xlabel: str = "Collective Variable / State",
    ylabel: str = "Energy",
    show_legend: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    grid: bool = False,
    save_path: Optional[str] = None,
    show=True,
    **kwargs
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
        grid (bool, optional): Whether to display a grid. Defaults to True.
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
    # --- 1. Create the figure and axes ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 2. Loop through and plot each energy curve ---
    # Use .items() to iterate over both keys (labels) and values (data) of the dictionary
    for label, (x_data, y_data) in data_sets.items():
        # Plot the curve, using the dictionary key directly as the label
        ax.plot(x_data, y_data, label=label, **kwargs)

    # --- 3. Configure the plot's appearance ---
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # If requested, display the legend
    if show_legend:
        ax.legend()

    # Set the grid
    if grid:
        ax.grid(True, linestyle='--', alpha=0.6)

    # --- 4. Save and display the plot ---
    if save_path:
        # Using bbox_inches='tight' prevents labels from being cut off
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show() if show else None

    # --- 5. Return the figure and axes objects ---
    return fig, ax


def energy_landscape_1d_animation(
    data_sets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    time_steps_per_second: int,
    fps: int = 30,
    title: str = "Evolving 1D Energy Landscape",
    xlabel: str = "Collective Variable / State",
    ylabel: str = "Energy",
    figsize: Tuple[int, int] = (10, 6),
    grid: bool = False,
    repeat: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
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

    # --- Initial Setup ---
    fig, ax = plt.subplots(figsize=figsize)
    lines = {}

    # Set stable axis limits to prevent jumping
    global_ymin, global_ymax = float('inf'), float('-inf')
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
        line, = ax.plot(x_data, ys_data[initial_sim_index, :], label=label, **kwargs)
        lines[label] = line

    # Configure plot appearance
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7))

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
        time_text.set_text(f'Time: {current_time_s:.2f} s')
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
        repeat=repeat
    )

    # --- Save or Show the Animation ---
    if save_path:
        try:
            writer = animation.PillowWriter(fps=fps)
            ani.save(save_path, writer=writer)
            print(f"Animation saved to: {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Please ensure you have a writer like 'Pillow' (for GIFs) or 'ffmpeg' (for MP4s) installed.")

    plt.show() if show else None
    return ani

def energy_landscape_2d_static(

):
    ...

def energy_landscape_2d_animation(

):
    ...

def raster_plot(

):
    ...

def average_firing_rate(

):
    ...

def tuning_curve(

):
    ...

def phase_plane_plot(

):
    ...