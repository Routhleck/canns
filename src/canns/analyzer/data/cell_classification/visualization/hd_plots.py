"""
Head Direction Cell Visualization

Plotting functions for head direction cell analysis.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_polar_tuning(
    angles: np.ndarray,
    rates: np.ndarray,
    preferred_direction: float | None = None,
    mvl: float | None = None,
    title: str = "Directional Tuning",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot directional tuning curve in polar coordinates.

    Parameters
    ----------
    angles : np.ndarray
        Angular bins in radians
    rates : np.ndarray
        Firing rates for each bin
    preferred_direction : float, optional
        Preferred direction to mark (radians)
    mvl : float, optional
        Mean Vector Length to display
    title : str, optional
        Plot title
    ax : plt.Axes, optional
        Polar axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : plt.Axes
        The axes object
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")

    # Plot tuning curve
    ax.plot(angles, rates, "b-", linewidth=2, label="Tuning curve")
    ax.fill_between(angles, 0, rates, alpha=0.3)

    # Mark preferred direction
    if preferred_direction is not None:
        max_rate = np.max(rates)
        ax.plot(
            [preferred_direction, preferred_direction],
            [0, max_rate],
            "r--",
            linewidth=2,
            label=f"Preferred: {np.rad2deg(preferred_direction):.1f}°",
        )

    # Add MVL to title
    if mvl is not None:
        title += f" (MVL: {mvl:.3f})"

    ax.set_title(title, pad=20)
    ax.set_theta_zero_location("E")  # 0° to the right
    ax.set_theta_direction(1)  # Counterclockwise
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    return ax


def plot_temporal_autocorr(
    lags: np.ndarray,
    acorr: np.ndarray,
    title: str = "Temporal Autocorrelation",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot temporal autocorrelation as bar plot.

    Parameters
    ----------
    lags : np.ndarray
        Time lags (ms or bins)
    acorr : np.ndarray
        Autocorrelation values
    title : str, optional
        Plot title
    ax : plt.Axes, optional
        Axes to plot on

    Returns
    -------
    ax : plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.bar(lags, acorr, width=np.diff(lags)[0] * 0.8, color="blue", alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel("Time Lag (ms)")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.3)

    return ax


def plot_hd_analysis(
    result,
    time_stamps: np.ndarray | None = None,
    head_directions: np.ndarray | None = None,
    spike_times: np.ndarray | None = None,
    figsize: tuple = (15, 5),
) -> plt.Figure:
    """
    Comprehensive head direction analysis plot.

    Parameters
    ----------
    result : HDCellResult
        Results from HeadDirectionAnalyzer
    time_stamps : np.ndarray, optional
        Time stamps for plotting trajectory
    head_directions : np.ndarray, optional
        Head direction time series
    spike_times : np.ndarray, optional
        Spike times
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object
    """
    fig = plt.figure(figsize=figsize)

    # Plot 1: Polar tuning curve
    ax1 = fig.add_subplot(131, projection="polar")
    bin_centers, firing_rates = result.tuning_curve
    plot_polar_tuning(
        bin_centers,
        firing_rates,
        preferred_direction=result.preferred_direction,
        mvl=result.mvl_hd,
        ax=ax1,
    )

    # Plot 2: Linear tuning curve
    ax2 = fig.add_subplot(132)
    ax2.plot(np.rad2deg(bin_centers), firing_rates, "b-", linewidth=2)
    ax2.fill_between(np.rad2deg(bin_centers), 0, firing_rates, alpha=0.3)
    if result.preferred_direction is not None:
        ax2.axvline(
            np.rad2deg(result.preferred_direction),
            color="r",
            linestyle="--",
            linewidth=2,
            label="Preferred direction",
        )
    ax2.set_xlabel("Head Direction (°)")
    ax2.set_ylabel("Firing Rate (Hz)")
    ax2.set_title("Linear Tuning Curve")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Statistics and trajectory (if provided)
    ax3 = fig.add_subplot(133)
    if time_stamps is not None and head_directions is not None:
        ax3.plot(time_stamps, np.rad2deg(head_directions), "k-", linewidth=0.5, alpha=0.5)
        if spike_times is not None:
            spike_hd = np.interp(spike_times, time_stamps, head_directions)
            ax3.plot(spike_times, np.rad2deg(spike_hd), "r.", markersize=3)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Head Direction (°)")
        ax3.set_title("HD Trajectory with Spikes")
        ax3.set_ylim([-180, 180])
    else:
        ax3.axis("off")
        stats_text = f"""
Head Direction Cell Analysis

Classification: {"HD Cell" if result.is_hd else "Non-HD Cell"}
MVL (HD): {result.mvl_hd:.3f}
{"MVL (Theta): " + f"{result.mvl_theta:.3f}" if result.mvl_theta else ""}
Preferred Direction: {np.rad2deg(result.preferred_direction):.1f}°
Rayleigh p-value: {result.rayleigh_p:.6f}

Peak Firing Rate: {np.max(firing_rates):.2f} Hz
Mean Firing Rate: {np.mean(firing_rates):.2f} Hz
        """
        ax3.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center", family="monospace")
        ax3.set_title("HD Statistics")

    plt.tight_layout()
    return fig
