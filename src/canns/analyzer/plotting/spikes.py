"""Spike train visualization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from .config import PlotConfig, PlotConfigs

__all__ = ["raster_plot", "average_firing_rate_plot"]


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


def raster_plot(
    spike_train: np.ndarray,
    config: PlotConfig | None = None,
    *,
    mode: str = "block",
    title: str = "Raster Plot",
    xlabel: str = "Time Step",
    ylabel: str = "Neuron Index",
    figsize: tuple[int, int] = (12, 6),
    color: str = "black",
    save_path: str | None = None,
    show: bool = True,
    **kwargs: Any,
):
    """Generate a raster plot from a spike train matrix.

    The explanatory text mirrors the former ``visualize`` module so callers see
    the same guidance after the reorganisation.

    Args:
        spike_train: Boolean/integer array of shape ``(timesteps, neurons)``.
        config: Optional :class:`PlotConfig` with shared styling options.
        mode: Either ``"scatter"`` or ``"block"`` to pick the rendering style.
        title: Plot title when ``config`` is not provided.
        xlabel: X-axis label when ``config`` is not provided.
        ylabel: Y-axis label when ``config`` is not provided.
        figsize: Figure size forwarded to Matplotlib when creating the axes.
        color: Spike colour (or "on" colour for block mode).
        save_path: Optional path used to persist the plot.
        show: Whether to display the plot interactively.
        **kwargs: Additional keyword arguments passed through to Matplotlib.
    """

    config = _ensure_plot_config(
        config,
        PlotConfigs.raster_plot,
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

    if not hasattr(config, "mode"):
        config.mode = mode

    if spike_train.ndim != 2:
        raise ValueError(f"Input spike_train must be a 2D array, but got shape {spike_train.shape}")
    if spike_train.size == 0:
        raise ValueError("Input spike_train must not be empty.")
    if config.mode not in {"block", "scatter"}:
        raise ValueError(f"Invalid mode '{config.mode}'. Choose 'scatter' or 'block'.")

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        ax.set_title(config.title, fontsize=16, fontweight="bold")
        ax.set_xlabel(config.xlabel, fontsize=12)
        ax.set_ylabel(config.ylabel, fontsize=12)

        if config.mode == "scatter":
            time_indices, neuron_indices = np.where(spike_train)
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
        else:
            data_to_show = spike_train.T
            cmap = config.kwargs.pop("cmap", ListedColormap(["white", config.color]))
            ax.imshow(
                data_to_show,
                aspect="auto",
                interpolation="none",
                cmap=cmap,
                **config.to_matplotlib_kwargs(),
            )
            ax.set_yticks(np.arange(spike_train.shape[1]))
            ax.set_yticklabels(np.arange(spike_train.shape[1]))
            if spike_train.shape[1] > 20:
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))

        if config.save_path:
            plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {config.save_path}")

        if config.show:
            plt.show()
    finally:
        plt.close(fig)

    return fig, ax


def average_firing_rate_plot(
    spike_train: np.ndarray,
    dt: float,
    config: PlotConfig | None = None,
    *,
    mode: str = "population",
    weights: np.ndarray | None = None,
    title: str = "Average Firing Rate",
    figsize: tuple[int, int] = (12, 5),
    save_path: str | None = None,
    show: bool = True,
    **kwargs: Any,
):
    """Calculate and plot average neural activity from a spike train.

    Args:
        spike_train: Boolean/integer array of shape ``(timesteps, neurons)``.
        dt: Simulation time step in seconds.
        config: Optional :class:`PlotConfig` with styling overrides.
        mode: One of ``"per_neuron"``, ``"population"`` or
            ``"weighted_average"``.
        weights: Neuron-wise weights required for ``"weighted_average"``.
        title: Plot title when ``config`` is not provided.
        figsize: Figure size forwarded to Matplotlib when creating the axes.
        save_path: Optional path used to persist the plot.
        show: Whether to display the plot interactively.
        **kwargs: Additional keyword arguments forwarded to Matplotlib.
    """

    config = _ensure_plot_config(
        config,
        PlotConfigs.average_firing_rate_plot,
        mode=mode,
        title=title,
        figsize=figsize,
        save_path=save_path,
        show=show,
        kwargs=kwargs,
    )

    if not hasattr(config, "mode"):
        config.mode = mode

    if spike_train.ndim != 2:
        raise ValueError("Input spike_train must be a 2D array.")

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        num_timesteps, num_neurons = spike_train.shape
        ax.set_title(config.title, fontsize=16, fontweight="bold")

        if config.mode == "per_neuron":
            duration_s = num_timesteps * dt
            total_spikes_per_neuron = np.sum(spike_train, axis=0)
            calculated_data = total_spikes_per_neuron / duration_s
            ax.plot(np.arange(num_neurons), calculated_data, **config.to_matplotlib_kwargs())
            ax.set_xlabel("Neuron Index", fontsize=12)
            ax.set_ylabel("Average Firing Rate (Hz)", fontsize=12)
            ax.set_xlim(0, num_neurons - 1)

        elif config.mode == "population":
            spikes_per_timestep = np.sum(spike_train, axis=1)
            calculated_data = spikes_per_timestep / dt
            time_vector = np.arange(num_timesteps) * dt
            ax.plot(time_vector, calculated_data, **config.to_matplotlib_kwargs())
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("Total Population Rate (Hz)", fontsize=12)
            ax.set_xlim(0, time_vector[-1])

        elif config.mode == "weighted_average":
            if weights is None:
                raise ValueError("'weights' argument is required for 'weighted_average' mode.")
            if weights.shape != (num_neurons,):
                raise ValueError(
                    f"Shape of 'weights' {weights.shape} must match num_neurons ({num_neurons})."
                )

            total_spikes_per_timestep = np.sum(spike_train, axis=1)
            weighted_sum_of_spikes = np.sum(spike_train * weights, axis=1)
            calculated_data = weighted_sum_of_spikes / (total_spikes_per_timestep + 1e-9)
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

    return fig, ax
