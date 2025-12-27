"""Tuning curve visualization utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic

from .core.config import PlotConfig, PlotConfigs

__all__ = ["tuning_curve"]


def _ensure_plot_config(
    config: PlotConfig | None,
    *,
    pref_stim: np.ndarray | None,
    num_bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[int, int],
    save_path: str | None,
    show: bool,
    kwargs: dict[str, Any] | None,
) -> PlotConfig:
    if config is None:
        return PlotConfigs.tuning_curve(
            pref_stim=pref_stim,
            num_bins=num_bins,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            save_path=save_path,
            show=show,
            kwargs=kwargs or {},
        )

    if not hasattr(config, "num_bins"):
        config.num_bins = num_bins
    if not hasattr(config, "pref_stim"):
        config.pref_stim = pref_stim
    if kwargs:
        config_kwargs = config.kwargs or {}
        config_kwargs.update(kwargs)
        config.kwargs = config_kwargs
    return config


def tuning_curve(
    stimulus: np.ndarray,
    firing_rates: np.ndarray,
    neuron_indices: np.ndarray | int,
    config: PlotConfig | None = None,
    *,
    pref_stim: np.ndarray | None = None,
    num_bins: int = 50,
    title: str = "Tuning Curve",
    xlabel: str = "Stimulus Value",
    ylabel: str = "Average Firing Rate",
    figsize: tuple[int, int] = (10, 6),
    save_path: str | None = None,
    show: bool = True,
    **kwargs: Any,
):
    """Plot the tuning curve for one or more neurons.

    The wording mirrors the original ``visualize`` module to avoid API drift and
    to keep existing references valid.

    Args:
        stimulus: 1D array with the stimulus value at each time step.
        firing_rates: 2D array of firing rates shaped ``(timesteps, neurons)``.
        neuron_indices: Integer or iterable of neuron indices to analyse.
        config: Optional :class:`PlotConfig` containing styling overrides.
        pref_stim: Optional 1D array of preferred stimuli used in legend text.
        num_bins: Number of bins when mapping stimulus to mean activity.
        title: Plot title when ``config`` is not provided.
        xlabel: X-axis label when ``config`` is not provided.
        ylabel: Y-axis label when ``config`` is not provided.
        figsize: Figure size forwarded to Matplotlib when creating the axes.
        save_path: Optional location where the figure should be stored.
        show: Whether to display the plot interactively.
        **kwargs: Additional keyword arguments passed through to ``ax.plot``.
    """

    config = _ensure_plot_config(
        config,
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

    if stimulus.ndim != 1:
        raise ValueError(f"stimulus must be a 1D array, but has {stimulus.ndim} dimensions.")
    if firing_rates.ndim != 2:
        raise ValueError(
            f"firing_rates must be a 2D array, but has {firing_rates.ndim} dimensions."
        )
    if stimulus.shape[0] != firing_rates.shape[0]:
        raise ValueError(
            "The first dimension (time steps) of stimulus and firing_rates must match: "
            f"{stimulus.shape[0]} != {firing_rates.shape[0]}"
        )

    if isinstance(neuron_indices, int):
        neuron_indices = [neuron_indices]
    elif not isinstance(neuron_indices, Iterable):
        raise TypeError(
            "neuron_indices must be an integer or an iterable (e.g., list, np.ndarray)."
        )

    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        for neuron_idx in neuron_indices:
            neuron_fr = firing_rates[:, neuron_idx]
            mean_rates, bin_edges, _ = binned_statistic(
                x=stimulus,
                values=neuron_fr,
                statistic="mean",
                bins=config.num_bins,
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            label = f"Neuron {neuron_idx}"
            if config.pref_stim is not None and neuron_idx < len(config.pref_stim):
                label += f" (pref_stim={config.pref_stim[neuron_idx]:.2f})"

            ax.plot(bin_centers, mean_rates, label=label, **config.to_matplotlib_kwargs())

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
        plt.close(fig)

    return fig, ax
