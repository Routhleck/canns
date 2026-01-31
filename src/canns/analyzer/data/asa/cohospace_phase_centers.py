"""Phase-center utilities for CohoSpace (skewed coordinates)."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ...visualization.core import PlotConfig, finalize_figure
from .cohospace_scatter import skew_transform_torus_scatter


def _ensure_plot_config(
    config: PlotConfig | None,
    factory,
    *args,
    **defaults,
) -> PlotConfig:
    if config is None:
        return factory(*args, **defaults)
    return config


def _ensure_parent_dir(save_path: str | None) -> None:
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def cohospace_phase_centers(cohospace_result: dict[str, Any]) -> dict[str, Any]:
    """
    Compute per-neuron CohoSpace phase centers and their skewed coordinates.

    Input
    -----
    cohospace_result : dict
        Output from `data.cohospace(...)` (must include `centers`).
    """
    centers = np.asarray(cohospace_result["centers"], dtype=float) % (2 * np.pi)
    centers_skew = skew_transform_torus_scatter(centers)
    return {
        "centers": centers,
        "centers_skew": centers_skew,
    }


def plot_cohospace_phase_centers(
    cohospace_result: dict[str, Any],
    *,
    neuron_id: int | None = None,
    show_all: bool = False,
    config: PlotConfig | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (5, 5),
    all_color: str = "tab:blue",
    highlight_color: str = "tab:red",
    alpha: float = 0.7,
    s: int = 12,
) -> plt.Figure:
    """
    Plot CohoSpace phase centers on the skewed torus domain.

    If neuron_id is None, plot all neurons. If neuron_id is provided, show_all controls
    whether all neurons are drawn lightly or only the selected neuron is shown.
    """
    centers_result = cohospace_phase_centers(cohospace_result)
    centers_skew = centers_result["centers_skew"]
    num_neurons = centers_skew.shape[0]

    if neuron_id is not None and (neuron_id < 0 or neuron_id >= num_neurons):
        raise ValueError(f"neuron_id out of range: {neuron_id}")

    title = "CohoSpace phase centers (skewed)"
    if neuron_id is not None and not show_all:
        title = f"CohoSpace phase center (neuron {neuron_id}, skewed)"
    elif neuron_id is not None and show_all:
        title = f"CohoSpace phase centers (all + neuron {neuron_id}, skewed)"

    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title=title,
        xlabel=r"$\theta_1 + \frac{1}{2}\theta_2$",
        ylabel=r"$\frac{\sqrt{3}}{2}\theta_2$",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    fig, ax = plt.subplots(figsize=config.figsize)

    # fundamental domain parallelogram
    e1 = np.array([2 * np.pi, 0.0])
    e2 = np.array([np.pi, np.sqrt(3) * np.pi])
    poly = np.vstack([[0.0, 0.0], e1, e1 + e2, e2, [0.0, 0.0]])
    ax.plot(poly[:, 0], poly[:, 1], lw=1.2, color="0.35")

    if neuron_id is None:
        ax.scatter(centers_skew[:, 0], centers_skew[:, 1], s=s, alpha=alpha, color=all_color)
    else:
        if show_all:
            ax.scatter(
                centers_skew[:, 0],
                centers_skew[:, 1],
                s=max(4, s - 4),
                alpha=0.25,
                color=all_color,
            )
        ax.scatter(
            centers_skew[neuron_id, 0],
            centers_skew[neuron_id, 1],
            s=max(10, s + 6),
            alpha=0.9,
            color=highlight_color,
        )

    base = np.vstack([[0.0, 0.0], e1, e2, e1 + e2])
    xmin, ymin = base.min(axis=0)
    xmax, ymax = base.max(axis=0)
    padx = 0.03 * (xmax - xmin)
    pady = 0.03 * (ymax - ymin)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal")
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)
    ax.set_xticks([])
    ax.set_yticks([])

    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return fig
