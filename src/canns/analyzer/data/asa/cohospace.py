from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

from ...visualization.core import PlotConfig, finalize_figure
from .coho import _ensure_parent_dir, _ensure_plot_config, _extract_coords_and_times


def cohospace(
    coords: np.ndarray | dict[str, Any],
    spikes: np.ndarray,
    *,
    times: np.ndarray | None = None,
    coords_key: str | None = None,
    bins: int = 51,
    coords_in_unit: bool = False,
    smooth_sigma: float = 0.0,
) -> dict[str, Any]:
    """
    Compute EcohoSpace rate maps and phase centers.

    Mirrors GridCellTorus get_ratemaps: mean activity in coho-space bins and
    a circular-mean center for each neuron. Optionally smooths the rate maps.
    """
    if isinstance(coords, dict):
        coords, times_box = _extract_coords_and_times(coords, coords_key)
        if times is None:
            times = times_box

    coords = np.asarray(coords, float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must be (N,2+) array, got {coords.shape}")

    if coords_in_unit:
        coords = coords * (2 * np.pi)

    spikes = np.asarray(spikes)
    if times is not None:
        spikes = spikes[np.asarray(times).astype(int)]

    if spikes.ndim == 1:
        spikes = spikes[:, np.newaxis]

    if spikes.shape[0] != coords.shape[0]:
        raise ValueError(
            f"spikes length must match coords length. Got {spikes.shape[0]} vs {coords.shape[0]}"
        )

    edges = np.linspace(0, 2 * np.pi, bins)
    bin_centers = edges[:-1] + (edges[1:] - edges[:-1]) / 2.0
    xv, yv = np.meshgrid(bin_centers, bin_centers)
    pos = np.stack([xv.ravel(), yv.ravel()], axis=1)
    ccos = np.cos(pos)
    csin = np.sin(pos)

    num_neurons = spikes.shape[1]
    maps = np.zeros((num_neurons, bins - 1, bins - 1))
    centers = np.zeros((num_neurons, 2))

    for n in range(num_neurons):
        mtot_tmp, x_edge, y_edge, _ = binned_statistic_2d(
            coords[:, 0],
            coords[:, 1],
            spikes[:, n],
            statistic="mean",
            bins=edges,
            range=None,
            expand_binnumbers=True,
        )
        mtot_tmp = np.rot90(mtot_tmp, 1).T
        if smooth_sigma and smooth_sigma > 0:
            nan_mask = np.isnan(mtot_tmp)
            mtot_tmp = np.nan_to_num(mtot_tmp, nan=0.0)
            mtot_tmp = gaussian_filter(mtot_tmp, smooth_sigma)
            mtot_tmp[nan_mask] = np.nan
        maps[n, :, :] = mtot_tmp.copy()

        flat = mtot_tmp.flatten()
        nans = ~np.isnan(flat)
        if np.any(nans):
            centcos = np.sum(ccos[nans, :] * flat[nans, np.newaxis], axis=0)
            centsin = np.sum(csin[nans, :] * flat[nans, np.newaxis], axis=0)
            centers[n, :] = np.arctan2(centsin, centcos) % (2 * np.pi)
        else:
            centers[n, :] = np.nan

    return {
        "rate_maps": maps,
        "centers": centers,
        "x_edge": x_edge,
        "y_edge": y_edge,
        "bins": bins,
        "smooth_sigma": smooth_sigma,
    }


def plot_cohospace(
    cohospace_result: dict[str, Any],
    *,
    neuron_id: int = 0,
    config: PlotConfig | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (5, 5),
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot a single-neuron EcohoSpace rate map.
    """
    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title="EcohoSpace",
        xlabel="",
        ylabel="",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    maps = cohospace_result["rate_maps"]
    x_edge = cohospace_result["x_edge"]
    y_edge = cohospace_result["y_edge"]

    if neuron_id < 0 or neuron_id >= maps.shape[0]:
        raise ValueError(f"neuron_id out of range: {neuron_id}")

    fig, ax = plt.subplots(1, 1, figsize=config.figsize)
    im = ax.imshow(
        maps[neuron_id],
        origin="lower",
        extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
        cmap=cmap,
    )
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"EcohoSpace Rate Map (neuron {neuron_id})", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean activity")

    fig.tight_layout()
    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return fig


def plot_cohospace_skewed(
    cohospace_result: dict[str, Any],
    *,
    neuron_id: int = 0,
    config: PlotConfig | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (5, 5),
    cmap: str = "viridis",
    show_grid: bool = True,
) -> plt.Figure:
    """
    Plot a single-neuron EcohoSpace rate map in skewed torus coordinates.
    """
    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title="EcohoSpace (Skewed)",
        xlabel=r"$\theta_1 + \frac{1}{2}\theta_2$",
        ylabel=r"$\frac{\sqrt{3}}{2}\theta_2$",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    maps = cohospace_result["rate_maps"]
    x_edge = cohospace_result["x_edge"]
    y_edge = cohospace_result["y_edge"]

    if neuron_id < 0 or neuron_id >= maps.shape[0]:
        raise ValueError(f"neuron_id out of range: {neuron_id}")

    th1, th2 = np.meshgrid(x_edge, y_edge, indexing="xy")
    X = th1 + 0.5 * th2
    Y = (np.sqrt(3) / 2.0) * th2

    fig, ax = plt.subplots(1, 1, figsize=config.figsize)
    im = ax.pcolormesh(X, Y, maps[neuron_id], shading="auto", cmap=cmap)

    if show_grid:
        e1 = np.array([2 * np.pi, 0.0])
        e2 = np.array([np.pi, np.sqrt(3) * np.pi])
        poly = np.vstack([np.zeros(2), e1, e1 + e2, e2, np.zeros(2)])
        ax.plot(poly[:, 0], poly[:, 1], lw=1.1, color="0.35")

    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"EcohoSpace Rate Map (skewed, neuron {neuron_id})", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean activity")

    corners = np.vstack(
        [
            [0.0, 0.0],
            [2 * np.pi, 0.0],
            [np.pi, np.sqrt(3) * np.pi],
            [3 * np.pi, np.sqrt(3) * np.pi],
        ]
    )
    xmin, ymin = corners.min(axis=0)
    xmax, ymax = corners.max(axis=0)
    padx = 0.02 * (xmax - xmin)
    pady = 0.02 * (ymax - ymin)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)

    fig.tight_layout()
    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return fig
