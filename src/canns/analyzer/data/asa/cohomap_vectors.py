"""Stripe vectors and diagnostics for CohoMap."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

from ...visualization.core import PlotConfig, finalize_figure
from .cohomap import fit_cohomap_stripes
from .utils import _ensure_parent_dir, _ensure_plot_config


def _rot_para(params1: np.ndarray, params2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform stripe fit parameters to canonical orientation.

    This function adjusts the orientation angles of stripe fit parameters to
    align them with a canonical coordinate system. Unlike `_rot_coord()` which
    transforms actual coordinate data, this operates on the fit parameters
    themselves (orientation angles, wavelengths, etc.).

    Parameters
    ----------
    params1 : np.ndarray
        Stripe fit parameters for first dimension. First element is orientation angle.
    params2 : np.ndarray
        Stripe fit parameters for second dimension. First element is orientation angle.

    Returns
    -------
    x : np.ndarray
        Transformed parameters for the dimension with stronger horizontal alignment.
    y : np.ndarray
        Transformed parameters for the other dimension.

    Notes
    -----
    The function selects which parameter set becomes 'x' and 'y' based on which
    has larger |cos(angle)|, then applies angle adjustments to ensure the stripe
    vectors are in a canonical orientation for visualization and analysis.
    """
    if abs(np.cos(params1[0])) < abs(np.cos(params2[0])):
        y = params1.copy()
        x = params2.copy()
    else:
        x = params1.copy()
        y = params2.copy()

    alpha = y[0] - x[0]
    if (alpha < 0) and (abs(alpha) > np.pi / 2):
        x[0] += np.pi
        x[0] = x[0] % (2 * np.pi)
    elif (alpha < 0) and (abs(alpha) < np.pi / 2):
        x[0] += np.pi * 4 / 3
        x[0] = x[0] % (2 * np.pi)
    elif abs(alpha) > np.pi / 2:
        y[0] -= np.pi / 3
        y[0] = y[0] % (2 * np.pi)

    if y[0] > np.pi / 2:
        y[0] -= np.pi / 6
        x[0] -= np.pi / 6
        x[0] = x[0] % (2 * np.pi)
        y[0] = y[0] % (2 * np.pi)
    if x[0] > np.pi / 2:
        y[0] += np.pi / 6
        x[0] += np.pi / 6
        x[0] = x[0] % (2 * np.pi)
        y[0] = y[0] % (2 * np.pi)

    return x, y


def cohomap_vectors(
    cohomap_result: dict[str, Any],
    *,
    grid_size: int | None = 151,
    trim: int = 25,
    angle_grid: int = 10,
    phase_grid: int = 10,
    spacing_grid: int = 10,
    spacing_range: tuple[float, float] = (1.0, 6.0),
) -> dict[str, Any]:
    """
    Fit CohoMap stripe parameters and compute parallelogram vectors (v, w).

    Returns a dict containing the stripe fit, rotated parameters, vector components,
    and angle (deg) following GridCellTorus conventions.
    """
    phase_map1 = np.asarray(cohomap_result["phase_map1"])
    phase_map2 = np.asarray(cohomap_result["phase_map2"])
    x_edge = np.asarray(cohomap_result["x_edge"])
    y_edge = np.asarray(cohomap_result["y_edge"])

    p1, f1 = fit_cohomap_stripes(
        phase_map1,
        grid_size=grid_size,
        trim=trim,
        angle_grid=angle_grid,
        phase_grid=phase_grid,
        spacing_grid=spacing_grid,
        spacing_range=spacing_range,
    )
    p2, f2 = fit_cohomap_stripes(
        phase_map2,
        grid_size=grid_size,
        trim=trim,
        angle_grid=angle_grid,
        phase_grid=phase_grid,
        spacing_grid=spacing_grid,
        spacing_range=spacing_range,
    )

    x_params, y_params = _rot_para(p1, p2)

    x_edge_shift = x_edge - float(x_edge.min())
    y_edge_shift = y_edge - float(y_edge.min())
    xmax = float(x_edge_shift.max())
    ymax = float(y_edge_shift.max())

    v = np.array(
        [
            (1 / x_params[2]) * np.cos(x_params[0]) * xmax,
            (1 / x_params[2]) * np.sin(x_params[0]) * xmax,
        ],
        dtype=float,
    )
    w = np.array(
        [
            (1 / y_params[2]) * np.cos(y_params[0]) * ymax,
            (1 / y_params[2]) * np.sin(y_params[0]) * ymax,
        ],
        dtype=float,
    )

    angle_deg = float(((y_params[0] - x_params[0]) / (2 * np.pi) * 360.0) % 360.0)

    return {
        "params1": p1,
        "params2": p2,
        "fit_error1": float(f1),
        "fit_error2": float(f2),
        "x_params": x_params,
        "y_params": y_params,
        "x_edge": x_edge,
        "y_edge": y_edge,
        "x_range": xmax,
        "y_range": ymax,
        "v": v,
        "w": w,
        "len_v": float(1 / x_params[2] * xmax),
        "len_w": float(1 / y_params[2] * ymax),
        "angle_deg": angle_deg,
        "grid_size": grid_size,
        "trim": trim,
    }


def plot_cohomap_vectors(
    cohomap_vectors_result: dict[str, Any],
    *,
    config: PlotConfig | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (5, 5),
    color: str = "#f28e2b",
) -> plt.Figure:
    """
    Plot v/w vectors and the parallelogram in spatial coordinates.
    """
    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title="CohoMap Vectors",
        xlabel="",
        ylabel="",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    vmax = float(cohomap_vectors_result["x_range"])
    wmax = float(cohomap_vectors_result["y_range"])
    v = np.asarray(cohomap_vectors_result["v"], dtype=float)
    w = np.asarray(cohomap_vectors_result["w"], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=config.figsize)

    ax.plot([0, 0], [0, wmax], "--", color="0.6", lw=1)
    ax.plot([vmax, vmax], [0, wmax], "--", color="0.6", lw=1)
    ax.plot([0, vmax], [0, 0], "--", color="0.6", lw=1)
    ax.plot([0, vmax], [wmax, wmax], "--", color="0.6", lw=1)

    ax.plot([0, v[0]], [0, v[1]], color=color, lw=3)
    ax.plot([0, w[0]], [0, w[1]], color=color, lw=3)
    ax.plot([v[0], v[0] + w[0]], [v[1], v[1] + w[1]], color=color, lw=3)
    ax.plot([w[0], v[0] + w[0]], [w[1], v[1] + w[1]], color=color, lw=3)

    pad_x = 0.05 * vmax if vmax > 0 else 1.0
    pad_y = 0.05 * wmax if wmax > 0 else 1.0
    ax.set_xlim(-pad_x, vmax + pad_x)
    ax.set_ylim(-pad_y, wmax + pad_y)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return fig


def _resolve_grid_size(phase_map: np.ndarray, grid_size: int | None, trim: int) -> int:
    """Determine grid size for stripe fitting.

    Parameters
    ----------
    phase_map : np.ndarray
        Phase map array (2D grid).
    grid_size : int, optional
        Explicit grid size. If None, inferred from phase_map shape.
    trim : int
        Number of edge bins to trim.

    Returns
    -------
    int
        Grid size to use for stripe fitting.
    """
    if grid_size is None:
        return int(phase_map.shape[0]) + 2 * trim + 1
    return int(grid_size)


def _stripe_fit_map(params: np.ndarray, grid_size: int, trim: int) -> np.ndarray:
    """Generate a synthetic stripe pattern from fit parameters.

    Creates a 2D cosine stripe pattern based on the fitted parameters (orientation,
    phase, and spatial frequency). Used for visualizing the fitted stripe model
    and comparing it with the actual phase map.

    Parameters
    ----------
    params : np.ndarray
        Stripe fit parameters: [angle, phase, frequency].
        - angle: Orientation angle in radians
        - phase: Phase offset in radians
        - frequency: Spatial frequency (inverse wavelength)
    grid_size : int
        Size of the grid to generate (before trimming).
    trim : int
        Number of edge bins to trim from the generated pattern.

    Returns
    -------
    np.ndarray
        2D array of shape (grid_size-2*trim, grid_size-2*trim) containing
        the cosine stripe pattern with values in [-1, 1].

    Notes
    -----
    The pattern is generated on a [0, 3π] × [0, 3π] domain to cover the
    extended torus space, then rotated by the fitted angle and trimmed.
    """
    numangsint = grid_size
    x, _ = np.meshgrid(
        np.linspace(0, 3 * np.pi, numangsint - 1),
        np.linspace(0, 3 * np.pi, numangsint - 1),
    )
    x_rot = rotate(x, params[0] * 360.0 / (2 * np.pi), reshape=False)
    return np.cos(params[2] * x_rot[trim:-trim, trim:-trim] + params[1])


def plot_cohomap_stripes(
    cohomap_result: dict[str, Any],
    *,
    cohomap_vectors_result: dict[str, Any] | None = None,
    grid_size: int | None = 151,
    trim: int = 25,
    angle_grid: int = 10,
    phase_grid: int = 10,
    spacing_grid: int = 10,
    spacing_range: tuple[float, float] = (1.0, 6.0),
    config: PlotConfig | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (10, 6),
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot stripe fit diagnostics for CohoMap (observed vs fitted stripes).
    """
    if cohomap_vectors_result is None:
        cohomap_vectors_result = cohomap_vectors(
            cohomap_result,
            grid_size=grid_size,
            trim=trim,
            angle_grid=angle_grid,
            phase_grid=phase_grid,
            spacing_grid=spacing_grid,
            spacing_range=spacing_range,
        )

    if "grid_size" in cohomap_vectors_result:
        grid_size = cohomap_vectors_result["grid_size"]
    if "trim" in cohomap_vectors_result:
        trim = cohomap_vectors_result["trim"]

    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title="CohoMap Stripe Fit",
        xlabel="",
        ylabel="",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    phase_map1 = np.asarray(cohomap_result["phase_map1"])
    phase_map2 = np.asarray(cohomap_result["phase_map2"])
    x_edge = np.asarray(cohomap_result["x_edge"])
    y_edge = np.asarray(cohomap_result["y_edge"])

    grid_size = _resolve_grid_size(phase_map1, grid_size, trim)
    expected = grid_size - 1 - 2 * trim
    if phase_map1.shape[0] != expected or phase_map2.shape[0] != expected:
        raise ValueError("phase_map shape does not match grid_size/trim")

    p1 = np.asarray(cohomap_vectors_result["params1"])
    p2 = np.asarray(cohomap_vectors_result["params2"])
    fit1 = _stripe_fit_map(p1, grid_size, trim)
    fit2 = _stripe_fit_map(p2, grid_size, trim)

    obs1 = np.cos(phase_map1)
    obs2 = np.cos(phase_map2)

    fig, axes = plt.subplots(2, 2, figsize=config.figsize)
    panels = [
        (obs1, "Phase Map 1 (cos)"),
        (fit1, "Stripe Fit 1"),
        (obs2, "Phase Map 2 (cos)"),
        (fit2, "Stripe Fit 2"),
    ]

    for ax, (img, title) in zip(axes.flat, panels, strict=True):
        ax.imshow(
            img,
            origin="lower",
            extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return fig
