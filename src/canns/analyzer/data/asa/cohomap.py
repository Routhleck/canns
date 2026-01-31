"""CohoMap (Ecoho-style) computation and plotting."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from scipy.optimize import minimize
from scipy.stats import binned_statistic_2d

from ...visualization.core import PlotConfig, finalize_figure
from .coho import (
    _circmean,
    _ensure_parent_dir,
    _ensure_plot_config,
    _extract_coords_and_times,
    _phase_map_valid_fraction,
    _smooth_circular_map,
)
from .path import parse_times_box_to_indices


def _select_phase_sign(
    phase_map: np.ndarray,
    params: np.ndarray,
    *,
    grid_size: int,
    trim: int,
) -> int:
    mtot = np.asarray(phase_map)
    expected = grid_size - 1 - 2 * trim
    if mtot.ndim != 2 or mtot.shape[0] != expected or mtot.shape[1] != expected:
        raise ValueError("phase_map shape does not match grid_size/trim for alignment")
    nnans = ~np.isnan(mtot)
    if not np.any(nnans):
        return 1
    x, _ = np.meshgrid(
        np.linspace(0, 3 * np.pi, grid_size - 1),
        np.linspace(0, 3 * np.pi, grid_size - 1),
    )
    x1 = rotate(x, params[0] * 360.0 / (2 * np.pi), reshape=False)
    base = params[2] * x1[trim:-trim, trim:-trim] + params[1]
    mtot_vals = (mtot[nnans]) % (2 * np.pi)
    pm1 = (base % (2 * np.pi))[nnans] - mtot_vals
    pm2 = ((2 * np.pi - base) % (2 * np.pi))[nnans] - mtot_vals
    if np.sum(np.abs(pm1)) > np.sum(np.abs(pm2)):
        return -1
    return 1


def _rot_coord(
    params1: np.ndarray,
    params2: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    p: tuple[int, int],
) -> np.ndarray:
    if abs(np.cos(params1[0])) < abs(np.cos(params2[0])):
        cc1 = c2.copy()
        cc2 = c1.copy()
        y = params1.copy()
        x = params2.copy()
        p = (p[1], p[0])
    else:
        cc1 = c1.copy()
        cc2 = c2.copy()
        x = params1.copy()
        y = params2.copy()

    if p[1] == -1:
        cc2 = 2 * np.pi - cc2
    if p[0] == -1:
        cc1 = 2 * np.pi - cc1

    alpha = y[0] - x[0]
    if (alpha < 0) and (abs(alpha) > np.pi / 2):
        cctmp = cc2.copy()
        cc2 = cc1.copy()
        cc1 = cctmp

    if (alpha < 0) and (abs(alpha) < np.pi / 2):
        cc1 = 2 * np.pi - cc1 + (np.pi / 3) * cc2
    elif abs(alpha) > np.pi / 2:
        cc2 = cc2 + (np.pi / 3) * cc1

    return np.stack([cc1, cc2], axis=1) % (2 * np.pi)


def _toroidal_align_coords(
    coords: np.ndarray,
    phase_map1: np.ndarray,
    phase_map2: np.ndarray,
    *,
    trim: int,
    grid_size: int | None,
) -> tuple[np.ndarray, float, float]:
    if phase_map1.shape != phase_map2.shape:
        raise ValueError("phase_map shapes do not match for alignment")
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must be (N,2+) array, got {coords.shape}")
    if grid_size is None:
        grid_size = int(phase_map1.shape[0]) + 2 * trim + 1
    p1, f1 = fit_cohomap_stripes(phase_map1, grid_size=grid_size, trim=trim)
    p2, f2 = fit_cohomap_stripes(phase_map2, grid_size=grid_size, trim=trim)
    s1 = _select_phase_sign(phase_map1, p1, grid_size=grid_size, trim=trim)
    s2 = _select_phase_sign(phase_map2, p2, grid_size=grid_size, trim=trim)
    aligned = _rot_coord(p1, p2, coords[:, 0], coords[:, 1], (s1, s2))
    return aligned, float(f1), float(f2)


def cohomap(
    decoding_result: dict[str, Any],
    position_data: dict[str, Any],
    *,
    coords_key: str | None = None,
    bins: int = 101,
    margin_frac: float = 0.0025,
    smooth_sigma: float = 1.0,
    fill_nan: bool = True,
    fill_sigma: float | None = None,
    fill_min_weight: float = 1e-3,
    align_torus: bool = True,
    align_trim: int = 25,
    align_grid_size: int | None = None,
    align_min_valid_frac: float | None = None,
    align_max_fit_error: float | None = None,
) -> dict[str, Any]:
    """
    Compute EcohoMap phase maps using circular-mean binning.

    This mirrors GridCellTorus get_ang_hist: bin spatial positions and compute the
    circular mean of each decoded angle within spatial bins, then smooth in sin/cos
    space. Optional toroidal alignment follows the GridCellTorus stripe fit + rotation.
    You can gate alignment by valid fraction or fit error thresholds.
    """
    coords, times_box = _extract_coords_and_times(decoding_result, coords_key)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"coords must be (N,2+) array, got {coords.shape}")

    xx = np.asarray(position_data["x"])
    yy = np.asarray(position_data["y"])

    if times_box is not None:
        if "t" in position_data:
            idx, _ = parse_times_box_to_indices(times_box, np.asarray(position_data["t"]))
            xx = xx[idx]
            yy = yy[idx]
        else:
            idx = np.asarray(times_box).astype(int)
            xx = xx[idx]
            yy = yy[idx]

    if len(xx) != coords.shape[0]:
        raise ValueError(
            "Length mismatch: coords length does not match position length after times_box."
        )

    x_min, x_max = float(np.min(xx)), float(np.max(xx))
    y_min, y_max = float(np.min(yy)), float(np.max(yy))
    x_pad = (x_max - x_min) * margin_frac
    y_pad = (y_max - y_min) * margin_frac

    binsx = np.linspace(x_min + x_pad, x_max - x_pad, bins)
    binsy = np.linspace(y_min + y_pad, y_max - y_pad, bins)

    def _angle_hist(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nnans = ~np.isnan(values)
        mtot, x_edge, y_edge, _ = binned_statistic_2d(
            xx[nnans],
            yy[nnans],
            values[nnans],
            statistic=_circmean,
            bins=(binsx, binsy),
            range=None,
            expand_binnumbers=True,
        )
        mtot = _smooth_circular_map(
            mtot,
            smooth_sigma,
            fill_nan=fill_nan,
            fill_sigma=fill_sigma,
            fill_min_weight=fill_min_weight,
        )
        return mtot, x_edge, y_edge

    coords_use = np.asarray(coords, float)
    m1_raw, x_edge, y_edge = _angle_hist(coords_use[:, 0])
    m2_raw, _, _ = _angle_hist(coords_use[:, 1])
    aligned = False
    align_error = None
    align_valid_frac1 = None
    align_valid_frac2 = None
    align_fit_error1 = None
    align_fit_error2 = None

    if align_torus:
        try:
            align_valid_frac1 = _phase_map_valid_fraction(m1_raw)
            align_valid_frac2 = _phase_map_valid_fraction(m2_raw)
            min_valid = min(align_valid_frac1, align_valid_frac2)

            if align_min_valid_frac is not None and min_valid < align_min_valid_frac:
                align_error = (
                    f"valid fraction too low ({min_valid:.3f} < {align_min_valid_frac:.3f})"
                )
            else:
                coords_aligned, f1, f2 = _toroidal_align_coords(
                    coords_use[:, :2],
                    m1_raw,
                    m2_raw,
                    trim=align_trim,
                    grid_size=align_grid_size,
                )
                align_fit_error1 = f1
                align_fit_error2 = f2
                if align_max_fit_error is not None and (
                    f1 > align_max_fit_error or f2 > align_max_fit_error
                ):
                    align_error = (
                        f"fit error too high ({f1:.4f}, {f2:.4f} > {align_max_fit_error:.4f})"
                    )
                else:
                    coords_use = coords_use.copy()
                    coords_use[:, :2] = coords_aligned
                    aligned = True
        except Exception as exc:
            align_error = str(exc)

    if aligned:
        m1, x_edge, y_edge = _angle_hist(coords_use[:, 0])
        m2, _, _ = _angle_hist(coords_use[:, 1])
    else:
        m1, m2 = m1_raw, m2_raw

    return {
        "phase_map1": m1,
        "phase_map2": m2,
        "phase_map1_raw": m1_raw,
        "phase_map2_raw": m2_raw,
        "x_edge": x_edge,
        "y_edge": y_edge,
        "bins": bins,
        "margin_frac": margin_frac,
        "smooth_sigma": smooth_sigma,
        "fill_nan": fill_nan,
        "fill_sigma": fill_sigma,
        "fill_min_weight": fill_min_weight,
        "aligned": aligned,
        "align_error": align_error,
        "align_min_valid_frac": align_min_valid_frac,
        "align_max_fit_error": align_max_fit_error,
        "align_valid_frac1": align_valid_frac1,
        "align_valid_frac2": align_valid_frac2,
        "align_fit_error1": align_fit_error1,
        "align_fit_error2": align_fit_error2,
    }


def fit_cohomap_stripes(
    phase_map: np.ndarray,
    *,
    grid_size: int | None = 151,
    trim: int = 25,
    angle_grid: int = 10,
    phase_grid: int = 10,
    spacing_grid: int = 10,
    spacing_range: tuple[float, float] = (1.0, 6.0),
) -> tuple[np.ndarray, float]:
    """
    Fit a cosine stripe model to a phase map, mirroring GridCellTorus fit_sine_wave.
    """
    mtot = np.asarray(phase_map)
    if mtot.ndim != 2:
        raise ValueError(f"phase_map must be 2D, got {mtot.shape}")

    if grid_size is None:
        grid_size = mtot.shape[0] + 2 * trim + 1

    expected = grid_size - 1 - 2 * trim
    if expected != mtot.shape[0]:
        raise ValueError(
            f"grid_size/trim incompatible with phase_map shape: "
            f"expected {expected} but got {mtot.shape[0]}"
        )

    numangsint = grid_size
    x, _ = np.meshgrid(
        np.linspace(0, 3 * np.pi, numangsint - 1),
        np.linspace(0, 3 * np.pi, numangsint - 1),
    )
    nnans = ~np.isnan(mtot)

    def cos_wave(p: np.ndarray) -> float:
        x1 = rotate(x, p[0] * 360.0 / (2 * np.pi), reshape=False)
        model = np.cos(p[2] * x1[trim:-trim, trim:-trim] + p[1])
        return float(np.mean(np.square(model[nnans] - np.cos(mtot[nnans]))))

    angle_space = np.linspace(0, np.pi, angle_grid)
    phase_space = np.linspace(0, 2 * np.pi, phase_grid)
    spacing_space = np.linspace(spacing_range[0], spacing_range[1], spacing_grid)

    grid = np.zeros((angle_grid, phase_grid, spacing_grid))
    for i, ang in enumerate(angle_space):
        for j, ph in enumerate(phase_space):
            for k, sp in enumerate(spacing_space):
                grid[i, j, k] = cos_wave(np.array([ang, ph, sp]))

    p_ind = np.unravel_index(np.argmin(grid), grid.shape)
    p0 = np.array([angle_space[p_ind[0]], phase_space[p_ind[1]], spacing_space[p_ind[2]]])
    res = minimize(cos_wave, p0, method="SLSQP", options={"disp": False})
    return res["x"], float(res["fun"])


def plot_cohomap(
    cohomap_result: dict[str, Any],
    *,
    config: PlotConfig | None = None,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (10, 4),
    cmap: str = "viridis",
    mode: str = "cos",
) -> plt.Figure:
    """
    Plot EcohoMap phase maps (two panels: phase_map1/phase_map2).

    mode:
        "phase" to show raw phase (radians),
        "cos" or "sin" to show cosine/sine of phase like GridCellTorus.
    """
    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title="EcohoMap",
        xlabel="",
        ylabel="",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    m1 = cohomap_result["phase_map1"]
    m2 = cohomap_result["phase_map2"]
    x_edge = cohomap_result["x_edge"]
    y_edge = cohomap_result["y_edge"]

    fig, ax = plt.subplots(1, 2, figsize=config.figsize)
    for i, (mtot, title) in enumerate(((m1, "Phase Map 1"), (m2, "Phase Map 2"))):
        if mode == "phase":
            plot_map = mtot
            cbar_label = "Phase (rad)"
            vmin, vmax = -np.pi, np.pi
        elif mode == "cos":
            plot_map = np.cos(mtot)
            cbar_label = "cos(phase)"
            vmin, vmax = -1.0, 1.0
        elif mode == "sin":
            plot_map = np.sin(mtot)
            cbar_label = "sin(phase)"
            vmin, vmax = -1.0, 1.0
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'phase', 'cos', or 'sin'.")
        im = ax[i].imshow(
            plot_map,
            origin="lower",
            extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax[i].set_title(title, fontsize=10)
        ax[i].set_aspect("equal", "box")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04, label=cbar_label)

    fig.tight_layout()
    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return fig


def cohomap_upgrade(*args, **kwargs) -> dict[str, Any]:
    """Legacy alias for EcohoMap (formerly cohomap_upgrade)."""
    return cohomap(*args, **kwargs)


def ecohomap(*args, **kwargs) -> dict[str, Any]:
    """Alias for EcohoMap (GridCellTorus-style)."""
    return cohomap(*args, **kwargs)


def fit_cohomap_stripes_upgrade(*args, **kwargs) -> tuple[np.ndarray, float]:
    """Legacy alias for EcohoMap stripe fitting."""
    return fit_cohomap_stripes(*args, **kwargs)


def plot_cohomap_upgrade(*args, **kwargs) -> plt.Figure:
    """Legacy alias for EcohoMap plotting (formerly plot_cohomap_upgrade)."""
    return plot_cohomap(*args, **kwargs)


def plot_ecohomap(*args, **kwargs) -> plt.Figure:
    """Alias for EcohoMap plotting."""
    return plot_cohomap(*args, **kwargs)
