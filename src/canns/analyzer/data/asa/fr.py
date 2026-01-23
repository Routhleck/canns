from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from ...visualization.core import PlotConfig, finalize_figure


def _ensure_parent_dir(save_path: str | None) -> None:
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

def _slice_range(r: Optional[Tuple[int, int]], length: int) -> slice:
    """Convert (start, end) into a safe Python slice within [0, length]."""
    if r is None:
        return slice(0, length)
    s, e = r
    s = 0 if s is None else int(s)
    e = length if e is None else int(e)
    s = max(0, min(length, s))
    e = max(0, min(length, e))
    if e < s:
        e = s
    return slice(s, e)


def compute_fr_heatmap_matrix(
    spike: np.ndarray,
    neuron_range: Optional[Tuple[int, int]] = None,
    time_range: Optional[Tuple[int, int]] = None,
    *,
    transpose: bool = True,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute a matrix for FR heatmap display from spike-like data.

    Parameters
    ----------
    spike : np.ndarray
        Shape (T, N). Can be continuous (float) or binned (int/float).
    neuron_range : (start, end) or None
        Neuron index range in [0, N]. End is exclusive.
    time_range : (start, end) or None
        Time index range in [0, T]. End is exclusive.
    transpose : bool
        If True, returns (N_sel, T_sel) which is convenient for imshow with
        neurons on Y and time on X (like your utils did with spike.T).
        If False, returns (T_sel, N_sel).
    normalize : {'zscore_per_neuron','minmax_per_neuron', None}
        Optional display normalization along time for each neuron.

    Returns
    -------
    M : np.ndarray
        Heatmap matrix. Default shape (N_sel, T_sel) if transpose=True.
    """
    spike = np.asarray(spike)
    if spike.ndim != 2:
        raise ValueError(f"spike must be 2D (T,N), got shape={spike.shape}")

    T, N = spike.shape
    t_sl = _slice_range(time_range, T)
    n_sl = _slice_range(neuron_range, N)

    sub = spike[t_sl, n_sl]  # (T_sel, N_sel)

    # normalization is only for display; does NOT change any downstream indices
    if normalize is not None:
        X = sub.astype(float, copy=False)
        if normalize == "zscore_per_neuron":
            mu = np.mean(X, axis=0, keepdims=True)
            sd = np.std(X, axis=0, keepdims=True)
            sd = np.where(sd == 0, 1.0, sd)
            sub = (X - mu) / sd
        elif normalize == "minmax_per_neuron":
            mn = np.min(X, axis=0, keepdims=True)
            mx = np.max(X, axis=0, keepdims=True)
            den = np.where((mx - mn) == 0, 1.0, (mx - mn))
            sub = (X - mn) / den
        else:
            raise ValueError(f"Unknown normalize={normalize!r}")

    return sub.T if transpose else sub


def save_fr_heatmap_png(
    M: np.ndarray,
    *,
    title: str = "Firing Rate Heatmap",
    xlabel: str = "Time",
    ylabel: str = "Neuron",
    cmap: str | None = None,
    interpolation: str | None = "nearest",
    origin: str | None = "lower",
    aspect: str | None = "auto",
    clabel: str | None = None,
    colorbar: bool = True,
    dpi: int = 200,
    show: bool | None = None,
    config: PlotConfig | None = None,
    **kwargs,
) -> None:
    """
    Save a heatmap PNG from a matrix (typically output of compute_fr_heatmap_matrix).

    Notes
    -----
    - Does not reorder neurons.
    - Uses matplotlib only here (ASA core stays compute-friendly).
    """
    import matplotlib.pyplot as plt  # local import to keep ASA light

    save_path = kwargs.pop("save_path", None)

    if config is None:
        show_val = False if show is None else show
        config = PlotConfig.for_static_plot(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=str(save_path) if save_path is not None else None,
            show=show_val,
        )
    else:
        if save_path is not None:
            config.save_path = str(save_path)
        if show is not None:
            config.show = show
        if not config.title:
            config.title = title
        if not config.xlabel:
            config.xlabel = xlabel
        if not config.ylabel:
            config.ylabel = ylabel

    if config.save_path is None:
        raise ValueError("save_path must be provided via config.save_path or as a keyword argument.")

    config.save_dpi = dpi

    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError(f"M must be 2D for heatmap display, got shape={M.shape}")
    fig, ax = plt.subplots(figsize=config.figsize)
    plot_kwargs = config.to_matplotlib_kwargs()
    if cmap is not None and "cmap" not in plot_kwargs:
        plot_kwargs["cmap"] = cmap
    if interpolation is not None and "interpolation" not in plot_kwargs:
        plot_kwargs["interpolation"] = interpolation
    if origin is not None and "origin" not in plot_kwargs:
        plot_kwargs["origin"] = origin
    if aspect is not None and "aspect" not in plot_kwargs:
        plot_kwargs["aspect"] = aspect
    if kwargs:
        plot_kwargs.update(kwargs)

    im = ax.imshow(M, **plot_kwargs)
    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    if colorbar:
        label = clabel if clabel is not None else config.clabel
        fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)


@dataclass
class FRMResult:
    frm: np.ndarray
    occupancy: np.ndarray
    spike_sum: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray


def compute_frm(
    spike: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    neuron_id: int,
    *,
    bins: int = 50,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    min_occupancy: int = 1,
    smoothing: bool = False,
    sigma: float = 1.0,
    nan_for_empty: bool = True,
) -> FRMResult:
    """
    Compute a single-neuron firing rate map (FRM) on 2D space.

    Parameters
    ----------
    spike : np.ndarray
        Shape (T, N). Can be continuous (float) or binned counts (int/float).
    x, y : np.ndarray
        Shape (T,). Position samples aligned with spike rows.
    neuron_id : int
        Neuron index in [0, N).
    bins : int
        Number of spatial bins per dimension.
    x_range, y_range : (min, max) or None
        Explicit ranges. If None, uses data min/max.
    min_occupancy : int
        Bins with occupancy < min_occupancy are treated as empty.
    smoothing : bool
        If True, apply Gaussian smoothing to frm (and optionally to occupancy/spike_sum if you want later).
    sigma : float
        Gaussian sigma for smoothing (in bin units).
    nan_for_empty : bool
        If True, empty bins become NaN; else 0.

    Returns
    -------
    FRMResult
        frm: 2D array (bins_x, bins_y) in Hz-like units per sample (relative scale).
    """
    spike = np.asarray(spike)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if spike.ndim != 2:
        raise ValueError(f"spike must be 2D (T,N), got shape={spike.shape}")
    T, N = spike.shape
    if len(x) != T or len(y) != T:
        raise ValueError(f"x/y length must match spike rows T={T}, got len(x)={len(x)}, len(y)={len(y)}")
    if not (0 <= int(neuron_id) < N):
        raise ValueError(f"neuron_id out of range: {neuron_id} for N={N}")

    fr = spike[:, int(neuron_id)].astype(float, copy=False)

    # ranges
    if x_range is None:
        x_min, x_max = float(np.min(x)), float(np.max(x))
    else:
        x_min, x_max = float(x_range[0]), float(x_range[1])

    if y_range is None:
        y_min, y_max = float(np.min(y)), float(np.max(y))
    else:
        y_min, y_max = float(y_range[0]), float(y_range[1])

    # Edges (bins+1)
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)

    # Bin indices
    xi = np.searchsorted(x_edges, x, side="right") - 1
    yi = np.searchsorted(y_edges, y, side="right") - 1

    # Keep only points inside range
    valid = (xi >= 0) & (xi < bins) & (yi >= 0) & (yi < bins)
    xi = xi[valid]
    yi = yi[valid]
    frv = fr[valid]

    occupancy = np.zeros((bins, bins), dtype=np.int64)
    spike_sum = np.zeros((bins, bins), dtype=np.float64)

    # accumulate
    # (Use np.add.at to avoid Python loops)
    np.add.at(occupancy, (xi, yi), 1)
    np.add.at(spike_sum, (xi, yi), frv)

    # rate = sum / occupancy
    with np.errstate(divide="ignore", invalid="ignore"):
        frm = spike_sum / occupancy

    # empty bins handling
    empty = occupancy < int(min_occupancy)
    if nan_for_empty:
        frm = frm.astype(np.float64, copy=False)
        frm[empty] = np.nan
    else:
        frm = np.where(empty, 0.0, frm)

    if smoothing:
        try:
            from scipy.ndimage import gaussian_filter
            # Smooth while respecting NaNs (simple approach: fill NaN -> 0, smooth weights)
            if np.any(np.isnan(frm)):
                val = np.nan_to_num(frm, nan=0.0)
                w = (~np.isnan(frm)).astype(np.float64)
                val_s = gaussian_filter(val, sigma=float(sigma))
                w_s = gaussian_filter(w, sigma=float(sigma))
                frm = np.divide(val_s, w_s, out=np.full_like(val_s, np.nan), where=(w_s > 1e-12))
            else:
                frm = gaussian_filter(frm, sigma=float(sigma))
        except Exception:
            # If scipy not available, just skip smoothing
            pass

    return FRMResult(frm=frm, occupancy=occupancy, spike_sum=spike_sum, x_edges=x_edges, y_edges=y_edges)


def plot_frm(
    frm: np.ndarray,
    *,
    title: str = "Firing Rate Map",
    dpi: int = 200,
    show: bool | None = None,
    config: PlotConfig | None = None,
    **kwargs,
) -> None:
    """
    Save FRM as PNG. Expects frm as 2D array (bins,bins).
    """
    from ...visualization import plot_firing_field_heatmap

    save_path = kwargs.pop("save_path", None)

    if config is None:
        show_val = False if show is None else show
        config = PlotConfig.for_static_plot(
            title=title,
            xlabel="X bin",
            ylabel="Y bin",
            save_path=str(save_path) if save_path is not None else None,
            show=show_val,
        )
    else:
        if save_path is not None:
            config.save_path = str(save_path)
        if show is not None:
            config.show = show
        if not config.title:
            config.title = title
        if not config.xlabel:
            config.xlabel = "X bin"
        if not config.ylabel:
            config.ylabel = "Y bin"

    if config.save_path is None:
        raise ValueError("save_path must be provided via config.save_path or as a keyword argument.")

    config.save_dpi = dpi

    frm = np.asarray(frm)
    plot_firing_field_heatmap(
        frm,
        config=config,
        origin="lower",
        **kwargs,
    )
