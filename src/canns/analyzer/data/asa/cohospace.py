from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import circvar

from ...visualization.core import PlotConfig, finalize_figure


def _ensure_plot_config(
    config: PlotConfig | None,
    factory,
    *,
    kwargs: dict | None = None,
    **defaults,
) -> PlotConfig:
    if config is None:
        defaults.update({"kwargs": kwargs or {}})
        return factory(**defaults)

    if kwargs:
        config_kwargs = config.kwargs or {}
        config_kwargs.update(kwargs)
        config.kwargs = config_kwargs
    return config


def _ensure_parent_dir(save_path: str | None) -> None:
    if save_path:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

# =====================================================================
# CohoSpace visualization and selectivity metrics (CohoScore)
# =====================================================================

def _coho_coords_to_degrees(coords: np.ndarray) -> np.ndarray:
    """
    Convert decoded coho coordinates (T x 2, radians) into degrees in [0, 360).
    """
    return np.degrees(coords % (2 * np.pi))


def _align_activity_to_coords(
    coords: np.ndarray,
    activity: np.ndarray,
    times: np.ndarray | None = None,
    *,
    label: str = "activity",
) -> np.ndarray:
    """
    Align activity to coords by optional time indices and validate lengths.
    """
    coords = np.asarray(coords)
    activity = np.asarray(activity)

    if times is not None:
        times = np.asarray(times)
        try:
            activity = activity[times]
        except Exception as exc:
            raise ValueError(
                f"Failed to index {label} with `times`. Ensure `times` indexes the original time axis."
            ) from exc

    if activity.shape[0] != coords.shape[0]:
        raise ValueError(
            f"{label} length must match coords length. Got {activity.shape[0]} vs {coords.shape[0]}. "
            "If coords are computed on a subset of timepoints (e.g., decode['times']), pass "
            "`times=decoding['times']` or slice the activity accordingly."
        )

    return activity


def plot_cohospace_trajectory(
    coords: np.ndarray,
    times: np.ndarray | None = None,
    subsample: int = 1,
    figsize: tuple[int, int] = (6, 6),
    cmap: str = "viridis",
    save_path: str | None = None,
    show: bool = False,
    config: PlotConfig | None = None,
) -> plt.Axes:
    """
    Plot a trajectory in cohomology space.

    Parameters
    ----------
    coords : ndarray, shape (T, 2)
        Decoded cohomology angles (theta1, theta2). Values may be in radians or in [0, 1] "unit circle"
        convention depending on upstream decoding; this function will convert to degrees for plotting.
    times : ndarray, optional, shape (T,)
        Optional time array used to color points. If None, uses arange(T).
    subsample : int
        Downsampling step (>1 reduces the number of plotted points).
    figsize : tuple
        Matplotlib figure size.
    cmap : str
        Matplotlib colormap name.
    save_path : str, optional
        If provided, saves the figure to this path.
    show : bool
        If True, calls plt.show(). If False, closes the figure and returns the Axes.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes containing the plot.
    """

    try:
        subsample_i = int(subsample)
    except Exception:
        subsample_i = 1
    if subsample_i < 1:
        subsample_i = 1

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"`coords` must have shape (T, 2). Got {coords.shape}.")

    theta_deg = _coho_coords_to_degrees(coords)
    if subsample_i > 1:
        theta_deg = theta_deg[::subsample_i]

    if times is None:
        times_vis = np.arange(theta_deg.shape[0])
    else:
        times_vis = np.asarray(times)
        if times_vis.shape[0] != coords.shape[0]:
            raise ValueError(
                f"`times` length must match coords length. Got times={times_vis.shape[0]}, coords={coords.shape[0]}."
            )
        if subsample_i > 1:
            times_vis = times_vis[::subsample_i]

    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title="CohoSpace trajectory",
        xlabel="theta1 (deg)",
        ylabel="theta2 (deg)",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    fig, ax = plt.subplots(figsize=config.figsize)
    sc = ax.scatter(
        theta_deg[:, 0],
        theta_deg[:, 1],
        c=times_vis,
        cmap=cmap,
        s=3,
        alpha=0.8,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time")

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)
    return ax

def plot_cohospace_neuron(
    coords: np.ndarray,
    activity: np.ndarray,
    neuron_id: int,
    mode: str = "fr",           # "fr" or "spike"
    top_percent: float = 5.0,   # Used in FR mode
    times: np.ndarray | None = None,
    figsize: tuple = (6, 6),
    cmap: str = "hot",
    save_path: str | None = None,
    show: bool = True,
    config: PlotConfig | None = None,
) -> plt.Figure:
    """
    Overlay a single neuron's activity on the cohomology-space trajectory.

    This is a visualization helper:
    - mode="fr": marks the top `top_percent`%% time points by firing rate for the given neuron.
    - mode="spike": marks all time points where spike > 0 for the given neuron.

    Parameters
    ----------
    coords : ndarray, shape (T, 2)
        Decoded cohomology angles (theta1, theta2), in radians.
    activity : ndarray, shape (T, N)
        Activity matrix (continuous firing rate or binned spikes).
    times : ndarray, optional, shape (T_coords,)
        Optional indices to align activity to coords when coords are computed on a subset of timepoints.
    neuron_id : int
        Neuron index to visualize.
    mode : {"fr", "spike"}
    top_percent : float
        Used only when mode="fr". For example, 5.0 means "top 5%%" time points.
    figsize, cmap, save_path, show : see `plot_cohospace_trajectory`.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    coords = np.asarray(coords)
    # activity = _align_activity_to_coords(coords, activity, times, label="activity")
    theta_deg = _coho_coords_to_degrees(coords)

    signal = activity[:, neuron_id]

    if mode == "fr":
        # Select the neuron's top `top_percent`% time points
        threshold = np.percentile(signal, 100 - top_percent)
        idx = signal >= threshold
        color = signal[idx]
        title = f"Neuron {neuron_id} FR top {top_percent:.1f}% on coho-space"
        use_cmap = cmap
    elif mode == "spike":
        idx = signal > 0
        color = None
        title = f"Neuron {neuron_id} spikes on coho-space"
        use_cmap = None
    else:
        raise ValueError("mode must be 'fr' or 'spike'")

    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title=title,
        xlabel="Theta 1 (°)",
        ylabel="Theta 2 (°)",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    fig, ax = plt.subplots(figsize=config.figsize)
    sc = ax.scatter(
        theta_deg[idx, 0],
        theta_deg[idx, 1],
        c=color if mode == "fr" else "red",
        cmap=use_cmap,
        s=5,
        alpha=0.9,
    )

    if mode == "fr":
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Firing rate")

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)

    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)

    return fig


def plot_cohospace_population(
    coords: np.ndarray,
    activity: np.ndarray,
    neuron_ids: list[int] | np.ndarray,
    mode: str = "fr",           # "fr" or "spike"
    top_percent: float = 5.0,   # Used in FR mode
    times: np.ndarray | None = None,
    figsize: tuple = (6, 6),
    cmap: str = "hot",
    save_path: str | None = None,
    show: bool = True,
    config: PlotConfig | None = None,
) -> plt.Figure:
    """
    Plot aggregated activity from multiple neurons in cohomology space.

    For mode="fr":
    - For each neuron, select its top `top_percent`%% time points by firing rate.
    - Aggregate (sum) firing rates over the selected points and plot as colors.

    For mode="spike":
    - For each neuron, count spikes at each time point (spike > 0).
    - Aggregate counts over neurons and plot as colors.

    Parameters
    ----------
    coords : ndarray, shape (T, 2)
    activity : ndarray, shape (T, N)
    times : ndarray, optional, shape (T_coords,)
        Optional indices to align activity to coords when coords are computed on a subset of timepoints.
    neuron_ids : iterable[int]
        Neuron indices to include (use range(N) to include all).
    mode : {"fr", "spike"}
    top_percent : float
        Used only when mode="fr".
    figsize, cmap, save_path, show : see `plot_cohospace_trajectory`.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    coords = np.asarray(coords)
    activity = _align_activity_to_coords(coords, activity, times, label="activity")
    neuron_ids = np.asarray(neuron_ids, dtype=int)

    theta_deg = _coho_coords_to_degrees(coords)

    T = activity.shape[0]
    mask = np.zeros(T, dtype=bool)
    agg_color = np.zeros(T, dtype=float)

    for n in neuron_ids:
        signal = activity[:, n]

        if mode == "fr":
            threshold = np.percentile(signal, 100 - top_percent)
            idx = signal >= threshold
            agg_color[idx] += signal[idx]
            mask |= idx
        elif mode == "spike":
            idx = signal > 0
            agg_color[idx] += 1.0
            mask |= idx
        else:
            raise ValueError("mode must be 'fr' or 'spike'")

    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title=f"{len(neuron_ids)} neurons on coho-space",
        xlabel="Theta 1 (°)",
        ylabel="Theta 2 (°)",
        figsize=figsize,
        save_path=save_path,
        show=show,
    )

    fig, ax = plt.subplots(figsize=config.figsize)
    sc = ax.scatter(
        theta_deg[mask, 0],
        theta_deg[mask, 1],
        c=agg_color[mask],
        cmap=cmap,
        s=5,
        alpha=0.9,
    )
    cbar = plt.colorbar(sc, ax=ax)
    label = "Aggregate FR" if mode == "fr" else "Spike count"
    cbar.set_label(label)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)

    _ensure_parent_dir(config.save_path)
    finalize_figure(fig, config)

    return fig


def compute_cohoscore(
    coords: np.ndarray,
    activity: np.ndarray,
    top_percent: float = 2.0,
    times: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a simple cohomology-space selectivity score (CohoScore) for each neuron.

    For each neuron:
    - Select "active" time points:
        - If top_percent is None: all time points with activity > 0
        - Else: top `top_percent`%% time points by activity value
    - Compute circular variance for theta1 and theta2 on the selected points.
    - CohoScore = 0.5 * (var(theta1) + var(theta2))

    Interpretation:
    - Smaller score => points are more concentrated in coho space => higher selectivity.

    Parameters
    ----------
    coords : ndarray, shape (T, 2)
        Decoded cohomology angles (theta1, theta2), in radians.
    activity : ndarray, shape (T, N)
    times : ndarray, optional, shape (T_coords,)
        Optional indices to align activity to coords when coords are computed on a subset of timepoints.
        Activity matrix (FR or spikes).
    top_percent : float | None
        Percentage for selecting active points (e.g., 2.0 means top 2%%). If None, use activity>0.

    Returns
    -------
    scores : ndarray, shape (N,)
        CohoScore per neuron (NaN for neurons with too few points).
    """
    coords = np.asarray(coords)
    activity = _align_activity_to_coords(coords, activity, times, label="activity")
    T, N = activity.shape

    theta = coords % (2 * np.pi)  # Ensure values are in [0, 2π)
    scores = np.zeros(N, dtype=float)

    for n in range(N):
        signal = activity[:, n]

        if top_percent is None:
            idx = signal > 0  # Use all time points with spikes
        else:
            threshold = np.percentile(signal, 100 - top_percent)
            idx = signal >= threshold

        if np.sum(idx) < 5:
            scores[n] = np.nan  # Too sparse; unreliable
            continue

        theta1 = theta[idx, 0]
        theta2 = theta[idx, 1]

        var1 = circvar(theta1, high=2 * np.pi, low=0)
        var2 = circvar(theta2, high=2 * np.pi, low=0)

        scores[n] = 0.5 * (var1 + var2)

    return scores




def skew_transform_torus(coords):
    """
    Convert torus angles (theta1, theta2) into coordinates in a skewed parallelogram fundamental domain.

    Given theta1, theta2 in radians, map:
        x = theta1 + 0.5 * theta2
        y = (sqrt(3)/2) * theta2

    This is a linear change of basis that turns the square [0, 2π)×[0, 2π) into a 60-degree
    parallelogram, which is convenient for visualizing wrap-around behavior on a 2-torus.

    Parameters
    ----------
    coords : ndarray, shape (T, 2)
        Angles (theta1, theta2) in radians.

    Returns
    -------
    xy : ndarray, shape (T, 2)
        Skewed planar coordinates.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (T,2), got {coords.shape}")

    theta1 = coords[:, 0]
    theta2 = coords[:, 1]

    # Linear change of basis (NO nonlinear scaling)
    x = theta1 + 0.5 * theta2
    y = (np.sqrt(3) / 2.0) * theta2

    return np.stack([x, y], axis=1)


def draw_torus_parallelogram_grid(
    ax,
    n_tiles=1,
    color="0.7",
    lw=1.0,
    alpha=0.8
):
    """
    Draw parallelogram grid corresponding to torus fundamental domain.

    Fundamental vectors:
        e1 = (2π, 0)
        e2 = (π, √3 π)

    Parameters
    ----------
    ax : matplotlib axis
    n_tiles : int
        How many tiles to draw in +/- directions (visual aid).
        n_tiles=1 means draw [-1, 0, 1] shifts.
    """
    e1 = np.array([2*np.pi, 0.0])
    e2 = np.array([np.pi, np.sqrt(3)*np.pi])

    shifts = range(-n_tiles, n_tiles + 1)

    for i in shifts:
        for j in shifts:
            origin = i * e1 + j * e2
            corners = np.array([
                origin,
                origin + e1,
                origin + e1 + e2,
                origin + e2,
                origin
            ])
            ax.plot(corners[:, 0], corners[:, 1],
                    color=color, lw=lw, alpha=alpha)


def tile_parallelogram_points(xy, n_tiles=1):
    """
    Tile points in the skewed (parallelogram) torus fundamental domain.

    This is mainly for static visualizations so you can visually inspect continuity
    across domain boundaries.

    Parameters
    ----------
    points : ndarray, shape (T, 2)
        Points in the skewed plane (same coordinates as returned by `skew_transform_torus`).
    n_tiles : int
        Number of tiles to extend around the base domain.
        - n_tiles=1 produces a 3x3 tiling
        - n_tiles=2 produces a 5x5 tiling

    Returns
    -------
    tiled : ndarray
        Tiled points.
    """
    xy = np.asarray(xy, dtype=float)

    e1 = np.array([2*np.pi, 0.0])
    e2 = np.array([np.pi, np.sqrt(3)*np.pi])

    out = []
    for i in range(-n_tiles, n_tiles + 1):
        for j in range(-n_tiles, n_tiles + 1):
            out.append(xy + i * e1 + j * e2)

    return np.vstack(out) if len(out) else xy


def plot_cohospace_neuron_skewed(
    coords,
    activity,
    neuron_id,
    mode="spike",
    top_percent=2.0,
    times: np.ndarray | None = None,
    save_path=None,
    show=None,
    ax=None,
    show_grid=True,
    n_tiles=1,
    s=6,
    alpha=0.8,
    config: PlotConfig | None = None,
):
    """
    Plot single-neuron CohoSpace on skewed torus domain.

    Parameters
    ----------
    coords : ndarray, shape (T, 2)
        Decoded circular coordinates (theta1, theta2), in radians.
    activity : ndarray, shape (T, N)
        Activity matrix aligned with coords.
    neuron_id : int
        Neuron index.
    mode : {"spike", "fr"}
        spike: use activity > 0
        fr: use top_percent threshold
    top_percent : float
        Percentile for FR thresholding.
    """
    coords = np.asarray(coords)
    activity = _align_activity_to_coords(coords, activity, times, label="activity")

    # --- normalize angles to [0, 2π)
    coords = coords % (2 * np.pi)

    # --- select neuron activity
    a = activity[:, neuron_id]

    if mode == "spike":
        mask = a > 0
    elif mode == "fr":
        thr = np.percentile(a, 100 - top_percent)
        mask = a >= thr
    else:
        raise ValueError(f"Unknown mode: {mode}")

    val = a[mask]  # Used for FR-mode coloring

    if config is None:
        config = PlotConfig.for_static_plot(
            title=f"Neuron {neuron_id} – CohoSpace (skewed, mode={mode})",
            xlabel=r"$\theta_1 + \frac{1}{2}\theta_2$",
            ylabel=r"$\frac{\sqrt{3}}{2}\theta_2$",
            figsize=(5, 5),
            save_path=save_path,
            show=bool(show) if show is not None else False,
        )
    else:
        if save_path is not None:
            config.save_path = save_path
        if show is not None:
            config.show = show
        if not config.title:
            config.title = f"Neuron {neuron_id} – CohoSpace (skewed, mode={mode})"
        if not config.xlabel:
            config.xlabel = r"$\theta_1 + \frac{1}{2}\theta_2$"
        if not config.ylabel:
            config.ylabel = r"$\frac{\sqrt{3}}{2}\theta_2$"

    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure

    # --- fundamental domain vectors in skew plane
    e1 = np.array([2 * np.pi, 0.0])
    e2 = np.array([np.pi, np.sqrt(3) * np.pi])

    def _draw_single_domain(ax):
        P00 = np.array([0.0, 0.0])
        P10 = e1
        P01 = e2
        P11 = e1 + e2
        poly = np.vstack([P00, P10, P11, P01, P00])
        ax.plot(poly[:, 0], poly[:, 1], lw=1.2, color="0.35")

    def _annotate_corners(ax):
        P00 = np.array([0.0, 0.0])
        P10 = e1
        P01 = e2
        P11 = e1 + e2

        corners = np.vstack([P00, P10, P01, P11])
        xmin, ymin = corners.min(axis=0)
        xmax, ymax = corners.max(axis=0)
        padx = 0.02 * (xmax - xmin)
        pady = 0.02 * (ymax - ymin)

        bbox = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0)

        ax.text(P00[0] + padx, P00[1] + pady, "(0,0)",
                fontsize=10, ha="left", va="bottom", bbox=bbox)
        ax.text(P10[0] - padx, P10[1] + pady, "(2π,0)",
                fontsize=10, ha="right", va="bottom", bbox=bbox)
        ax.text(P01[0] + padx, P01[1] - pady, "(0,2π)",
                fontsize=10, ha="left", va="top", bbox=bbox)
        ax.text(P11[0] - padx, P11[1] - pady, "(2π,2π)",
                fontsize=10, ha="right", va="top", bbox=bbox)

    # --- skew transform
    xy = skew_transform_torus(coords[mask])

    # Tiling: if points are tiled, values must be tiled too (FR mode) to keep lengths consistent
    if n_tiles and n_tiles > 0:
        xy = tile_parallelogram_points(xy, n_tiles=n_tiles)
        if mode == "fr":
            val = np.tile(val, (2 * n_tiles + 1) ** 2)

    # --- scatter
    if mode == "fr":
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=s, alpha=alpha, cmap="viridis")
        fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, label="activity")
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=s, alpha=alpha, color="tab:blue")

    # Always draw the base domain boundary
    _draw_single_domain(ax)

    # Grid is optional (debug aid); when tiles=0 only the base domain is drawn
    if show_grid:
        draw_torus_parallelogram_grid(ax, n_tiles=n_tiles)

    _annotate_corners(ax)

    # Fix view limits: tiles=0 shows base domain; tiles>0 shows the tiled extent
    base = np.vstack([[0, 0], e1, e2, e1 + e2])

    if n_tiles and n_tiles > 0:
        # Expand view by n_tiles rings around the base domain
        # Translation vectors for tiling are i*e1 + j*e2
        shifts = []
        for i in range(-n_tiles, n_tiles + 1):
            for j in range(-n_tiles, n_tiles + 1):
                shifts.append(i * e1 + j * e2)
        shifts = np.asarray(shifts)  # ((2n+1)^2, 2)

        all_corners = (base[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
        xmin, ymin = all_corners.min(axis=0)
        xmax, ymax = all_corners.max(axis=0)
    else:
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

    if ax is None:
        _ensure_parent_dir(config.save_path)
        finalize_figure(fig, config)
    else:
        if config.save_path is not None:
            _ensure_parent_dir(config.save_path)
            fig.savefig(config.save_path, **config.to_savefig_kwargs())
        if config.show:
            plt.show()

    return ax


def plot_cohospace_population_skewed(
    coords,
    activity,
    neuron_ids,
    mode="spike",
    top_percent=2.0,
    times: np.ndarray | None = None,
    save_path=None,
    show=False,
    ax=None,
    show_grid=True,
    n_tiles=1,
    s=4,
    alpha=0.5,
    config: PlotConfig | None = None,
):
    """
    Plot population CohoSpace on skewed torus domain.

    neuron_ids : list or ndarray
        Neurons to include (e.g. top-K by CohoScore).
    """
    coords = np.asarray(coords)
    activity = _align_activity_to_coords(coords, activity, times, label="activity")
    coords = coords % (2 * np.pi)

    if config is None:
        config = PlotConfig.for_static_plot(
            title=f"Population CohoSpace (skewed, n={len(neuron_ids)}, mode={mode})",
            xlabel=r"$\theta_1 + \frac{1}{2}\theta_2$",
            ylabel=r"$\frac{\sqrt{3}}{2}\theta_2$",
            figsize=(5, 5),
            save_path=save_path,
            show=show,
        )
    else:
        if save_path is not None:
            config.save_path = save_path
        if show is not None:
            config.show = show
        if not config.title:
            config.title = f"Population CohoSpace (skewed, n={len(neuron_ids)}, mode={mode})"
        if not config.xlabel:
            config.xlabel = r"$\theta_1 + \frac{1}{2}\theta_2$"
        if not config.ylabel:
            config.ylabel = r"$\frac{\sqrt{3}}{2}\theta_2$"

    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.figure

    # --- fundamental domain vectors in skew plane
    e1 = np.array([2 * np.pi, 0.0])
    e2 = np.array([np.pi, np.sqrt(3) * np.pi])

    def _draw_single_domain(ax):
        P00 = np.array([0.0, 0.0])
        P10 = e1
        P01 = e2
        P11 = e1 + e2
        poly = np.vstack([P00, P10, P11, P01, P00])
        ax.plot(poly[:, 0], poly[:, 1], lw=1.2, color="0.35")

    # --- scatter each neuron
    for nid in neuron_ids:
        a = activity[:, nid]
        if mode == "spike":
            mask = a > 0
        else:
            thr = np.percentile(a, 100 - top_percent)
            mask = a >= thr

        xy = skew_transform_torus(coords[mask])

        if n_tiles and n_tiles > 0:
            xy = tile_parallelogram_points(xy, n_tiles=n_tiles)

        ax.scatter(xy[:, 0], xy[:, 1], s=s, alpha=alpha)

    # Always draw the base domain boundary
    _draw_single_domain(ax)

    if show_grid:
        draw_torus_parallelogram_grid(ax, n_tiles=n_tiles)

    # Fix view limits: tiles=0 shows base domain; tiles>0 shows the tiled extent
    base = np.vstack([[0, 0], e1, e2, e1 + e2])

    if n_tiles and n_tiles > 0:
        # Expand view by n_tiles rings around the base domain
        # Translation vectors for tiling are i*e1 + j*e2
        shifts = []
        for i in range(-n_tiles, n_tiles + 1):
            for j in range(-n_tiles, n_tiles + 1):
                shifts.append(i * e1 + j * e2)
        shifts = np.asarray(shifts)  # ((2n+1)^2, 2)

        all_corners = (base[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
        xmin, ymin = all_corners.min(axis=0)
        xmax, ymax = all_corners.max(axis=0)
    else:
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

    if ax is None:
        _ensure_parent_dir(config.save_path)
        finalize_figure(fig, config)
    else:
        if config.save_path is not None:
            _ensure_parent_dir(config.save_path)
            fig.savefig(config.save_path, **config.to_savefig_kwargs())
        if config.show:
            plt.show()

    return ax
