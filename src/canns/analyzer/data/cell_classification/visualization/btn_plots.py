"""BTN visualization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from canns.analyzer.visualization.core.config import PlotConfig, finalize_figure

_DEFAULT_BTN_COLORS = {
    "B": "#1f77b4",
    "T": "#000000",
    "N": "#2ca02c",
}


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


def _canonical_label(label: str) -> str:
    lab = label.strip().lower()
    if lab in ("b", "bursty"):
        return "B"
    if lab in ("t", "theta", "theta-modulated", "theta_modulated", "theta modulated"):
        return "T"
    if lab in ("n", "nonbursty", "non-bursty", "non_bursty"):
        return "N"
    return label


def _cluster_order(labels: np.ndarray, mapping: dict[int, str] | None) -> list[int]:
    cids = [int(c) for c in np.unique(labels)]
    if mapping is None:
        return sorted(cids)

    def _key(cid: int) -> tuple[int, str]:
        lab = _canonical_label(mapping.get(int(cid), str(cid)))
        order = {"B": 0, "T": 1, "N": 2}.get(lab, 999)
        return (order, str(lab))

    return sorted(cids, key=_key)


def _label_color(
    label: str,
    colors: dict[str, str] | None,
    fallback_idx: int,
) -> str:
    if colors and label in colors:
        return colors[label]
    if label in _DEFAULT_BTN_COLORS:
        return _DEFAULT_BTN_COLORS[label]
    cmap = plt.get_cmap("tab10")
    return cmap(fallback_idx % 10)


def _normalize_rows(acorr: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None or mode == "none":
        return acorr
    if mode == "probability":
        denom = acorr.sum(axis=1, keepdims=True)
    elif mode == "peak":
        denom = acorr.max(axis=1, keepdims=True)
    elif mode == "first":
        denom = acorr[:, :1]
    else:
        raise ValueError(f"Unknown normalize mode: {mode!r}")
    denom = np.where(denom == 0, 1.0, denom)
    return acorr / denom


def plot_btn_distance_matrix(
    *,
    dist: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    mapping: dict[int, str] | None = None,
    sort_by_label: bool = True,
    title: str = "BTN distance matrix",
    cmap: str = "afmhot",
    figsize: tuple[int, int] = (5, 5),
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
    config: PlotConfig | None = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Plot a distance matrix heatmap sorted by BTN cluster labels."""
    if dist is None or labels is None:
        raise ValueError("dist and labels are required.")

    labels = np.asarray(labels).astype(int)

    if sort_by_label:
        cids = _cluster_order(labels, mapping)
        order = np.concatenate([np.where(labels == c)[0] for c in cids])
    else:
        order = np.arange(len(labels))

    dist_sorted = dist[np.ix_(order, order)]

    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title=title,
        figsize=figsize,
        save_path=save_path,
        show=show,
        kwargs={},
    )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=config.figsize)
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(dist_sorted, cmap=cmap, origin="lower", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Cosine distance")
    ax.set_title(config.title)
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")

    if sort_by_label:
        sizes = [np.sum(labels == c) for c in cids]
        boundaries = np.cumsum(sizes)[:-1]
        for b in boundaries:
            ax.axhline(b - 0.5, color="w", linewidth=0.6, alpha=0.7)
            ax.axvline(b - 0.5, color="w", linewidth=0.6, alpha=0.7)

    if created_fig:
        fig.tight_layout()
        finalize_figure(fig, config, rasterize_artists=[im] if config.rasterized else None)

    return fig, ax, order


def plot_btn_autocorr_summary(
    *,
    acorr: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    bin_times: np.ndarray | None = None,
    res: float | None = None,
    mapping: dict[int, str] | None = None,
    colors: dict[str, str] | None = None,
    normalize: str | None = "probability",
    smooth_sigma: float | None = None,
    long_max_ms: float | None = 200.0,
    short_max_ms: float | None = None,
    title: str = "BTN temporal autocorr",
    figsize: tuple[int, int] = (8, 3),
    save_path: str | None = None,
    show: bool = True,
    config: PlotConfig | None = None,
) -> plt.Figure:
    """Plot class-averaged ISI autocorr curves (mean +/- SEM)."""
    if acorr is None or labels is None:
        raise ValueError("acorr and labels are required.")

    labels = np.asarray(labels).astype(int)
    acorr = np.asarray(acorr)
    acorr_plot = _normalize_rows(acorr.astype(float, copy=False), normalize)
    if smooth_sigma is not None:
        acorr_plot = gaussian_filter1d(acorr_plot, sigma=float(smooth_sigma), axis=1)

    if bin_times is not None:
        bin_times = np.asarray(bin_times)
        x = 0.5 * (bin_times[:-1] + bin_times[1:])
    elif res is not None:
        x = np.arange(acorr.shape[1]) * float(res)
    else:
        raise ValueError("Provide bin_times or res to define lag axis.")

    x_ms = x * 1000.0

    cids = _cluster_order(labels, mapping)
    label_strings = []
    for c in cids:
        if mapping is not None:
            label_strings.append(_canonical_label(mapping.get(int(c), str(c))))
        else:
            label_strings.append(str(c))

    show_short = short_max_ms is not None
    ncols = 2 if show_short else 1

    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title=title,
        figsize=figsize if ncols == 1 else (figsize[0] * 1.6, figsize[1]),
        save_path=save_path,
        show=show,
        kwargs={},
    )

    fig, axes = plt.subplots(1, ncols, figsize=config.figsize)
    if ncols == 1:
        axes = [axes]

    def _plot_panel(ax: plt.Axes, max_ms: float | None, panel_title: str):
        if max_ms is None:
            mask = np.ones_like(x_ms, dtype=bool)
        else:
            mask = x_ms <= max_ms

        for idx, (cid, label_str) in enumerate(zip(cids, label_strings, strict=False)):
            rows = acorr_plot[labels == cid]
            if rows.size == 0:
                continue
            mean = rows.mean(axis=0)
            sem = rows.std(axis=0) / np.sqrt(rows.shape[0])
            color = _label_color(label_str, colors, idx)
            ax.plot(x_ms[mask], mean[mask], color=color, lw=2, label=label_str)
            ax.fill_between(
                x_ms[mask],
                (mean - sem)[mask],
                (mean + sem)[mask],
                color=color,
                alpha=0.25,
                linewidth=0,
            )

        ax.set_xlabel("Lag (ms)")
        ax.set_title(panel_title)
        ax.grid(False)

    _plot_panel(axes[0], long_max_ms, "Long lag")
    ylabel = "Probability" if normalize == "probability" else "Autocorr (norm)"
    axes[0].set_ylabel(ylabel)

    if show_short:
        _plot_panel(axes[1], float(short_max_ms), "Short lag")

    for ax in axes:
        ax.legend(frameon=False)

    fig.suptitle(config.title)
    fig.tight_layout()
    finalize_figure(fig, config)
    return fig
