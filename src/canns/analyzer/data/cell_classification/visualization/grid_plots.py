"""Grid cell visualization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

from canns.analyzer.visualization.core.config import PlotConfig, PlotConfigs, finalize_figure


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


def plot_autocorrelogram(
    autocorr: np.ndarray,
    config: PlotConfig | None = None,
    *,
    gridness_score: float | None = None,
    center_radius: float | None = None,
    peak_locations: np.ndarray | None = None,
    title: str = "Spatial Autocorrelation",
    xlabel: str = "X Lag (bins)",
    ylabel: str = "Y Lag (bins)",
    figsize: tuple[int, int] = (6, 6),
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 2D autocorrelogram with optional annotations."""
    config = _ensure_plot_config(
        config,
        PlotConfigs.grid_autocorrelation,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        save_path=save_path,
        show=show,
        kwargs=kwargs,
    )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=config.figsize)
        created_fig = True
    else:
        fig = ax.figure

    # Plot autocorrelogram
    plot_kwargs = config.to_matplotlib_kwargs()
    plot_kwargs.setdefault("origin", "lower")
    plot_kwargs.setdefault("interpolation", "bilinear")
    im = ax.imshow(autocorr, **plot_kwargs)
    fig.colorbar(im, ax=ax, label="Correlation")

    # Add center circle if radius provided
    if center_radius is not None:
        center = np.array(autocorr.shape) / 2
        circle = patches.Circle(
            (center[1], center[0]), center_radius,
            fill=False, edgecolor="red", linewidth=2, linestyle="--",
            label="Center field"
        )
        ax.add_patch(circle)

    # Mark peak locations if provided
    if peak_locations is not None:
        ax.plot(
            peak_locations[:, 0],
            peak_locations[:, 1],
            "r+",
            markersize=10,
            markeredgewidth=2,
            label="Grid peaks",
        )

    # Add gridness score to title
    title_text = config.title
    if gridness_score is not None:
        title_text = f"{config.title} (Gridness: {gridness_score:.3f})"

    ax.set_title(title_text)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)

    if center_radius is not None or peak_locations is not None:
        ax.legend(loc='upper right')

    if created_fig:
        fig.tight_layout()
        finalize_figure(fig, config, rasterize_artists=[im] if config.rasterized else None)

    return fig, ax


def plot_gridness_analysis(
    rate_map: np.ndarray,
    autocorr: np.ndarray,
    result,
    config: PlotConfig | None = None,
    *,
    title: str = "Grid Cell Analysis",
    figsize: tuple[int, int] = (15, 5),
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Comprehensive grid analysis plot with rate map, autocorr, and statistics."""
    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title=title,
        figsize=figsize,
        save_path=save_path,
        show=show,
        kwargs={},
    )

    fig, axes = plt.subplots(1, 3, figsize=config.figsize)

    # Plot 1: Rate map
    im1 = axes[0].imshow(rate_map, cmap='hot', origin='lower')
    axes[0].set_title('Firing Rate Map')
    axes[0].set_xlabel('X (bins)')
    axes[0].set_ylabel('Y (bins)')
    plt.colorbar(im1, ax=axes[0], label='Rate (Hz)')

    # Plot 2: Autocorrelogram with annotations
    plot_autocorrelogram(
        autocorr,
        gridness_score=result.score,
        center_radius=result.center_radius,
        peak_locations=result.peak_locations,
        title='Autocorrelogram',
        ax=axes[1]
    )

    # Plot 3: Grid statistics
    axes[2].axis('off')
    stats_text = f"""
Grid Cell Analysis

Gridness Score: {result.score:.3f}
Center Radius: {result.center_radius:.1f} bins
Optimal Radius: {result.optimal_radius:.1f} bins

Grid Spacing (bins):
  {result.spacing[0]:.2f}
  {result.spacing[1]:.2f}
  {result.spacing[2]:.2f}

Grid Orientation (°):
  {result.orientation[0]:.1f}
  {result.orientation[1]:.1f}
  {result.orientation[2]:.1f}

Ellipse Parameters:
  Center: ({result.ellipse[0]:.1f}, {result.ellipse[1]:.1f})
  Radii: ({result.ellipse[2]:.1f}, {result.ellipse[3]:.1f})
  Angle: {result.ellipse_theta_deg:.1f}°
    """
    axes[2].text(0.1, 0.5, stats_text, fontsize=10,
                verticalalignment='center', family='monospace')
    axes[2].set_title('Grid Statistics')

    fig.suptitle(config.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    finalize_figure(fig, config)
    return fig


def plot_rate_map(
    rate_map: np.ndarray,
    config: PlotConfig | None = None,
    *,
    title: str = "Firing Field (Rate Map)",
    xlabel: str = "X Position (bins)",
    ylabel: str = "Y Position (bins)",
    figsize: tuple[int, int] = (6, 6),
    colorbar: bool = True,
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 2D spatial firing rate map."""
    config = _ensure_plot_config(
        config,
        PlotConfigs.firing_field_heatmap,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        save_path=save_path,
        show=show,
        kwargs=kwargs,
    )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=config.figsize)
        created_fig = True
    else:
        fig = ax.figure

    plot_kwargs = config.to_matplotlib_kwargs()
    plot_kwargs.setdefault("origin", "lower")
    plot_kwargs.setdefault("interpolation", "bilinear")
    im = ax.imshow(rate_map, **plot_kwargs)

    if colorbar:
        fig.colorbar(im, ax=ax, label="Firing Rate (Hz)")

    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)

    if created_fig:
        fig.tight_layout()
        finalize_figure(fig, config, rasterize_artists=[im] if config.rasterized else None)

    return fig, ax


def plot_grid_score_histogram(
    scores: np.ndarray,
    config: PlotConfig | None = None,
    *,
    bins: int = 30,
    title: str = "Grid Score Distribution",
    xlabel: str = "Grid Score",
    ylabel: str = "Count",
    figsize: tuple[int, int] = (6, 4),
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot histogram of gridness scores."""
    config = _ensure_plot_config(
        config,
        PlotConfig.for_static_plot,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        save_path=save_path,
        show=show,
        kwargs=kwargs,
    )

    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=config.figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.hist(scores, bins=bins, **config.to_matplotlib_kwargs())
    ax.set_title(config.title)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)

    if created_fig:
        fig.tight_layout()
        finalize_figure(fig, config)

    return fig, ax
