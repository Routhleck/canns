"""Spatial visualization functions for neural firing field heatmaps.

This module provides plotting utilities for visualizing spatial firing patterns
of neural populations, particularly for grid cells, place cells, and band cells.
"""

import numpy as np
from matplotlib import pyplot as plt

from .config import PlotConfig

__all__ = ["plot_firing_field_heatmap"]


def plot_firing_field_heatmap(
    heatmap: np.ndarray,
    config: PlotConfig | None = None,
    # Backward compatibility parameters
    figsize: tuple[int, int] = (5, 5),
    cmap: str = "jet",
    interpolation: str = "nearest",
    origin: str = "lower",
    show: bool = True,
    save_path: str | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a single spatial firing field heatmap.

    This function creates a publication-quality heatmap visualization of neural
    spatial firing patterns. It supports both modern PlotConfig-based configuration
    and legacy keyword arguments for backward compatibility.

    Args:
        heatmap (np.ndarray): 2D array of shape (M, K) representing spatial
            firing rates in each bin.
        config (PlotConfig | None): Unified configuration object. If None,
            uses backward compatibility parameters.
        figsize (tuple[int, int]): Figure size (width, height) in inches.
            Defaults to (5, 5).
        cmap (str): Colormap name for the heatmap. Defaults to 'jet'.
        interpolation (str): Interpolation method for imshow. Defaults to 'nearest'.
        origin (str): Origin position for imshow ('lower' or 'upper').
            Defaults to 'lower'.
        show (bool): Whether to display the plot. Defaults to True.
        save_path (str | None): Path to save the figure. If None, figure is not saved.
        **kwargs: Additional keyword arguments passed to plt.imshow().

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axis objects for further customization.

    Example:
        >>> from canns.analyzer.spatial import compute_firing_field
        >>> from canns.analyzer.plotting import plot_firing_field_heatmap, PlotConfig
        >>> # Compute firing field
        >>> heatmaps = compute_firing_field(activity, positions, 5.0, 5.0, 50, 50)
        >>> # Plot single neuron with PlotConfig
        >>> config = PlotConfig(figsize=(6, 6), save_path='neuron_0.png', show=False)
        >>> fig, ax = plot_firing_field_heatmap(heatmaps[0], config=config)
        >>> # Plot with legacy parameters
        >>> fig, ax = plot_firing_field_heatmap(heatmaps[1], cmap='viridis', save_path='neuron_1.png')
    """
    # Handle configuration
    if config is None:
        config = PlotConfig(
            figsize=figsize,
            show=show,
            save_path=save_path,
            kwargs={"cmap": cmap, "interpolation": interpolation, "origin": origin, **kwargs},
        )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=config.figsize)

    try:
        # Extract plotting parameters
        plot_kwargs = config.to_matplotlib_kwargs()
        if "cmap" not in plot_kwargs:
            plot_kwargs["cmap"] = cmap
        if "interpolation" not in plot_kwargs:
            plot_kwargs["interpolation"] = interpolation
        if "origin" not in plot_kwargs:
            plot_kwargs["origin"] = origin

        # Plot heatmap
        im = ax.imshow(heatmap.T, **plot_kwargs)

        # Remove ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])

        # Tight layout for better appearance
        fig.tight_layout()

        # Save if path provided
        if config.save_path:
            plt.savefig(config.save_path, bbox_inches="tight")

        # Show if requested
        if config.show:
            plt.show()
        else:
            # Close figure to avoid memory accumulation when batch saving
            plt.close(fig)

        return fig, ax

    except Exception as e:
        plt.close(fig)
        raise e
