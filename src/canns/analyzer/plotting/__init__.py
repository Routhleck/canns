"""High-level plotting helpers for analyzer functionality."""

from .config import PlotConfig, PlotConfigs
from .energy import (
    energy_landscape_1d_animation,
    energy_landscape_1d_static,
    energy_landscape_2d_animation,
    energy_landscape_2d_static,
)
from .spatial import plot_firing_field_heatmap
from .spikes import average_firing_rate_plot, raster_plot
from .tuning import tuning_curve

__all__ = [
    "PlotConfig",
    "PlotConfigs",
    "energy_landscape_1d_animation",
    "energy_landscape_1d_static",
    "energy_landscape_2d_animation",
    "energy_landscape_2d_static",
    "average_firing_rate_plot",
    "plot_firing_field_heatmap",
    "raster_plot",
    "tuning_curve",
]
