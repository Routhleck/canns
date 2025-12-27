"""Model visualization utilities."""

# Re-export core components for backward compatibility
from .core import (
    AnimationConfig,
    OptimizedAnimationBase,
    OptimizedAnimationWriter,
    ParallelAnimationRenderer,
    PlotConfig,
    PlotConfigs,
    create_optimized_writer,
    display_animation_in_jupyter,
    get_recommended_format,
    is_jupyter_environment,
    warn_double_rendering,
    warn_gif_format,
)
from .energy_plots import (
    energy_landscape_1d_animation,
    energy_landscape_1d_static,
    energy_landscape_2d_animation,
    energy_landscape_2d_static,
)
from .spatial_plots import (
    create_grid_cell_tracking_animation,
    plot_autocorrelation,
    plot_firing_field_heatmap,
    plot_grid_score,
    plot_grid_spacing_analysis,
)
from .spike_plots import average_firing_rate_plot, population_activity_heatmap, raster_plot
from .theta_sweep_plots import (
    create_theta_sweep_grid_cell_animation,
    create_theta_sweep_place_cell_animation,
    plot_grid_cell_manifold,
    plot_population_activity_with_theta,
)
from .tuning_plots import tuning_curve

__all__ = [
    # Core components (re-exported)
    "PlotConfig",
    "PlotConfigs",
    "AnimationConfig",
    "OptimizedAnimationBase",
    "ParallelAnimationRenderer",
    "OptimizedAnimationWriter",
    "create_optimized_writer",
    "get_recommended_format",
    "warn_double_rendering",
    "warn_gif_format",
    "is_jupyter_environment",
    "display_animation_in_jupyter",
    # Visualization functions
    "energy_landscape_1d_animation",
    "energy_landscape_1d_static",
    "energy_landscape_2d_animation",
    "energy_landscape_2d_static",
    "plot_firing_field_heatmap",
    "plot_autocorrelation",
    "plot_grid_score",
    "plot_grid_spacing_analysis",
    "create_grid_cell_tracking_animation",
    "raster_plot",
    "average_firing_rate_plot",
    "population_activity_heatmap",
    "tuning_curve",
    "create_theta_sweep_grid_cell_animation",
    "create_theta_sweep_place_cell_animation",
    "plot_grid_cell_manifold",
    "plot_population_activity_with_theta",
]
