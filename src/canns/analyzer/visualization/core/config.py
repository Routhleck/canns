"""Reusable plotting configuration utilities for analyzer visualizations."""

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["PlotConfig", "PlotConfigs", "AnimationConfig"]


@dataclass
class PlotConfig:
    """Unified configuration class for all plotting helpers in ``canns.analyzer``.

    This mirrors the behaviour of the previous ``visualize`` module so that
    reorganising the files does not affect the public API. The attributes map
    directly to keyword arguments exposed by the high-level plotting functions,
    allowing users to keep existing configuration objects unchanged after the
    reorganisation.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: tuple[int, int] = (10, 6)
    grid: bool = False
    save_path: str | None = None
    show: bool = True

    time_steps_per_second: int | None = None
    fps: int = 30
    repeat: bool = True
    show_progress_bar: bool = True

    show_legend: bool = True
    color: str = "black"
    clabel: str = "Value"

    kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kwargs is None:
            self.kwargs = {}

    @classmethod
    def for_static_plot(cls, **kwargs: Any) -> "PlotConfig":
        """Return configuration tailored for static plots."""

        config = cls(**kwargs)
        config.time_steps_per_second = None
        return config

    @classmethod
    def for_animation(cls, time_steps_per_second: int, **kwargs: Any) -> "PlotConfig":
        """Return configuration tailored for animations."""

        return cls(time_steps_per_second=time_steps_per_second, **kwargs)

    def to_matplotlib_kwargs(self) -> dict[str, Any]:
        """Materialize matplotlib keyword arguments from the config."""

        return self.kwargs.copy() if self.kwargs else {}


@dataclass
class AnimationConfig:
    """Configuration for animation rendering.

    Provides unified settings for optimized animation rendering with automatic
    quality presets and parallel rendering support.

    Attributes:
        fps: Frames per second for the animation
        enable_blitting: Whether to use blitting optimization (auto-detected by default)
        use_parallel: Force parallel rendering even for short animations
        num_workers: Number of worker processes for parallel rendering
        quality: Quality preset - 'draft', 'medium', or 'high'
        npoints_multiplier: Resolution multiplier (< 1.0 for draft mode)
        auto_parallel_threshold: Auto-enable parallel rendering for animations with
                                more than this many frames

    Example:
        >>> # High-quality animation (default)
        >>> config = AnimationConfig(fps=30, quality='high')
        >>>
        >>> # Fast draft mode for quick iteration
        >>> draft_config = AnimationConfig(quality='draft')  # Auto: 15 FPS, 0.5x resolution
        >>>
        >>> # Force parallel rendering
        >>> parallel_config = AnimationConfig(use_parallel=True, num_workers=8)
    """

    fps: int = 30
    enable_blitting: bool = True  # Auto-detect backend support
    use_parallel: bool = False  # Auto-enable for long animations
    num_workers: int = 4
    quality: str = "high"  # Options: 'draft', 'medium', 'high'
    npoints_multiplier: float = 1.0  # Automatically set to 0.5 for draft mode
    auto_parallel_threshold: int = 500  # Enable parallel for > 500 frames

    def __post_init__(self):
        """Automatically adjust settings based on quality preset."""
        if self.quality == "draft":
            self.npoints_multiplier = 0.5
            self.fps = max(15, self.fps // 2)
        elif self.quality == "medium":
            self.npoints_multiplier = 0.75


class PlotConfigs:
    """Collection of commonly used plot configurations.

    These helpers mirror the presets that existed in ``canns.analyzer.visualize``
    so that callers relying on them continue to receive the exact same defaults.
    """

    @staticmethod
    def energy_landscape_1d_static(**kwargs: Any) -> PlotConfig:
        defaults: dict[str, Any] = {
            "title": "1D Energy Landscape",
            "xlabel": "Collective Variable / State",
            "ylabel": "Energy",
            "figsize": (10, 6),
        }
        defaults.update(kwargs)
        return PlotConfig.for_static_plot(**defaults)

    @staticmethod
    def energy_landscape_1d_animation(**kwargs: Any) -> PlotConfig:
        defaults: dict[str, Any] = {
            "title": "Evolving 1D Energy Landscape",
            "xlabel": "Collective Variable / State",
            "ylabel": "Energy",
            "figsize": (10, 6),
            "fps": 30,
        }
        time_steps = kwargs.pop("time_steps_per_second", 1000)
        defaults.update(kwargs)
        return PlotConfig.for_animation(time_steps, **defaults)

    @staticmethod
    def energy_landscape_2d_static(**kwargs: Any) -> PlotConfig:
        defaults: dict[str, Any] = {
            "title": "2D Static Landscape",
            "xlabel": "X-Index",
            "ylabel": "Y-Index",
            "clabel": "Value",
            "figsize": (8, 7),
        }
        defaults.update(kwargs)
        return PlotConfig.for_static_plot(**defaults)

    @staticmethod
    def energy_landscape_2d_animation(**kwargs: Any) -> PlotConfig:
        defaults: dict[str, Any] = {
            "title": "Evolving 2D Landscape",
            "xlabel": "X-Index",
            "ylabel": "Y-Index",
            "clabel": "Value",
            "figsize": (8, 7),
            "fps": 30,
        }
        time_steps = kwargs.pop("time_steps_per_second", 1000)
        defaults.update(kwargs)
        return PlotConfig.for_animation(time_steps, **defaults)

    @staticmethod
    def raster_plot(mode: str = "block", **kwargs: Any) -> PlotConfig:
        defaults: dict[str, Any] = {
            "title": "Raster Plot",
            "xlabel": "Time Step",
            "ylabel": "Neuron Index",
            "figsize": (12, 6),
            "color": "black",
        }
        defaults.update(kwargs)
        config = PlotConfig.for_static_plot(**defaults)
        config.mode = mode
        return config

    @staticmethod
    def average_firing_rate_plot(mode: str = "per_neuron", **kwargs: Any) -> PlotConfig:
        defaults: dict[str, Any] = {
            "title": "Average Firing Rate",
            "figsize": (12, 5),
        }
        defaults.update(kwargs)
        config = PlotConfig.for_static_plot(**defaults)
        config.mode = mode
        return config

    @staticmethod
    def theta_population_activity_static(**kwargs: Any) -> PlotConfig:
        defaults: dict[str, Any] = {
            "title": "Population Activity with Theta",
            "xlabel": "Time (s)",
            "ylabel": "Direction (°)",
            "figsize": (12, 4),
        }
        plot_kwargs: dict[str, Any] = {"cmap": "jet"}
        plot_kwargs.update(kwargs.pop("kwargs", {}))
        defaults["kwargs"] = plot_kwargs
        defaults.update(kwargs)
        return PlotConfig.for_static_plot(**defaults)

    @staticmethod
    def grid_cell_manifold_static(**kwargs: Any) -> PlotConfig:
        defaults: dict[str, Any] = {
            "title": "Grid Cell Activity on Manifold",
            "figsize": (8, 6),
        }
        plot_kwargs: dict[str, Any] = {"cmap": "jet", "add_colorbar": True}
        plot_kwargs.update(kwargs.pop("kwargs", {}))
        defaults["kwargs"] = plot_kwargs
        defaults.update(kwargs)
        return PlotConfig.for_static_plot(**defaults)

    @staticmethod
    def theta_sweep_animation(**kwargs: Any) -> PlotConfig:
        defaults: dict[str, Any] = {
            "figsize": (12, 3),
            "fps": 10,
            "show_progress_bar": True,
        }
        animation_kwargs: dict[str, Any] = {
            "cmap": "jet",
            "alpha": 0.8,
            "trajectory_color": "#FFFFFF",
            "trajectory_outline": "#1A1A1A",
            "current_marker_color": "#FF2D00",
        }
        animation_kwargs.update(kwargs.pop("kwargs", {}))
        defaults["kwargs"] = animation_kwargs
        time_steps = kwargs.pop("time_steps_per_second", None)
        defaults.update(kwargs)
        defaults["time_steps_per_second"] = time_steps
        return PlotConfig(**defaults)

    @staticmethod
    def tuning_curve(
        num_bins: int = 50,
        pref_stim: np.ndarray | None = None,
        **kwargs: Any,
    ) -> PlotConfig:
        defaults: dict[str, Any] = {
            "title": "Tuning Curve",
            "xlabel": "Stimulus Value",
            "ylabel": "Average Firing Rate",
            "figsize": (10, 6),
        }
        defaults.update(kwargs)
        config = PlotConfig.for_static_plot(**defaults)
        config.num_bins = num_bins
        config.pref_stim = pref_stim
        return config

    @staticmethod
    def grid_autocorrelation(**kwargs: Any) -> PlotConfig:
        """Configuration for spatial autocorrelation heatmap visualization.

        Used to visualize hexagonal periodicity patterns in grid cell firing fields.
        Applies diverging colormap (RdBu_r) suitable for correlation values [-1, 1].

        Args:
            **kwargs: Additional configuration parameters to override defaults.

        Returns:
            PlotConfig: Configuration object for autocorrelation plots.

        Example:
            >>> from canns.analyzer.visualization import PlotConfigs
            >>> config = PlotConfigs.grid_autocorrelation(
            ...     title="Grid Cell Autocorrelation",
            ...     save_path="autocorr.png"
            ... )
        """
        defaults: dict[str, Any] = {
            "title": "Spatial Autocorrelation",
            "xlabel": "X Lag (bins)",
            "ylabel": "Y Lag (bins)",
            "figsize": (6, 6),
        }
        plot_kwargs: dict[str, Any] = {"cmap": "RdBu_r", "vmin": -1, "vmax": 1}
        plot_kwargs.update(kwargs.pop("kwargs", {}))
        defaults["kwargs"] = plot_kwargs
        defaults.update(kwargs)
        return PlotConfig.for_static_plot(**defaults)

    @staticmethod
    def grid_score_plot(**kwargs: Any) -> PlotConfig:
        """Configuration for grid score bar chart visualization.

        Displays rotational correlations at different angles used to compute grid score.
        Highlights hexagonal angles (60°, 120°) versus non-hexagonal angles.

        Args:
            **kwargs: Additional configuration parameters to override defaults.

        Returns:
            PlotConfig: Configuration object for grid score plots.

        Example:
            >>> config = PlotConfigs.grid_score_plot(
            ...     title="Grid Cell Quality Assessment",
            ...     save_path="grid_score.png"
            ... )
        """
        defaults: dict[str, Any] = {
            "title": "Grid Score Analysis",
            "xlabel": "Rotation Angle (°)",
            "ylabel": "Correlation",
            "figsize": (8, 5),
            "grid": True,
        }
        defaults.update(kwargs)
        return PlotConfig.for_static_plot(**defaults)

    @staticmethod
    def grid_spacing_plot(**kwargs: Any) -> PlotConfig:
        """Configuration for grid spacing radial profile visualization.

        Shows how autocorrelation decays with distance from center, revealing
        the periodic spacing of grid fields.

        Args:
            **kwargs: Additional configuration parameters to override defaults.

        Returns:
            PlotConfig: Configuration object for spacing analysis plots.

        Example:
            >>> config = PlotConfigs.grid_spacing_plot(
            ...     title="Grid Field Spacing",
            ...     save_path="spacing.png"
            ... )
        """
        defaults: dict[str, Any] = {
            "title": "Grid Spacing Analysis",
            "xlabel": "Distance (bins)",
            "ylabel": "Autocorrelation",
            "figsize": (8, 5),
            "grid": True,
        }
        defaults.update(kwargs)
        return PlotConfig.for_static_plot(**defaults)

    @staticmethod
    def grid_cell_tracking_animation(**kwargs: Any) -> PlotConfig:
        """Configuration for grid cell tracking animation.

        Creates 3-panel synchronized animation showing trajectory, activity time course,
        and rate map with position overlay for analyzing grid cell behavior.

        Args:
            **kwargs: Additional configuration parameters to override defaults.
                Must include 'time_steps_per_second' if not using default.

        Returns:
            PlotConfig: Configuration object for tracking animations.

        Example:
            >>> config = PlotConfigs.grid_cell_tracking_animation(
            ...     time_steps_per_second=1000,  # dt=1ms
            ...     fps=20,
            ...     save_path="tracking.gif"
            ... )
        """
        defaults: dict[str, Any] = {
            "title": "Grid Cell Tracking",
            "figsize": (15, 5),
            "fps": 20,
            "show_progress_bar": True,
        }
        time_steps = kwargs.pop("time_steps_per_second", None)
        defaults.update(kwargs)
        if time_steps is not None:
            return PlotConfig.for_animation(time_steps, **defaults)
        else:
            # Return config without time_steps_per_second set
            # Will be validated when animation function is called
            return PlotConfig(**defaults)
