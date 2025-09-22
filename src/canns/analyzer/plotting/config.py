"""Reusable plotting configuration utilities for analyzer visualizations."""

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["PlotConfig", "PlotConfigs"]


@dataclass
class PlotConfig:
    """Unified configuration class for analyzer plotting helpers."""

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


class PlotConfigs:
    """Collection of high-level presets for common analyzer plots."""

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
