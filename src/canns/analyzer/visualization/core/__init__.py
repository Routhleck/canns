"""
Visualization core infrastructure.

This module provides foundational components for all visualization functions:
- Configuration classes (PlotConfig, AnimationConfig, PlotConfigs)
- Animation framework (OptimizedAnimationBase)
- Parallel rendering (ParallelAnimationRenderer)
- Optimized writers (create_optimized_writer, OptimizedAnimationWriter)

All core components are re-exported at the parent visualization level for
backward compatibility and convenience.
"""

from .animation import OptimizedAnimationBase
from .config import AnimationConfig, PlotConfig, PlotConfigs
from .jupyter_utils import display_animation_in_jupyter, is_jupyter_environment
from .rendering import ParallelAnimationRenderer
from .writers import (
    OptimizedAnimationWriter,
    create_optimized_writer,
    get_matplotlib_writer,
    get_recommended_format,
    warn_double_rendering,
    warn_gif_format,
)

__all__ = [
    # Configuration
    "PlotConfig",
    "AnimationConfig",
    "PlotConfigs",
    # Animation framework
    "OptimizedAnimationBase",
    # Rendering
    "ParallelAnimationRenderer",
    # Writers
    "OptimizedAnimationWriter",
    "create_optimized_writer",
    "get_matplotlib_writer",
    "get_recommended_format",
    "warn_double_rendering",
    "warn_gif_format",
    # Jupyter utilities
    "is_jupyter_environment",
    "display_animation_in_jupyter",
]
