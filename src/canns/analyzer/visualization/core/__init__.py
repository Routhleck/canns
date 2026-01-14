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
from .backend import (
    AnimationBackend,
    BackendSelection,
    emit_backend_warnings,
    get_imageio_writer_kwargs,
    get_multiprocessing_context,
    get_optimal_worker_count,
    select_animation_backend,
)
from .config import AnimationConfig, PlotConfig, PlotConfigs, finalize_figure
from .jupyter_utils import display_animation_in_jupyter, is_jupyter_environment
from .rendering import ParallelAnimationRenderer, render_animation_parallel
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
    "finalize_figure",
    # Animation framework
    "OptimizedAnimationBase",
    # Backend management
    "AnimationBackend",
    "BackendSelection",
    "select_animation_backend",
    "get_imageio_writer_kwargs",
    "get_optimal_worker_count",
    "get_multiprocessing_context",
    "emit_backend_warnings",
    # Rendering
    "ParallelAnimationRenderer",
    "render_animation_parallel",
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
