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

from .config import PlotConfig, AnimationConfig, PlotConfigs
from .animation import OptimizedAnimationBase
from .rendering import ParallelAnimationRenderer
from .writers import (
    OptimizedAnimationWriter,
    create_optimized_writer,
    get_recommended_format,
    warn_double_rendering,
    warn_gif_format,
)

__all__ = [
    # Configuration
    'PlotConfig',
    'AnimationConfig',
    'PlotConfigs',
    # Animation framework
    'OptimizedAnimationBase',
    # Rendering
    'ParallelAnimationRenderer',
    # Writers
    'OptimizedAnimationWriter',
    'create_optimized_writer',
    'get_recommended_format',
    'warn_double_rendering',
    'warn_gif_format',
]
