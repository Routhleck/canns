"""
High-performance animation framework with blitting and parallel rendering support.

This module provides base classes and utilities for creating optimized matplotlib
animations using blitting and optional parallel rendering for long animations.
"""

import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

from .config import AnimationConfig


class OptimizedAnimationBase(ABC):
    """
    High-performance animation base class with blitting support.

    This abstract base class enforces best practices for matplotlib animations
    by separating artist creation from frame updates, enabling automatic blitting
    for significant performance improvements (2-10x faster rendering).

    The class provides:
    - Automatic blitting support detection and fallback
    - Optional parallel rendering for long animations (>500 frames)
    - Consistent API for creating optimized animations
    - Integration with imageio and matplotlib backends

    Subclasses must implement
    -----------------------
    create_artists()
        Pre-create all artist objects with animated=True. Called once before
        animation starts. Should create and return all plot objects (lines,
        scatter, images, etc.) that will be animated.
    
    update_frame(frame_idx)
        Update artist data for a specific frame. Called for each frame.
        Should only modify data (via set_data(), set_array(), etc.), never
        recreate artists or call ax.clear().

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure to animate
    ax : matplotlib.axes.Axes
        Matplotlib axes to animate
    config : AnimationConfig or None, default=None
        Animation configuration with fps, blitting, and parallel rendering
        settings. Uses AnimationConfig() defaults if None.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure being animated
    ax : matplotlib.axes.Axes
        The axes being animated
    config : AnimationConfig
        Configuration for animation rendering
    artists : list[Artist]
        List of animated artist objects (populated by create_artists())

    Methods
    -------
    render_animation(nframes, interval=None, repeat=True, save_path=None, **save_kwargs)
        Render the complete animation with automatic optimization
    create_artists()
        Abstract method - must be implemented by subclasses
    update_frame(frame_idx)
        Abstract method - must be implemented by subclasses

    Example
    -------
    Create a simple line animation:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from canns.analyzer.visualization.core.animation import OptimizedAnimationBase
    >>> from canns.analyzer.visualization.core.config import AnimationConfig
    >>> 
    >>> class SineWaveAnimation(OptimizedAnimationBase):
    ...     def __init__(self, fig, ax, config=None):
    ...         super().__init__(fig, ax, config)
    ...         self.x = np.linspace(0, 2*np.pi, 100)
    ...         self.line = None
    ...     
    ...     def create_artists(self):
    ...         # Pre-create line with empty data
    ...         self.line, = self.ax.plot([], [], 'b-', animated=True)
    ...         self.ax.set_xlim(0, 2*np.pi)
    ...         self.ax.set_ylim(-1.5, 1.5)
    ...         return [self.line]
    ...     
    ...     def update_frame(self, frame_idx):
    ...         # Only update data, don't recreate
    ...         phase = frame_idx * 0.1
    ...         y = np.sin(self.x + phase)
    ...         self.line.set_data(self.x, y)
    ...         return (self.line,)
    >>> 
    >>> # Create and render animation
    >>> fig, ax = plt.subplots()
    >>> config = AnimationConfig(fps=30, enable_blitting=True)
    >>> anim = SineWaveAnimation(fig, ax, config)
    >>> animation = anim.render_animation(
    ...     nframes=100,
    ...     save_path='sine_wave.mp4'
    ... )

    Heatmap animation with blitting:

    >>> class HeatmapAnimation(OptimizedAnimationBase):
    ...     def __init__(self, fig, ax, data_func, config=None):
    ...         super().__init__(fig, ax, config)
    ...         self.data_func = data_func
    ...         self.im = None
    ...     
    ...     def create_artists(self):
    ...         # Pre-create image with initial data
    ...         initial_data = self.data_func(0)
    ...         self.im = self.ax.imshow(
    ...             initial_data,
    ...             animated=True,
    ...             cmap='viridis'
    ...         )
    ...         plt.colorbar(self.im, ax=self.ax)
    ...         return [self.im]
    ...     
    ...     def update_frame(self, frame_idx):
    ...         # Only update array data
    ...         data = self.data_func(frame_idx)
    ...         self.im.set_array(data)
    ...         return (self.im,)
    >>> 
    >>> # Use with dynamic data
    >>> def generate_heatmap(frame):
    ...     return np.random.randn(50, 50) * (frame / 100)
    >>> 
    >>> fig, ax = plt.subplots()
    >>> anim = HeatmapAnimation(fig, ax, generate_heatmap)
    >>> animation = anim.render_animation(nframes=200)

    See Also
    --------
    AnimationConfig : Configuration for animation rendering
    ParallelAnimationRenderer : Parallel rendering for long animations
    create_optimized_writer : Create optimized video writer

    Notes
    -----
    **Performance Benefits:**
    - Blitting: 2-10x faster than standard matplotlib animations
    - Parallel rendering: 3-4x additional speedup for >500 frames
    - Artist pre-creation eliminates per-frame overhead

    **Best Practices:**
    1. Create all artists once in create_artists()
    2. Only update data in update_frame(), never recreate objects
    3. Set animated=True on all artists that will change
    4. Use numpy arrays for efficient data updates
    5. Avoid calling ax.clear() or fig.clear() in update_frame()

    **Blitting Requirements:**
    - Backend must support blitting (TkAgg, Qt5Agg work; some don't)
    - Artists must have animated=True
    - Only works with backends that have canvas.copy_from_bbox()
    - Automatically disabled if backend doesn't support it

    **When to Use Parallel Rendering:**
    - Animations with >500 frames
    - Each frame takes significant time to compute
    - Multi-core CPU available
    - Using imageio backend (matplotlib doesn't support parallel)

    References
    ----------
    .. [1] Matplotlib Animation Tutorial: https://matplotlib.org/stable/api/animation_api.html
    .. [2] Blitting for speed: https://matplotlib.org/stable/tutorials/advanced/blitting.html
    """

    def __init__(self, fig, ax, config: AnimationConfig | None = None):
        """Initialize the animation base.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to animate
        ax : matplotlib.axes.Axes
            Matplotlib axes to animate
        config : AnimationConfig or None, default=None
            Animation configuration (uses defaults if None)
        """
        self.fig = fig
        self.ax = ax
        self.config = config or AnimationConfig()
        self.artists: list[Artist] = []

        # Check backend blitting support
        self._blitting_supported = self._check_blitting_support()
        if not self._blitting_supported and self.config.enable_blitting:
            warnings.warn(
                "Backend does not support blitting. Falling back to non-blitted mode.",
                UserWarning,
                stacklevel=2,
            )
            self.config.enable_blitting = False

    def _check_blitting_support(self) -> bool:
        """Check if the current backend supports blitting.

        Returns:
            True if blitting is supported, False otherwise
        """
        try:
            return self.fig.canvas.supports_blit
        except AttributeError:
            return False

    @abstractmethod
    def create_artists(self) -> list[Artist]:
        """Pre-create all artist objects for the animation.

        This method should:
        1. Create all plot objects (lines, scatter, images, etc.)
        2. Set animated=True for objects that will be updated
        3. Set initial data (can be empty with [], [])
        4. Return list of all animated artists

        Returns:
            List of artist objects that will be animated
        """
        pass

    @abstractmethod
    def update_frame(self, frame_idx: int) -> tuple[Artist, ...]:
        """Update artists for a specific frame.

        This method should:
        1. Compute data for the current frame
        2. Update artist data using set_data(), set_array(), etc.
        3. Return tuple of all modified artists

        Important: Do NOT call ax.clear() or recreate artists here!

        Args:
            frame_idx: Index of the current frame

        Returns:
            Tuple of modified artist objects
        """
        pass

    def init_func(self) -> tuple[Artist, ...]:
        """Initialize animation (called by FuncAnimation).

        Returns:
            Tuple of all animated artists
        """
        return tuple(self.artists)

    def render_animation(
        self,
        nframes: int,
        interval: int | None = None,
        repeat: bool = True,
        save_path: str | None = None,
        **save_kwargs,
    ) -> FuncAnimation:
        """Render the animation with automatic optimization selection.

        Args:
            nframes: Total number of frames
            interval: Milliseconds between frames (computed from fps if None)
            repeat: Whether to loop the animation
            save_path: Path to save animation (None to skip saving)
            **save_kwargs: Additional arguments for animation.save()

        Returns:
            FuncAnimation object
        """
        # Pre-create all artists
        self.artists = self.create_artists()

        # Compute interval from fps if not specified
        if interval is None:
            interval = 1000 // self.config.fps

        # Decide whether to use parallel rendering
        use_parallel = self.config.use_parallel or nframes > self.config.auto_parallel_threshold

        if use_parallel and save_path:
            # Use parallel rendering for long animations
            return self._render_parallel(nframes, save_path, **save_kwargs)
        else:
            # Use standard FuncAnimation with blitting
            ani = FuncAnimation(
                self.fig,
                self.update_frame,
                frames=nframes,
                init_func=self.init_func,
                blit=self.config.enable_blitting,
                interval=interval,
                repeat=repeat,
            )

            if save_path:
                ani.save(save_path, **save_kwargs)

            return ani

    def _render_parallel(self, nframes: int, save_path: str, **save_kwargs) -> FuncAnimation:
        """Render animation using parallel workers (for long animations).

        Args:
            nframes: Total number of frames
            save_path: Path to save animation
            **save_kwargs: Additional arguments for saving

        Returns:
            FuncAnimation object (for API compatibility)
        """
        # Import here to avoid circular dependency
        from .rendering import ParallelAnimationRenderer

        renderer = ParallelAnimationRenderer(num_workers=self.config.num_workers)

        renderer.render(
            animation_base=self,
            nframes=nframes,
            fps=self.config.fps,
            save_path=save_path,
            **save_kwargs,
        )

        # Return a dummy FuncAnimation for API compatibility
        # (in parallel mode, we've already saved the animation)
        return FuncAnimation(
            self.fig,
            self.update_frame,
            frames=1,  # Single frame to satisfy API
            blit=False,
        )


def supports_blitting() -> bool:
    """Check if the current matplotlib backend supports blitting.

    Returns:
        True if blitting is supported, False otherwise
    """
    try:
        fig = plt.figure()
        result = fig.canvas.supports_blit
        plt.close(fig)
        return result
    except (AttributeError, ValueError):
        return False


def create_buffer(shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """Pre-allocate a numpy buffer for efficient in-place updates.

    Args:
        shape: Shape of the buffer array
        dtype: Data type (default: float32 for memory efficiency)

    Returns:
        Pre-allocated numpy array
    """
    return np.zeros(shape, dtype=dtype)


def optimize_colormap(
    data: np.ndarray,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Pre-compute colormap normalization for efficient color mapping.

    Args:
        data: Data array to normalize
        cmap_name: Name of the colormap
        vmin: Minimum value for normalization (computed if None)
        vmax: Maximum value for normalization (computed if None)

    Returns:
        Tuple of (colormap function, vmin, vmax)
    """
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))

    cmap = plt.get_cmap(cmap_name)

    return cmap, vmin, vmax
