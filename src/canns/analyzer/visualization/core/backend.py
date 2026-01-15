"""
Unified animation backend selection and management.

This module provides a centralized system for choosing the optimal rendering backend
(imageio vs matplotlib) based on file format, available dependencies, and user preferences.
"""

import os
import platform
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class AnimationBackend(Enum):
    """
    Available animation rendering backends.

    Enumeration of supported backends for rendering matplotlib animations
    to video files. Each backend has different performance characteristics
    and dependency requirements.

    Attributes
    ----------
    IMAGEIO : str
        Use imageio library for rendering. Supports parallel rendering for
        significant speedup on multi-core systems. Requires imageio package.
        Best for: Large animations (>500 frames), when imageio is installed.
    MATPLOTLIB : str
        Use matplotlib's built-in writers. Single-threaded but always
        available with no extra dependencies. Generally slower than imageio.
        Best for: Small animations, when imageio is unavailable.
    AUTO : str
        Automatically select the best backend based on file format and
        available dependencies. Preferred option for most users.

    Example
    -------
    Specify backend explicitly:

    >>> from canns.analyzer.visualization.core.backend import AnimationBackend
    >>> from canns.analyzer.visualization.core.backend import select_animation_backend
    >>> 
    >>> # Request specific backend
    >>> selection = select_animation_backend(
    ...     "output.mp4",
    ...     requested_backend=AnimationBackend.IMAGEIO.value
    ... )
    >>> print(f"Using: {selection.backend}")
    Using: imageio

    Use AUTO selection (recommended):

    >>> # Let system choose optimal backend
    >>> selection = select_animation_backend(
    ...     "output.gif",
    ...     requested_backend=AnimationBackend.AUTO.value
    ... )
    >>> print(f"Selected {selection.backend}: {selection.reason}")
    Selected imageio: Parallel rendering available for GIF

    See Also
    --------
    BackendSelection : Result of backend selection process
    select_animation_backend : Smart backend selection function

    Notes
    -----
    - **IMAGEIO**: 3-4x faster for large animations, requires `pip install imageio`
    - **MATPLOTLIB**: Always available, single-threaded
    - **AUTO**: Recommended - selects imageio when available, falls back to matplotlib
    """

    IMAGEIO = "imageio"  # Supports parallel rendering, requires imageio package
    MATPLOTLIB = "matplotlib"  # Single-threaded, always available
    AUTO = "auto"  # Automatically select best backend


@dataclass
class BackendSelection:
    """
    Result of backend selection process.

    Contains the selected backend along with metadata about the selection,
    including performance capabilities and any warnings for the user.

    Attributes
    ----------
    backend : Literal["imageio", "matplotlib"]
        The selected backend name
    supports_parallel : bool
        Whether this backend supports parallel frame rendering
    reason : str
        Human-readable explanation of why this backend was selected
    warnings : list[str]
        Any warnings or suggestions for the user about the selection

    Example
    -------
    Get backend selection information:

    >>> from canns.analyzer.visualization.core.backend import select_animation_backend
    >>> 
    >>> # Select backend for GIF
    >>> selection = select_animation_backend("output.gif")
    >>> print(f"Backend: {selection.backend}")
    Backend: imageio
    >>> print(f"Supports parallel: {selection.supports_parallel}")
    Supports parallel: True
    >>> print(f"Reason: {selection.reason}")
    Reason: Parallel rendering available for GIF
    >>> 
    >>> # Check for warnings
    >>> if selection.warnings:
    ...     print("Warnings:", selection.warnings)

    Handle fallback to matplotlib:

    >>> # When imageio unavailable
    >>> selection = select_animation_backend("output.mp4", "auto")
    >>> if selection.backend == "matplotlib":
    ...     print(f"Using fallback: {selection.reason}")
    ...     for warning in selection.warnings:
    ...         print(f"Warning: {warning}")

    Use selection result:

    >>> # Use selected backend in rendering
    >>> if selection.supports_parallel and num_frames > 500:
    ...     # Use parallel rendering
    ...     renderer = ParallelAnimationRenderer()
    ... else:
    ...     # Use standard matplotlib rendering
    ...     anim = FuncAnimation(fig, update_frame, frames=num_frames)

    See Also
    --------
    AnimationBackend : Enumeration of available backends
    select_animation_backend : Function that creates BackendSelection

    Notes
    -----
    - This is a dataclass with immutable fields (frozen=False by default)
    - warnings list may be empty if selection is optimal
    - supports_parallel indicates potential speedup for large animations
    - reason provides context for debugging and user communication
    """

    backend: Literal["imageio", "matplotlib"]
    """The selected backend."""

    supports_parallel: bool
    """Whether this backend supports parallel rendering."""

    reason: str
    """Why this backend was selected."""

    warnings: list[str]
    """Any warnings or suggestions for the user."""


def select_animation_backend(
    save_path: str | None,
    requested_backend: str | None = None,
    check_imageio_plugins: bool = True,
) -> BackendSelection:
    """
    Select the optimal animation rendering backend.

    This function implements smart backend selection logic:
    1. If user explicitly requests a backend, validate and use it
    2. Otherwise, auto-select based on file format and available dependencies
    3. For GIF: prefer imageio (parallel rendering)
    4. For MP4: prefer imageio if plugins available, else matplotlib
    5. Always fallback gracefully with helpful warnings

    Args:
        save_path: Output file path (determines format)
        requested_backend: User's backend preference ('imageio', 'matplotlib', 'auto', or None)
        check_imageio_plugins: Whether to verify imageio can write the format

    Returns:
        BackendSelection with backend choice and metadata

    Example:
        >>> selection = select_animation_backend("output.mp4")
        >>> print(f"Using {selection.backend}: {selection.reason}")
        Using imageio: Parallel rendering available for MP4

        >>> selection = select_animation_backend("output.gif", "matplotlib")
        >>> print(selection.warnings)
        ['Consider using imageio backend for faster GIF rendering']
    """
    warnings_list = []

    # Normalize requested backend
    backend_requested = (requested_backend or "auto").lower()
    auto_select = backend_requested in {"auto", "none", ""}

    # Get file extension
    file_ext = _get_file_extension(save_path) if save_path else None

    # Check if imageio is available
    imageio_available = _check_imageio_available()

    if not auto_select:
        # User explicitly requested a backend - validate and use it
        if backend_requested not in {"imageio", "matplotlib"}:
            raise ValueError(
                f"Invalid render_backend='{backend_requested}'. "
                f"Must be 'imageio', 'matplotlib', or 'auto'."
            )

        if backend_requested == "imageio":
            if not imageio_available:
                raise ImportError(
                    "render_backend='imageio' requires the imageio package. "
                    "Install with: uv add imageio"
                )

            # Check if imageio can handle this format
            if file_ext not in {".gif", None} and check_imageio_plugins:
                can_write = _check_imageio_format_support(file_ext)
                if not can_write:
                    raise ValueError(
                        f"imageio cannot write '{file_ext}' format (missing plugin). "
                        f"Install with: uv add 'imageio[ffmpeg]' or uv add 'imageio[pyav]'. "
                        f"Or use render_backend='matplotlib'."
                    )

            return BackendSelection(
                backend="imageio",
                supports_parallel=True,
                reason=f"User explicitly requested imageio backend",
                warnings=[],
            )

        # User requested matplotlib
        return BackendSelection(
            backend="matplotlib",
            supports_parallel=False,
            reason="User explicitly requested matplotlib backend",
            warnings=[],
        )

    # Auto-selection logic
    if not imageio_available:
        # imageio not installed - must use matplotlib
        warnings_list.append(
            "Using matplotlib backend (single-threaded). "
            "For faster rendering, install: uv add imageio"
        )
        return BackendSelection(
            backend="matplotlib",
            supports_parallel=False,
            reason="imageio not installed",
            warnings=warnings_list,
        )

    # imageio is available - check format support
    if file_ext == ".gif":
        # GIF: imageio is ideal (always works, supports parallel)
        return BackendSelection(
            backend="imageio",
            supports_parallel=True,
            reason="imageio provides optimal GIF rendering with parallel processing",
            warnings=[],
        )

    elif file_ext in {".mp4", ".m4v", ".mov", ".avi", ".webm"}:
        # Video format: check if imageio has required plugins
        if check_imageio_plugins:
            can_write = _check_imageio_format_support(file_ext)
            if not can_write:
                # imageio can't write this format - fallback to matplotlib
                warnings_list.append(
                    f"imageio cannot write '{file_ext}' (missing plugin). Using matplotlib. "
                    f"For faster parallel rendering, install: uv add 'imageio[ffmpeg]'"
                )
                return BackendSelection(
                    backend="matplotlib",
                    supports_parallel=False,
                    reason=f"imageio missing plugin for {file_ext}",
                    warnings=warnings_list,
                )

        # imageio can handle this format
        return BackendSelection(
            backend="imageio",
            supports_parallel=True,
            reason=f"imageio provides parallel rendering for {file_ext}",
            warnings=[],
        )

    else:
        # Unknown or no format - prefer imageio for parallel rendering
        return BackendSelection(
            backend="imageio",
            supports_parallel=True,
            reason="imageio supports parallel rendering",
            warnings=[],
        )


def get_imageio_writer_kwargs(save_path: str, fps: int) -> tuple[dict, str | None]:
    """
    Get appropriate kwargs for imageio.get_writer() based on file format.

    Determines optimal imageio writer configuration based on the output file
    extension. Returns format-specific parameters for GIF or video encoding.

    Parameters
    ----------
    save_path : str
        Output file path (extension determines format)
    fps : int
        Frames per second for the animation

    Returns
    -------
    tuple[dict, str | None]
        Tuple of (writer_kwargs, mode):
        - writer_kwargs: Dictionary of parameters for imageio.get_writer()
        - mode: Mode string for get_writer() ('I' for GIF, None for video)

    Example
    -------
    Get writer configuration and create imageio writer:

    >>> from canns.analyzer.visualization.core.backend import get_imageio_writer_kwargs
    >>> import imageio
    >>> 
    >>> # Get configuration for GIF output
    >>> kwargs, mode = get_imageio_writer_kwargs("animation.gif", fps=10)
    >>> print(kwargs)
    {'duration': 0.1, 'loop': 0}
    >>> print(mode)
    'I'
    >>> 
    >>> # Create writer with configuration
    >>> with imageio.get_writer("animation.gif", mode=mode, **kwargs) as writer:
    ...     for frame in frames:
    ...         writer.append_data(frame)

    MP4 video configuration:

    >>> # Get configuration for MP4 output
    >>> kwargs, mode = get_imageio_writer_kwargs("video.mp4", fps=30)
    >>> print(kwargs)
    {'fps': 30, 'codec': 'libx264', 'pixelformat': 'yuv420p'}
    >>> print(mode)
    None
    >>> 
    >>> # MP4 writer uses different parameters
    >>> with imageio.get_writer("video.mp4", **kwargs) as writer:
    ...     for frame in frames:
    ...         writer.append_data(frame)

    See Also
    --------
    select_animation_backend : High-level backend selection
    create_optimized_writer : Create optimized animation writer

    Notes
    -----
    **GIF format** (.gif):
    - duration: Time per frame (1/fps)
    - loop: 0 for infinite loop
    - mode: 'I' (image mode)

    **Video formats** (.mp4, .avi, etc.):
    - fps: Frames per second
    - codec: libx264 (H.264 encoding)
    - pixelformat: yuv420p (widely compatible)
    - mode: None (default)

    The returned kwargs can be passed directly to imageio.get_writer() via **kwargs unpacking.
    """
    file_ext = _get_file_extension(save_path)

    if file_ext == ".gif":
        # GIF-specific parameters
        return {
            "duration": 1.0 / fps,
            "loop": 0,
        }, "I"
    else:
        # MP4/video parameters
        return {
            "fps": fps,
            "codec": "libx264",
            "pixelformat": "yuv420p",
        }, None


def _get_file_extension(save_path: str | None) -> str | None:
    """Extract lowercase file extension from path."""
    if save_path is None:
        return None
    return os.path.splitext(str(save_path))[1].lower()


def _check_imageio_available() -> bool:
    """Check if imageio package is available."""
    try:
        import imageio  # noqa: F401

        return True
    except ImportError:
        return False


def _check_imageio_format_support(file_ext: str) -> bool:
    """
    Check if imageio can write the given format.

    This does a quick check by trying to create a writer.
    """
    if file_ext == ".gif":
        # GIF is always supported by imageio
        return True

    try:
        import imageio

        # Try to create a writer for this format
        test_path = f"_test_writer_check{file_ext}"
        writer = imageio.get_writer(test_path, fps=1)
        writer.close()

        # Clean up test file
        if os.path.exists(test_path):
            os.remove(test_path)

        return True
    except Exception:
        return False


def get_optimal_worker_count() -> int:
    """
    Get optimal number of parallel workers for this system.

    Calculates the recommended worker count for parallel rendering by using
    cpu_count - 1 (leaving one core free for the main process), with a
    minimum of 1 worker.

    Returns
    -------
    int
        Number of workers (cpu_count - 1, minimum 1)

    Example
    -------
    Determine optimal parallelism for animation rendering:

    >>> from canns.analyzer.visualization.core.backend import get_optimal_worker_count
    >>> 
    >>> # Get recommended worker count
    >>> num_workers = get_optimal_worker_count()
    >>> print(f"Using {num_workers} workers for parallel rendering")
    Using 3 workers for parallel rendering
    >>> 
    >>> # Use in parallel rendering configuration
    >>> from canns.analyzer.visualization.core.rendering import ParallelAnimationRenderer
    >>> renderer = ParallelAnimationRenderer(num_workers=num_workers)

    See Also
    --------
    get_multiprocessing_context : Get appropriate multiprocessing context
    ParallelAnimationRenderer : Multi-process parallel renderer

    Notes
    -----
    - Leaves one CPU core free to avoid system slowdown
    - Minimum of 1 worker ensures functionality on single-core systems
    - Worker count affects memory usage (each worker needs separate memory)
    """
    import multiprocessing as mp

    return max(mp.cpu_count() - 1, 1)


def get_multiprocessing_context(prefer_fork: bool = False):
    """
    Get appropriate multiprocessing context for this platform.

    Selects the optimal multiprocessing start method based on platform and
    loaded libraries. Automatically handles JAX compatibility issues and
    platform-specific constraints.

    Parameters
    ----------
    prefer_fork : bool, default=False
        Whether to prefer 'fork' over 'spawn' on Linux. Fork is faster but
        incompatible with some libraries (e.g., JAX). Only effective on Linux.

    Returns
    -------
    multiprocessing.context.BaseContext or None
        Multiprocessing context suitable for this platform, or None if
        multiprocessing is unavailable

    Example
    -------
    Create multiprocessing context for parallel rendering:

    >>> from canns.analyzer.visualization.core.backend import get_multiprocessing_context
    >>> import multiprocessing as mp
    >>> 
    >>> # Get appropriate context
    >>> ctx = get_multiprocessing_context()
    >>> if ctx:
    ...     # Use context to create worker pool
    ...     with ctx.Pool(processes=4) as pool:
    ...         results = pool.map(process_frame, frame_indices)
    >>> 
    >>> # Force spawn method (safer with JAX/TensorFlow)
    >>> ctx = get_multiprocessing_context(prefer_fork=False)

    Using with parallel animation rendering:

    >>> from canns.analyzer.visualization.core import ParallelAnimationRenderer
    >>> 
    >>> # Context is automatically handled internally
    >>> renderer = ParallelAnimationRenderer(num_workers=4)
    >>> # render() will use appropriate context based on platform

    See Also
    --------
    get_optimal_worker_count : Get recommended number of workers
    ParallelAnimationRenderer : Renderer using multiprocessing

    Notes
    -----
    - **'spawn'** (default): Works on all platforms, slower startup
    - **'fork'** (Linux only): Faster startup but incompatible with JAX
    - Automatically detects JAX and avoids 'fork' to prevent deadlocks
    - Returns None if multiprocessing is unavailable
    - Context must be obtained before creating worker pools

    Warnings
    --------
    - Using 'fork' with JAX/TensorFlow can cause deadlocks
    - The function automatically detects and warns about this
    """
    import multiprocessing as mp

    # Determine best start method
    if prefer_fork and platform.system() == "Linux":
        try:
            # Check for JAX which doesn't work with fork
            import sys

            if any(name.startswith("jax") for name in sys.modules):
                warnings.warn(
                    "Detected JAX; using 'spawn' instead of 'fork' to avoid deadlocks.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return mp.get_context("spawn")
            return mp.get_context("fork")
        except (RuntimeError, ValueError):
            pass

    # Default to spawn (works everywhere)
    try:
        return mp.get_context("spawn")
    except (RuntimeError, ValueError):
        return None


def emit_backend_warnings(warnings_list: list[str], stacklevel: int = 2):
    """
    Emit all backend selection warnings to the user.

    Iterates through a list of warning messages and emits each as a
    RuntimeWarning. Used by backend selection logic to inform users about
    sub-optimal configurations or missing dependencies.

    Parameters
    ----------
    warnings_list : list[str]
        List of warning messages to emit
    stacklevel : int, default=2
        Stack level for warning origin (2 points to caller of the function
        that called emit_backend_warnings)

    Example
    -------
    Emit warnings from backend selection:

    >>> from canns.analyzer.visualization.core.backend import emit_backend_warnings
    >>> 
    >>> # Collect warnings during backend selection
    >>> warnings = []
    >>> if not imageio_available:
    ...     warnings.append("imageio not found; using slower matplotlib backend")
    >>> if file_format == "gif":
    ...     warnings.append("GIF encoding is slower than MP4; consider using .mp4")
    >>> 
    >>> # Emit all warnings at once
    >>> emit_backend_warnings(warnings)

    Internal usage in select_animation_backend:

    >>> from canns.analyzer.visualization.core.backend import select_animation_backend
    >>> 
    >>> selection = select_animation_backend("output.gif", "matplotlib")
    >>> # Warnings automatically emitted about sub-optimal choice
    >>> print(selection.warnings)
    ['Consider using imageio backend for faster GIF rendering']

    See Also
    --------
    select_animation_backend : Smart backend selection with automatic warnings
    BackendSelection : Result object containing warning list

    Notes
    -----
    - All warnings are emitted as RuntimeWarning type
    - Stacklevel controls where the warning appears to originate
    - Empty list is safe (no warnings emitted)
    - Used internally by backend selection logic
    """
    for warning_msg in warnings_list:
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=stacklevel)
