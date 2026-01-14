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
    """Available animation rendering backends."""

    IMAGEIO = "imageio"  # Supports parallel rendering, requires imageio package
    MATPLOTLIB = "matplotlib"  # Single-threaded, always available
    AUTO = "auto"  # Automatically select best backend


@dataclass
class BackendSelection:
    """Result of backend selection process."""

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

    Args:
        save_path: Output file path
        fps: Frames per second

    Returns:
        Tuple of (writer_kwargs, mode) where mode is for get_writer()

    Example:
        >>> kwargs, mode = get_imageio_writer_kwargs("output.gif", 10)
        >>> writer = imageio.get_writer("output.gif", mode=mode, **kwargs)
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

    Returns:
        Number of workers (cpu_count - 1, minimum 1)
    """
    import multiprocessing as mp

    return max(mp.cpu_count() - 1, 1)


def get_multiprocessing_context(prefer_fork: bool = False):
    """
    Get appropriate multiprocessing context for this platform.

    Args:
        prefer_fork: Whether to prefer 'fork' over 'spawn' (Linux only)

    Returns:
        Multiprocessing context or None if unavailable
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
    """Emit all backend selection warnings."""
    for warning_msg in warnings_list:
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=stacklevel)
