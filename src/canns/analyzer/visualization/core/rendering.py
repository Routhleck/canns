"""
Parallel frame rendering engine for long matplotlib animations.

This module provides multi-process rendering capabilities for animations with
hundreds or thousands of frames, achieving 3-4x speedup on multi-core CPUs.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Note: Backend is set to 'Agg' inside worker processes, not at module import time

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    warnings.warn(
        "imageio not available. Install with 'pip install imageio' for parallel rendering.",
        ImportWarning,
        stacklevel=2,
    )


class ParallelAnimationRenderer:
    """Multi-process parallel renderer for matplotlib animations.

    This renderer creates separate processes to render frames in parallel,
    then combines them into a video file using imageio. Best for animations
    with >500 frames where the rendering bottleneck is matplotlib itself.

    Performance: Achieves ~3-4x speedup on 4-core CPUs.
    """

    def __init__(self, num_workers: int | None = None):
        """Initialize the parallel renderer.

        Args:
            num_workers: Number of worker processes (uses CPU count if None)
        """
        self.num_workers = num_workers or cpu_count()

    def render(
        self,
        animation_base: Any,  # OptimizedAnimationBase instance
        nframes: int,
        fps: int,
        save_path: str,
        writer: str = "ffmpeg",
        codec: str = "libx264",
        bitrate: int | None = None,
        show_progress: bool = True,
    ) -> None:
        """Render animation frames in parallel and save to file.

        Args:
            animation_base: OptimizedAnimationBase instance with update_frame method
            nframes: Total number of frames to render
            fps: Frames per second
            save_path: Output file path
            writer: Video writer to use ('ffmpeg' or 'pillow')
            codec: Video codec (for ffmpeg writer)
            bitrate: Video bitrate in kbps (None for automatic)
            show_progress: Whether to show progress bar
        """
        if not IMAGEIO_AVAILABLE:
            raise ImportError(
                "imageio is required for parallel rendering. Install with: pip install imageio"
            )

        # Warn about experimental status
        warnings.warn(
            "Parallel rendering is experimental and may not work for all animation types "
            "due to matplotlib object pickling limitations. If you encounter errors, "
            "use standard rendering (disable use_parallel in AnimationConfig).",
            UserWarning,
            stacklevel=3,
        )

        # Create frame rendering tasks
        print(f"Rendering {nframes} frames using {self.num_workers} workers...")

        # Use ProcessPoolExecutor for parallel rendering
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all frame rendering tasks
            future_to_frame = {
                executor.submit(_render_single_frame_worker, animation_base, frame_idx): frame_idx
                for frame_idx in range(nframes)
            }

            # Collect rendered frames in order
            frames = [None] * nframes
            completed = 0

            for future in as_completed(future_to_frame):
                frame_idx = future_to_frame[future]
                try:
                    frame_data = future.result()
                    frames[frame_idx] = frame_data
                    completed += 1

                    if show_progress and completed % 10 == 0:
                        print(f"Rendered {completed}/{nframes} frames...")

                except Exception as e:
                    warnings.warn(
                        f"Failed to render frame {frame_idx}: {e}", RuntimeWarning, stacklevel=2
                    )
                    # Create blank frame as fallback
                    frames[frame_idx] = np.zeros((480, 640, 3), dtype=np.uint8)

        # Save frames to video file
        print(f"Saving animation to {save_path}...")
        self._save_video(frames, save_path, fps, writer, codec, bitrate)
        print("Animation saved successfully!")

    def _save_video(
        self,
        frames: list[np.ndarray],
        save_path: str,
        fps: int,
        writer: str,
        codec: str,
        bitrate: int | None,
    ) -> None:
        """Save rendered frames to video file using imageio.

        Args:
            frames: List of frame arrays (H, W, 3) in RGB format
            save_path: Output file path
            fps: Frames per second
            writer: Video writer ('ffmpeg' or 'pillow')
            codec: Video codec
            bitrate: Video bitrate in kbps
        """
        # Configure writer based on file extension and settings
        if writer == "ffmpeg" and save_path.endswith(".mp4"):
            writer_kwargs = {
                "fps": fps,
                "codec": codec,
                "pixelformat": "yuv420p",
            }
            if bitrate:
                writer_kwargs["bitrate"] = f"{bitrate}k"

            with imageio.get_writer(save_path, **writer_kwargs) as video_writer:
                for frame in frames:
                    if frame is not None:
                        # Ensure RGB format
                        if frame.shape[-1] == 4:  # RGBA
                            frame = frame[:, :, :3]
                        video_writer.append_data(frame)

        elif save_path.endswith(".gif"):
            # Use Pillow writer for GIF
            with imageio.get_writer(save_path, mode="I", fps=fps) as video_writer:
                for frame in frames:
                    if frame is not None:
                        if frame.shape[-1] == 4:  # RGBA
                            frame = frame[:, :, :3]
                        video_writer.append_data(frame)

        else:
            # Default: use imageio's auto-detection
            with imageio.get_writer(save_path, fps=fps) as video_writer:
                for frame in frames:
                    if frame is not None:
                        if frame.shape[-1] == 4:  # RGBA
                            frame = frame[:, :, :3]
                        video_writer.append_data(frame)


def _render_single_frame_worker(animation_base: Any, frame_idx: int) -> np.ndarray:
    """Worker function to render a single frame in a separate process.

    This function is called by ProcessPoolExecutor workers. Each worker
    creates its own matplotlib figure, renders one frame, and returns
    the pixel data.

    Args:
        animation_base: OptimizedAnimationBase instance
        frame_idx: Index of the frame to render

    Returns:
        Frame data as numpy array (H, W, 3) in RGB format

    Note:
        Parallel rendering is experimental. The animation_base instance must be
        picklable, which may not work for all animation types due to matplotlib
        object serialization limitations.
    """
    # Set non-interactive backend for this worker process
    matplotlib.use("Agg")

    # Each worker needs to recreate the figure and setup
    # (Can't pickle matplotlib objects across processes)
    fig = Figure(figsize=animation_base.fig.get_size_inches(), dpi=animation_base.fig.dpi)
    ax = fig.add_subplot(111)

    # Copy relevant plot settings
    ax.set_xlim(animation_base.ax.get_xlim())
    ax.set_ylim(animation_base.ax.get_ylim())
    if hasattr(animation_base.ax, "get_zlim"):
        ax.set_zlim(animation_base.ax.get_zlim())

    # Create artists for this worker
    worker_animation = animation_base.__class__(fig, ax, animation_base.config)
    worker_animation.artists = worker_animation.create_artists()

    # Update frame
    worker_animation.update_frame(frame_idx)

    # Render to canvas
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Extract pixel data
    buf = canvas.buffer_rgba()
    frame_data = np.frombuffer(buf, dtype=np.uint8)
    w, h = canvas.get_width_height()
    frame_data = frame_data.reshape((h, w, 4))

    # Convert RGBA to RGB
    frame_rgb = frame_data[:, :, :3].copy()

    # Clean up
    plt.close(fig)

    return frame_rgb


def estimate_parallel_speedup(nframes: int, num_workers: int = 4) -> float:
    """Estimate speedup from parallel rendering.

    Args:
        nframes: Number of frames
        num_workers: Number of parallel workers

    Returns:
        Estimated speedup factor
    """
    # Parallel rendering has overhead, so speedup is sublinear
    # Empirically: ~3-4x speedup with 4 workers for long animations
    if nframes < 100:
        return 1.0  # No benefit for short animations
    elif nframes < 500:
        return min(2.0, num_workers * 0.6)
    else:
        # Long animations see best speedup
        return min(num_workers * 0.8, num_workers)


def should_use_parallel(
    nframes: int, estimated_frame_time: float, threshold_seconds: float = 30.0
) -> bool:
    """Determine if parallel rendering would be beneficial.

    Args:
        nframes: Number of frames
        estimated_frame_time: Estimated time per frame in seconds
        threshold_seconds: Use parallel if total time exceeds this

    Returns:
        True if parallel rendering is recommended
    """
    estimated_total_time = nframes * estimated_frame_time
    return estimated_total_time > threshold_seconds


def render_animation_parallel(
    render_frame_func,
    frame_data,
    num_frames: int,
    save_path: str,
    fps: int = 10,
    num_workers: int | None = None,
    show_progress: bool = True,
    file_format: str | None = None,
):
    """Universal parallel animation renderer for analyzer animations.

    Args:
        render_frame_func: Callable that renders a single frame:
            ``func(frame_idx, frame_data) -> np.ndarray (H, W, 3 or 4)``.
        frame_data: Data needed by ``render_frame_func`` (passed to workers).
        num_frames: Total number of frames to render.
        save_path: Output file path (extension determines format).
        fps: Frames per second.
        num_workers: Number of parallel workers (None = auto-detect).
        show_progress: Whether to show progress bar.
        file_format: Override file format detection ('gif', 'mp4', etc.).

    Returns:
        None (saves animation to file).

    Examples:
        >>> import numpy as np
        >>> import tempfile
        >>> from pathlib import Path
        >>> from canns.analyzer.visualization.core.rendering import render_animation_parallel
        >>> from canns.analyzer.visualization.core import rendering
        >>>
        >>> def render_frame(idx, data):
        ...     frame = data[idx]
        ...     return frame  # (H, W, 3)
        >>>
        >>> frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)]
        >>> # Save a tiny animation if imageio is available
        >>> if rendering.IMAGEIO_AVAILABLE:
        ...     with tempfile.TemporaryDirectory() as tmpdir:
        ...         save_path = Path(tmpdir) / "demo.gif"
        ...         render_animation_parallel(
        ...             render_frame, frames, num_frames=2, save_path=str(save_path), fps=2
        ...         )
        ...         print("saved")
        ... else:
        ...     print("imageio not available")
    """
    import os
    import multiprocessing as mp
    import platform
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    # Detect file format
    if file_format is None:
        ext = os.path.splitext(save_path)[1].lower()
        if ext in {".gif"}:
            file_format = "gif"
        elif ext in {".mp4", ".m4v", ".mov", ".avi", ".webm"}:
            file_format = "mp4"
        else:
            file_format = "mp4"  # default

    # Auto-detect number of workers
    if num_workers is None:
        num_workers = max(mp.cpu_count() - 1, 1)

    # Determine if we should use parallel rendering
    use_parallel = num_frames >= 50 and num_workers > 1

    # Setup progress bar
    progress_bar = None
    if show_progress:
        desc = f"<render_animation> Saving to {os.path.basename(save_path)}"
        progress_bar = tqdm(total=num_frames, desc=desc)

    try:
        if file_format == "gif":
            # GIF: Use imageio with direct parallel write
            _render_gif_parallel(
                render_frame_func,
                frame_data,
                num_frames,
                save_path,
                fps,
                num_workers if use_parallel else 1,
                progress_bar,
            )
        else:
            # MP4: Use parallel render + FFMpegWriter
            _render_mp4_parallel(
                render_frame_func,
                frame_data,
                num_frames,
                save_path,
                fps,
                num_workers if use_parallel else 1,
                progress_bar,
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()


def _render_gif_parallel(
    render_frame_func,
    frame_data,
    num_frames: int,
    save_path: str,
    fps: int,
    num_workers: int,
    progress_bar,
):
    """Render GIF with parallel processing using imageio."""
    import multiprocessing as mp
    import platform

    if not IMAGEIO_AVAILABLE:
        raise ImportError(
            "imageio is required for GIF rendering. Install with: uv pip install imageio"
        )

    writer_kwargs = {"duration": 1.0 / fps, "loop": 0}

    use_parallel = num_workers > 1
    ctx = None
    if use_parallel:
        try:
            start_method = "fork" if platform.system() == "Linux" else "spawn"
            ctx = mp.get_context(start_method)
        except (RuntimeError, ValueError):
            use_parallel = False
            warnings.warn(
                "Multiprocessing unavailable; falling back to sequential rendering.",
                RuntimeWarning,
                stacklevel=3,
            )

    with imageio.get_writer(save_path, mode="I", **writer_kwargs) as writer:
        if use_parallel and ctx is not None:
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                for frame_image in executor.map(
                    render_frame_func,
                    range(num_frames),
                    [frame_data] * num_frames,
                ):
                    writer.append_data(frame_image)
                    if progress_bar is not None:
                        progress_bar.update(1)
        else:
            for frame_idx in range(num_frames):
                frame_image = render_frame_func(frame_idx, frame_data)
                writer.append_data(frame_image)
                if progress_bar is not None:
                    progress_bar.update(1)


def _render_mp4_parallel(
    render_frame_func,
    frame_data,
    num_frames: int,
    save_path: str,
    fps: int,
    num_workers: int,
    progress_bar,
):
    """Render MP4 with parallel frame rendering then write with imageio/FFMpeg."""
    import multiprocessing as mp
    import platform

    use_parallel = num_workers > 1

    # Step 1: Parallel render frames to memory
    frames = []
    if use_parallel:
        try:
            start_method = "fork" if platform.system() == "Linux" else "spawn"
            ctx = mp.get_context(start_method)
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                # Use map for ordered results
                for frame in executor.map(
                    render_frame_func,
                    range(num_frames),
                    [frame_data] * num_frames,
                ):
                    frames.append(frame)
                    if progress_bar is not None:
                        progress_bar.update(1)
        except Exception as e:
            warnings.warn(
                f"Parallel rendering failed: {e}. Falling back to sequential.",
                RuntimeWarning,
                stacklevel=3,
            )
            use_parallel = False
            frames = []  # Clear partial results

    if not use_parallel:
        # Sequential rendering fallback
        for frame_idx in range(num_frames):
            frame = render_frame_func(frame_idx, frame_data)
            frames.append(frame)
            if progress_bar is not None:
                progress_bar.update(1)

    # Step 2: Write frames to MP4
    if IMAGEIO_AVAILABLE:
        # Try imageio first (simpler, more reliable if ffmpeg plugin available)
        try:
            writer_kwargs = {"fps": fps, "codec": "libx264", "pixelformat": "yuv420p", "bitrate": "5000k"}
            with imageio.get_writer(save_path, **writer_kwargs) as writer:
                for frame in frames:
                    # Ensure RGB format
                    if frame.shape[-1] == 4:  # RGBA
                        frame = frame[:, :, :3]
                    writer.append_data(frame)
            return  # Success!
        except Exception as e:
            # imageio failed (probably missing ffmpeg plugin), fall back to matplotlib
            warnings.warn(
                f"imageio MP4 writing failed ({e}). Falling back to matplotlib FFMpegWriter. "
                "For better performance, install imageio-ffmpeg: uv pip install imageio[ffmpeg]",
                RuntimeWarning,
                stacklevel=3,
            )

    # Fallback to matplotlib's FFMpegWriter
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter

    # Get frame dimensions
    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    writer = FFMpegWriter(fps=fps, codec="h264", bitrate=5000)
    with writer.saving(fig, save_path, dpi=100):
        for frame in frames:
            ax.clear()
            ax.imshow(frame)
            ax.axis('off')
            writer.grab_frame()

    plt.close(fig)
