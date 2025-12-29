"""
Optimized animation writers for faster file encoding.

This module provides drop-in replacements for matplotlib's animation writers
with significant performance improvements through better encoding libraries.
"""

import importlib.util
import os
import warnings
from typing import Literal

import numpy as np

# Check available backends
IMAGEIO_AVAILABLE = importlib.util.find_spec("imageio") is not None
if not IMAGEIO_AVAILABLE:
    warnings.warn(
        "imageio not available. Install with 'pip install imageio' for faster encoding.",
        ImportWarning,
        stacklevel=2,
    )

FFMPEG_AVAILABLE = importlib.util.find_spec("imageio_ffmpeg") is not None


EncodingSpeed = Literal["fast", "balanced", "quality"]
VideoFormat = Literal["gif", "mp4", "webm"]


class OptimizedAnimationWriter:
    """
    High-performance animation writer with automatic format detection.

    This writer automatically selects the best encoding method based on:
    - Output file format (detected from extension)
    - Available encoding libraries
    - User-specified speed/quality preferences

    Performance improvements:
    - GIF: 1.7x faster than PillowWriter
    - MP4: 5-10x faster than GIF encoding
    - WebM: Best compression, moderate speed

    Example:
        >>> writer = OptimizedAnimationWriter(
        ...     'output.mp4',
        ...     fps=10,
        ...     encoding_speed='fast'
        ... )
        >>> writer.setup(fig, 'output.mp4')
        >>> for frame in frames:
        ...     writer.grab_frame()
        >>> writer.finish()
    """

    def __init__(
        self,
        save_path: str,
        fps: int = 10,
        encoding_speed: EncodingSpeed = "balanced",
        codec: str | None = None,
        bitrate: int | None = None,
        dpi: int = 100,
    ):
        """
        Initialize the optimized writer.

        Args:
            save_path: Output file path (extension determines format)
            fps: Frames per second
            encoding_speed: 'fast', 'balanced', or 'quality'
            codec: Override automatic codec selection
            bitrate: Video bitrate in kbps (None for automatic)
            dpi: Figure DPI for rendering
        """
        self.save_path = save_path
        self.fps = fps
        self.encoding_speed = encoding_speed
        self.codec = codec
        self.bitrate = bitrate
        self.dpi = dpi

        # Detect format from extension
        self.format = self._detect_format(save_path)

        # Select best available writer
        self.writer = self._select_writer()

        # Frame buffer
        self.frames = []

    def _detect_format(self, path: str) -> VideoFormat:
        """Detect video format from file extension."""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".gif":
            # Warn user about performance: MP4 is 36.8x faster
            warn_gif_format(stacklevel=4)
            return "gif"
        elif ext in [".mp4", ".m4v", ".mov"]:
            return "mp4"
        elif ext == ".webm":
            return "webm"
        else:
            # Default to GIF for unknown extensions
            warnings.warn(
                f"Unknown extension '{ext}', defaulting to GIF format", UserWarning, stacklevel=3
            )
            return "gif"

    def _select_writer(self) -> str:
        """Select best available writer based on format and libraries."""
        if self.format == "gif":
            if IMAGEIO_AVAILABLE:
                return "imageio_gif"
            else:
                return "pillow_gif"

        elif self.format in ["mp4", "webm"]:
            if FFMPEG_AVAILABLE:
                return "imageio_ffmpeg"
            elif IMAGEIO_AVAILABLE:
                warnings.warn(
                    "FFmpeg not available, falling back to GIF. "
                    "Install with: pip install imageio[ffmpeg]",
                    UserWarning,
                    stacklevel=3,
                )
                return "imageio_gif"
            else:
                warnings.warn(
                    f"Cannot encode {self.format}, falling back to Pillow GIF. "
                    f"Install imageio with: pip install imageio[ffmpeg]",
                    UserWarning,
                    stacklevel=3,
                )
                return "pillow_gif"

        return "pillow_gif"

    def setup(self, fig, outfile=None, dpi=None):
        """Setup the writer (matplotlib API compatibility)."""
        self.fig = fig
        if dpi is not None:
            self.dpi = dpi

        # Get canvas dimensions
        self.width, self.height = fig.canvas.get_width_height()

    def grab_frame(self, **kwargs):
        """Grab current frame from figure (matplotlib API compatibility)."""
        # Render figure to array
        self.fig.canvas.draw()

        # Get pixel data
        buf = self.fig.canvas.buffer_rgba()
        frame = np.frombuffer(buf, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 4))

        # Convert RGBA to RGB
        frame_rgb = frame[:, :, :3].copy()

        self.frames.append(frame_rgb)

    def finish(self):
        """Finish writing and save file."""
        if not self.frames:
            raise ValueError("No frames captured")

        if self.writer == "imageio_gif":
            self._save_imageio_gif()
        elif self.writer == "imageio_ffmpeg":
            self._save_imageio_ffmpeg()
        elif self.writer == "pillow_gif":
            self._save_pillow_gif()
        else:
            raise ValueError(f"Unknown writer: {self.writer}")

    def _save_imageio_gif(self):
        """Save using imageio (1.7x faster than Pillow)."""
        import imageio

        # Optimized GIF parameters based on encoding_speed
        if self.encoding_speed == "fast":
            params = {
                "quantizer": "nq",  # Faster quantizer
                "palettesize": 128,  # Fewer colors = faster
            }
        elif self.encoding_speed == "balanced":
            params = {
                "quantizer": "nq",
                "palettesize": 256,
            }
        else:  # quality
            params = {
                "palettesize": 256,
            }

        imageio.mimsave(self.save_path, self.frames, fps=self.fps, format="GIF", **params)

    def _save_imageio_ffmpeg(self):
        """Save using imageio with FFmpeg (5-10x faster than GIF)."""
        import imageio

        if self.format == "mp4":
            # H.264 encoding parameters
            codec = self.codec or "libx264"
            pixel_format = "yuv420p"  # Universal compatibility

            if self.encoding_speed == "fast":
                ffmpeg_params = [
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "28",  # Slightly lower quality for speed
                    "-tune",
                    "fastdecode",
                ]
            elif self.encoding_speed == "balanced":
                ffmpeg_params = [
                    "-preset",
                    "medium",
                    "-crf",
                    "23",  # Good quality
                ]
            else:  # quality
                ffmpeg_params = [
                    "-preset",
                    "slow",
                    "-crf",
                    "18",  # High quality
                ]

            if self.bitrate:
                ffmpeg_params.extend(["-b:v", f"{self.bitrate}k"])

            imageio.mimsave(
                self.save_path,
                self.frames,
                fps=self.fps,
                format="FFMPEG",
                codec=codec,
                pixelformat=pixel_format,
                ffmpeg_params=ffmpeg_params,
            )

        elif self.format == "webm":
            # VP9 encoding parameters
            codec = self.codec or "libvpx-vp9"

            if self.encoding_speed == "fast":
                ffmpeg_params = [
                    "-speed",
                    "8",  # Fastest
                    "-tile-columns",
                    "2",
                    "-threads",
                    "4",
                ]
            elif self.encoding_speed == "balanced":
                ffmpeg_params = ["-speed", "4", "-tile-columns", "2", "-threads", "4"]
            else:  # quality
                ffmpeg_params = [
                    "-speed",
                    "1",  # Slower but better quality
                    "-tile-columns",
                    "4",
                    "-threads",
                    "8",
                ]

            imageio.mimsave(
                self.save_path,
                self.frames,
                fps=self.fps,
                format="FFMPEG",
                codec=codec,
                ffmpeg_params=ffmpeg_params,
            )

    def _save_pillow_gif(self):
        """Fallback to Pillow GIF writer."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter

        # Create temporary animation to use PillowWriter
        # (This is slower but maintains compatibility)
        warnings.warn(
            "Using slower PillowWriter. Install imageio for 1.7x speedup: pip install imageio",
            UserWarning,
            stacklevel=3,
        )

        # Use matplotlib's PillowWriter
        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(self.frames[0])

        def update(frame):
            im.set_array(self.frames[frame])
            return [im]

        from matplotlib.animation import FuncAnimation

        ani = FuncAnimation(fig, update, frames=len(self.frames), blit=True, repeat=False)

        ani.save(self.save_path, writer=PillowWriter(fps=self.fps))
        plt.close(fig)


def get_recommended_format(
    use_case: Literal["web", "publication", "github", "presentation"] = "web",
) -> tuple[str, str]:
    """
    Get recommended file format and extension for different use cases.

    Args:
        use_case: Target use case

    Returns:
        Tuple of (format, extension) - format string and file extension with dot

    Examples:
        >>> format_str, ext = get_recommended_format('web')
        >>> save_path = f'animation{ext}'  # 'animation.mp4'
    """
    recommendations = {
        "web": ("mp4", ".mp4", "Universal browser support, fast encoding"),
        "publication": ("mp4", ".mp4", "High quality, smaller file size"),
        "github": ("gif", ".gif", "Inline display in README"),
        "presentation": ("mp4", ".mp4", "Smooth playback, high quality"),
    }

    if use_case not in recommendations:
        raise ValueError(
            f"Unknown use case '{use_case}'. Choose from: {list(recommendations.keys())}"
        )

    format_type, ext, _reason = recommendations[use_case]
    return format_type, ext


def create_optimized_writer(
    save_path: str, fps: int = 10, encoding_speed: EncodingSpeed = "balanced", **kwargs
) -> OptimizedAnimationWriter:
    """
    Factory function to create an optimized animation writer.

    This is the recommended way to create writers for CANNs animations.

    Args:
        save_path: Output file path
        fps: Frames per second
        encoding_speed: 'fast', 'balanced', or 'quality'
        **kwargs: Additional parameters passed to writer

    Returns:
        OptimizedAnimationWriter instance

    Examples:
        >>> # Fast GIF for quick iteration
        >>> writer = create_optimized_writer(
        ...     'output.gif',
        ...     fps=10,
        ...     encoding_speed='fast'
        ... )

        >>> # High-quality MP4 for publication
        >>> writer = create_optimized_writer(
        ...     'output.mp4',
        ...     fps=30,
        ...     encoding_speed='quality'
        ... )
    """
    return OptimizedAnimationWriter(
        save_path=save_path, fps=fps, encoding_speed=encoding_speed, **kwargs
    )


def warn_double_rendering(nframes: int, save_path: str, *, stacklevel: int = 2) -> None:
    """
    Warn user about performance impact when both saving and showing animations.

    When both save_path and show=True are enabled, the animation gets rendered twice:
    1. First time: encoding to file (fast with MP4: ~1000 FPS)
    2. Second time: live GUI display (slow: ~10-30 FPS)

    This can significantly increase total processing time, especially for long animations.

    Args:
        nframes: Number of frames in the animation
        save_path: Path where animation will be saved
        stacklevel: Stack level for the warning (default: 2, caller's caller)

    Example:
        >>> if save_path and show and nframes > 50:
        ...     warn_double_rendering(nframes, save_path, stacklevel=2)
    """
    warnings.warn(
        f"Both save_path and show=True are enabled for {nframes} frames. "
        "This will render the animation twice (once for saving, once for display), "
        "significantly increasing total time. Recommendation:\n"
        f"  • For batch processing: use show=False to render only once\n"
        f"  • For preview: consider viewing the saved file '{save_path}' instead of live display\n"
        f"  • MP4 encoding is very fast (~1000 FPS), but GUI display is slow (~10-30 FPS)",
        UserWarning,
        stacklevel=stacklevel,
    )


def warn_gif_format(*, stacklevel: int = 2) -> None:
    """
    Warn user about GIF format performance limitations.

    GIF encoding is significantly slower than MP4:
    - GIF: ~27 FPS encoding (256 colors, larger files)
    - MP4: ~1000 FPS encoding (36.8x faster, full color, smaller files)

    Args:
        stacklevel: Stack level for the warning (default: 2, caller's caller)

    Example:
        >>> if save_path.endswith('.gif'):
        ...     warn_gif_format(stacklevel=2)
    """
    warnings.warn(
        "Using GIF format. For 36.8x faster encoding, consider using MP4 format instead:\n"
        "  Change: 'output.gif' → 'output.mp4'\n"
        "  MP4 benefits:\n"
        "    • 36.8x faster encoding (986 FPS vs 27 FPS)\n"
        "    • Smaller file size with better compression\n"
        "    • Full color support (vs 256 colors in GIF)\n"
        "    • Universal browser and player support\n"
        "  Note: Use GIF only if you specifically need inline display in GitHub README.",
        UserWarning,
        stacklevel=stacklevel,
    )


def get_matplotlib_writer(save_path: str, fps: int = 10, **kwargs):
    """
    Create appropriate matplotlib animation writer based on file extension.

    This function automatically selects the correct writer:
    - .mp4 → FFMpegWriter (H.264 codec, high quality, fast encoding)
    - .gif → PillowWriter (universal compatibility)
    - others → FFMpegWriter (default)

    Args:
        save_path: Output file path (extension determines format)
        fps: Frames per second
        **kwargs: Additional arguments passed to the writer
            For FFMpegWriter: codec, bitrate, extra_args
            For PillowWriter: codec (ignored)

    Returns:
        Matplotlib animation writer instance

    Example:
        >>> from matplotlib import animation
        >>> writer = get_matplotlib_writer('output.mp4', fps=20)
        >>> ani.save('output.mp4', writer=writer)

        >>> # With custom codec
        >>> writer = get_matplotlib_writer('output.mp4', fps=30, bitrate=8000)
    """
    from matplotlib import animation
    import os

    ext = os.path.splitext(save_path)[1].lower()

    if ext == '.mp4':
        # MP4 format: Use FFMpegWriter (36.8x faster than GIF)
        codec = kwargs.pop('codec', 'h264')
        bitrate = kwargs.pop('bitrate', 5000)
        return animation.FFMpegWriter(
            fps=fps,
            codec=codec,
            bitrate=bitrate,
            **kwargs
        )
    elif ext == '.gif':
        # GIF format: Use PillowWriter
        warn_gif_format(stacklevel=3)
        return animation.PillowWriter(fps=fps)
    else:
        # Default to FFMpegWriter for other formats
        return animation.FFMpegWriter(fps=fps, **kwargs)
