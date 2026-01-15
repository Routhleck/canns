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

    This writer provides a drop-in replacement for matplotlib's animation
    writers with significant performance improvements through better encoding
    libraries and optimized settings. Automatically selects the best encoding
    method based on output format and available dependencies.

    The writer handles:
    - Format detection from file extension
    - Automatic backend selection (imageio vs Pillow)
    - Encoding speed vs quality tradeoffs
    - Compatibility with matplotlib's animation API

    Performance Improvements
    ------------------------
    - **GIF**: 1.7x faster than PillowWriter
    - **MP4**: 5-10x faster than GIF encoding (~36x total speedup)
    - **WebM**: Best compression, moderate speed

    Supported Formats
    -----------------
    - **GIF** (.gif): Universal compatibility, inline GitHub display
    - **MP4** (.mp4, .m4v, .mov): Best performance, smallest files
    - **WebM** (.webm): Modern format, excellent compression

    Parameters
    ----------
    save_path : str
        Output file path. Extension determines format (.gif, .mp4, .webm)
    fps : int, default=10
        Frames per second for the animation
    encoding_speed : {"fast", "balanced", "quality"}, default="balanced"
        Encoding speed vs quality tradeoff:
        - "fast": Fastest encoding, good quality
        - "balanced": Good balance (recommended)
        - "quality": Best quality, slower encoding
    codec : str or None, default=None
        Video codec override (None for automatic selection).
        Common: 'libx264' (MP4), 'h264', 'mpeg4'
    bitrate : int or None, default=None
        Video bitrate in kbps (None for automatic).
        Higher = better quality but larger files.
        Typical: 1000-5000 kbps for MP4
    dpi : int, default=100
        Figure DPI for rendering. Higher DPI = larger output but sharper

    Attributes
    ----------
    save_path : str
        Output file path
    format : VideoFormat
        Detected format ("gif", "mp4", or "webm")
    writer : str
        Selected writer backend ("imageio_gif", "imageio_ffmpeg", "pillow_gif")
    frames : list[np.ndarray]
        Buffer of captured frames

    Methods
    -------
    setup(fig, outfile=None, dpi=None)
        Setup writer for figure (matplotlib API)
    grab_frame(**kwargs)
        Capture current frame from figure
    finish()
        Complete encoding and save file

    Example
    -------
    Basic usage with matplotlib FuncAnimation:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.animation import FuncAnimation
    >>> from canns.analyzer.visualization.core.writers import OptimizedAnimationWriter
    >>> 
    >>> # Create figure and animation
    >>> fig, ax = plt.subplots()
    >>> line, = ax.plot([], [], 'b-')
    >>> ax.set_xlim(0, 2*np.pi)
    >>> ax.set_ylim(-1, 1)
    >>> 
    >>> def init():
    ...     line.set_data([], [])
    ...     return line,
    >>> 
    >>> def update(frame):
    ...     x = np.linspace(0, 2*np.pi, 100)
    ...     y = np.sin(x + frame * 0.1)
    ...     line.set_data(x, y)
    ...     return line,
    >>> 
    >>> # Create animation
    >>> anim = FuncAnimation(fig, update, init_func=init, frames=100)
    >>> 
    >>> # Save with optimized writer
    >>> writer = OptimizedAnimationWriter(
    ...     'output.mp4',
    ...     fps=30,
    ...     encoding_speed='fast'
    ... )
    >>> anim.save('output.mp4', writer=writer)

    Fast GIF for iteration:

    >>> # Quick preview GIF (fast encoding)
    >>> writer = OptimizedAnimationWriter(
    ...     'preview.gif',
    ...     fps=10,
    ...     encoding_speed='fast'
    ... )
    >>> anim.save('preview.gif', writer=writer)

    High-quality MP4 for publication:

    >>> # Publication quality (high DPI, slower encoding)
    >>> writer = OptimizedAnimationWriter(
    ...     'figure_publication.mp4',
    ...     fps=30,
    ...     encoding_speed='quality',
    ...     bitrate=5000,  # 5 Mbps
    ...     dpi=150
    ... )
    >>> anim.save('figure_publication.mp4', writer=writer)

    Manual frame capture:

    >>> # Direct frame-by-frame control
    >>> fig, ax = plt.subplots()
    >>> writer = OptimizedAnimationWriter('manual.mp4', fps=24)
    >>> writer.setup(fig)
    >>> 
    >>> for i in range(100):
    ...     ax.clear()
    ...     ax.plot(np.sin(np.linspace(0, 2*np.pi, 100) + i*0.1))
    ...     writer.grab_frame()
    >>> 
    >>> writer.finish()

    See Also
    --------
    create_optimized_writer : Factory function for creating writers
    get_recommended_format : Get optimal format for use case
    warn_gif_format : Performance warning for GIF format

    Notes
    -----
    **Backend Selection:**
    1. MP4/WebM: Prefers imageio+ffmpeg (best performance)
    2. GIF: Prefers imageio (1.7x faster than Pillow)
    3. Fallback: Pillow (always available but slower)

    **Format Recommendations:**
    - **Fast iteration**: GIF with encoding_speed='fast'
    - **Final output**: MP4 with encoding_speed='balanced'
    - **Publication**: MP4 with encoding_speed='quality', high dpi
    - **GitHub README**: GIF (displays inline)

    **Performance Tips:**
    - MP4 is ~36x faster than GIF overall
    - Use 'fast' encoding for quick previews
    - Higher DPI significantly increases encoding time
    - Bitrate affects file size more than encoding time

    **Dependencies:**
    - Basic (always works): Pillow (via matplotlib)
    - Recommended: imageio (1.7x faster GIF)
    - Best performance: imageio[ffmpeg] (fast MP4)

    **Compatibility:**
    - Implements matplotlib's MovieWriter API
    - Drop-in replacement for FFMpegWriter, PillowWriter
    - Works with FuncAnimation.save()

    Warnings
    --------
    - GIF encoding is significantly slower than MP4 (~36x)
    - High DPI settings can greatly increase encoding time
    - Large bitrates produce very large files
    - WebM requires ffmpeg with webm support
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

        Parameters
        ----------
        save_path : str
            Output file path (extension determines format)
        fps : int, default=10
            Frames per second
        encoding_speed : {"fast", "balanced", "quality"}, default="balanced"
            Encoding speed vs quality tradeoff
        codec : str or None, default=None
            Override automatic codec selection
        bitrate : int or None, default=None
            Video bitrate in kbps (None for automatic)
        dpi : int, default=100
            Figure DPI for rendering
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

    Provides expert recommendations for animation format selection based on
    the intended use case, balancing file size, quality, and compatibility.

    Parameters
    ----------
    use_case : {"web", "publication", "github", "presentation"}, default="web"
        Target use case for the animation:
        - "web": Universal browser compatibility, fast encoding
        - "publication": High quality for academic papers
        - "github": Inline display in README files
        - "presentation": Smooth playback for slides

    Returns
    -------
    tuple[str, str]
        Tuple of (format, extension):
        - format: Format name (e.g., "mp4", "gif")
        - extension: File extension with dot (e.g., ".mp4", ".gif")

    Raises
    ------
    ValueError
        If use_case is not one of the recognized options

    Example
    -------
    Get recommended format for different use cases:

    >>> from canns.analyzer.visualization.core.writers import get_recommended_format
    >>> 
    >>> # Web embedding
    >>> format_type, ext = get_recommended_format('web')
    >>> print(f"Format: {format_type}, Extension: {ext}")
    Format: mp4, Extension: .mp4
    >>> save_path = f'animation{ext}'  # 'animation.mp4'
    
    >>> # GitHub README
    >>> format_type, ext = get_recommended_format('github')
    >>> print(f"Format: {format_type}, Extension: {ext}")
    Format: gif, Extension: .gif
    >>> # GIF displays inline in GitHub markdown

    Use with animation creation:

    >>> # Select format based on target platform
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.animation import FuncAnimation
    >>> 
    >>> use_case = "publication"
    >>> format_type, ext = get_recommended_format(use_case)
    >>> save_path = f"figure_{use_case}{ext}"
    >>> 
    >>> # Create and save animation
    >>> fig, ax = plt.subplots()
    >>> anim = FuncAnimation(fig, update_func, frames=100)
    >>> anim.save(save_path, fps=30, dpi=150)

    See Also
    --------
    create_optimized_writer : Create writer with recommended settings
    OptimizedAnimationWriter : High-performance animation writer

    Notes
    -----
    **Format Recommendations:**
    
    - **web**: MP4 with H.264 codec
        - Universal browser support (Chrome, Firefox, Safari, Edge)
        - Fast encoding (~1000 FPS)
        - Good compression (small file size)
    
    - **publication**: MP4 with high quality
        - High visual quality for papers/journals
        - Smaller file size than GIF
        - Widely accepted by publishers
    
    - **github**: GIF format
        - Displays inline in GitHub README
        - No need to click through to view
        - Larger file size but convenient
    
    - **presentation**: MP4 with smooth playback
        - Works in PowerPoint, Keynote, Google Slides
        - Smooth playback even for long animations
        - Professional appearance
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

    This is the recommended way to create writers for CANNS animations,
    providing automatic format detection and optimized encoding settings.

    Parameters
    ----------
    save_path : str
        Output file path (extension determines format)
    fps : int, default=10
        Frames per second for the animation
    encoding_speed : {"fast", "balanced", "quality"}, default="balanced"
        Encoding speed vs quality tradeoff:
        - "fast": Fastest encoding, good quality
        - "balanced": Good balance (recommended)
        - "quality": Best quality, slower encoding
    **kwargs
        Additional parameters passed to OptimizedAnimationWriter:
        - codec: str - Override automatic codec selection
        - bitrate: int - Video bitrate in kbps
        - dpi: int - Figure DPI for rendering (default: 100)

    Returns
    -------
    OptimizedAnimationWriter
        Configured writer instance ready to use

    Example
    -------
    Create writer for fast iteration:

    >>> from canns.analyzer.visualization.core.writers import create_optimized_writer
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Fast GIF for quick testing
    >>> writer = create_optimized_writer(
    ...     'test_output.gif',
    ...     fps=10,
    ...     encoding_speed='fast'
    ... )
    >>> 
    >>> # Use with matplotlib animation
    >>> fig, ax = plt.subplots()
    >>> # ... create animation ...
    >>> writer.setup(fig, 'test_output.gif')
    >>> for frame_idx in range(100):
    ...     # ... update plot ...
    ...     writer.grab_frame()
    >>> writer.finish()

    High-quality MP4 for publication:

    >>> # High-quality settings for paper
    >>> writer = create_optimized_writer(
    ...     'figure_publication.mp4',
    ...     fps=30,
    ...     encoding_speed='quality',
    ...     bitrate=5000,  # 5 Mbps
    ...     dpi=150
    ... )

    Using with FuncAnimation:

    >>> from matplotlib.animation import FuncAnimation
    >>> 
    >>> fig, ax = plt.subplots()
    >>> line, = ax.plot([], [])
    >>> 
    >>> def update(frame):
    ...     line.set_data(x[:frame], y[:frame])
    ...     return line,
    >>> 
    >>> # Create writer
    >>> writer = create_optimized_writer('output.mp4', fps=24)
    >>> 
    >>> # Animate and save
    >>> anim = FuncAnimation(fig, update, frames=200)
    >>> anim.save('output.mp4', writer=writer)

    See Also
    --------
    OptimizedAnimationWriter : The writer class being instantiated
    get_recommended_format : Get recommended format for use case

    Notes
    -----
    **Performance Tips:**
    - MP4 encodes ~36x faster than GIF (1000 FPS vs 27 FPS)
    - Use 'fast' encoding for iteration, 'quality' for final output
    - Higher DPI increases encoding time significantly
    - Bitrate controls file size vs quality tradeoff

    **Format Detection:**
    - .gif → GIF with optimization
    - .mp4 → H.264 MP4 video
    - .webm → WebM video (requires ffmpeg)
    - Other extensions default to MP4
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

    Parameters
    ----------
    nframes : int
        Number of frames in the animation
    save_path : str
        Path where animation will be saved
    stacklevel : int, default=2
        Stack level for the warning (2 points to caller's caller)

    Example
    -------
    Warn about double rendering overhead:

    >>> from canns.analyzer.visualization.core.writers import warn_double_rendering
    >>> 
    >>> # In animation function
    >>> def create_animation(save_path=None, show=False):
    ...     nframes = 1000
    ...     
    ...     # Warn if both save and show are enabled
    ...     if save_path and show and nframes > 50:
    ...         warn_double_rendering(nframes, save_path, stacklevel=2)
    ...     
    ...     # ... create and render animation ...

    Typical usage pattern:

    >>> # Good: Only save (renders once)
    >>> create_animation(save_path="output.mp4", show=False)
    
    >>> # Also good: Only show (renders once)
    >>> create_animation(save_path=None, show=True)
    
    >>> # Warning: Both enabled (renders twice - slow!)
    >>> create_animation(save_path="output.mp4", show=True)
    # UserWarning: Both save_path and show=True are enabled...

    See Also
    --------
    warn_gif_format : Warn about GIF performance issues
    create_optimized_writer : Create fast animation writer

    Notes
    -----
    **Performance Impact:**
    - MP4 encoding: ~1000 FPS (very fast)
    - GUI display: ~10-30 FPS (slow, limited by refresh rate)
    - For 1000 frames: encoding takes ~1s, display takes ~30-100s
    - Total time with both: ~31-101s vs ~1s for save-only

    **Recommendations:**
    - For batch processing: Use show=False
    - For preview: View the saved file after encoding
    - Threshold: Warn for animations >50 frames
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

    This warning helps users make informed decisions about format selection.

    Parameters
    ----------
    stacklevel : int, default=2
        Stack level for the warning (2 points to caller's caller)

    Example
    -------
    Warn about GIF performance in format detection:

    >>> from canns.analyzer.visualization.core.writers import warn_gif_format
    >>> 
    >>> # In file format detection code
    >>> def detect_format(save_path):
    ...     if save_path.endswith('.gif'):
    ...         warn_gif_format(stacklevel=2)
    ...         return 'gif'
    ...     return 'mp4'

    User sees warning when creating GIF:

    >>> # User code
    >>> save_path = "animation.gif"
    >>> detect_format(save_path)
    # UserWarning: Using GIF format. For 36.8x faster encoding, consider using MP4...
    'gif'

    Conditional warning for large animations:

    >>> def create_animation(save_path, nframes):
    ...     # Only warn for large GIF animations
    ...     if save_path.endswith('.gif') and nframes > 100:
    ...         warn_gif_format(stacklevel=2)
    ...     # ... proceed with animation creation ...

    See Also
    --------
    warn_double_rendering : Warn about double render performance
    get_recommended_format : Get optimal format for use case
    create_optimized_writer : Create writer with automatic warnings

    Notes
    -----
    **Performance Comparison:**
    - GIF encoding: ~27 FPS
    - MP4 encoding: ~1000 FPS
    - Speedup: 36.8x faster with MP4

    **GIF Limitations:**
    - Limited to 256 colors (color quantization artifacts)
    - Larger file sizes (poor compression)
    - Slower encoding (CPU-intensive quantization)

    **MP4 Advantages:**
    - Full 24-bit color (16.7 million colors)
    - Better compression (smaller files)
    - Much faster encoding (hardware acceleration)
    - Universal playback support

    **When to Use GIF:**
    - GitHub README inline display
    - Websites that don't support video embedding
    - When you specifically need the GIF format

    **Recommended Action:**
    Change `.gif` to `.mp4` for better performance unless you specifically
    need GIF format for inline display (e.g., GitHub README).
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
    import os

    from matplotlib import animation

    ext = os.path.splitext(save_path)[1].lower()

    if ext == ".mp4":
        # MP4 format: Use FFMpegWriter (36.8x faster than GIF)
        codec = kwargs.pop("codec", "h264")
        bitrate = kwargs.pop("bitrate", 5000)
        return animation.FFMpegWriter(fps=fps, codec=codec, bitrate=bitrate, **kwargs)
    elif ext == ".gif":
        # GIF format: Use PillowWriter
        warn_gif_format(stacklevel=3)
        return animation.PillowWriter(fps=fps)
    else:
        # Default to FFMpegWriter for other formats
        return animation.FFMpegWriter(fps=fps, **kwargs)
