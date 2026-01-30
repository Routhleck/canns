canns.analyzer.visualization.core.rendering
===========================================

.. py:module:: canns.analyzer.visualization.core.rendering

.. autoapi-nested-parse::

   Parallel frame rendering engine for long matplotlib animations.

   This module provides multi-process rendering capabilities for animations with
   hundreds or thousands of frames, achieving 3-4x speedup on multi-core CPUs.



Attributes
----------

.. autoapisummary::

   canns.analyzer.visualization.core.rendering.IMAGEIO_AVAILABLE


Classes
-------

.. autoapisummary::

   canns.analyzer.visualization.core.rendering.ParallelAnimationRenderer


Functions
---------

.. autoapisummary::

   canns.analyzer.visualization.core.rendering.estimate_parallel_speedup
   canns.analyzer.visualization.core.rendering.render_animation_parallel
   canns.analyzer.visualization.core.rendering.should_use_parallel


Module Contents
---------------

.. py:class:: ParallelAnimationRenderer(num_workers = None)

   Multi-process parallel renderer for matplotlib animations.

   This renderer creates separate processes to render frames in parallel,
   then combines them into a video file using imageio. Best for animations
   with >500 frames where the rendering bottleneck is matplotlib itself.

   Performance: Achieves ~3-4x speedup on 4-core CPUs.

   Initialize the parallel renderer.

   :param num_workers: Number of worker processes (uses CPU count if None)


   .. py:method:: render(animation_base, nframes, fps, save_path, writer = 'ffmpeg', codec = 'libx264', bitrate = None, show_progress = True)

      Render animation frames in parallel and save to file.

      :param animation_base: OptimizedAnimationBase instance with update_frame method
      :param nframes: Total number of frames to render
      :param fps: Frames per second
      :param save_path: Output file path
      :param writer: Video writer to use ('ffmpeg' or 'pillow')
      :param codec: Video codec (for ffmpeg writer)
      :param bitrate: Video bitrate in kbps (None for automatic)
      :param show_progress: Whether to show progress bar



   .. py:attribute:: num_workers


.. py:function:: estimate_parallel_speedup(nframes, num_workers = 4)

   Estimate speedup from parallel rendering.

   :param nframes: Number of frames
   :param num_workers: Number of parallel workers

   :returns: Estimated speedup factor


.. py:function:: render_animation_parallel(render_frame_func, frame_data, num_frames, save_path, fps = 10, num_workers = None, show_progress = True, file_format = None)

   Universal parallel animation renderer for analyzer animations.

   :param render_frame_func: Callable that renders a single frame:
                             ``func(frame_idx, frame_data) -> np.ndarray (H, W, 3 or 4)``.
   :param frame_data: Data needed by ``render_frame_func`` (passed to workers).
   :param num_frames: Total number of frames to render.
   :param save_path: Output file path (extension determines format).
   :param fps: Frames per second.
   :param num_workers: Number of parallel workers (None = auto-detect).
   :param show_progress: Whether to show progress bar.
   :param file_format: Override file format detection ('gif', 'mp4', etc.).

   :returns: None (saves animation to file).

   .. rubric:: Examples

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


.. py:function:: should_use_parallel(nframes, estimated_frame_time, threshold_seconds = 30.0)

   Determine if parallel rendering would be beneficial.

   :param nframes: Number of frames
   :param estimated_frame_time: Estimated time per frame in seconds
   :param threshold_seconds: Use parallel if total time exceeds this

   :returns: True if parallel rendering is recommended


.. py:data:: IMAGEIO_AVAILABLE
   :value: True


