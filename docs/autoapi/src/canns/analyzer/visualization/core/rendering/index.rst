src.canns.analyzer.visualization.core.rendering
===============================================

.. py:module:: src.canns.analyzer.visualization.core.rendering

.. autoapi-nested-parse::

   Parallel frame rendering engine for long matplotlib animations.

   This module provides multi-process rendering capabilities for animations with
   hundreds or thousands of frames, achieving 3-4x speedup on multi-core CPUs.



Attributes
----------

.. autoapisummary::

   src.canns.analyzer.visualization.core.rendering.IMAGEIO_AVAILABLE


Classes
-------

.. autoapisummary::

   src.canns.analyzer.visualization.core.rendering.ParallelAnimationRenderer


Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.core.rendering.estimate_parallel_speedup
   src.canns.analyzer.visualization.core.rendering.should_use_parallel


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


.. py:function:: should_use_parallel(nframes, estimated_frame_time, threshold_seconds = 30.0)

   Determine if parallel rendering would be beneficial.

   :param nframes: Number of frames
   :param estimated_frame_time: Estimated time per frame in seconds
   :param threshold_seconds: Use parallel if total time exceeds this

   :returns: True if parallel rendering is recommended


.. py:data:: IMAGEIO_AVAILABLE
   :value: True


