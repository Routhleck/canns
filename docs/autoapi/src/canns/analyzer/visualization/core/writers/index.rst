src.canns.analyzer.visualization.core.writers
=============================================

.. py:module:: src.canns.analyzer.visualization.core.writers

.. autoapi-nested-parse::

   Optimized animation writers for faster file encoding.

   This module provides drop-in replacements for matplotlib's animation writers
   with significant performance improvements through better encoding libraries.



Attributes
----------

.. autoapisummary::

   src.canns.analyzer.visualization.core.writers.EncodingSpeed
   src.canns.analyzer.visualization.core.writers.FFMPEG_AVAILABLE
   src.canns.analyzer.visualization.core.writers.IMAGEIO_AVAILABLE
   src.canns.analyzer.visualization.core.writers.VideoFormat


Classes
-------

.. autoapisummary::

   src.canns.analyzer.visualization.core.writers.OptimizedAnimationWriter


Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.core.writers.create_optimized_writer
   src.canns.analyzer.visualization.core.writers.get_matplotlib_writer
   src.canns.analyzer.visualization.core.writers.get_recommended_format
   src.canns.analyzer.visualization.core.writers.warn_double_rendering
   src.canns.analyzer.visualization.core.writers.warn_gif_format


Module Contents
---------------

.. py:class:: OptimizedAnimationWriter(save_path, fps = 10, encoding_speed = 'balanced', codec = None, bitrate = None, dpi = 100)

   High-performance animation writer with automatic format detection.

   This writer automatically selects the best encoding method based on:
   - Output file format (detected from extension)
   - Available encoding libraries
   - User-specified speed/quality preferences

   Performance improvements:
   - GIF: 1.7x faster than PillowWriter
   - MP4: 5-10x faster than GIF encoding
   - WebM: Best compression, moderate speed

   .. rubric:: Example

   >>> writer = OptimizedAnimationWriter(
   ...     'output.mp4',
   ...     fps=10,
   ...     encoding_speed='fast'
   ... )
   >>> writer.setup(fig, 'output.mp4')
   >>> for frame in frames:
   ...     writer.grab_frame()
   >>> writer.finish()

   Initialize the optimized writer.

   :param save_path: Output file path (extension determines format)
   :param fps: Frames per second
   :param encoding_speed: 'fast', 'balanced', or 'quality'
   :param codec: Override automatic codec selection
   :param bitrate: Video bitrate in kbps (None for automatic)
   :param dpi: Figure DPI for rendering


   .. py:method:: finish()

      Finish writing and save file.



   .. py:method:: grab_frame(**kwargs)

      Grab current frame from figure (matplotlib API compatibility).



   .. py:method:: setup(fig, outfile=None, dpi=None)

      Setup the writer (matplotlib API compatibility).



   .. py:attribute:: bitrate
      :value: None



   .. py:attribute:: codec
      :value: None



   .. py:attribute:: dpi
      :value: 100



   .. py:attribute:: encoding_speed
      :value: 'balanced'



   .. py:attribute:: format
      :value: 'gif'



   .. py:attribute:: fps
      :value: 10



   .. py:attribute:: frames
      :value: []



   .. py:attribute:: save_path


   .. py:attribute:: writer
      :value: 'imageio_gif'



.. py:function:: create_optimized_writer(save_path, fps = 10, encoding_speed = 'balanced', **kwargs)

   Factory function to create an optimized animation writer.

   This is the recommended way to create writers for CANNs animations.

   :param save_path: Output file path
   :param fps: Frames per second
   :param encoding_speed: 'fast', 'balanced', or 'quality'
   :param \*\*kwargs: Additional parameters passed to writer

   :returns: OptimizedAnimationWriter instance

   .. rubric:: Examples

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


.. py:function:: get_matplotlib_writer(save_path, fps = 10, **kwargs)

   Create appropriate matplotlib animation writer based on file extension.

   This function automatically selects the correct writer:
   - .mp4 → FFMpegWriter (H.264 codec, high quality, fast encoding)
   - .gif → PillowWriter (universal compatibility)
   - others → FFMpegWriter (default)

   :param save_path: Output file path (extension determines format)
   :param fps: Frames per second
   :param \*\*kwargs: Additional arguments passed to the writer
                      For FFMpegWriter: codec, bitrate, extra_args
                      For PillowWriter: codec (ignored)

   :returns: Matplotlib animation writer instance

   .. rubric:: Example

   >>> from matplotlib import animation
   >>> writer = get_matplotlib_writer('output.mp4', fps=20)
   >>> ani.save('output.mp4', writer=writer)

   >>> # With custom codec
   >>> writer = get_matplotlib_writer('output.mp4', fps=30, bitrate=8000)


.. py:function:: get_recommended_format(use_case = 'web')

   Get recommended file format and extension for different use cases.

   :param use_case: Target use case

   :returns: Tuple of (format, extension) - format string and file extension with dot

   .. rubric:: Examples

   >>> format_str, ext = get_recommended_format('web')
   >>> save_path = f'animation{ext}'  # 'animation.mp4'


.. py:function:: warn_double_rendering(nframes, save_path, *, stacklevel = 2)

   Warn user about performance impact when both saving and showing animations.

   When both save_path and show=True are enabled, the animation gets rendered twice:
   1. First time: encoding to file (fast with MP4: ~1000 FPS)
   2. Second time: live GUI display (slow: ~10-30 FPS)

   This can significantly increase total processing time, especially for long animations.

   :param nframes: Number of frames in the animation
   :param save_path: Path where animation will be saved
   :param stacklevel: Stack level for the warning (default: 2, caller's caller)

   .. rubric:: Example

   >>> if save_path and show and nframes > 50:
   ...     warn_double_rendering(nframes, save_path, stacklevel=2)


.. py:function:: warn_gif_format(*, stacklevel = 2)

   Warn user about GIF format performance limitations.

   GIF encoding is significantly slower than MP4:
   - GIF: ~27 FPS encoding (256 colors, larger files)
   - MP4: ~1000 FPS encoding (36.8x faster, full color, smaller files)

   :param stacklevel: Stack level for the warning (default: 2, caller's caller)

   .. rubric:: Example

   >>> if save_path.endswith('.gif'):
   ...     warn_gif_format(stacklevel=2)


.. py:data:: EncodingSpeed

.. py:data:: FFMPEG_AVAILABLE

.. py:data:: IMAGEIO_AVAILABLE

.. py:data:: VideoFormat

