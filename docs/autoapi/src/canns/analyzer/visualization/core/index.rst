src.canns.analyzer.visualization.core
=====================================

.. py:module:: src.canns.analyzer.visualization.core

.. autoapi-nested-parse::

   Visualization core infrastructure.

   This module provides foundational components for all visualization functions:
   - Configuration classes (PlotConfig, AnimationConfig, PlotConfigs)
   - Animation framework (OptimizedAnimationBase)
   - Parallel rendering (ParallelAnimationRenderer)
   - Optimized writers (create_optimized_writer, OptimizedAnimationWriter)

   All core components are re-exported at the parent visualization level for
   backward compatibility and convenience.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/analyzer/visualization/core/animation/index
   /autoapi/src/canns/analyzer/visualization/core/config/index
   /autoapi/src/canns/analyzer/visualization/core/jupyter_utils/index
   /autoapi/src/canns/analyzer/visualization/core/rendering/index
   /autoapi/src/canns/analyzer/visualization/core/writers/index


Classes
-------

.. autoapisummary::

   src.canns.analyzer.visualization.core.AnimationConfig
   src.canns.analyzer.visualization.core.OptimizedAnimationBase
   src.canns.analyzer.visualization.core.OptimizedAnimationWriter
   src.canns.analyzer.visualization.core.ParallelAnimationRenderer
   src.canns.analyzer.visualization.core.PlotConfig
   src.canns.analyzer.visualization.core.PlotConfigs


Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.core.create_optimized_writer
   src.canns.analyzer.visualization.core.display_animation_in_jupyter
   src.canns.analyzer.visualization.core.get_matplotlib_writer
   src.canns.analyzer.visualization.core.get_recommended_format
   src.canns.analyzer.visualization.core.is_jupyter_environment
   src.canns.analyzer.visualization.core.warn_double_rendering
   src.canns.analyzer.visualization.core.warn_gif_format


Package Contents
----------------

.. py:class:: AnimationConfig

   Configuration for animation rendering.

   Provides unified settings for optimized animation rendering with automatic
   quality presets and parallel rendering support.

   .. attribute:: fps

      Frames per second for the animation

   .. attribute:: enable_blitting

      Whether to use blitting optimization (auto-detected by default)

   .. attribute:: use_parallel

      Force parallel rendering even for short animations

   .. attribute:: num_workers

      Number of worker processes for parallel rendering

   .. attribute:: quality

      Quality preset - 'draft', 'medium', or 'high'

   .. attribute:: npoints_multiplier

      Resolution multiplier (< 1.0 for draft mode)

   .. attribute:: auto_parallel_threshold

      Auto-enable parallel rendering for animations with
      more than this many frames

   .. rubric:: Example

   >>> # High-quality animation (default)
   >>> config = AnimationConfig(fps=30, quality='high')
   >>>
   >>> # Fast draft mode for quick iteration
   >>> draft_config = AnimationConfig(quality='draft')  # Auto: 15 FPS, 0.5x resolution
   >>>
   >>> # Force parallel rendering
   >>> parallel_config = AnimationConfig(use_parallel=True, num_workers=8)


   .. py:method:: __post_init__()

      Automatically adjust settings based on quality preset.



   .. py:attribute:: auto_parallel_threshold
      :type:  int
      :value: 500



   .. py:attribute:: enable_blitting
      :type:  bool
      :value: True



   .. py:attribute:: fps
      :type:  int
      :value: 30



   .. py:attribute:: npoints_multiplier
      :type:  float
      :value: 1.0



   .. py:attribute:: num_workers
      :type:  int
      :value: 4



   .. py:attribute:: quality
      :type:  str
      :value: 'high'



   .. py:attribute:: use_parallel
      :type:  bool
      :value: False



.. py:class:: OptimizedAnimationBase(fig, ax, config = None)

   Bases: :py:obj:`abc.ABC`


   High-performance animation base class with blitting support.

   This abstract base class enforces best practices for matplotlib animations:
   - Artists are pre-created in create_artists()
   - Frame updates only modify data, never rebuild objects
   - Automatic blitting support detection
   - Optional parallel rendering for long animations

   Subclasses must implement:
   - create_artists(): Pre-create all artist objects with animated=True
   - update_frame(frame_idx): Update artist data and return modified artists

   Initialize the animation base.

   :param fig: Matplotlib figure
   :param ax: Matplotlib axes
   :param config: Animation configuration (uses defaults if None)


   .. py:method:: create_artists()
      :abstractmethod:


      Pre-create all artist objects for the animation.

      This method should:
      1. Create all plot objects (lines, scatter, images, etc.)
      2. Set animated=True for objects that will be updated
      3. Set initial data (can be empty with [], [])
      4. Return list of all animated artists

      :returns: List of artist objects that will be animated



   .. py:method:: init_func()

      Initialize animation (called by FuncAnimation).

      :returns: Tuple of all animated artists



   .. py:method:: render_animation(nframes, interval = None, repeat = True, save_path = None, **save_kwargs)

      Render the animation with automatic optimization selection.

      :param nframes: Total number of frames
      :param interval: Milliseconds between frames (computed from fps if None)
      :param repeat: Whether to loop the animation
      :param save_path: Path to save animation (None to skip saving)
      :param \*\*save_kwargs: Additional arguments for animation.save()

      :returns: FuncAnimation object



   .. py:method:: update_frame(frame_idx)
      :abstractmethod:


      Update artists for a specific frame.

      This method should:
      1. Compute data for the current frame
      2. Update artist data using set_data(), set_array(), etc.
      3. Return tuple of all modified artists

      Important: Do NOT call ax.clear() or recreate artists here!

      :param frame_idx: Index of the current frame

      :returns: Tuple of modified artist objects



   .. py:attribute:: artists
      :type:  list[matplotlib.artist.Artist]
      :value: []



   .. py:attribute:: ax


   .. py:attribute:: config


   .. py:attribute:: fig


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


.. py:class:: PlotConfig

   Unified configuration class for all plotting helpers in ``canns.analyzer``.

   This mirrors the behaviour of the previous ``visualize`` module so that
   reorganising the files does not affect the public API. The attributes map
   directly to keyword arguments exposed by the high-level plotting functions,
   allowing users to keep existing configuration objects unchanged after the
   reorganisation.


   .. py:method:: __post_init__()


   .. py:method:: for_animation(time_steps_per_second, **kwargs)
      :classmethod:


      Return configuration tailored for animations.



   .. py:method:: for_static_plot(**kwargs)
      :classmethod:


      Return configuration tailored for static plots.



   .. py:method:: to_matplotlib_kwargs()

      Materialize matplotlib keyword arguments from the config.



   .. py:attribute:: clabel
      :type:  str
      :value: 'Value'



   .. py:attribute:: color
      :type:  str
      :value: 'black'



   .. py:attribute:: figsize
      :type:  tuple[int, int]
      :value: (10, 6)



   .. py:attribute:: fps
      :type:  int
      :value: 30



   .. py:attribute:: grid
      :type:  bool
      :value: False



   .. py:attribute:: kwargs
      :type:  dict[str, Any] | None
      :value: None



   .. py:attribute:: repeat
      :type:  bool
      :value: True



   .. py:attribute:: save_path
      :type:  str | None
      :value: None



   .. py:attribute:: show
      :type:  bool
      :value: True



   .. py:attribute:: show_legend
      :type:  bool
      :value: True



   .. py:attribute:: show_progress_bar
      :type:  bool
      :value: True



   .. py:attribute:: time_steps_per_second
      :type:  int | None
      :value: None



   .. py:attribute:: title
      :type:  str
      :value: ''



   .. py:attribute:: xlabel
      :type:  str
      :value: ''



   .. py:attribute:: ylabel
      :type:  str
      :value: ''



.. py:class:: PlotConfigs

   Collection of commonly used plot configurations.

   These helpers mirror the presets that existed in ``canns.analyzer.visualize``
   so that callers relying on them continue to receive the exact same defaults.


   .. py:method:: average_firing_rate_plot(mode = 'per_neuron', **kwargs)
      :staticmethod:



   .. py:method:: direction_cell_polar(**kwargs)
      :staticmethod:


      Configuration for direction cell polar plot visualization.

      Creates polar coordinate plots showing directional tuning of head direction
      cells or other orientation-selective neurons.

      :param \*\*kwargs: Additional configuration parameters to override defaults.

      :returns: Configuration object for polar plots.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> config = PlotConfigs.direction_cell_polar(
      ...     title="Head Direction Cell",
      ...     save_path="direction_cell.png"
      ... )



   .. py:method:: energy_landscape_1d_animation(**kwargs)
      :staticmethod:



   .. py:method:: energy_landscape_1d_static(**kwargs)
      :staticmethod:



   .. py:method:: energy_landscape_2d_animation(**kwargs)
      :staticmethod:



   .. py:method:: energy_landscape_2d_static(**kwargs)
      :staticmethod:



   .. py:method:: firing_field_heatmap(**kwargs)
      :staticmethod:


      Configuration for firing field (rate map) heatmap visualization.

      Displays spatial firing rate distribution for grid cells, place cells, or
      other spatially-tuned neurons. Uses 'jet' colormap for high-contrast
      visualization of firing fields.

      :param \*\*kwargs: Additional configuration parameters to override defaults.

      :returns: Configuration object for firing field heatmaps.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> from canns.analyzer.visualization import PlotConfigs
      >>> config = PlotConfigs.firing_field_heatmap(
      ...     title="Grid Cell Firing Field",
      ...     save_path="ratemap.png"
      ... )



   .. py:method:: grid_autocorrelation(**kwargs)
      :staticmethod:


      Configuration for spatial autocorrelation heatmap visualization.

      Used to visualize hexagonal periodicity patterns in grid cell firing fields.
      Applies diverging colormap (RdBu_r) suitable for correlation values [-1, 1].

      :param \*\*kwargs: Additional configuration parameters to override defaults.

      :returns: Configuration object for autocorrelation plots.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> from canns.analyzer.visualization import PlotConfigs
      >>> config = PlotConfigs.grid_autocorrelation(
      ...     title="Grid Cell Autocorrelation",
      ...     save_path="autocorr.png"
      ... )



   .. py:method:: grid_cell_manifold_static(**kwargs)
      :staticmethod:



   .. py:method:: grid_cell_tracking_animation(**kwargs)
      :staticmethod:


      Configuration for grid cell tracking animation.

      Creates 3-panel synchronized animation showing trajectory, activity time course,
      and rate map with position overlay for analyzing grid cell behavior.

      :param \*\*kwargs: Additional configuration parameters to override defaults.
                         Must include 'time_steps_per_second' if not using default.

      :returns: Configuration object for tracking animations.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> config = PlotConfigs.grid_cell_tracking_animation(
      ...     time_steps_per_second=1000,  # dt=1ms
      ...     fps=20,
      ...     save_path="tracking.gif"
      ... )



   .. py:method:: grid_score_plot(**kwargs)
      :staticmethod:


      Configuration for grid score bar chart visualization.

      Displays rotational correlations at different angles used to compute grid score.
      Highlights hexagonal angles (60°, 120°) versus non-hexagonal angles.

      :param \*\*kwargs: Additional configuration parameters to override defaults.

      :returns: Configuration object for grid score plots.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> config = PlotConfigs.grid_score_plot(
      ...     title="Grid Cell Quality Assessment",
      ...     save_path="grid_score.png"
      ... )



   .. py:method:: grid_spacing_plot(**kwargs)
      :staticmethod:


      Configuration for grid spacing radial profile visualization.

      Shows how autocorrelation decays with distance from center, revealing
      the periodic spacing of grid fields.

      :param \*\*kwargs: Additional configuration parameters to override defaults.

      :returns: Configuration object for spacing analysis plots.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> config = PlotConfigs.grid_spacing_plot(
      ...     title="Grid Field Spacing",
      ...     save_path="spacing.png"
      ... )



   .. py:method:: population_activity_heatmap(**kwargs)
      :staticmethod:


      Configuration for population activity heatmap visualization.

      Displays neural population activity over time as a 2D heatmap where
      rows represent neurons and columns represent time points.

      :param \*\*kwargs: Additional configuration parameters to override defaults.

      :returns: Configuration object for population activity heatmaps.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> config = PlotConfigs.population_activity_heatmap(
      ...     title="Network Activity",
      ...     save_path="activity.png"
      ... )



   .. py:method:: raster_plot(mode = 'block', **kwargs)
      :staticmethod:



   .. py:method:: theta_population_activity_static(**kwargs)
      :staticmethod:



   .. py:method:: theta_sweep_animation(**kwargs)
      :staticmethod:



   .. py:method:: theta_sweep_place_cell_animation(**kwargs)
      :staticmethod:


      Configuration for theta sweep place cell animation.

      Creates synchronized 2-panel animation showing trajectory with place cell
      activity overlay and population activity heatmap.

      :param \*\*kwargs: Additional configuration parameters to override defaults.
                         Must include 'time_steps_per_second' if not using default.

      :returns: Configuration object for place cell animations.
      :rtype: PlotConfig

      .. rubric:: Example

      >>> config = PlotConfigs.theta_sweep_place_cell_animation(
      ...     time_steps_per_second=1000,
      ...     fps=10,
      ...     save_path="place_cell_sweep.gif"
      ... )



   .. py:method:: tuning_curve(num_bins = 50, pref_stim = None, **kwargs)
      :staticmethod:



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


.. py:function:: display_animation_in_jupyter(animation, format = 'html5')

   Display a matplotlib animation in Jupyter notebook.

   Performance comparison (100 frames):
       - html5 (default): 1.3s, 134 KB - Fast encoding, small size, smooth playback
       - jshtml: 2.6s, 4837 KB - Slower, 36x larger, but works without FFmpeg

   :param animation: matplotlib.animation.FuncAnimation object
   :param format: Display format - 'html5' (default, MP4 video) or 'jshtml' (JS animation)

   :returns: IPython.display.HTML object if successful, None otherwise

   .. note::

      'html5' format requires FFmpeg to be installed. If FFmpeg is not available,
      it will automatically fall back to 'jshtml'.


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


.. py:function:: is_jupyter_environment()

   Detect if code is running in a Jupyter notebook environment.

   :returns: True if running in Jupyter/IPython notebook, False otherwise.
   :rtype: bool


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


