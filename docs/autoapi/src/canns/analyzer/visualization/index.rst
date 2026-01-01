src.canns.analyzer.visualization
================================

.. py:module:: src.canns.analyzer.visualization

.. autoapi-nested-parse::

   Model visualization utilities.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/analyzer/visualization/core/index
   /autoapi/src/canns/analyzer/visualization/energy_plots/index
   /autoapi/src/canns/analyzer/visualization/spatial_plots/index
   /autoapi/src/canns/analyzer/visualization/spike_plots/index
   /autoapi/src/canns/analyzer/visualization/theta_sweep_plots/index
   /autoapi/src/canns/analyzer/visualization/tuning_plots/index


Classes
-------

.. autoapisummary::

   src.canns.analyzer.visualization.AnimationConfig
   src.canns.analyzer.visualization.OptimizedAnimationBase
   src.canns.analyzer.visualization.OptimizedAnimationWriter
   src.canns.analyzer.visualization.ParallelAnimationRenderer
   src.canns.analyzer.visualization.PlotConfig
   src.canns.analyzer.visualization.PlotConfigs


Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.average_firing_rate_plot
   src.canns.analyzer.visualization.create_grid_cell_tracking_animation
   src.canns.analyzer.visualization.create_optimized_writer
   src.canns.analyzer.visualization.create_theta_sweep_grid_cell_animation
   src.canns.analyzer.visualization.create_theta_sweep_place_cell_animation
   src.canns.analyzer.visualization.display_animation_in_jupyter
   src.canns.analyzer.visualization.energy_landscape_1d_animation
   src.canns.analyzer.visualization.energy_landscape_1d_static
   src.canns.analyzer.visualization.energy_landscape_2d_animation
   src.canns.analyzer.visualization.energy_landscape_2d_static
   src.canns.analyzer.visualization.get_recommended_format
   src.canns.analyzer.visualization.is_jupyter_environment
   src.canns.analyzer.visualization.plot_autocorrelation
   src.canns.analyzer.visualization.plot_firing_field_heatmap
   src.canns.analyzer.visualization.plot_grid_cell_manifold
   src.canns.analyzer.visualization.plot_grid_score
   src.canns.analyzer.visualization.plot_grid_spacing_analysis
   src.canns.analyzer.visualization.plot_population_activity_with_theta
   src.canns.analyzer.visualization.population_activity_heatmap
   src.canns.analyzer.visualization.raster_plot
   src.canns.analyzer.visualization.tuning_curve
   src.canns.analyzer.visualization.warn_double_rendering
   src.canns.analyzer.visualization.warn_gif_format


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



.. py:function:: average_firing_rate_plot(spike_train, dt, config = None, *, mode = 'population', weights = None, title = 'Average Firing Rate', figsize = (12, 5), save_path = None, show = True, **kwargs)

   Calculate and plot average neural activity from a spike train.

   :param spike_train: Boolean/integer array of shape ``(timesteps, neurons)``.
   :param dt: Simulation time step in seconds.
   :param config: Optional :class:`PlotConfig` with styling overrides.
   :param mode: One of ``"per_neuron"``, ``"population"`` or
                ``"weighted_average"``.
   :param weights: Neuron-wise weights required for ``"weighted_average"``.
   :param title: Plot title when ``config`` is not provided.
   :param figsize: Figure size forwarded to Matplotlib when creating the axes.
   :param save_path: Optional path used to persist the plot.
   :param show: Whether to display the plot interactively.
   :param \*\*kwargs: Additional keyword arguments forwarded to Matplotlib.


.. py:function:: create_grid_cell_tracking_animation(position, activity, rate_map, config = None, *, time_steps_per_second = None, fps = 20, title = 'Grid Cell Tracking', figsize = (15, 5), env_size = 1.0, dt = 1.0, repeat = True, save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create 3-panel animation showing grid cell tracking behavior.

   Creates a synchronized animation with three panels:
   1. Left: Trajectory with current position marker
   2. Center: Firing rate time course
   3. Right: Rate map with position overlay

   :param position: Trajectory array of shape (T, 2) with (x, y) coordinates.
   :type position: np.ndarray
   :param activity: Neural activity time series of shape (T,).
   :type activity: np.ndarray
   :param rate_map: Spatial firing field of shape (M, K).
   :type rate_map: np.ndarray
   :param config: Unified configuration object.
   :type config: PlotConfig | None
   :param time_steps_per_second: Number of simulation steps per second
                                 (e.g., 1000 for dt=1ms). Required unless in config.
   :type time_steps_per_second: int | None
   :param fps: Frames per second for the animation. Defaults to 20.
   :type fps: int
   :param title: Overall plot title. Defaults to "Grid Cell Tracking".
   :type title: str
   :param figsize: Figure size. Defaults to (15, 5).
   :type figsize: tuple[int, int]
   :param env_size: Environment size for trajectory plot. Defaults to 1.0.
   :type env_size: float
   :param dt: Time step size in milliseconds. Defaults to 1.0.
   :type dt: float
   :param repeat: Whether animation should loop. Defaults to True.
   :type repeat: bool
   :param save_path: Path to save animation (e.g., 'tracking.gif').
   :type save_path: str | None
   :param show: Whether to display the animation. Defaults to True.
   :type show: bool
   :param show_progress_bar: Whether to show progress bar during save. Defaults to True.
   :type show_progress_bar: bool
   :param \*\*kwargs: Additional keyword arguments.

   :returns: Animation object, or None if displayed in Jupyter.
   :rtype: FuncAnimation | None

   .. rubric:: Example

   >>> from canns.analyzer.visualization import create_grid_cell_tracking_animation, PlotConfigs
   >>> # Create animation
   >>> config = PlotConfigs.grid_cell_tracking_animation(
   ...     time_steps_per_second=1000,  # dt=1.0ms
   ...     fps=20,
   ...     save_path="tracking.gif"
   ... )
   >>> anim = create_grid_cell_tracking_animation(
   ...     position, activity, rate_map,
   ...     config=config,
   ...     env_size=3.0
   ... )


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


.. py:function:: create_theta_sweep_grid_cell_animation(position_data, direction_data, dc_activity_data, gc_activity_data, gc_network, env_size, mapping_ratio, dt = 0.001, config = None, n_step = 10, fps = 10, figsize = (12, 3), save_path = None, show = True, show_progress_bar = True, render_backend = 'auto', output_dpi = 150, render_workers = None, render_start_method = None, **kwargs)

   Create comprehensive theta sweep animation with 4 panels (optimized for speed):
   1. Animal trajectory
   2. Direction cell polar plot
   3. Grid cell activity on manifold
   4. Grid cell activity in real space

   :param position_data: Animal position data (time, 2)
   :param direction_data: Direction data (time,)
   :param dc_activity_data: Direction cell activity (time, neurons)
   :param gc_activity_data: Grid cell activity (time, neurons)
   :param gc_network: GridCellNetwork instance for coordinate transformations
   :param env_size: Environment size
   :param mapping_ratio: Mapping ratio for grid cells
   :param dt: Time step size
   :param config: PlotConfig object for unified configuration
   :param n_step: Subsample every n_step frames for animation
   :param render_backend: Rendering backend. Use 'matplotlib', 'imageio', or 'auto'/'None' for auto-detect.
   :param output_dpi: Target DPI when rendering frames with non-interactive backends
   :param render_workers: Worker processes for imageio backend. ``None`` auto-selects, 0 disables.
   :param render_start_method: Multiprocessing start method ('fork', 'spawn', 'forkserver') or None for auto
   :param \*\*kwargs: Additional parameters for backward compatibility

   :returns: Matplotlib animation object for interactive backend, otherwise None
   :rtype: FuncAnimation | None


.. py:function:: create_theta_sweep_place_cell_animation(position_data, pc_activity_data, pc_network, navigation_task, dt = 0.001, config = None, n_step = 10, fps = 10, figsize = (12, 4), save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create theta sweep animation for place cell network with 2 panels:
   1. Environment trajectory with place cell bump overlay
   2. Population activity heatmap over time

   :param position_data: Animal position data (time, 2)
   :param pc_activity_data: Place cell activity (time, num_cells)
   :param pc_network: PlaceCellNetwork instance
   :param navigation_task: BaseNavigationTask instance for environment visualization
   :param dt: Time step size
   :param config: PlotConfig object for unified configuration
   :param n_step: Subsample every n_step frames for animation
   :param fps: Frames per second for animation
   :param figsize: Figure size (width, height)
   :param save_path: Path to save animation (GIF or MP4)
   :param show: Whether to display animation
   :param show_progress_bar: Whether to show progress bar during saving
   :param \*\*kwargs: Additional parameters (cmap, alpha, etc.)

   :returns: Matplotlib animation object
   :rtype: FuncAnimation


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


.. py:function:: energy_landscape_1d_animation(data_sets, time_steps_per_second = None, config = None, *, fps = 30, title = 'Evolving 1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', figsize = (10, 6), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create an animation of an evolving 1D energy landscape.

   The docstring intentionally preserves the guidance from the previous
   implementation so existing callers can rely on the same parameter
   explanations.

   :param data_sets: Dictionary whose keys are legend labels and values are
                     ``(x_data, y_data)`` tuples where ``y_data`` is shaped as
                     ``(time, state)``.
   :param time_steps_per_second: Number of simulation time steps per second of
                                 wall-clock time (e.g., ``1/dt``).
   :param config: Optional :class:`PlotConfig` with shared styling overrides.
   :param fps: Frames per second to render in the resulting animation.
   :param title: Title used when ``config`` is not provided.
   :param xlabel: X-axis label used when ``config`` is not provided.
   :param ylabel: Y-axis label used when ``config`` is not provided.
   :param figsize: Figure size passed to Matplotlib when building the canvas.
   :param grid: Whether to overlay a grid on the animation axes.
   :param repeat: Whether the animation should loop once it finishes.
   :param save_path: Optional path to persist the animation (``.gif`` / ``.mp4``).
   :param show: Whether to display the animation interactively.
   :param show_progress_bar: Whether to show a ``tqdm`` progress bar when saving.
   :param \*\*kwargs: Further keyword arguments passed through to ``ax.plot``.

   :returns: The constructed animation.
   :rtype: ``matplotlib.animation.FuncAnimation``


.. py:function:: energy_landscape_1d_static(data_sets, config = None, *, title = '1D Energy Landscape', xlabel = 'Collective Variable / State', ylabel = 'Energy', show_legend = True, figsize = (10, 6), grid = False, save_path = None, show = True, **kwargs)

   Plot a 1D static energy landscape using Matplotlib.

   This mirrors the long-form description from the pre-reorganisation module so
   existing documentation references stay accurate. The function accepts a
   dictionary of datasets, plotting each curve on the same set of axes while
   honouring the ``PlotConfig`` defaults callers relied on previously.

   :param data_sets: Mapping of series labels to ``(x, y)`` tuples representing
                     the energy curve to draw.
   :param config: Optional :class:`PlotConfig` carrying shared styling.
   :param title: Plot title when no config override is supplied.
   :param xlabel: X-axis label when no config override is supplied.
   :param ylabel: Y-axis label when no config override is supplied.
   :param show_legend: Whether to display the legend for labelled curves.
   :param figsize: Figure size forwarded to Matplotlib when creating the axes.
   :param grid: Whether to enable a grid background.
   :param save_path: Optional path for persisting the plot to disk.
   :param show: Whether to display the generated figure.
   :param \*\*kwargs: Additional keyword arguments forwarded to ``ax.plot``.

   :returns: The created figure and axes handles.
   :rtype: Tuple[plt.Figure, plt.Axes]


.. py:function:: energy_landscape_2d_animation(zs_data, config = None, *, time_steps_per_second = None, fps = 30, title = 'Evolving 2D Landscape', xlabel = 'X-Index', ylabel = 'Y-Index', clabel = 'Value', figsize = (8, 7), grid = False, repeat = True, save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create an animation of an evolving 2D landscape.

   The long-form description mirrors the previous implementation to maintain
   backwards-compatible documentation for downstream users.

   :param zs_data: Array of shape ``(timesteps, dim_y, dim_x)`` describing the
                   landscape at each simulation step.
   :param config: Optional :class:`PlotConfig` carrying display preferences.
   :param time_steps_per_second: Number of simulation steps per second of
                                 simulated time; required unless encoded in ``config``.
   :param fps: Frames per second in the generated animation.
   :param title: Title used when ``config`` is not provided.
   :param xlabel: X-axis label used when ``config`` is not provided.
   :param ylabel: Y-axis label used when ``config`` is not provided.
   :param clabel: Colorbar label used when ``config`` is not provided.
   :param figsize: Figure size passed to Matplotlib.
   :param grid: Whether to overlay a grid on the heatmap.
   :param repeat: Whether the animation should loop.
   :param save_path: Optional output path (``.gif`` / ``.mp4``).
   :param show: Whether to display the animation interactively.
   :param show_progress_bar: Whether to render a ``tqdm`` progress bar during save.
   :param \*\*kwargs: Additional keyword arguments forwarded to ``ax.imshow``.

   :returns: The constructed animation.
   :rtype: ``matplotlib.animation.FuncAnimation``


.. py:function:: energy_landscape_2d_static(z_data, config = None, *, title = '2D Static Landscape', xlabel = 'X-Index', ylabel = 'Y-Index', clabel = 'Value', figsize = (8, 7), grid = False, save_path = None, show = True, **kwargs)

   Plot a static 2D landscape from a 2D array as a heatmap.

   :param z_data: 2D array ``(dim_y, dim_x)`` representing the landscape.
   :param config: Optional :class:`PlotConfig` with pre-set styling.
   :param title: Plot title when ``config`` is not provided.
   :param xlabel: X-axis label when ``config`` is not provided.
   :param ylabel: Y-axis label when ``config`` is not provided.
   :param clabel: Colorbar label when ``config`` is not provided.
   :param figsize: Figure size forwarded to Matplotlib when allocating the canvas.
   :param grid: Whether to draw a grid overlay.
   :param save_path: Optional path that triggers saving the figure to disk.
   :param show: Whether to display the figure interactively.
   :param \*\*kwargs: Additional keyword arguments passed through to ``ax.imshow``.

   :returns: The Matplotlib figure and axes objects.
   :rtype: Tuple[plt.Figure, plt.Axes]


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


.. py:function:: plot_autocorrelation(autocorr, config = None, *, title = 'Spatial Autocorrelation', xlabel = 'X Lag (bins)', ylabel = 'Y Lag (bins)', figsize = (6, 6), save_path = None, show = True, **kwargs)

   Plot 2D spatial autocorrelation heatmap.

   Visualizes the spatial autocorrelation map which reveals periodic patterns
   in grid cell firing fields. For grid cells, this will show a characteristic
   hexagonal pattern of peaks indicating 60-degree rotational symmetry.

   :param autocorr: 2D spatial autocorrelation map, normalized to [-1, 1].
   :type autocorr: np.ndarray
   :param config: Unified configuration object. If None,
                  uses backward compatibility parameters.
   :type config: PlotConfig | None
   :param title: Plot title. Defaults to "Spatial Autocorrelation".
   :type title: str
   :param xlabel: X-axis label. Defaults to "X Lag (bins)".
   :type xlabel: str
   :param ylabel: Y-axis label. Defaults to "Y Lag (bins)".
   :type ylabel: str
   :param figsize: Figure size (width, height) in inches.
                   Defaults to (6, 6).
   :type figsize: tuple[int, int]
   :param save_path: Path to save the figure. If None, not saved.
   :type save_path: str | None
   :param show: Whether to display the plot. Defaults to True.
   :type show: bool
   :param \*\*kwargs: Additional keyword arguments passed to plt.imshow().

   :returns: Figure and axes objects.
   :rtype: tuple[plt.Figure, plt.Axes]

   .. rubric:: Example

   >>> from canns.analyzer.metrics.spatial_metrics import compute_spatial_autocorrelation
   >>> from canns.analyzer.visualization import plot_autocorrelation, PlotConfigs
   >>> autocorr = compute_spatial_autocorrelation(rate_map)
   >>> # Modern approach
   >>> config = PlotConfigs.grid_autocorrelation(save_path='autocorr.png')
   >>> fig, ax = plot_autocorrelation(autocorr, config=config)
   >>> # Legacy approach
   >>> fig, ax = plot_autocorrelation(autocorr, cmap='RdBu_r', save_path='autocorr.png')

   .. rubric:: References

   Sargolini et al. (2006). Conjunctive representation of position, direction,
   and velocity in entorhinal cortex. Science, 312(5774), 758-762.


.. py:function:: plot_firing_field_heatmap(heatmap, config = None, figsize = (5, 5), cmap = 'jet', interpolation = 'nearest', origin = 'lower', show = True, save_path = None, **kwargs)

   Plot a single spatial firing field heatmap.

   This function creates a publication-quality heatmap visualization of neural
   spatial firing patterns. It supports both modern PlotConfig-based configuration
   and legacy keyword arguments for backward compatibility.

   :param heatmap: 2D array of shape (M, K) representing spatial
                   firing rates in each bin.
   :type heatmap: np.ndarray
   :param config: Unified configuration object. If None,
                  uses backward compatibility parameters.
   :type config: PlotConfig | None
   :param figsize: Figure size (width, height) in inches.
                   Defaults to (5, 5).
   :type figsize: tuple[int, int]
   :param cmap: Colormap name for the heatmap. Defaults to 'jet'.
   :type cmap: str
   :param interpolation: Interpolation method for imshow. Defaults to 'nearest'.
   :type interpolation: str
   :param origin: Origin position for imshow ('lower' or 'upper').
                  Defaults to 'lower'.
   :type origin: str
   :param show: Whether to display the plot. Defaults to True.
   :type show: bool
   :param save_path: Path to save the figure. If None, figure is not saved.
   :type save_path: str | None
   :param \*\*kwargs: Additional keyword arguments passed to plt.imshow().

   :returns: The figure and axis objects for further customization.
   :rtype: tuple[plt.Figure, plt.Axes]

   .. rubric:: Example

   >>> from canns.analyzer.metrics.spatial_metrics import compute_firing_field
   >>> from canns.analyzer.visualization import plot_firing_field_heatmap, PlotConfig
   >>> # Compute firing field
   >>> heatmaps = compute_firing_field(activity, positions, 5.0, 5.0, 50, 50)
   >>> # Plot single neuron with PlotConfig
   >>> config = PlotConfig(figsize=(6, 6), save_path='neuron_0.png', show=False)
   >>> fig, ax = plot_firing_field_heatmap(heatmaps[0], config=config)
   >>> # Plot with legacy parameters
   >>> fig, ax = plot_firing_field_heatmap(heatmaps[1], cmap='viridis', save_path='neuron_1.png')


.. py:function:: plot_grid_cell_manifold(value_grid_twisted, grid_cell_activity, config = None, ax = None, title = 'Grid Cell Activity on Manifold', figsize = (8, 6), cmap = 'jet', show = True, save_path = None, **kwargs)

   Plot grid cell activity on the twisted torus manifold.

   :param value_grid_twisted: Coordinates on twisted manifold
   :param grid_cell_activity: 2D array of grid cell activities
   :param config: PlotConfig object for unified configuration
   :param ax: Optional axis to draw on instead of creating a new figure
   :param \*\*kwargs: Additional parameters for backward compatibility

   :returns: (figure, axis) objects
   :rtype: tuple


.. py:function:: plot_grid_score(rotated_corrs, grid_score, config = None, *, title = 'Grid Score Analysis', xlabel = 'Rotation Angle (°)', ylabel = 'Correlation', figsize = (8, 5), grid = True, save_path = None, show = True, **kwargs)

   Plot bar chart of rotational correlations with grid score.

   Visualizes the correlations at different rotation angles used to compute
   the grid score. Highlights 60° and 120° (hexagonal angles) which should
   be high for grid cells, versus 30°, 90°, and 150° which should be lower.

   :param rotated_corrs: Dictionary mapping rotation angles
                         to correlation values. Keys: 30, 60, 90, 120, 150.
   :type rotated_corrs: dict[int, float]
   :param grid_score: Computed grid score value.
   :type grid_score: float
   :param config: Unified configuration object.
   :type config: PlotConfig | None
   :param title: Plot title. Defaults to "Grid Score Analysis".
   :type title: str
   :param xlabel: X-axis label. Defaults to "Rotation Angle (°)".
   :type xlabel: str
   :param ylabel: Y-axis label. Defaults to "Correlation".
   :type ylabel: str
   :param figsize: Figure size. Defaults to (8, 5).
   :type figsize: tuple[int, int]
   :param grid: Whether to show grid lines. Defaults to True.
   :type grid: bool
   :param save_path: Path to save the figure.
   :type save_path: str | None
   :param show: Whether to display the plot. Defaults to True.
   :type show: bool
   :param \*\*kwargs: Additional keyword arguments.

   :returns: Figure and axes objects.
   :rtype: tuple[plt.Figure, plt.Axes]

   .. rubric:: Example

   >>> from canns.analyzer.metrics.spatial_metrics import compute_grid_score
   >>> from canns.analyzer.visualization import plot_grid_score
   >>> grid_score, rotated_corrs = compute_grid_score(autocorr)
   >>> fig, ax = plot_grid_score(rotated_corrs, grid_score)
   >>> print(f"Grid score: {grid_score:.3f}")
   Grid score: 0.456


.. py:function:: plot_grid_spacing_analysis(autocorr, spacing_bins, bin_size = None, config = None, *, title = 'Grid Spacing Analysis', xlabel = 'Distance (bins)', ylabel = 'Autocorrelation', figsize = (8, 5), grid = True, save_path = None, show = True, **kwargs)

   Plot radial profile of autocorrelation with spacing markers.

   Visualizes how autocorrelation changes with distance from center,
   revealing the periodic spacing of grid fields. The detected spacing
   is marked with a vertical line.

   :param autocorr: 2D autocorrelation map.
   :type autocorr: np.ndarray
   :param spacing_bins: Detected grid spacing in bins.
   :type spacing_bins: float
   :param bin_size: Size of spatial bins in real units (e.g., meters).
                    If provided, shows dual x-axis with real distance.
   :type bin_size: float | None
   :param config: Unified configuration object.
   :type config: PlotConfig | None
   :param title: Plot title. Defaults to "Grid Spacing Analysis".
   :type title: str
   :param xlabel: X-axis label. Defaults to "Distance (bins)".
   :type xlabel: str
   :param ylabel: Y-axis label. Defaults to "Autocorrelation".
   :type ylabel: str
   :param figsize: Figure size. Defaults to (8, 5).
   :type figsize: tuple[int, int]
   :param grid: Whether to show grid lines. Defaults to True.
   :type grid: bool
   :param save_path: Path to save the figure.
   :type save_path: str | None
   :param show: Whether to display the plot. Defaults to True.
   :type show: bool
   :param \*\*kwargs: Additional keyword arguments.

   :returns: Figure and axes objects.
   :rtype: tuple[plt.Figure, plt.Axes]

   .. rubric:: Example

   >>> from canns.analyzer.metrics.spatial_metrics import find_grid_spacing
   >>> from canns.analyzer.visualization import plot_grid_spacing_analysis
   >>> spacing_bins, spacing_m = find_grid_spacing(autocorr, bin_size=0.06)
   >>> fig, ax = plot_grid_spacing_analysis(autocorr, spacing_bins, bin_size=0.06)
   >>> print(f"Spacing: {spacing_m:.3f}m")


.. py:function:: plot_population_activity_with_theta(time_steps, theta_phase, net_activity, direction, config = None, add_lines = True, atol = 0.01, title = 'Population Activity with Theta', xlabel = 'Time (s)', ylabel = 'Direction (°)', figsize = (12, 4), cmap = 'jet', show = True, save_path = None, **kwargs)

   Plot neural population activity with theta oscillation markers and direction trace.

   :param time_steps: Array of time points
   :param theta_phase: Array of theta phase values [-π, π]
   :param net_activity: 2D array of network activity (time, neurons)
   :param direction: Array of direction values
   :param config: PlotConfig object for unified configuration
   :param add_lines: Whether to add vertical lines at theta phase zeros
   :param atol: Tolerance for detecting theta phase zeros
   :param \*\*kwargs: Additional parameters for backward compatibility

   :returns: (figure, axis) objects
   :rtype: tuple


.. py:function:: population_activity_heatmap(activity_data, dt, config = None, *, title = 'Population Activity', xlabel = 'Time (s)', ylabel = 'Neuron Index', figsize = (10, 6), cmap = 'viridis', save_path = None, show = True, **kwargs)

   Generate a heatmap of population firing rate activity over time.

   This function creates a 2D visualization where each row represents a neuron
   and each column represents a time point, with color indicating the firing rate
   or activity level.

   :param activity_data: 2D array of shape ``(timesteps, neurons)`` containing
                         firing rates or activity values.
   :param dt: Simulation time step in seconds.
   :param config: Optional :class:`PlotConfig` with styling overrides.
   :param title: Plot title when ``config`` is not provided.
   :param xlabel: X-axis label when ``config`` is not provided.
   :param ylabel: Y-axis label when ``config`` is not provided.
   :param figsize: Figure size forwarded to Matplotlib when creating the axes.
   :param cmap: Colormap name (default: "viridis").
   :param save_path: Optional path used to persist the plot.
   :param show: Whether to display the plot interactively.
   :param \*\*kwargs: Additional keyword arguments forwarded to Matplotlib.

   :returns: (figure, axis) objects.
   :rtype: tuple

   .. rubric:: Example

   >>> import numpy as np
   >>> from canns.analyzer.visualization.spike_plots import population_activity_heatmap
   >>> # Simulate some activity data
   >>> activity = np.random.rand(1000, 100)  # 1000 timesteps, 100 neurons
   >>> fig, ax = population_activity_heatmap(activity, dt=0.001)


.. py:function:: raster_plot(spike_train, config = None, *, mode = 'block', title = 'Raster Plot', xlabel = 'Time Step', ylabel = 'Neuron Index', figsize = (12, 6), color = 'black', save_path = None, show = True, **kwargs)

   Generate a raster plot from a spike train matrix.

   The explanatory text mirrors the former ``visualize`` module so callers see
   the same guidance after the reorganisation.

   :param spike_train: Boolean/integer array of shape ``(timesteps, neurons)``.
   :param config: Optional :class:`PlotConfig` with shared styling options.
   :param mode: Either ``"scatter"`` or ``"block"`` to pick the rendering style.
   :param title: Plot title when ``config`` is not provided.
   :param xlabel: X-axis label when ``config`` is not provided.
   :param ylabel: Y-axis label when ``config`` is not provided.
   :param figsize: Figure size forwarded to Matplotlib when creating the axes.
   :param color: Spike colour (or "on" colour for block mode).
   :param save_path: Optional path used to persist the plot.
   :param show: Whether to display the plot interactively.
   :param \*\*kwargs: Additional keyword arguments passed through to Matplotlib.


.. py:function:: tuning_curve(stimulus, firing_rates, neuron_indices, config = None, *, pref_stim = None, num_bins = 50, title = 'Tuning Curve', xlabel = 'Stimulus Value', ylabel = 'Average Firing Rate', figsize = (10, 6), save_path = None, show = True, **kwargs)

   Plot the tuning curve for one or more neurons.

   The wording mirrors the original ``visualize`` module to avoid API drift and
   to keep existing references valid.

   :param stimulus: 1D array with the stimulus value at each time step.
   :param firing_rates: 2D array of firing rates shaped ``(timesteps, neurons)``.
   :param neuron_indices: Integer or iterable of neuron indices to analyse.
   :param config: Optional :class:`PlotConfig` containing styling overrides.
   :param pref_stim: Optional 1D array of preferred stimuli used in legend text.
   :param num_bins: Number of bins when mapping stimulus to mean activity.
   :param title: Plot title when ``config`` is not provided.
   :param xlabel: X-axis label when ``config`` is not provided.
   :param ylabel: Y-axis label when ``config`` is not provided.
   :param figsize: Figure size forwarded to Matplotlib when creating the axes.
   :param save_path: Optional location where the figure should be stored.
   :param show: Whether to display the plot interactively.
   :param \*\*kwargs: Additional keyword arguments passed through to ``ax.plot``.


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


