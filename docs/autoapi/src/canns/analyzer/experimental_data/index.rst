src.canns.analyzer.experimental_data
====================================

.. py:module:: src.canns.analyzer.experimental_data


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/analyzer/experimental_data/cann1d/index
   /autoapi/src/canns/analyzer/experimental_data/cann2d/index


Classes
-------

.. autoapisummary::

   src.canns.analyzer.experimental_data.CANN1DPlotConfig
   src.canns.analyzer.experimental_data.CANN2DPlotConfig
   src.canns.analyzer.experimental_data.PlotConfig
   src.canns.analyzer.experimental_data.SpikeEmbeddingConfig
   src.canns.analyzer.experimental_data.TDAConfig


Functions
---------

.. autoapisummary::

   src.canns.analyzer.experimental_data.bump_fits
   src.canns.analyzer.experimental_data.create_1d_bump_animation
   src.canns.analyzer.experimental_data.decode_circular_coordinates
   src.canns.analyzer.experimental_data.embed_spike_trains
   src.canns.analyzer.experimental_data.load_grid_data
   src.canns.analyzer.experimental_data.load_roi_data
   src.canns.analyzer.experimental_data.plot_3d_bump_on_torus
   src.canns.analyzer.experimental_data.plot_projection
   src.canns.analyzer.experimental_data.tda_vis
   src.canns.analyzer.experimental_data.validate_grid_data
   src.canns.analyzer.experimental_data.validate_roi_data


Package Contents
----------------

.. py:class:: CANN1DPlotConfig

   Bases: :py:obj:`src.canns.analyzer.plotting.PlotConfig`


   Specialized PlotConfig for CANN1D visualizations.


   .. py:method:: for_bump_animation(**kwargs)
      :classmethod:


      Create configuration for 1D CANN bump animation.



   .. py:attribute:: bump_selection
      :type:  str
      :value: 'strongest'



   .. py:attribute:: max_height_value
      :type:  float
      :value: 0.5



   .. py:attribute:: max_width_range
      :type:  int
      :value: 40



   .. py:attribute:: nframes
      :type:  int | None
      :value: None



   .. py:attribute:: npoints
      :type:  int
      :value: 300



.. py:class:: CANN2DPlotConfig

   Bases: :py:obj:`src.canns.analyzer.plotting.PlotConfig`


   Specialized PlotConfig for CANN2D visualizations.


   .. py:method:: for_projection_3d(**kwargs)
      :classmethod:


      Create configuration for 3D projection plots.



   .. py:method:: for_torus_animation(**kwargs)
      :classmethod:


      Create configuration for 3D torus bump animations.



   .. py:attribute:: dpi
      :type:  int
      :value: 300



   .. py:attribute:: frame_step
      :type:  int
      :value: 5



   .. py:attribute:: n_frames
      :type:  int
      :value: 20



   .. py:attribute:: numangsint
      :type:  int
      :value: 51



   .. py:attribute:: r1
      :type:  float
      :value: 1.5



   .. py:attribute:: r2
      :type:  float
      :value: 1.0



   .. py:attribute:: window_size
      :type:  int
      :value: 300



   .. py:attribute:: zlabel
      :type:  str
      :value: 'Component 3'



.. py:class:: PlotConfig

   Unified configuration class for analyzer plotting helpers.


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



.. py:class:: SpikeEmbeddingConfig

   Configuration for spike train embedding.


   .. py:attribute:: dt
      :type:  int
      :value: 1000



   .. py:attribute:: min_speed
      :type:  float
      :value: 2.5



   .. py:attribute:: res
      :type:  int
      :value: 100000



   .. py:attribute:: sigma
      :type:  int
      :value: 5000



   .. py:attribute:: smooth
      :type:  bool
      :value: True



   .. py:attribute:: speed_filter
      :type:  bool
      :value: True



.. py:class:: TDAConfig

   Configuration for Topological Data Analysis.


   .. py:attribute:: active_times
      :type:  int
      :value: 15000



   .. py:attribute:: coeff
      :type:  int
      :value: 47



   .. py:attribute:: dim
      :type:  int
      :value: 6



   .. py:attribute:: do_shuffle
      :type:  bool
      :value: False



   .. py:attribute:: k
      :type:  int
      :value: 1000



   .. py:attribute:: maxdim
      :type:  int
      :value: 1



   .. py:attribute:: metric
      :type:  str
      :value: 'cosine'



   .. py:attribute:: n_points
      :type:  int
      :value: 1200



   .. py:attribute:: nbs
      :type:  int
      :value: 800



   .. py:attribute:: num_shuffles
      :type:  int
      :value: 1000



   .. py:attribute:: num_times
      :type:  int
      :value: 5



   .. py:attribute:: progress_bar
      :type:  bool
      :value: True



   .. py:attribute:: show
      :type:  bool
      :value: True



.. py:function:: bump_fits(data, config = None, save_path=None, **kwargs)

   Fit CANN1D bumps to data using MCMC optimization.

   :param data: numpy.ndarray
                Input data for bump fitting
   :param config: BumpFitsConfig, optional
                  Configuration object with all fitting parameters
   :param save_path: str, optional
                     Path to save the results
   :param \*\*kwargs: backward compatibility parameters

   :returns:

             list
                 List of fitted bump objects
             fits_array : numpy.ndarray
                 Array of fitted bump parameters
             nbump_array : numpy.ndarray
                 Array of bump counts and reconstructed signals
             centrbump_array : numpy.ndarray
                 Array of centered bump data
   :rtype: bumps


.. py:function:: create_1d_bump_animation(fits_data, config = None, save_path=None, **kwargs)

   Create 1D CANN bump animation using vectorized operations.

   :param fits_data: numpy.ndarray
                     Shape (n_fits, 4) array with columns [time, position, amplitude, kappa]
   :param config: AnimationConfig, optional
                  Configuration object with all animation parameters
   :param save_path: str, optional
                     Output path for the generated GIF
   :param \*\*kwargs: backward compatibility parameters

   :returns:

             matplotlib.animation.FuncAnimation
                 The animation object


.. py:function:: decode_circular_coordinates(persistence_result, spike_data, real_ground = True, real_of = True, save_path = None)

   Decode circular coordinates (bump positions) from cohomology.

   :param persistence_result: dict containing persistence analysis results with keys:
                              - 'persistence': persistent homology result
                              - 'indstemp': indices of sampled points
                              - 'movetimes': selected time points
                              - 'n_points': number of sampled points
   :param spike_data: dict, optional
                      Spike data dictionary containing 'spike', 't', and optionally 'x', 'y'
   :param real_ground: bool
                       Whether x, y, t ground truth exists
   :param real_of: bool
                   Whether experiment was performed in open field
   :param save_path: str, optional
                     Path to save decoding results. If None, saves to 'Results/spikes_decoding.npz'

   :returns:

             Dictionary containing decoding results with keys:
                 - 'coords': decoded coordinates for all timepoints
                 - 'coordsbox': decoded coordinates for box timepoints
                 - 'times': time indices for coords
                 - 'times_box': time indices for coordsbox
                 - 'centcosall': cosine centroids
                 - 'centsinall': sine centroids
   :rtype: dict


.. py:function:: embed_spike_trains(spike_trains, config = None, **kwargs)

   Load and preprocess spike train data from npz file.

   This function converts raw spike times into a time-binned spike matrix,
   optionally applying Gaussian smoothing and filtering based on animal movement speed.

   :param spike_trains: dict containing 'spike', 't', and optionally 'x', 'y'.
   :param config: SpikeEmbeddingConfig, optional configuration object
   :param \*\*kwargs: backward compatibility parameters

   :returns: Binned and optionally smoothed spike matrix of shape (T, N).
             xx (ndarray, optional): X coordinates (if speed_filter=True).
             yy (ndarray, optional): Y coordinates (if speed_filter=True).
             tt (ndarray, optional): Time points (if speed_filter=True).
   :rtype: spikes_bin (ndarray)


.. py:function:: load_grid_data(source = None, dataset_key = 'grid_1')

   Load grid cell data for 2D CANN analysis.

   :param source: Data source. Can be:
                  - URL string: downloads and loads from URL
                  - Path: loads from local file
                  - None: uses default CANNS dataset
   :type source: str, Path, or None
   :param dataset_key: Which default dataset to use ('grid_1' or 'grid_2') when source is None.
   :type dataset_key: str

   :returns: Dictionary containing spike data and metadata if successful, None otherwise.
             Expected keys: 'spike', 't', and optionally 'x', 'y' for position data.
   :rtype: dict or None

   .. rubric:: Examples

   >>> # Load default dataset
   >>> grid_data = load_grid_data()
   >>>
   >>> # Load from URL
   >>> grid_data = load_grid_data('https://example.com/grid_data.npz')
   >>>
   >>> # Load specific default dataset
   >>> grid_data = load_grid_data(dataset_key='grid_2')


.. py:function:: load_roi_data(source = None)

   Load ROI data for 1D CANN analysis.

   :param source: Data source. Can be:
                  - URL string: downloads and loads from URL
                  - Path: loads from local file
                  - None: uses default CANNS dataset
   :type source: str, Path, or None

   :returns: ROI data array if successful, None otherwise.
   :rtype: ndarray or None

   .. rubric:: Examples

   >>> # Load default dataset
   >>> roi_data = load_roi_data()
   >>>
   >>> # Load from URL
   >>> roi_data = load_roi_data('https://example.com/roi_data.txt')
   >>>
   >>> # Load from local file
   >>> roi_data = load_roi_data('./my_roi_data.txt')


.. py:function:: plot_3d_bump_on_torus(decoding_result, spike_data, config = None, save_path = None, numangsint = 51, r1 = 1.5, r2 = 1.0, window_size = 300, frame_step = 5, n_frames = 20, fps = 5, show_progress = True, show = True, figsize = (8, 8), **kwargs)

   Visualize the movement of the neural activity bump on a torus using matplotlib animation.

   This function follows the canns.analyzer.plotting patterns for animation generation
   with progress tracking and proper resource cleanup.

   :param decoding_result: dict or str
                           Dictionary containing decoding results with 'coordsbox' and 'times_box' keys,
                           or path to .npz file containing these results
   :param spike_data: dict, optional
                      Spike data dictionary containing spike information
   :param config: PlotConfig, optional
                  Configuration object for unified plotting parameters
   :param \*\*kwargs: backward compatibility parameters
   :param save_path: str, optional
                     Path to save the animation (e.g., 'animation.gif' or 'animation.mp4')
   :param numangsint: int
                      Grid resolution for the torus surface
   :param r1: float
              Major radius of the torus
   :param r2: float
              Minor radius of the torus
   :param window_size: int
                       Time window (in number of time points) for each frame
   :param frame_step: int
                      Step size to slide the time window between frames
   :param n_frames: int
                    Total number of frames in the animation
   :param fps: int
               Frames per second for the output animation
   :param show_progress: bool
                         Whether to show progress bar during generation
   :param show: bool
                Whether to display the animation
   :param figsize: tuple[int, int]
                   Figure size for the animation

   :returns: The animation object
   :rtype: matplotlib.animation.FuncAnimation


.. py:function:: plot_projection(reduce_func, embed_data, config = None, title='Projection (3D)', xlabel='Component 1', ylabel='Component 2', zlabel='Component 3', save_path=None, show=True, dpi=300, figsize=(10, 8), **kwargs)

   Plot a 3D projection of the embedded data.

   :param reduce_func: Function to reduce the dimensionality of the data.
   :type reduce_func: callable
   :param embed_data: Data to be projected.
   :type embed_data: ndarray
   :param config: Configuration object for unified plotting parameters
   :type config: PlotConfig, optional
   :param \*\*kwargs: backward compatibility parameters
   :param title: Title of the plot.
   :type title: str
   :param xlabel: Label for the x-axis.
   :type xlabel: str
   :param ylabel: Label for the y-axis.
   :type ylabel: str
   :param zlabel: Label for the z-axis.
   :type zlabel: str
   :param save_path: Path to save the plot. If None, plot will not be saved.
   :type save_path: str, optional
   :param show: Whether to display the plot.
   :type show: bool
   :param dpi: Dots per inch for saving the figure.
   :type dpi: int
   :param figsize: Size of the figure.
   :type figsize: tuple

   :returns: The created figure object.
   :rtype: fig


.. py:function:: tda_vis(embed_data, config = None, **kwargs)

   Topological Data Analysis visualization with optional shuffle testing.

   :param embed_data: ndarray
                      Embedded spike train data.
   :param config: TDAConfig, optional
                  Configuration object with all TDA parameters
   :param \*\*kwargs: backward compatibility parameters

   :returns:

             Dictionary containing:
                 - persistence: persistence diagrams from real data
                 - indstemp: indices of sampled points
                 - movetimes: selected time points
                 - n_points: number of sampled points
                 - shuffle_max: shuffle analysis results (if do_shuffle=True, otherwise None)
   :rtype: dict


.. py:function:: validate_grid_data(data)

   Validate grid data format for 2D CANN analysis.

   :param data: Grid data dictionary.
   :type data: dict

   :returns: True if data is valid, False otherwise.
   :rtype: bool


.. py:function:: validate_roi_data(data)

   Validate ROI data format for 1D CANN analysis.

   :param data: ROI data array.
   :type data: ndarray

   :returns: True if data is valid, False otherwise.
   :rtype: bool


