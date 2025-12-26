src.canns.analyzer.data.cann2d
==============================

.. py:module:: src.canns.analyzer.data.cann2d


Attributes
----------

.. autoapisummary::

   src.canns.analyzer.data.cann2d.HAS_NUMBA
   src.canns.analyzer.data.cann2d.data


Exceptions
----------

.. autoapisummary::

   src.canns.analyzer.data.cann2d.CANN2DError
   src.canns.analyzer.data.cann2d.DataLoadError
   src.canns.analyzer.data.cann2d.ProcessingError


Classes
-------

.. autoapisummary::

   src.canns.analyzer.data.cann2d.CANN2DPlotConfig
   src.canns.analyzer.data.cann2d.Constants
   src.canns.analyzer.data.cann2d.SpikeEmbeddingConfig
   src.canns.analyzer.data.cann2d.TDAConfig


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.cann2d.decode_circular_coordinates
   src.canns.analyzer.data.cann2d.embed_spike_trains
   src.canns.analyzer.data.cann2d.plot_3d_bump_on_torus
   src.canns.analyzer.data.cann2d.plot_cohomap
   src.canns.analyzer.data.cann2d.plot_projection
   src.canns.analyzer.data.cann2d.tda_vis


Module Contents
---------------

.. py:exception:: CANN2DError

   Bases: :py:obj:`Exception`


   Base exception for CANN2D analysis errors.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:exception:: DataLoadError

   Bases: :py:obj:`CANN2DError`


   Raised when data loading fails.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:exception:: ProcessingError

   Bases: :py:obj:`CANN2DError`


   Raised when data processing fails.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: CANN2DPlotConfig

   Bases: :py:obj:`src.canns.analyzer.visualization.PlotConfig`


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



.. py:class:: Constants

   Constants used throughout CANN2D analysis.


   .. py:attribute:: DEFAULT_DPI
      :value: 300



   .. py:attribute:: DEFAULT_FIGSIZE
      :value: (10, 8)



   .. py:attribute:: GAUSSIAN_SIGMA_FACTOR
      :value: 100



   .. py:attribute:: MULTIPROCESSING_CORES
      :value: 4



   .. py:attribute:: SPEED_CONVERSION_FACTOR
      :value: 100



   .. py:attribute:: TIME_CONVERSION_FACTOR
      :value: 0.01



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


.. py:function:: plot_cohomap(decoding_result, position_data, save_path = None, show = False, figsize = (10, 4), dpi = 300, subsample = 10)

   Visualize CohoMap 1.0: decoded circular coordinates mapped onto spatial trajectory.

   Creates a two-panel visualization showing how the two decoded circular coordinates
   vary across the animal's spatial trajectory. Each panel displays the spatial path
   colored by the cosine of one circular coordinate dimension.

   :param decoding_result: dict
                           Dictionary from decode_circular_coordinates() containing:
                           - 'coordsbox': decoded coordinates for box timepoints (n_times x n_dims)
                           - 'times_box': time indices for coordsbox
   :param position_data: dict
                         Position data containing 'x' and 'y' arrays for spatial coordinates
   :param save_path: str, optional
                     Path to save the visualization. If None, no save performed
   :param show: bool, default=False
                Whether to display the visualization
   :param figsize: tuple[int, int], default=(10, 4)
                   Figure size (width, height) in inches
   :param dpi: int, default=300
               Resolution for saved figure
   :param subsample: int, default=10
                     Subsampling interval for plotting (plot every Nth timepoint)

   :returns: The matplotlib figure object
   :rtype: plt.Figure

   :raises KeyError: If required keys are missing from input dictionaries
   :raises ValueError: If data dimensions are inconsistent
   :raises IndexError: If time indices are out of bounds

   .. rubric:: Examples

   >>> # Decode coordinates
   >>> decoding = decode_circular_coordinates(persistence_result, spike_data)
   >>> # Visualize with trajectory data
   >>> fig = plot_cohomap(
   ...     decoding,
   ...     position_data={'x': xx, 'y': yy},
   ...     save_path='cohomap.png',
   ...     show=True
   ... )


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


.. py:data:: HAS_NUMBA
   :value: True


.. py:data:: data
   :value: None


