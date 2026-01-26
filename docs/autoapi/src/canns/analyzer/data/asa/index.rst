src.canns.analyzer.data.asa
===========================

.. py:module:: src.canns.analyzer.data.asa


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/analyzer/data/asa/cohospace/index
   /autoapi/src/canns/analyzer/data/asa/config/index
   /autoapi/src/canns/analyzer/data/asa/decode/index
   /autoapi/src/canns/analyzer/data/asa/embedding/index
   /autoapi/src/canns/analyzer/data/asa/filters/index
   /autoapi/src/canns/analyzer/data/asa/fr/index
   /autoapi/src/canns/analyzer/data/asa/path/index
   /autoapi/src/canns/analyzer/data/asa/plotting/index
   /autoapi/src/canns/analyzer/data/asa/tda/index


Exceptions
----------

.. autoapisummary::

   src.canns.analyzer.data.asa.CANN2DError
   src.canns.analyzer.data.asa.DataLoadError
   src.canns.analyzer.data.asa.ProcessingError


Classes
-------

.. autoapisummary::

   src.canns.analyzer.data.asa.CANN2DPlotConfig
   src.canns.analyzer.data.asa.Constants
   src.canns.analyzer.data.asa.FRMResult
   src.canns.analyzer.data.asa.SpikeEmbeddingConfig
   src.canns.analyzer.data.asa.TDAConfig


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.asa.align_coords_to_position
   src.canns.analyzer.data.asa.apply_angle_scale
   src.canns.analyzer.data.asa.compute_cohoscore
   src.canns.analyzer.data.asa.compute_fr_heatmap_matrix
   src.canns.analyzer.data.asa.compute_frm
   src.canns.analyzer.data.asa.decode_circular_coordinates
   src.canns.analyzer.data.asa.decode_circular_coordinates_multi
   src.canns.analyzer.data.asa.embed_spike_trains
   src.canns.analyzer.data.asa.plot_2d_bump_on_manifold
   src.canns.analyzer.data.asa.plot_3d_bump_on_torus
   src.canns.analyzer.data.asa.plot_cohomap
   src.canns.analyzer.data.asa.plot_cohomap_multi
   src.canns.analyzer.data.asa.plot_cohospace_neuron
   src.canns.analyzer.data.asa.plot_cohospace_population
   src.canns.analyzer.data.asa.plot_cohospace_trajectory
   src.canns.analyzer.data.asa.plot_frm
   src.canns.analyzer.data.asa.plot_path_compare
   src.canns.analyzer.data.asa.plot_projection
   src.canns.analyzer.data.asa.save_fr_heatmap_png
   src.canns.analyzer.data.asa.tda_vis


Package Contents
----------------

.. py:exception:: CANN2DError

   Bases: :py:obj:`Exception`


   Base exception for CANN2D analysis errors.

   .. rubric:: Examples

   >>> try:  # doctest: +SKIP
   ...     raise CANN2DError("boom")
   ... except CANN2DError:
   ...     pass

   Initialize self.  See help(type(self)) for accurate signature.


.. py:exception:: DataLoadError

   Bases: :py:obj:`CANN2DError`


   Raised when data loading fails.

   .. rubric:: Examples

   >>> try:  # doctest: +SKIP
   ...     raise DataLoadError("missing data")
   ... except DataLoadError:
   ...     pass

   Initialize self.  See help(type(self)) for accurate signature.


.. py:exception:: ProcessingError

   Bases: :py:obj:`CANN2DError`


   Raised when data processing fails.

   .. rubric:: Examples

   >>> try:  # doctest: +SKIP
   ...     raise ProcessingError("processing failed")
   ... except ProcessingError:
   ...     pass

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: CANN2DPlotConfig

   Bases: :py:obj:`src.canns.analyzer.visualization.PlotConfig`


   Specialized PlotConfig for CANN2D visualizations.

   Extends :class:`canns.analyzer.visualization.PlotConfig` with fields that
   control 3D projection and torus animation parameters.

   .. rubric:: Examples

   >>> from canns.analyzer.data import CANN2DPlotConfig
   >>> cfg = CANN2DPlotConfig.for_projection_3d(title="Projection")
   >>> cfg.zlabel
   'Component 3'


   .. py:method:: for_projection_3d(**kwargs)
      :classmethod:


      Create configuration for 3D projection plots.

      .. rubric:: Examples

      >>> cfg = CANN2DPlotConfig.for_projection_3d(figsize=(6, 5))
      >>> cfg.figsize
      (6, 5)



   .. py:method:: for_torus_animation(**kwargs)
      :classmethod:


      Create configuration for 3D torus bump animations.

      .. rubric:: Examples

      >>> cfg = CANN2DPlotConfig.for_torus_animation(fps=10, n_frames=50)
      >>> cfg.fps, cfg.n_frames
      (10, 50)



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

   .. rubric:: Examples

   >>> from canns.analyzer.data import Constants
   >>> Constants.DEFAULT_DPI
   300


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



.. py:class:: FRMResult

   Return object for firing-rate map computation.

   .. attribute:: frm

      Firing rate map (bins_x, bins_y).

      :type: np.ndarray

   .. attribute:: occupancy

      Occupancy counts per spatial bin.

      :type: np.ndarray

   .. attribute:: spike_sum

      Spike counts per spatial bin.

      :type: np.ndarray

   .. attribute:: x_edges, y_edges

      Bin edges used for the FRM computation.

      :type: np.ndarray

   .. rubric:: Examples

   >>> res = FRMResult(frm=None, occupancy=None, spike_sum=None, x_edges=None, y_edges=None)  # doctest: +SKIP


   .. py:attribute:: frm
      :type:  numpy.ndarray


   .. py:attribute:: occupancy
      :type:  numpy.ndarray


   .. py:attribute:: spike_sum
      :type:  numpy.ndarray


   .. py:attribute:: x_edges
      :type:  numpy.ndarray


   .. py:attribute:: y_edges
      :type:  numpy.ndarray


.. py:class:: SpikeEmbeddingConfig

   Configuration for spike train embedding.

   .. attribute:: res

      Time scaling factor that converts seconds to integer bins.

      :type: int

   .. attribute:: dt

      Bin width in the same scaled units as ``res``.

      :type: int

   .. attribute:: sigma

      Gaussian smoothing width (scaled units).

      :type: int

   .. attribute:: smooth

      Whether to apply temporal smoothing to spike counts.

      :type: bool

   .. attribute:: speed_filter

      Whether to filter by animal speed (requires x/y/t in the input).

      :type: bool

   .. attribute:: min_speed

      Minimum speed threshold for ``speed_filter`` (cm/s by convention).

      :type: float

   .. rubric:: Examples

   >>> from canns.analyzer.data import SpikeEmbeddingConfig
   >>> cfg = SpikeEmbeddingConfig(smooth=False, speed_filter=False)
   >>> cfg.min_speed
   2.5


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

   Configuration for Topological Data Analysis (TDA).

   .. attribute:: dim

      Target PCA dimension before TDA.

      :type: int

   .. attribute:: num_times

      Downsampling stride in time.

      :type: int

   .. attribute:: active_times

      Number of most active time points to keep.

      :type: int

   .. attribute:: k

      Number of neighbors used in denoising.

      :type: int

   .. attribute:: n_points

      Number of points sampled for persistent homology.

      :type: int

   .. attribute:: metric

      Distance metric for point cloud (e.g., "cosine").

      :type: str

   .. attribute:: nbs

      Number of neighbors for distance matrix construction.

      :type: int

   .. attribute:: maxdim

      Maximum homology dimension for persistence.

      :type: int

   .. attribute:: coeff

      Field coefficient for persistent homology.

      :type: int

   .. attribute:: show

      Whether to show barcode plots.

      :type: bool

   .. attribute:: do_shuffle

      Whether to run shuffle analysis.

      :type: bool

   .. attribute:: num_shuffles

      Number of shuffles for null distribution.

      :type: int

   .. attribute:: progress_bar

      Whether to show progress bars.

      :type: bool

   .. rubric:: Examples

   >>> from canns.analyzer.data import TDAConfig
   >>> cfg = TDAConfig(maxdim=1, do_shuffle=False, show=False)
   >>> cfg.maxdim
   1


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



.. py:function:: align_coords_to_position(t_full, x_full, y_full, coords2, use_box, times_box, interp_to_full)

   Align decoded coordinates to the original (x, y, t) trajectory.

   :param t_full: Full-length trajectory arrays of shape (T,).
   :type t_full: np.ndarray
   :param x_full: Full-length trajectory arrays of shape (T,).
   :type x_full: np.ndarray
   :param y_full: Full-length trajectory arrays of shape (T,).
   :type y_full: np.ndarray
   :param coords2: Decoded coordinates of shape (K, 2) or (T, 2).
   :type coords2: np.ndarray
   :param use_box: Whether to use ``times_box`` to align to the original trajectory.
   :type use_box: bool
   :param times_box: Time indices or timestamps corresponding to ``coords2`` when ``use_box=True``.
   :type times_box: np.ndarray | None
   :param interp_to_full: If True, interpolate decoded coords back to full length; otherwise return a subset.
   :type interp_to_full: bool

   :returns: ``(t_aligned, x_aligned, y_aligned, coords_aligned, tag)`` where ``tag`` describes
             the alignment path used.
   :rtype: tuple

   .. rubric:: Examples

   >>> t, x, y, coords2, tag = align_coords_to_position(  # doctest: +SKIP
   ...     t_full, x_full, y_full, coords2,
   ...     use_box=True, times_box=decoding["times_box"], interp_to_full=True
   ... )
   >>> coords2.shape[1]
   2


.. py:function:: apply_angle_scale(coords2, scale)

   Convert angle units to radians before wrapping.

   :param coords2: Angle array of shape (T, 2) in the given ``scale``.
   :type coords2: np.ndarray
   :param scale: ``rad``  : already in radians.
                 ``deg``  : degrees -> radians.
                 ``unit`` : unit circle in [0, 1] -> radians.
                 ``auto`` : infer unit circle if values look like [0, 1].
   :type scale: {"rad", "deg", "unit", "auto"}

   :returns: Angles in radians.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> apply_angle_scale([[0.25, 0.5]], "unit")  # doctest: +SKIP


.. py:function:: compute_cohoscore(coords, activity, top_percent = 2.0, times = None, auto_filter = True)

   Compute a simple cohomology-space selectivity score (CohoScore) for each neuron.

   For each neuron:
   - Select "active" time points:
       - If top_percent is None: all time points with activity > 0
       - Else: top `top_percent`%% time points by activity value
   - Compute circular variance for theta1 and theta2 on the selected points.
   - CohoScore = 0.5 * (var(theta1) + var(theta2))

   Interpretation:
   - Smaller score => points are more concentrated in coho space => higher selectivity.

   :param coords: Decoded cohomology angles (theta1, theta2), in radians.
   :type coords: ndarray, shape (T, 2)
   :param activity:
   :type activity: ndarray, shape (T, N)
   :param times: Optional indices to align activity to coords when coords are computed on a subset of timepoints.
   :type times: ndarray, optional, shape (T_coords,)
   :param auto_filter: If True and lengths mismatch, auto-filter activity with activity>0 to mimic decode filtering.
                       Activity matrix (FR or spikes).
   :type auto_filter: bool
   :param top_percent: Percentage for selecting active points (e.g., 2.0 means top 2%%). If None, use activity>0.
   :type top_percent: float | None

   :returns: **scores** -- CohoScore per neuron (NaN for neurons with too few points).
   :rtype: ndarray, shape (N,)

   .. rubric:: Examples

   >>> scores = compute_cohoscore(coords, spikes)  # doctest: +SKIP
   >>> scores.shape[0]  # doctest: +SKIP


.. py:function:: compute_fr_heatmap_matrix(spike, neuron_range = None, time_range = None, *, transpose = True, normalize = None)

   Compute a matrix for FR heatmap display from spike-like data.

   :param spike: Shape (T, N). Can be continuous (float) or binned (int/float).
   :type spike: np.ndarray
   :param neuron_range: Neuron index range in [0, N]. End is exclusive.
   :type neuron_range: (start, end) or None
   :param time_range: Time index range in [0, T]. End is exclusive.
   :type time_range: (start, end) or None
   :param transpose: If True, returns (N_sel, T_sel) which is convenient for imshow with
                     neurons on Y and time on X (like your utils did with spike.T).
                     If False, returns (T_sel, N_sel).
   :type transpose: bool
   :param normalize: Optional display normalization along time for each neuron.
   :type normalize: {'zscore_per_neuron','minmax_per_neuron', None}

   :returns: **M** -- Heatmap matrix. Default shape (N_sel, T_sel) if transpose=True.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> M = compute_fr_heatmap_matrix(spikes, transpose=True)  # doctest: +SKIP
   >>> M.ndim
   2


.. py:function:: compute_frm(spike, x, y, neuron_id, *, bins = 50, x_range = None, y_range = None, min_occupancy = 1, smoothing = False, sigma = 1.0, nan_for_empty = True)

   Compute a single-neuron firing rate map (FRM) on 2D space.

   :param spike: Shape (T, N). Can be continuous (float) or binned counts (int/float).
   :type spike: np.ndarray
   :param x: Shape (T,). Position samples aligned with spike rows.
   :type x: np.ndarray
   :param y: Shape (T,). Position samples aligned with spike rows.
   :type y: np.ndarray
   :param neuron_id: Neuron index in [0, N).
   :type neuron_id: int
   :param bins: Number of spatial bins per dimension.
   :type bins: int
   :param x_range: Explicit ranges. If None, uses data min/max.
   :type x_range: (min, max) or None
   :param y_range: Explicit ranges. If None, uses data min/max.
   :type y_range: (min, max) or None
   :param min_occupancy: Bins with occupancy < min_occupancy are treated as empty.
   :type min_occupancy: int
   :param smoothing: If True, apply Gaussian smoothing to frm (and optionally to occupancy/spike_sum if you want later).
   :type smoothing: bool
   :param sigma: Gaussian sigma for smoothing (in bin units).
   :type sigma: float
   :param nan_for_empty: If True, empty bins become NaN; else 0.
   :type nan_for_empty: bool

   :returns: frm: 2D array (bins_x, bins_y) in Hz-like units per sample (relative scale).
   :rtype: FRMResult

   .. rubric:: Examples

   >>> res = compute_frm(spikes, x, y, neuron_id=0)  # doctest: +SKIP
   >>> res.frm.shape  # doctest: +SKIP


.. py:function:: decode_circular_coordinates(persistence_result, spike_data, real_ground = True, real_of = True, save_path = None)

   Decode circular coordinates (bump positions) from cohomology.

   :param persistence_result: Output from :func:`canns.analyzer.data.tda_vis`, containing keys:
                              ``persistence``, ``indstemp``, ``movetimes``, ``n_points``.
   :type persistence_result: dict
   :param spike_data: Spike data dictionary containing ``'spike'``, ``'t'`` and optionally ``'x'``/``'y'``.
   :type spike_data: dict
   :param real_ground: Whether x/y/t ground-truth exists (controls whether speed filtering is applied).
   :type real_ground: bool
   :param real_of: Whether the experiment is open-field (controls box coordinate handling).
   :type real_of: bool
   :param save_path: Path to save decoding results. Defaults to ``Results/spikes_decoding.npz``.
   :type save_path: str, optional

   :returns: Dictionary containing:
             - ``coords``: decoded coordinates for all timepoints.
             - ``coordsbox``: decoded coordinates for box timepoints.
             - ``times``: time indices for ``coords``.
             - ``times_box``: time indices for ``coordsbox``.
             - ``centcosall`` / ``centsinall``: cosine/sine centroids.
   :rtype: dict

   .. rubric:: Examples

   >>> from canns.analyzer.data import tda_vis, decode_circular_coordinates
   >>> persistence = tda_vis(embed_spikes, config=tda_cfg)  # doctest: +SKIP
   >>> decoding = decode_circular_coordinates(persistence, spike_data)  # doctest: +SKIP
   >>> decoding["coords"].shape  # doctest: +SKIP


.. py:function:: decode_circular_coordinates_multi(persistence_result, spike_data, save_path = None, num_circ = 2)

   Decode multiple circular coordinates from TDA persistence.

   :param persistence_result: Output from :func:`canns.analyzer.data.tda_vis`, containing keys:
                              ``persistence``, ``indstemp``, ``movetimes``, ``n_points``.
   :type persistence_result: dict
   :param spike_data: Spike data dictionary containing ``'spike'``, ``'t'`` and optionally ``'x'``/``'y'``.
   :type spike_data: dict
   :param save_path: Path to save decoding results. Defaults to ``Results/spikes_decoding.npz``.
   :type save_path: str, optional
   :param num_circ: Number of H1 cocycles/circular coordinates to decode.
   :type num_circ: int

   :returns: Dictionary with ``coords``, ``coordsbox``, ``times``, ``times_box`` and centroid terms.
   :rtype: dict

   .. rubric:: Examples

   >>> decoding = decode_circular_coordinates_multi(persistence, spike_data, num_circ=2)  # doctest: +SKIP
   >>> decoding["coords"].shape  # doctest: +SKIP


.. py:function:: embed_spike_trains(spike_trains, config = None, **kwargs)

   Load and preprocess spike train data from npz file.

   This function converts raw spike times into a time-binned spike matrix,
   optionally applying Gaussian smoothing and filtering based on animal movement speed.

   :param spike_trains: Dictionary containing ``'spike'`` and ``'t'``, and optionally ``'x'``/``'y'``.
                        ``'spike'`` can be a dict of neuron->spike_times, a list/array of arrays, or
                        a numpy object array from ``np.load``.
   :type spike_trains: dict
   :param config: Configuration object controlling binning, smoothing, and speed filtering.
   :type config: SpikeEmbeddingConfig, optional
   :param \*\*kwargs: Legacy keyword parameters (``res``, ``dt``, ``sigma``, ``smooth0``, ``speed0``,
                      ``min_speed``). Prefer ``config`` in new code.
   :type \*\*kwargs: Any

   :returns: ``(spikes_bin, xx, yy, tt)`` where:
             - ``spikes_bin`` is a (T, N) binned spike matrix.
             - ``xx``, ``yy``, ``tt`` are position/time arrays when ``speed_filter=True``,
               otherwise ``None``.
   :rtype: tuple

   .. rubric:: Examples

   >>> from canns.analyzer.data import SpikeEmbeddingConfig, embed_spike_trains
   >>> cfg = SpikeEmbeddingConfig(smooth=False, speed_filter=False)
   >>> spikes, xx, yy, tt = embed_spike_trains(mock_data, config=cfg)  # doctest: +SKIP
   >>> spikes.ndim
   2


.. py:function:: plot_2d_bump_on_manifold(decoding_result, spike_data, save_path = None, fps = 20, show = True, mode = 'fast', window_size = 10, frame_step = 5, numangsint = 20, figsize = (8, 6), show_progress = False, config = None, render_backend = 'auto', output_dpi = 150, render_workers = None)

   Create 2D projection animation of CANN2D bump activity with full blitting support.

   This function provides a fast 2D heatmap visualization as an alternative to the
   3D torus animation. It achieves 10-20x speedup using matplotlib blitting
   optimization, making it ideal for rapid prototyping and daily analysis.

   :param decoding_result: Decoding results containing coords and times (dict or file path)
   :param spike_data: Dictionary containing spike train data
   :param save_path: Path to save animation (None to skip saving)
   :param fps: Frames per second
   :param show: Whether to display the animation
   :param mode: Visualization mode - 'fast' for 2D heatmap (default), '3d' falls back to 3D
   :param window_size: Time window for activity aggregation
   :param frame_step: Time step between frames
   :param numangsint: Number of angular bins for spatial discretization
   :param figsize: Figure size (width, height) in inches
   :param show_progress: Show progress bar during processing

   :returns: * *matplotlib.animation.FuncAnimation | None* -- Animation object (or None in Jupyter when showing).
             * *Raises* -- ProcessingError: If mode is invalid or animation generation fails

   .. rubric:: Examples

   >>> # Fast 2D visualization (recommended for daily use)
   >>> ani = plot_2d_bump_on_manifold(
   ...     decoding_result, spike_data,
   ...     save_path='bump_2d.mp4', mode='fast'
   ... )
   >>> # For publication-ready 3D visualization, use mode='3d'
   >>> ani = plot_2d_bump_on_manifold(
   ...     decoding_result, spike_data, mode='3d'
   ... )


.. py:function:: plot_3d_bump_on_torus(decoding_result, spike_data, config = None, save_path = None, numangsint = 51, r1 = 1.5, r2 = 1.0, window_size = 300, frame_step = 5, n_frames = 20, fps = 5, show_progress = True, show = True, figsize = (8, 8), render_backend = 'auto', output_dpi = 150, render_workers = None, **kwargs)

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

   :returns: The animation object, or None when shown in Jupyter.
   :rtype: matplotlib.animation.FuncAnimation | None

   .. rubric:: Examples

   >>> ani = plot_3d_bump_on_torus(decoding, spike_data, show=False)  # doctest: +SKIP


.. py:function:: plot_cohomap(decoding_result, position_data, config = None, save_path = None, show = False, figsize = (10, 4), dpi = 300, subsample = 10)

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

   :returns: * *matplotlib.figure.Figure* -- The matplotlib figure object.
             * *Raises* -- KeyError : If required keys are missing from input dictionaries
               ValueError : If data dimensions are inconsistent
               IndexError : If time indices are out of bounds

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


.. py:function:: plot_cohomap_multi(decoding_result, position_data, config = None, save_path = None, show = False, figsize = (10, 4), dpi = 300, subsample = 10)

   Visualize CohoMap with N-dimensional decoded coordinates.

   Each subplot shows the spatial trajectory colored by ``cos(coord_i)`` for a single
   circular coordinate.

   :param decoding_result: Dictionary containing ``coordsbox`` and ``times_box``.
   :type decoding_result: dict
   :param position_data: Position data containing ``x`` and ``y`` arrays.
   :type position_data: dict
   :param config: Plot configuration for styling, saving, and showing.
   :type config: PlotConfig, optional
   :param save_path: Path to save the figure.
   :type save_path: str, optional
   :param show: Whether to show the figure.
   :type show: bool
   :param figsize: Figure size in inches.
   :type figsize: tuple[int, int]
   :param dpi: Save DPI.
   :type dpi: int
   :param subsample: Subsample stride for plotting.
   :type subsample: int

   :returns: The created figure.
   :rtype: matplotlib.figure.Figure

   .. rubric:: Examples

   >>> fig = plot_cohomap_multi(decoding, {"x": xx, "y": yy}, show=False)  # doctest: +SKIP


.. py:function:: plot_cohospace_neuron(coords, activity, neuron_id, mode = 'fr', top_percent = 5.0, times = None, auto_filter = True, figsize = (6, 6), cmap = 'hot', save_path = None, show = True, config = None)

   Overlay a single neuron's activity on the cohomology-space trajectory.

   This is a visualization helper:
   - mode="fr": marks the top `top_percent`%% time points by firing rate for the given neuron.
   - mode="spike": marks all time points where spike > 0 for the given neuron.

   :param coords: Decoded cohomology angles (theta1, theta2), in radians.
   :type coords: ndarray, shape (T, 2)
   :param activity: Activity matrix (continuous firing rate or binned spikes).
   :type activity: ndarray, shape (T, N)
   :param times: Optional indices to align activity to coords when coords are computed on a subset of timepoints.
   :type times: ndarray, optional, shape (T_coords,)
   :param auto_filter: If True and lengths mismatch, auto-filter activity with activity>0 to mimic decode filtering.
   :type auto_filter: bool
   :param neuron_id: Neuron index to visualize.
   :type neuron_id: int
   :param mode:
   :type mode: {"fr", "spike"}
   :param top_percent: Used only when mode="fr". For example, 5.0 means "top 5%%" time points.
   :type top_percent: float
   :param figsize:
   :type figsize: see `plot_cohospace_trajectory`.
   :param cmap:
   :type cmap: see `plot_cohospace_trajectory`.
   :param save_path:
   :type save_path: see `plot_cohospace_trajectory`.
   :param show:
   :type show: see `plot_cohospace_trajectory`.

   :returns: **ax**
   :rtype: matplotlib.axes.Axes

   .. rubric:: Examples

   >>> plot_cohospace_neuron(coords, spikes, neuron_id=0, show=False)  # doctest: +SKIP


.. py:function:: plot_cohospace_population(coords, activity, neuron_ids, mode = 'fr', top_percent = 5.0, times = None, auto_filter = True, figsize = (6, 6), cmap = 'hot', save_path = None, show = True, config = None)

   Plot aggregated activity from multiple neurons in cohomology space.

   For mode="fr":
   - For each neuron, select its top `top_percent`%% time points by firing rate.
   - Aggregate (sum) firing rates over the selected points and plot as colors.

   For mode="spike":
   - For each neuron, count spikes at each time point (spike > 0).
   - Aggregate counts over neurons and plot as colors.

   :param coords:
   :type coords: ndarray, shape (T, 2)
   :param activity:
   :type activity: ndarray, shape (T, N)
   :param times: Optional indices to align activity to coords when coords are computed on a subset of timepoints.
   :type times: ndarray, optional, shape (T_coords,)
   :param auto_filter: If True and lengths mismatch, auto-filter activity with activity>0 to mimic decode filtering.
   :type auto_filter: bool
   :param neuron_ids: Neuron indices to include (use range(N) to include all).
   :type neuron_ids: iterable[int]
   :param mode:
   :type mode: {"fr", "spike"}
   :param top_percent: Used only when mode="fr".
   :type top_percent: float
   :param figsize:
   :type figsize: see `plot_cohospace_trajectory`.
   :param cmap:
   :type cmap: see `plot_cohospace_trajectory`.
   :param save_path:
   :type save_path: see `plot_cohospace_trajectory`.
   :param show:
   :type show: see `plot_cohospace_trajectory`.

   :returns: **ax**
   :rtype: matplotlib.axes.Axes

   .. rubric:: Examples

   >>> plot_cohospace_population(coords, spikes, neuron_ids=[0, 1, 2], show=False)  # doctest: +SKIP


.. py:function:: plot_cohospace_trajectory(coords, times = None, subsample = 1, figsize = (6, 6), cmap = 'viridis', save_path = None, show = False, config = None)

   Plot a trajectory in cohomology space.

   :param coords: Decoded cohomology angles (theta1, theta2). Values may be in radians or in [0, 1] "unit circle"
                  convention depending on upstream decoding; this function will convert to degrees for plotting.
   :type coords: ndarray, shape (T, 2)
   :param times: Optional time array used to color points. If None, uses arange(T).
   :type times: ndarray, optional, shape (T,)
   :param subsample: Downsampling step (>1 reduces the number of plotted points).
   :type subsample: int
   :param figsize: Matplotlib figure size.
   :type figsize: tuple
   :param cmap: Matplotlib colormap name.
   :type cmap: str
   :param save_path: If provided, saves the figure to this path.
   :type save_path: str, optional
   :param show: If True, calls plt.show(). If False, closes the figure and returns the Axes.
   :type show: bool

   :returns: **ax** -- The Axes containing the plot.
   :rtype: matplotlib.axes.Axes

   .. rubric:: Examples

   >>> fig = plot_cohospace_trajectory(coords, subsample=2, show=False)  # doctest: +SKIP


.. py:function:: plot_frm(frm, *, title = 'Firing Rate Map', dpi = 200, show = None, config = None, **kwargs)

   Save FRM as PNG. Expects frm as 2D array (bins,bins).

   :param frm: Firing rate map (2D).
   :type frm: np.ndarray
   :param title: Figure title (used when ``config`` is None or missing fields).
   :type title: str
   :param dpi: Save DPI.
   :type dpi: int
   :param show: Whether to show the plot (overrides ``config.show`` if not None).
   :type show: bool | None
   :param config: Plot configuration. Use ``config.save_path`` to specify output file.
   :type config: PlotConfig, optional
   :param \*\*kwargs: Additional ``imshow`` keyword arguments. ``save_path`` may be provided here
                      as a fallback if not set in ``config``.
   :type \*\*kwargs: Any

   .. rubric:: Examples

   >>> cfg = PlotConfig.for_static_plot(save_path="frm.png", show=False)  # doctest: +SKIP
   >>> plot_frm(frm, config=cfg)  # doctest: +SKIP


.. py:function:: plot_path_compare(x, y, coords, config = None, *, title = 'Path Compare', figsize = (12, 5), show = True, save_path = None)

   Plot physical path vs decoded coho-space path side-by-side.

   :param x: Physical position arrays of shape (T,).
   :type x: np.ndarray
   :param y: Physical position arrays of shape (T,).
   :type y: np.ndarray
   :param coords: Decoded circular coordinates, shape (T, 1) or (T, 2).
   :type coords: np.ndarray
   :param config: Plot configuration. If None, a default config is created.
   :type config: PlotConfig, optional
   :param title: Backward-compatibility parameters.
   :type title: optional
   :param figsize: Backward-compatibility parameters.
   :type figsize: optional
   :param show: Backward-compatibility parameters.
   :type show: optional
   :param save_path: Backward-compatibility parameters.
   :type save_path: optional

   :returns: Figure and axes array.
   :rtype: (Figure, ndarray)

   .. rubric:: Examples

   >>> fig, axes = plot_path_compare(x, y, coords, show=False)  # doctest: +SKIP


.. py:function:: plot_projection(reduce_func, embed_data, config = None, title='Projection (3D)', xlabel='Component 1', ylabel='Component 2', zlabel='Component 3', save_path=None, show=True, dpi=300, figsize=(10, 8), **kwargs)

   Plot a 3D projection of the embedded data.

   :param reduce_func (callable):
   :type reduce_func (callable): Function to reduce the dimensionality of the data.
   :param embed_data (ndarray):
   :type embed_data (ndarray): Data to be projected.
   :param config (PlotConfig:
   :type config (PlotConfig: Configuration object for unified plotting parameters
   :param optional):
   :type optional): Configuration object for unified plotting parameters
   :param \*\*kwargs:
   :type \*\*kwargs: backward compatibility parameters
   :param title (str):
   :type title (str): Title of the plot.
   :param xlabel (str):
   :type xlabel (str): Label for the x-axis.
   :param ylabel (str):
   :type ylabel (str): Label for the y-axis.
   :param zlabel (str):
   :type zlabel (str): Label for the z-axis.
   :param save_path (str:
   :type save_path (str: Path to save the plot. If None, plot will not be saved.
   :param optional):
   :type optional): Path to save the plot. If None, plot will not be saved.
   :param show (bool):
   :type show (bool): Whether to display the plot.
   :param dpi (int):
   :type dpi (int): Dots per inch for saving the figure.
   :param figsize (tuple):
   :type figsize (tuple): Size of the figure.

   :returns: The created figure.
   :rtype: matplotlib.figure.Figure

   .. rubric:: Examples

   >>> fig = plot_projection(reduce_func, embed_data, show=False)  # doctest: +SKIP


.. py:function:: save_fr_heatmap_png(M, *, title = 'Firing Rate Heatmap', xlabel = 'Time', ylabel = 'Neuron', cmap = None, interpolation = 'nearest', origin = 'lower', aspect = 'auto', clabel = None, colorbar = True, dpi = 200, show = None, config = None, **kwargs)

   Save a heatmap PNG from a matrix (typically output of compute_fr_heatmap_matrix).

   :param M: Heatmap matrix (2D).
   :type M: np.ndarray
   :param title: Plot labels (used when ``config`` is None or missing fields).
   :type title: str
   :param xlabel: Plot labels (used when ``config`` is None or missing fields).
   :type xlabel: str
   :param ylabel: Plot labels (used when ``config`` is None or missing fields).
   :type ylabel: str
   :param cmap: Matplotlib imshow options.
   :type cmap: str, optional
   :param interpolation: Matplotlib imshow options.
   :type interpolation: str, optional
   :param origin: Matplotlib imshow options.
   :type origin: str, optional
   :param aspect: Matplotlib imshow options.
   :type aspect: str, optional
   :param clabel: Colorbar label (defaults to ``config.clabel``).
   :type clabel: str, optional
   :param colorbar: Whether to draw a colorbar.
   :type colorbar: bool
   :param dpi: Save DPI.
   :type dpi: int
   :param show: Whether to show the plot (overrides ``config.show`` if not None).
   :type show: bool | None
   :param config: Plot configuration. Use ``config.save_path`` to specify output file.
   :type config: PlotConfig, optional
   :param \*\*kwargs: Additional ``imshow`` keyword arguments. ``save_path`` may be provided here
                      as a fallback if not set in ``config``.
   :type \*\*kwargs: Any

   .. rubric:: Notes

   - Does not reorder neurons.
   - Uses matplotlib only here (ASA core stays compute-friendly).

   .. rubric:: Examples

   >>> config = PlotConfig.for_static_plot(save_path="fr.png", show=False)  # doctest: +SKIP
   >>> save_fr_heatmap_png(M, config=config)  # doctest: +SKIP


.. py:function:: tda_vis(embed_data, config = None, **kwargs)

   Topological Data Analysis visualization with optional shuffle testing.

   :param embed_data: Embedded spike train data of shape (T, N).
   :type embed_data: np.ndarray
   :param config: Configuration object with all TDA parameters. If None, legacy kwargs are used.
   :type config: TDAConfig, optional
   :param \*\*kwargs: Legacy keyword parameters (``dim``, ``num_times``, ``active_times``, ``k``,
                      ``n_points``, ``metric``, ``nbs``, ``maxdim``, ``coeff``, ``show``,
                      ``do_shuffle``, ``num_shuffles``, ``progress_bar``).
   :type \*\*kwargs: Any

   :returns: Dictionary containing:
             - ``persistence``: persistence diagrams from real data.
             - ``indstemp``: indices of sampled points.
             - ``movetimes``: selected time points.
             - ``n_points``: number of sampled points.
             - ``shuffle_max``: shuffle analysis results (if ``do_shuffle=True``), else ``None``.
   :rtype: dict

   .. rubric:: Examples

   >>> from canns.analyzer.data import TDAConfig, tda_vis
   >>> cfg = TDAConfig(maxdim=1, do_shuffle=False, show=False)
   >>> result = tda_vis(embed_data, config=cfg)  # doctest: +SKIP
   >>> sorted(result.keys())
   ['indstemp', 'movetimes', 'n_points', 'persistence', 'shuffle_max']


