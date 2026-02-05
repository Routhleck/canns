canns.analyzer.data.asa.cohospace_scatter
=========================================

.. py:module:: canns.analyzer.data.asa.cohospace_scatter

.. autoapi-nested-parse::

   Scatter-style CohoSpace plots and cohoscore utilities.



Functions
---------

.. autoapisummary::

   canns.analyzer.data.asa.cohospace_scatter.compute_cohoscore_scatter_1d
   canns.analyzer.data.asa.cohospace_scatter.compute_cohoscore_scatter_2d
   canns.analyzer.data.asa.cohospace_scatter.draw_torus_parallelogram_grid_scatter
   canns.analyzer.data.asa.cohospace_scatter.plot_cohospace_scatter_neuron_1d
   canns.analyzer.data.asa.cohospace_scatter.plot_cohospace_scatter_neuron_2d
   canns.analyzer.data.asa.cohospace_scatter.plot_cohospace_scatter_neuron_skewed
   canns.analyzer.data.asa.cohospace_scatter.plot_cohospace_scatter_population_1d
   canns.analyzer.data.asa.cohospace_scatter.plot_cohospace_scatter_population_2d
   canns.analyzer.data.asa.cohospace_scatter.plot_cohospace_scatter_population_skewed
   canns.analyzer.data.asa.cohospace_scatter.plot_cohospace_scatter_trajectory_1d
   canns.analyzer.data.asa.cohospace_scatter.plot_cohospace_scatter_trajectory_2d
   canns.analyzer.data.asa.cohospace_scatter.tile_parallelogram_points_scatter


Module Contents
---------------

.. py:function:: compute_cohoscore_scatter_1d(coords, activity, top_percent = 2.0, times = None, auto_filter = True)

   Compute 1D cohomology-space selectivity score (CohoScore) for each neuron.

   For each neuron, select active time points (top_percent or activity > 0), compute
   circular variance for theta on the selected points, and use it as the score.


.. py:function:: compute_cohoscore_scatter_2d(coords, activity, top_percent = 2.0, times = None, auto_filter = True)

   Compute a simple cohomology-space selectivity score (CohoScore) for each neuron.

   For each neuron, select active time points (top_percent or activity > 0), compute
   circular variance for theta1 and theta2 on the selected points, and average them.

   Interpretation: smaller score means points are more concentrated in coho space
   and the neuron is more selective.

   :param coords: Decoded cohomology angles (theta1, theta2), in radians.
   :type coords: ndarray, shape (T, 2)
   :param activity:
   :type activity: ndarray, shape (T, N)
   :param times: Optional indices to align activity to coords when coords are computed on a subset of timepoints.
   :type times: ndarray, optional, shape (T_coords,)
   :param auto_filter: If True and lengths mismatch, auto-filter activity with activity>0 to mimic decode filtering.
                       Activity matrix (FR or spikes).
   :type auto_filter: bool
   :param top_percent: Percentage for selecting active points (e.g., 2.0 means top 2%). If None, use activity>0.
   :type top_percent: float | None

   :returns: **scores** -- CohoScore per neuron (NaN for neurons with too few points).
   :rtype: ndarray, shape (N,)

   .. rubric:: Examples

   >>> scores = compute_cohoscore_scatter_2d(coords, spikes)  # doctest: +SKIP
   >>> scores.shape[0]  # doctest: +SKIP


.. py:function:: draw_torus_parallelogram_grid_scatter(ax, n_tiles=1, color='0.7', lw=1.0, alpha=0.8)

   Draw parallelogram grid corresponding to torus fundamental domain.

   Fundamental vectors:
       e1 = (2π, 0)
       e2 = (π, √3 π)

   :param ax:
   :type ax: matplotlib axis
   :param n_tiles: How many tiles to draw in +/- directions (visual aid).
                   n_tiles=1 means draw [-1, 0, 1] shifts.
   :type n_tiles: int


.. py:function:: plot_cohospace_scatter_neuron_1d(coords, activity, neuron_id, mode = 'fr', top_percent = 5.0, times = None, auto_filter = True, figsize = (6, 6), cmap = 'hot', save_path = None, show = True, config = None)

   Overlay a single neuron's activity on the 1D cohomology trajectory (unit circle).


.. py:function:: plot_cohospace_scatter_neuron_2d(coords, activity, neuron_id, mode = 'fr', top_percent = 5.0, times = None, auto_filter = True, figsize = (6, 6), cmap = 'hot', save_path = None, show = True, config = None)

   Overlay a single neuron's activity on the cohomology-space trajectory.

   This is a visualization helper. In "fr" mode it marks the top top_percent% time points
   by firing rate for the neuron. In "spike" mode it marks all time points where spike > 0.

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
   :param top_percent: Used only when mode="fr". For example, 5.0 means "top 5%" time points.
   :type top_percent: float
   :param figsize:
   :type figsize: see `plot_cohospace_scatter_trajectory_2d`.
   :param cmap:
   :type cmap: see `plot_cohospace_scatter_trajectory_2d`.
   :param save_path:
   :type save_path: see `plot_cohospace_scatter_trajectory_2d`.
   :param show:
   :type show: see `plot_cohospace_scatter_trajectory_2d`.

   :returns: **ax**
   :rtype: matplotlib.axes.Axes

   .. rubric:: Examples

   >>> plot_cohospace_scatter_neuron_2d(coords, spikes, neuron_id=0, show=False)  # doctest: +SKIP


.. py:function:: plot_cohospace_scatter_neuron_skewed(coords, activity, neuron_id, mode='spike', top_percent=2.0, times = None, auto_filter = True, save_path=None, show=None, ax=None, show_grid=True, n_tiles=1, s=6, alpha=0.8, config = None)

   Plot single-neuron CohoSpace on skewed torus domain.

   :param coords: Decoded circular coordinates (theta1, theta2), in radians.
   :type coords: ndarray, shape (T, 2)
   :param activity: Activity matrix aligned with coords.
   :type activity: ndarray, shape (T, N)
   :param neuron_id: Neuron index.
   :type neuron_id: int
   :param mode: spike: use activity > 0
                fr: use top_percent threshold
   :type mode: {"spike", "fr"}
   :param top_percent: Percentile for FR thresholding.
   :type top_percent: float
   :param auto_filter: If True and lengths mismatch, auto-filter activity with activity>0 to mimic decode filtering.
   :type auto_filter: bool


.. py:function:: plot_cohospace_scatter_population_1d(coords, activity, neuron_ids, mode = 'fr', top_percent = 5.0, times = None, auto_filter = True, figsize = (6, 6), cmap = 'hot', save_path = None, show = True, config = None)

   Plot aggregated activity from multiple neurons on the 1D cohomology trajectory.


.. py:function:: plot_cohospace_scatter_population_2d(coords, activity, neuron_ids, mode = 'fr', top_percent = 5.0, times = None, auto_filter = True, figsize = (6, 6), cmap = 'hot', save_path = None, show = True, config = None)

   Plot aggregated activity from multiple neurons in cohomology space.

   In "fr" mode, select each neuron's top top_percent% time points by firing rate and
   aggregate (sum) firing rates over the selected points for coloring. In "spike" mode,
   count spikes at each time point (spike > 0) and aggregate counts over neurons.

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
   :type figsize: see `plot_cohospace_scatter_trajectory_2d`.
   :param cmap:
   :type cmap: see `plot_cohospace_scatter_trajectory_2d`.
   :param save_path:
   :type save_path: see `plot_cohospace_scatter_trajectory_2d`.
   :param show:
   :type show: see `plot_cohospace_scatter_trajectory_2d`.

   :returns: **ax**
   :rtype: matplotlib.axes.Axes

   .. rubric:: Examples

   >>> plot_cohospace_scatter_population_2d(coords, spikes, neuron_ids=[0, 1, 2], show=False)  # doctest: +SKIP


.. py:function:: plot_cohospace_scatter_population_skewed(coords, activity, neuron_ids, mode='spike', top_percent=2.0, times = None, auto_filter = True, save_path=None, show=False, ax=None, show_grid=True, n_tiles=1, s=4, alpha=0.5, config = None)

   Plot population CohoSpace on skewed torus domain.

   neuron_ids : list or ndarray
       Neurons to include (e.g. top-K by CohoScore).
   auto_filter : bool
       If True and lengths mismatch, auto-filter activity with activity>0 to mimic decode filtering.


.. py:function:: plot_cohospace_scatter_trajectory_1d(coords, times = None, subsample = 1, figsize = (6, 6), cmap = 'viridis', save_path = None, show = False, config = None)

   Plot a 1D cohomology trajectory on the unit circle.

   :param coords: Decoded cohomology angles (theta). Values may be in radians or in [0, 1] "unit circle"
                  convention depending on upstream decoding; this function will plot on the unit circle.
   :type coords: ndarray, shape (T,) or (T, 1)
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


.. py:function:: plot_cohospace_scatter_trajectory_2d(coords, times = None, subsample = 1, figsize = (6, 6), cmap = 'viridis', save_path = None, show = False, config = None)

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

   >>> fig = plot_cohospace_scatter_trajectory_2d(coords, subsample=2, show=False)  # doctest: +SKIP


.. py:function:: tile_parallelogram_points_scatter(xy, n_tiles=1)

   Tile points in the skewed (parallelogram) torus fundamental domain.

   This is mainly for static visualizations so you can visually inspect continuity
   across domain boundaries.

   :param points: Points in the skewed plane (same coordinates as returned by `skew_transform`).
   :type points: ndarray, shape (T, 2)
   :param n_tiles: Number of tiles to extend around the base domain.
                   - n_tiles=1 produces a 3x3 tiling
                   - n_tiles=2 produces a 5x5 tiling
   :type n_tiles: int

   :returns: **tiled** -- Tiled points.
   :rtype: ndarray


