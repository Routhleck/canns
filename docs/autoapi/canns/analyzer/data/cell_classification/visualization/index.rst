canns.analyzer.data.cell_classification.visualization
=====================================================

.. py:module:: canns.analyzer.data.cell_classification.visualization

.. autoapi-nested-parse::

   Visualization modules.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/canns/analyzer/data/cell_classification/visualization/grid_plots/index
   /autoapi/canns/analyzer/data/cell_classification/visualization/hd_plots/index


Functions
---------

.. autoapisummary::

   canns.analyzer.data.cell_classification.visualization.plot_autocorrelogram
   canns.analyzer.data.cell_classification.visualization.plot_grid_score_histogram
   canns.analyzer.data.cell_classification.visualization.plot_gridness_analysis
   canns.analyzer.data.cell_classification.visualization.plot_hd_analysis
   canns.analyzer.data.cell_classification.visualization.plot_polar_tuning
   canns.analyzer.data.cell_classification.visualization.plot_rate_map
   canns.analyzer.data.cell_classification.visualization.plot_temporal_autocorr


Package Contents
----------------

.. py:function:: plot_autocorrelogram(autocorr, config = None, *, gridness_score = None, center_radius = None, peak_locations = None, title = 'Spatial Autocorrelation', xlabel = 'X Lag (bins)', ylabel = 'Y Lag (bins)', figsize = (6, 6), save_path = None, show = True, ax = None, **kwargs)

   Plot 2D autocorrelogram with optional annotations.


.. py:function:: plot_grid_score_histogram(scores, config = None, *, bins = 30, title = 'Grid Score Distribution', xlabel = 'Grid Score', ylabel = 'Count', figsize = (6, 4), save_path = None, show = True, ax = None, **kwargs)

   Plot histogram of gridness scores.


.. py:function:: plot_gridness_analysis(rate_map, autocorr, result, config = None, *, title = 'Grid Cell Analysis', figsize = (15, 5), save_path = None, show = True)

   Comprehensive grid analysis plot with rate map, autocorr, and statistics.


.. py:function:: plot_hd_analysis(result, time_stamps = None, head_directions = None, spike_times = None, figsize = (15, 5))

   Comprehensive head direction analysis plot.

   :param result: Results from HeadDirectionAnalyzer
   :type result: HDCellResult
   :param time_stamps: Time stamps for plotting trajectory
   :type time_stamps: np.ndarray, optional
   :param head_directions: Head direction time series
   :type head_directions: np.ndarray, optional
   :param spike_times: Spike times
   :type spike_times: np.ndarray, optional
   :param figsize: Figure size
   :type figsize: tuple, optional

   :returns: **fig** -- The figure object
   :rtype: plt.Figure


.. py:function:: plot_polar_tuning(angles, rates, preferred_direction = None, mvl = None, title = 'Directional Tuning', ax = None)

   Plot directional tuning curve in polar coordinates.

   :param angles: Angular bins in radians
   :type angles: np.ndarray
   :param rates: Firing rates for each bin
   :type rates: np.ndarray
   :param preferred_direction: Preferred direction to mark (radians)
   :type preferred_direction: float, optional
   :param mvl: Mean Vector Length to display
   :type mvl: float, optional
   :param title: Plot title
   :type title: str, optional
   :param ax: Polar axes to plot on. If None, creates new figure.
   :type ax: plt.Axes, optional

   :returns: **ax** -- The axes object
   :rtype: plt.Axes


.. py:function:: plot_rate_map(rate_map, config = None, *, title = 'Firing Field (Rate Map)', xlabel = 'X Position (bins)', ylabel = 'Y Position (bins)', figsize = (6, 6), colorbar = True, save_path = None, show = True, ax = None, **kwargs)

   Plot 2D spatial firing rate map.


.. py:function:: plot_temporal_autocorr(lags, acorr, title = 'Temporal Autocorrelation', ax = None)

   Plot temporal autocorrelation as bar plot.

   :param lags: Time lags (ms or bins)
   :type lags: np.ndarray
   :param acorr: Autocorrelation values
   :type acorr: np.ndarray
   :param title: Plot title
   :type title: str, optional
   :param ax: Axes to plot on
   :type ax: plt.Axes, optional

   :returns: **ax** -- The axes object
   :rtype: plt.Axes


