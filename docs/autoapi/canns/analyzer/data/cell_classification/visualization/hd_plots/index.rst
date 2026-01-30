canns.analyzer.data.cell_classification.visualization.hd_plots
==============================================================

.. py:module:: canns.analyzer.data.cell_classification.visualization.hd_plots

.. autoapi-nested-parse::

   Head Direction Cell Visualization

   Plotting functions for head direction cell analysis.



Functions
---------

.. autoapisummary::

   canns.analyzer.data.cell_classification.visualization.hd_plots.plot_hd_analysis
   canns.analyzer.data.cell_classification.visualization.hd_plots.plot_polar_tuning
   canns.analyzer.data.cell_classification.visualization.hd_plots.plot_temporal_autocorr


Module Contents
---------------

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


