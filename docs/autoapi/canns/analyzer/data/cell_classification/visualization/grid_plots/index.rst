canns.analyzer.data.cell_classification.visualization.grid_plots
================================================================

.. py:module:: canns.analyzer.data.cell_classification.visualization.grid_plots

.. autoapi-nested-parse::

   Grid cell visualization utilities.



Functions
---------

.. autoapisummary::

   canns.analyzer.data.cell_classification.visualization.grid_plots.plot_autocorrelogram
   canns.analyzer.data.cell_classification.visualization.grid_plots.plot_grid_score_histogram
   canns.analyzer.data.cell_classification.visualization.grid_plots.plot_gridness_analysis
   canns.analyzer.data.cell_classification.visualization.grid_plots.plot_rate_map


Module Contents
---------------

.. py:function:: plot_autocorrelogram(autocorr, config = None, *, gridness_score = None, center_radius = None, peak_locations = None, title = 'Spatial Autocorrelation', xlabel = 'X Lag (bins)', ylabel = 'Y Lag (bins)', figsize = (6, 6), save_path = None, show = True, ax = None, **kwargs)

   Plot 2D autocorrelogram with optional annotations.


.. py:function:: plot_grid_score_histogram(scores, config = None, *, bins = 30, title = 'Grid Score Distribution', xlabel = 'Grid Score', ylabel = 'Count', figsize = (6, 4), save_path = None, show = True, ax = None, **kwargs)

   Plot histogram of gridness scores.


.. py:function:: plot_gridness_analysis(rate_map, autocorr, result, config = None, *, title = 'Grid Cell Analysis', figsize = (15, 5), save_path = None, show = True)

   Comprehensive grid analysis plot with rate map, autocorr, and statistics.


.. py:function:: plot_rate_map(rate_map, config = None, *, title = 'Firing Field (Rate Map)', xlabel = 'X Position (bins)', ylabel = 'Y Position (bins)', figsize = (6, 6), colorbar = True, save_path = None, show = True, ax = None, **kwargs)

   Plot 2D spatial firing rate map.


