canns.analyzer.data.cell_classification.visualization.btn_plots
===============================================================

.. py:module:: canns.analyzer.data.cell_classification.visualization.btn_plots

.. autoapi-nested-parse::

   BTN visualization utilities.



Functions
---------

.. autoapisummary::

   canns.analyzer.data.cell_classification.visualization.btn_plots.plot_btn_autocorr_summary
   canns.analyzer.data.cell_classification.visualization.btn_plots.plot_btn_distance_matrix


Module Contents
---------------

.. py:function:: plot_btn_autocorr_summary(*, acorr = None, labels = None, bin_times = None, res = None, mapping = None, colors = None, normalize = 'probability', smooth_sigma = None, long_max_ms = 200.0, short_max_ms = None, title = 'BTN temporal autocorr', figsize = (8, 3), save_path = None, show = True, config = None)

   Plot class-averaged ISI autocorr curves (mean +/- SEM).


.. py:function:: plot_btn_distance_matrix(*, dist = None, labels = None, mapping = None, sort_by_label = True, title = 'BTN distance matrix', cmap = 'afmhot', figsize = (5, 5), save_path = None, show = True, ax = None, config = None)

   Plot a distance matrix heatmap sorted by BTN cluster labels.


