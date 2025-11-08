src.canns.analyzer.plotting.spatial
===================================

.. py:module:: src.canns.analyzer.plotting.spatial

.. autoapi-nested-parse::

   Spatial visualization functions for neural firing field heatmaps.

   This module provides plotting utilities for visualizing spatial firing patterns
   of neural populations, particularly for grid cells, place cells, and band cells.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.plotting.spatial.plot_firing_field_heatmap


Module Contents
---------------

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

   >>> from canns.analyzer.spatial import compute_firing_field
   >>> from canns.analyzer.plotting import plot_firing_field_heatmap, PlotConfig
   >>> # Compute firing field
   >>> heatmaps = compute_firing_field(activity, positions, 5.0, 5.0, 50, 50)
   >>> # Plot single neuron with PlotConfig
   >>> config = PlotConfig(figsize=(6, 6), save_path='neuron_0.png', show=False)
   >>> fig, ax = plot_firing_field_heatmap(heatmaps[0], config=config)
   >>> # Plot with legacy parameters
   >>> fig, ax = plot_firing_field_heatmap(heatmaps[1], cmap='viridis', save_path='neuron_1.png')


