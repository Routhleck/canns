src.canns.analyzer.visualization.spatial_plots
==============================================

.. py:module:: src.canns.analyzer.visualization.spatial_plots

.. autoapi-nested-parse::

   Spatial visualization functions for neural firing field heatmaps.

   This module provides plotting utilities for visualizing spatial firing patterns
   of neural populations, particularly for grid cells, place cells, and band cells.
   Includes specialized grid cell analysis visualizations (autocorrelation, grid score,
   spacing analysis) and tracking animations.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.spatial_plots.create_grid_cell_tracking_animation
   src.canns.analyzer.visualization.spatial_plots.plot_autocorrelation
   src.canns.analyzer.visualization.spatial_plots.plot_firing_field_heatmap
   src.canns.analyzer.visualization.spatial_plots.plot_grid_score
   src.canns.analyzer.visualization.spatial_plots.plot_grid_spacing_analysis


Module Contents
---------------

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


