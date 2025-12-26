src.canns.analyzer.visualization.theta_sweep_plots
==================================================

.. py:module:: src.canns.analyzer.visualization.theta_sweep_plots

.. autoapi-nested-parse::

   Theta sweep specific visualization functions for CANNs models.

   This module contains specialized plotting functions for analyzing theta-modulated
   neural activity, particularly for direction cell and grid cell networks.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.theta_sweep_plots.create_theta_sweep_grid_cell_animation
   src.canns.analyzer.visualization.theta_sweep_plots.create_theta_sweep_place_cell_animation
   src.canns.analyzer.visualization.theta_sweep_plots.plot_direction_cell_polar
   src.canns.analyzer.visualization.theta_sweep_plots.plot_grid_cell_manifold
   src.canns.analyzer.visualization.theta_sweep_plots.plot_population_activity_with_theta


Module Contents
---------------

.. py:function:: create_theta_sweep_grid_cell_animation(position_data, direction_data, dc_activity_data, gc_activity_data, gc_network, env_size, mapping_ratio, dt = 0.001, config = None, n_step = 10, fps = 10, figsize = (12, 3), save_path = None, show = True, show_progress_bar = True, render_backend = 'auto', output_dpi = 150, render_workers = None, render_start_method = None, **kwargs)

   Create comprehensive theta sweep animation with 4 panels (optimized for speed):
   1. Animal trajectory
   2. Direction cell polar plot
   3. Grid cell activity on manifold
   4. Grid cell activity in real space

   :param position_data: Animal position data (time, 2)
   :param direction_data: Direction data (time,)
   :param dc_activity_data: Direction cell activity (time, neurons)
   :param gc_activity_data: Grid cell activity (time, neurons)
   :param gc_network: GridCellNetwork instance for coordinate transformations
   :param env_size: Environment size
   :param mapping_ratio: Mapping ratio for grid cells
   :param dt: Time step size
   :param config: PlotConfig object for unified configuration
   :param n_step: Subsample every n_step frames for animation
   :param render_backend: Rendering backend. Use 'matplotlib', 'imageio', or 'auto'/'None' for auto-detect.
   :param output_dpi: Target DPI when rendering frames with non-interactive backends
   :param render_workers: Worker processes for imageio backend. ``None`` auto-selects, 0 disables.
   :param render_start_method: Multiprocessing start method ('fork', 'spawn', 'forkserver') or None for auto
   :param \*\*kwargs: Additional parameters for backward compatibility

   :returns: Matplotlib animation object for interactive backend, otherwise None
   :rtype: FuncAnimation | None


.. py:function:: create_theta_sweep_place_cell_animation(position_data, pc_activity_data, pc_network, navigation_task, dt = 0.001, config = None, n_step = 10, fps = 10, figsize = (12, 4), save_path = None, show = True, show_progress_bar = True, **kwargs)

   Create theta sweep animation for place cell network with 2 panels:
   1. Environment trajectory with place cell bump overlay
   2. Population activity heatmap over time

   :param position_data: Animal position data (time, 2)
   :param pc_activity_data: Place cell activity (time, num_cells)
   :param pc_network: PlaceCellNetwork instance
   :param navigation_task: BaseNavigationTask instance for environment visualization
   :param dt: Time step size
   :param config: PlotConfig object for unified configuration
   :param n_step: Subsample every n_step frames for animation
   :param fps: Frames per second for animation
   :param figsize: Figure size (width, height)
   :param save_path: Path to save animation (GIF or MP4)
   :param show: Whether to display animation
   :param show_progress_bar: Whether to show progress bar during saving
   :param \*\*kwargs: Additional parameters (cmap, alpha, etc.)

   :returns: Matplotlib animation object
   :rtype: FuncAnimation


.. py:function:: plot_direction_cell_polar(direction_bins, direction_activity, true_direction, config = None, title = 'Direction Cell Activity', figsize = (6, 6), show = True, save_path = None, **kwargs)

   Plot direction cell activity in polar coordinates.

   :param direction_bins: Array of direction bins (radians)
   :param direction_activity: Array of activity values for each direction
   :param true_direction: True direction value (radians)
   :param config: PlotConfig object for unified configuration
   :param \*\*kwargs: Additional parameters for backward compatibility

   :returns: (figure, axis) objects
   :rtype: tuple


.. py:function:: plot_grid_cell_manifold(value_grid_twisted, grid_cell_activity, config = None, ax = None, title = 'Grid Cell Activity on Manifold', figsize = (8, 6), cmap = 'jet', show = True, save_path = None, **kwargs)

   Plot grid cell activity on the twisted torus manifold.

   :param value_grid_twisted: Coordinates on twisted manifold
   :param grid_cell_activity: 2D array of grid cell activities
   :param config: PlotConfig object for unified configuration
   :param ax: Optional axis to draw on instead of creating a new figure
   :param \*\*kwargs: Additional parameters for backward compatibility

   :returns: (figure, axis) objects
   :rtype: tuple


.. py:function:: plot_population_activity_with_theta(time_steps, theta_phase, net_activity, direction, config = None, add_lines = True, atol = 0.01, title = 'Population Activity with Theta', xlabel = 'Time (s)', ylabel = 'Direction (°)', figsize = (12, 4), cmap = 'jet', show = True, save_path = None, **kwargs)

   Plot neural population activity with theta oscillation markers and direction trace.

   :param time_steps: Array of time points
   :param theta_phase: Array of theta phase values [-π, π]
   :param net_activity: 2D array of network activity (time, neurons)
   :param direction: Array of direction values
   :param config: PlotConfig object for unified configuration
   :param add_lines: Whether to add vertical lines at theta phase zeros
   :param atol: Tolerance for detecting theta phase zeros
   :param \*\*kwargs: Additional parameters for backward compatibility

   :returns: (figure, axis) objects
   :rtype: tuple


