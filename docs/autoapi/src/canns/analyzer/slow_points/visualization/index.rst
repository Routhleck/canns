src.canns.analyzer.slow_points.visualization
============================================

.. py:module:: src.canns.analyzer.slow_points.visualization

.. autoapi-nested-parse::

   Visualization functions for fixed point analysis.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.slow_points.visualization.plot_fixed_points_2d
   src.canns.analyzer.slow_points.visualization.plot_fixed_points_3d


Module Contents
---------------

.. py:function:: plot_fixed_points_2d(fixed_points, state_traj, config = None, plot_batch_idx = None, plot_start_time = 0)

   Plot fixed points and trajectories in 2D using PCA.

   :param fixed_points: FixedPoints object containing analysis results.
   :param state_traj: State trajectories [n_batch x n_time x n_states].
   :param config: Plot configuration. If None, uses default config.
   :param plot_batch_idx: Batch indices to plot trajectories. If None, plots first 30.
   :param plot_start_time: Starting time index for trajectory plotting.

   :returns: matplotlib Figure object.

   .. rubric:: Example

   >>> from canns.analyzer.slow_points import plot_fixed_points_2d, FixedPoints
   >>> from canns.analyzer.visualization import PlotConfig
   >>> config = PlotConfig(
   ...     title="Fixed Points Analysis",
   ...     figsize=(10, 8),
   ...     save_path="fps_2d.png"
   ... )
   >>> fig = plot_fixed_points_2d(unique_fps, hiddens, config=config)


.. py:function:: plot_fixed_points_3d(fixed_points, state_traj, config = None, plot_batch_idx = None, plot_start_time = 0)

   Plot fixed points and trajectories in 3D using PCA.

   :param fixed_points: FixedPoints object containing analysis results.
   :param state_traj: State trajectories [n_batch x n_time x n_states].
   :param config: Plot configuration. If None, uses default config.
   :param plot_batch_idx: Batch indices to plot trajectories. If None, plots first 30.
   :param plot_start_time: Starting time index for trajectory plotting.

   :returns: matplotlib Figure object.

   .. rubric:: Example

   >>> from canns.analyzer.slow_points import plot_fixed_points_3d, FixedPoints
   >>> from canns.analyzer.visualization import PlotConfig
   >>> config = PlotConfig(
   ...     title="Fixed Points 3D",
   ...     figsize=(12, 10),
   ...     save_path="fps_3d.png"
   ... )
   >>> fig = plot_fixed_points_3d(unique_fps, hiddens, config=config)


