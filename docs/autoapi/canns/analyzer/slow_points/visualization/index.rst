canns.analyzer.slow_points.visualization
========================================

.. py:module:: canns.analyzer.slow_points.visualization

.. autoapi-nested-parse::

   Visualization functions for fixed point analysis.



Functions
---------

.. autoapisummary::

   canns.analyzer.slow_points.visualization.plot_fixed_points_2d
   canns.analyzer.slow_points.visualization.plot_fixed_points_3d


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

   >>> import numpy as np
   >>> from canns.analyzer.slow_points import plot_fixed_points_2d, FixedPoints
   >>> from canns.analyzer.visualization import PlotConfig
   >>>
   >>> # Dummy inputs based on fixed-point tests
   >>> state_traj = np.random.rand(4, 10, 3).astype(np.float32)
   >>> fixed_points = FixedPoints(
   ...     xstar=np.random.rand(2, 3).astype(np.float32),
   ...     is_stable=np.array([True, False]),
   ... )
   >>> config = PlotConfig(title="Fixed Points (2D)", show=False)
   >>> fig = plot_fixed_points_2d(fixed_points, state_traj, config=config)
   >>> print(fig is not None)
   True


.. py:function:: plot_fixed_points_3d(fixed_points, state_traj, config = None, plot_batch_idx = None, plot_start_time = 0)

   Plot fixed points and trajectories in 3D using PCA.

   :param fixed_points: FixedPoints object containing analysis results.
   :param state_traj: State trajectories [n_batch x n_time x n_states].
   :param config: Plot configuration. If None, uses default config.
   :param plot_batch_idx: Batch indices to plot trajectories. If None, plots first 30.
   :param plot_start_time: Starting time index for trajectory plotting.

   :returns: matplotlib Figure object.

   .. rubric:: Example

   >>> import numpy as np
   >>> from canns.analyzer.slow_points import plot_fixed_points_3d, FixedPoints
   >>> from canns.analyzer.visualization import PlotConfig
   >>>
   >>> # Dummy inputs based on fixed-point tests
   >>> state_traj = np.random.rand(3, 8, 4).astype(np.float32)
   >>> fixed_points = FixedPoints(
   ...     xstar=np.random.rand(2, 4).astype(np.float32),
   ...     is_stable=np.array([True, False]),
   ... )
   >>> config = PlotConfig(title="Fixed Points (3D)", show=False)
   >>> fig = plot_fixed_points_3d(fixed_points, state_traj, config=config)
   >>> print(fig is not None)
   True


