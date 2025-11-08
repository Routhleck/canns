src.canns.task.navigation_base
==============================

.. py:module:: src.canns.task.navigation_base

.. autoapi-nested-parse::

   Base navigation task with geodesic distance computation capabilities.



Classes
-------

.. autoapisummary::

   src.canns.task.navigation_base.BaseNavigationTask
   src.canns.task.navigation_base.GeodesicDistanceResult
   src.canns.task.navigation_base.MovementCostGrid


Module Contents
---------------

.. py:class:: BaseNavigationTask(start_pos=(2.5, 2.5), width=5, height=5, dimensionality='2D', boundary_conditions='solid', scale=None, dx=0.01, grid_dx = None, grid_dy = None, boundary=None, walls=None, holes=None, objects=None, dt=None, speed_mean=0.04, speed_std=0.016, speed_coherence_time=0.7, rotational_velocity_coherence_time=0.08, rotational_velocity_std=120 * np.pi / 180, head_direction_smoothing_timescale=0.15, initial_head_direction = None, thigmotaxis=0.5, wall_repel_distance=0.1, wall_repel_strength=1.0, data_class=None)

   Bases: :py:obj:`src.canns.task._base.Task`


   Base class for navigation tasks with geodesic distance computation support.

   This class provides common functionality for both open-loop and closed-loop
   navigation tasks, including environment setup, agent initialization, and
   geodesic distance computation on discretized grids.

   Initializes the Task instance.

   :param data_class: A dataclass type for structured data.
                      If provided, the task will use this
                      class to structure the loaded or
                      generated data.
   :type data_class: type, optional


   .. py:method:: build_movement_cost_grid(*, refresh = False)

      Construct a grid-based movement cost map for the configured environment.

      A cell weight of ``1`` indicates free space, while ``INT32_MAX`` marks an
      impassable cell (intersecting a wall/hole or lying outside the boundary).

      :param refresh: Force recomputation even if a cached grid is available.

      :returns: MovementCostGrid describing the discretised environment.



   .. py:method:: compute_geodesic_distance_matrix(dx = None, dy = None, *, refresh = False)

      Compute pairwise geodesic distances between traversable grid cells.

      The computation treats each traversable cell (weight ``1``) as a graph node
      connected to its four axis-aligned neighbours. Horizontal steps cost ``dx``
      and vertical steps cost ``dy``. Impassable cells (``INT32_MAX``) are ignored.

      When Numba is available, this method uses parallelized Dijkstra computation
      across CPU cores for significant speedup (typically 4-8x on multi-core systems).
      Without Numba, it falls back to sequential Python implementation with a
      progress bar.

      :param dx: Grid cell width along the x axis. When ``None`` the existing
                 ``grid_dx`` attribute is used.
      :param dy: Grid cell height along the y axis. When ``None`` the existing
                 ``grid_dy`` attribute is used.
      :param refresh: Force recomputation even if cached results exist.

      :returns: GeodesicDistanceResult containing the distance matrix and metadata.

      .. note::

         The parallel Numba implementation cannot show a progress bar during
         computation, but prints start/end messages instead.



   .. py:method:: get_geodesic_index_by_pos(pos, *, refresh = False)

      Get the index of the grid cell containing the given position.

      :param pos: (x, y) coordinates of the position.
      :param refresh: Recompute the cached grid before querying the index.

      :returns: Index of the grid cell in the geodesic distance matrix, or None if
                the position is out of bounds or in an impassable cell.



   .. py:method:: set_grid_resolution(dx, dy)

      Update the stored grid resolution and invalidate cached data.



   .. py:method:: show_data(show = True, save_path = None, *, overlay_movement_cost = False, cost_grid = None, free_color = '#f8f9fa', blocked_color = '#f94144', gridline_color = '#2b2d42', cost_alpha = 0.6, show_colorbar = False, cost_legend_loc = None)

      Display the agent's trajectory with optional movement cost grid overlay.

      :param show: Whether to display the plot.
      :param save_path: Path to save the figure. If None, the figure is not saved.
      :param overlay_movement_cost: Whether to overlay the movement cost grid.
      :param cost_grid: Pre-computed cost grid. If None and overlay_movement_cost is True,
                        the grid will be built on demand.
      :param free_color: Color for free (accessible) cells in the cost grid.
      :param blocked_color: Color for blocked (inaccessible) cells in the cost grid.
      :param gridline_color: Color for grid lines.
      :param cost_alpha: Transparency of the cost grid overlay (0=transparent, 1=opaque).
      :param show_colorbar: Whether to show a colorbar for the cost grid.
      :param cost_legend_loc: Location of the legend for the cost grid (e.g., 'upper right').
                              If None, no legend is shown.



   .. py:method:: show_geodesic_distance_matrix(dx = None, dy = None, *, show = True, save_path = None, cmap = 'viridis', normalize = False, colorbar = True, refresh = False)

      Visualise the geodesic distance matrix for the discretised environment.



   .. py:attribute:: agent


   .. py:attribute:: agent_params


   .. py:attribute:: aspect
      :value: 1.0



   .. py:attribute:: boundary


   .. py:attribute:: boundary_conditions
      :value: 'solid'



   .. py:attribute:: cost_grid
      :type:  MovementCostGrid | None
      :value: None



   .. py:attribute:: dimensionality
      :value: ''



   .. py:attribute:: dt


   .. py:attribute:: dx
      :value: 0.01



   .. py:attribute:: env


   .. py:attribute:: env_params


   .. py:attribute:: geodesic_result
      :type:  GeodesicDistanceResult | None
      :value: None



   .. py:attribute:: grid_dx
      :value: 0.01



   .. py:attribute:: grid_dy
      :value: 0.01



   .. py:attribute:: head_direction_smoothing_timescale
      :value: 0.15



   .. py:attribute:: height
      :value: 5



   .. py:attribute:: holes


   .. py:attribute:: initial_head_direction
      :value: None



   .. py:attribute:: objects


   .. py:attribute:: rotational_velocity_coherence_time
      :value: 0.08



   .. py:attribute:: rotational_velocity_std


   .. py:attribute:: scale
      :value: 5



   .. py:attribute:: speed_coherence_time
      :value: 0.7



   .. py:attribute:: speed_mean
      :value: 0.04



   .. py:attribute:: speed_std
      :value: 0.016



   .. py:attribute:: start_pos
      :value: (2.5, 2.5)



   .. py:attribute:: thigmotaxis
      :value: 0.5



   .. py:attribute:: wall_repel_distance
      :value: 0.1



   .. py:attribute:: wall_repel_strength
      :value: 1.0



   .. py:attribute:: walls


   .. py:attribute:: width
      :value: 5



.. py:class:: GeodesicDistanceResult

   .. py:attribute:: accessible_indices
      :type:  numpy.ndarray


   .. py:attribute:: cost_grid
      :type:  MovementCostGrid


   .. py:attribute:: distances
      :type:  numpy.ndarray


.. py:class:: MovementCostGrid

   .. py:method:: get_cell_index(pos)

      Get the geodesic index of the grid cell containing the given position.

      This method is JAX-compatible and can be used inside jitted functions.

      :param pos: (x, y) coordinates of the position.

      :returns: Index of the grid cell in the accessible_indices array, or -1 if
                the position is out of bounds or in an impassable cell.

      .. note::

         Returns -1 (instead of None) for JAX compatibility. The caller should
         check for negative values to detect invalid positions.



   .. py:attribute:: accessible_indices
      :type:  numpy.ndarray | None
      :value: None



   .. py:property:: accessible_mask
      :type: numpy.ndarray



   .. py:attribute:: costs
      :type:  numpy.ndarray


   .. py:attribute:: dx
      :type:  float


   .. py:attribute:: dy
      :type:  float


   .. py:property:: shape
      :type: tuple[int, int]



   .. py:property:: x_centers
      :type: numpy.ndarray



   .. py:attribute:: x_edges
      :type:  numpy.ndarray


   .. py:property:: y_centers
      :type: numpy.ndarray



   .. py:attribute:: y_edges
      :type:  numpy.ndarray


