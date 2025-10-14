src.canns.task.closed_loop_navigation
=====================================

.. py:module:: src.canns.task.closed_loop_navigation


Attributes
----------

.. autoapisummary::

   src.canns.task.closed_loop_navigation.EPSILON
   src.canns.task.closed_loop_navigation.INT32_MAX


Classes
-------

.. autoapisummary::

   src.canns.task.closed_loop_navigation.ClosedLoopNavigationTask
   src.canns.task.closed_loop_navigation.GeodesicDistanceResult
   src.canns.task.closed_loop_navigation.MovementCostGrid
   src.canns.task.closed_loop_navigation.TMazeClosedLoopNavigationTask


Module Contents
---------------

.. py:class:: ClosedLoopNavigationTask(start_pos=(2.5, 2.5), width=5, height=5, dimensionality='2D', boundary_conditions='solid', scale=None, dx=0.01, boundary=None, walls=None, holes=None, objects=None, dt=None, speed_mean=0.04, speed_std=0.016, speed_coherence_time=0.7, rotational_velocity_coherence_time=0.08, rotational_velocity_std=120 * np.pi / 180, head_direction_smoothing_timescale=0.15, thigmotaxis=0.5, wall_repel_distance=0.1, wall_repel_strength=1.0)

   Bases: :py:obj:`src.canns.task._base.Task`


   A generic Task Abstract Base Class (ABC) designed to standardize the data
   handling workflow for various AI tasks.

   This class defines a standard interface that any concrete task class
   inheriting from it must implement, covering the core methods for data

   acquisition, processing, visualization, and access. It also provides a
   universal data-saving functionality.

   .. attribute:: data

      A container for the loaded or generated data, typically a
      NumPy array or a dictionary of arrays.

   Initializes the Task instance.

   :param data_class: A dataclass type for structured data.
                      If provided, the task will use this
                      class to structure the loaded or
                      generated data.
   :type data_class: type, optional


   .. py:method:: build_movement_cost_grid(dx, dy)

      Construct a grid-based movement cost map for the configured environment.

      A cell weight of ``1`` indicates free space, while ``INT32_MAX`` marks an
      impassable cell (intersecting a wall/hole or lying outside the boundary).

      :param dx: Grid cell width along the x axis.
      :param dy: Grid cell height along the y axis.

      :returns: MovementCostGrid describing the discretised environment.



   .. py:method:: compute_geodesic_distance_matrix(dx, dy)

      Compute pairwise geodesic distances between traversable grid cells.

      The computation treats each traversable cell (weight ``1``) as a graph node
      connected to its four axis-aligned neighbours. Horizontal steps cost ``dx``
      and vertical steps cost ``dy``. Impassable cells (``INT32_MAX``) are ignored.

      :param dx: Grid cell width along the x axis.
      :param dy: Grid cell height along the y axis.

      :returns: GeodesicDistanceResult containing the distance matrix and metadata.



   .. py:method:: get_data()
      :abstractmethod:


      Abstract core method for data acquisition.

      Subclasses must implement this method. Depending on the task type,
      the implementation could be:
      - Downloading and loading data from the web.
      - Reading data from the local filesystem.
      - Generating synthetic data in real-time.

      After execution, this method should assign the processed data to `self.data`.



   .. py:method:: show_data(show = True, save_path = None, *, overlay_movement_cost = False, cost_dx = None, cost_dy = None, cost_grid = None, free_color = '#f8f9fa', blocked_color = '#f94144', gridline_color = '#2b2d42', cost_alpha = 0.6, show_colorbar = False)

      Abstract method to display a task.

      Subclasses must implement this to visualize a sample in a way that is
      appropriate for its data type (e.g., plotting an image, a waveform,
      or printing text).



   .. py:method:: show_geodesic_distance_matrix(dx, dy, *, show = True, save_path = None, cmap = 'viridis', normalize = False, colorbar = True)

      Visualise the geodesic distance matrix for the discretised environment.



   .. py:method:: step_by_pos(new_pos)


   .. py:attribute:: agent


   .. py:attribute:: agent_params


   .. py:attribute:: aspect
      :value: 1.0



   .. py:attribute:: boundary


   .. py:attribute:: boundary_conditions
      :value: 'solid'



   .. py:attribute:: dimensionality
      :value: ''



   .. py:attribute:: dt
      :value: None



   .. py:attribute:: dx
      :value: 0.01



   .. py:attribute:: env


   .. py:attribute:: env_params


   .. py:attribute:: head_direction_smoothing_timescale
      :value: 0.15



   .. py:attribute:: height
      :value: 5



   .. py:attribute:: holes


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



   .. py:attribute:: total_steps
      :value: 1



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


.. py:class:: TMazeClosedLoopNavigationTask(w=0.3, l_s=1.0, l_arm=0.75, t=0.3, start_pos=(0.0, 0.15), dt=None, **kwargs)

   Bases: :py:obj:`ClosedLoopNavigationTask`


   A generic Task Abstract Base Class (ABC) designed to standardize the data
   handling workflow for various AI tasks.

   This class defines a standard interface that any concrete task class
   inheriting from it must implement, covering the core methods for data

   acquisition, processing, visualization, and access. It also provides a
   universal data-saving functionality.

   .. attribute:: data

      A container for the loaded or generated data, typically a
      NumPy array or a dictionary of arrays.

   Initializes the Task instance.

   :param data_class: A dataclass type for structured data.
                      If provided, the task will use this
                      class to structure the loaded or
                      generated data.
   :type data_class: type, optional


.. py:data:: EPSILON
   :value: 1e-12


.. py:data:: INT32_MAX

