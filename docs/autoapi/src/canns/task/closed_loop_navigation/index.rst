src.canns.task.closed_loop_navigation
=====================================

.. py:module:: src.canns.task.closed_loop_navigation


Classes
-------

.. autoapisummary::

   src.canns.task.closed_loop_navigation.ClosedLoopNavigationTask
   src.canns.task.closed_loop_navigation.TMazeClosedLoopNavigationTask
   src.canns.task.closed_loop_navigation.TMazeRecessClosedLoopNavigationTask


Module Contents
---------------

.. py:class:: ClosedLoopNavigationTask(start_pos=(2.5, 2.5), width=5, height=5, dimensionality='2D', boundary_conditions='solid', scale=None, dx=0.01, grid_dx = None, grid_dy = None, boundary=None, walls=None, holes=None, objects=None, dt=None, speed_mean=0.04, speed_std=0.016, speed_coherence_time=0.7, rotational_velocity_coherence_time=0.08, rotational_velocity_std=120 * np.pi / 180, head_direction_smoothing_timescale=0.15, thigmotaxis=0.5, wall_repel_distance=0.1, wall_repel_strength=1.0)

   Bases: :py:obj:`src.canns.task.navigation_base.BaseNavigationTask`


   Closed-loop navigation task driven by external control.

   The agent moves step-by-step using commands supplied at runtime rather than
   following a pre-generated trajectory.

   Workflow:
       Setup -> Create a task and define environment boundaries.
       Execute -> Call ``step_by_pos`` for each new position.
       Result -> Use geodesic tools or agent history for analysis.

   .. rubric:: Examples

   >>> from canns.task.closed_loop_navigation import ClosedLoopNavigationTask
   >>>
   >>> task = ClosedLoopNavigationTask(
   ...     boundary=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
   ...     dt=0.1,
   ... )
   >>> task.step_by_pos((0.2, 0.2))
   >>> task.set_grid_resolution(0.5, 0.5)
   >>> grid = task.build_movement_cost_grid()
   >>> result = task.compute_geodesic_distance_matrix()
   >>> grid.costs.ndim
   2
   >>> result.distances.shape[0] == result.distances.shape[1]
   True


   .. py:method:: get_data()
      :abstractmethod:



   .. py:method:: step_by_pos(new_pos)


   .. py:attribute:: total_steps
      :value: 1



.. py:class:: TMazeClosedLoopNavigationTask(w=0.3, l_s=1.0, l_arm=0.75, t=0.3, start_pos=(0.0, 0.15), dt=None, **kwargs)

   Bases: :py:obj:`ClosedLoopNavigationTask`


   Closed-loop navigation task in a T-maze environment.

   Workflow:
       Setup -> Create a T-maze task.
       Execute -> Step the agent position.
       Result -> Build movement-cost grids or geodesic distances.

   .. rubric:: Examples

   >>> from canns.task.closed_loop_navigation import TMazeClosedLoopNavigationTask
   >>>
   >>> task = TMazeClosedLoopNavigationTask(dt=0.1)
   >>> task.step_by_pos(task.start_pos)
   >>> task.set_grid_resolution(0.5, 0.5)
   >>> grid = task.build_movement_cost_grid()
   >>> grid.costs.ndim
   2

   Initialize T-maze closed-loop navigation task.

   :param w: Width of the corridor (default: 0.3)
   :param l_s: Length of the stem (default: 1.0)
   :param l_arm: Length of each arm (default: 0.75)
   :param t: Thickness of the walls (default: 0.3)
   :param start_pos: Starting position of the agent (default: (0.0, 0.15))
   :param dt: Time step (default: None, uses bm.get_dt())
   :param \*\*kwargs: Additional keyword arguments passed to ClosedLoopNavigationTask


.. py:class:: TMazeRecessClosedLoopNavigationTask(w=0.3, l_s=1.0, l_arm=0.75, t=0.3, recess_width=None, recess_depth=None, start_pos=(0.0, 0.15), dt=None, **kwargs)

   Bases: :py:obj:`TMazeClosedLoopNavigationTask`


   Closed-loop navigation task in a T-maze with recesses at the junction.

   Workflow:
       Setup -> Create the recess T-maze task.
       Execute -> Step the agent position.
       Result -> Access environment-derived grids for analysis.

   .. rubric:: Examples

   >>> from canns.task.closed_loop_navigation import TMazeRecessClosedLoopNavigationTask
   >>>
   >>> task = TMazeRecessClosedLoopNavigationTask(dt=0.1)
   >>> task.step_by_pos(task.start_pos)
   >>> task.set_grid_resolution(0.5, 0.5)
   >>> grid = task.build_movement_cost_grid()
   >>> grid.costs.ndim
   2

   Initialize T-maze with recesses closed-loop navigation task.

   :param w: Width of the corridor (default: 0.3)
   :param l_s: Length of the stem (default: 1.0)
   :param l_arm: Length of each arm (default: 0.75)
   :param t: Thickness of the walls (default: 0.3)
   :param recess_width: Width of recesses at stem-arm junctions (default: t/4)
   :param recess_depth: Depth of recesses extending downward (default: t/4)
   :param start_pos: Starting position of the agent (default: (0.0, 0.15))
   :param dt: Time step (default: None, uses bm.get_dt())
   :param \*\*kwargs: Additional keyword arguments passed to ClosedLoopNavigationTask


