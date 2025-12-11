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


   Closed-loop navigation task that incorporates real-time feedback from a controller.

   In this task, the agent's movement is controlled step-by-step by external commands
   rather than following a pre-generated trajectory.


   .. py:method:: get_data()
      :abstractmethod:



   .. py:method:: step_by_pos(new_pos)


   .. py:attribute:: total_steps
      :value: 1



.. py:class:: TMazeClosedLoopNavigationTask(w=0.3, l_s=1.0, l_arm=0.75, t=0.3, start_pos=(0.0, 0.15), dt=None, **kwargs)

   Bases: :py:obj:`ClosedLoopNavigationTask`


   Closed-loop navigation task in a T-maze environment.

   This subclass configures the environment with a T-maze boundary, which is useful
   for studying decision-making and spatial navigation in a controlled setting.

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


   Closed-loop navigation task in a T-maze environment with recesses at stem-arm junctions.

   This variant adds small rectangular indentations at the T-junction, creating
   additional spatial features that may be useful for studying spatial navigation
   and decision-making.

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


