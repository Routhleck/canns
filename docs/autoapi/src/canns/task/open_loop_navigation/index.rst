src.canns.task.open_loop_navigation
===================================

.. py:module:: src.canns.task.open_loop_navigation


Classes
-------

.. autoapisummary::

   src.canns.task.open_loop_navigation.ActionPolicy
   src.canns.task.open_loop_navigation.CustomOpenLoopNavigationTask
   src.canns.task.open_loop_navigation.OpenLoopNavigationData
   src.canns.task.open_loop_navigation.OpenLoopNavigationTask
   src.canns.task.open_loop_navigation.RasterScanNavigationTask
   src.canns.task.open_loop_navigation.StateAwareRasterScanPolicy
   src.canns.task.open_loop_navigation.TMazeOpenLoopNavigationTask
   src.canns.task.open_loop_navigation.TMazeRecessOpenLoopNavigationTask


Functions
---------

.. autoapisummary::

   src.canns.task.open_loop_navigation.map2pi


Module Contents
---------------

.. py:class:: ActionPolicy

   Bases: :py:obj:`abc.ABC`


   Abstract base class for action policies that control agent movement.

   Action policies compute parameters for agent.update() at each simulation step,
   enabling reusable, testable, and composable control strategies.

   .. rubric:: Example

   ```python
   class ConstantDriftPolicy(ActionPolicy):
       def __init__(self, drift_direction):
           self.drift = np.array(drift_direction)

       def compute_action(self, step_idx, agent):
           return {'drift_velocity': self.drift}

   task = CustomOpenLoopNavigationTask(
       duration=100, action_policy=ConstantDriftPolicy([0.1, 0.0])
   )
   ```


   .. py:method:: compute_action(step_idx, agent)
      :abstractmethod:


      Compute action parameters for the current simulation step.

      :param step_idx: Current simulation step (0 to total_steps-1)
      :param agent: Agent instance (for state-aware policies)

      :returns:

                Keyword arguments for agent.update()
                      Supported keys:
                      - drift_velocity: np.ndarray of shape (2,) for 2D drift
                      - drift_to_random_strength_ratio: float (typically 5.0-20.0)
                      - forced_next_position: np.ndarray of shape (2,)
      :rtype: dict



.. py:class:: CustomOpenLoopNavigationTask(*args, action_policy = None, **kwargs)

   Bases: :py:obj:`OpenLoopNavigationTask`


   Template class for policy-based open-loop navigation tasks.

   This class enables custom action policies by accepting an ActionPolicy object
   that controls agent movement at each simulation step.

   :param action_policy: ActionPolicy instance controlling agent movement
   :param \*\*kwargs: All other arguments passed to OpenLoopNavigationTask

   .. rubric:: Example

   ```python
   # Define custom policy
   class MyPolicy(ActionPolicy):
       def compute_action(self, step_idx, agent):
           return {'drift_velocity': np.array([0.1, 0.0])}

   # Use it
   task = CustomOpenLoopNavigationTask(
       duration=100, action_policy=MyPolicy()
   )
   task.get_data()
   ```


   .. py:attribute:: action_policy
      :value: None



.. py:class:: OpenLoopNavigationData

   Container for the inputs recorded during the open-loop navigation task.
   It contains the position, velocity, speed, movement direction, head direction,
   and rotational velocity of the agent.

   Additional fields for theta sweep analysis:
   - ang_velocity: Angular velocity calculated using unwrap method
   - linear_speed_gains: Normalized linear speed gains [0,1]
   - ang_speed_gains: Normalized angular speed gains [-1,1]


   .. py:attribute:: ang_speed_gains
      :type:  numpy.ndarray | None
      :value: None



   .. py:attribute:: ang_velocity
      :type:  numpy.ndarray | None
      :value: None



   .. py:attribute:: hd_angle
      :type:  numpy.ndarray


   .. py:attribute:: linear_speed_gains
      :type:  numpy.ndarray | None
      :value: None



   .. py:attribute:: movement_direction
      :type:  numpy.ndarray


   .. py:attribute:: position
      :type:  numpy.ndarray


   .. py:attribute:: rot_vel
      :type:  numpy.ndarray


   .. py:attribute:: speed
      :type:  numpy.ndarray


   .. py:attribute:: velocity
      :type:  numpy.ndarray


.. py:class:: OpenLoopNavigationTask(duration=20.0, start_pos=(2.5, 2.5), initial_head_direction=None, progress_bar=True, width=5, height=5, dimensionality='2D', boundary_conditions='solid', scale=None, dx=0.01, grid_dx = None, grid_dy = None, boundary=None, walls=None, holes=None, objects=None, dt=None, speed_mean=0.04, speed_std=0.016, speed_coherence_time=0.7, rotational_velocity_coherence_time=0.08, rotational_velocity_std=120 * np.pi / 180, head_direction_smoothing_timescale=0.15, thigmotaxis=0.5, wall_repel_distance=0.1, wall_repel_strength=1.0)

   Bases: :py:obj:`src.canns.task.navigation_base.BaseNavigationTask`


   Open-loop spatial navigation task that synthesises trajectories without
   incorporating real-time feedback from a controller.


   .. py:method:: calculate_theta_sweep_data()

      Calculate additional fields needed for theta sweep analysis.
      This should be called after get_data() to add ang_velocity,
      linear_speed_gains, and ang_speed_gains to the data.



   .. py:method:: get_data()

      Generates the inputs for the agent based on its current position.



   .. py:method:: get_empty_trajectory()

      Returns an empty trajectory data structure with the same shape as the generated trajectory.
      This is useful for initializing the trajectory data structure without any actual data.



   .. py:method:: import_data(position_data, times = None, dt = None, head_direction = None, initial_pos = None)

      Import external position coordinates and calculate derived features.

      This method allows importing external trajectory data (e.g., from experimental
      recordings or other simulations) instead of using the built-in random motion model.
      The imported data will be processed to calculate velocity, speed, movement direction,
      head direction, and rotational velocity.

      :param position_data: Array of position coordinates with shape (n_steps, 2)
                            for 2D trajectories or (n_steps, 1) for 1D trajectories.
      :type position_data: np.ndarray
      :param times: Array of time points corresponding to position_data.
                    If None, uniform time steps with dt will be assumed.
      :type times: np.ndarray, optional
      :param dt: Time step between consecutive positions. If None, uses
                 self.dt. Required if times is None.
      :type dt: float, optional
      :param head_direction: Array of head direction angles in radians
                             with shape (n_steps,). If None, head direction
                             will be derived from movement direction.
      :type head_direction: np.ndarray, optional
      :param initial_pos: Initial position for the agent. If None,
                          uses the first position from position_data.
      :type initial_pos: np.ndarray, optional

      :raises ValueError: If position_data has invalid dimensions or if required parameters
          are missing.

      .. rubric:: Example

      ```python
      # Import experimental trajectory data
      positions = np.array([[0, 0], [0.1, 0.05], [0.2, 0.1], ...])  # shape: (n_steps, 2)
      times = np.array([0, 0.1, 0.2, ...])  # shape: (n_steps,)

      task = OpenLoopNavigationTask(...)
      task.import_data(position_data=positions, times=times)

      # Or with uniform time steps
      task.import_data(position_data=positions, dt=0.1)
      ```



   .. py:method:: reset()

      Resets the agent's position to the starting position.



   .. py:method:: show_trajectory_analysis(show = True, save_path = None, figsize = (12, 3), smooth_window = 50, **kwargs)

      Display comprehensive trajectory analysis including position, speed, and direction changes.

      :param show: Whether to display the plot
      :param save_path: Path to save the figure
      :param figsize: Figure size (width, height)
      :param smooth_window: Window size for smoothing speed and direction plots (set to 0 to disable smoothing)
      :param \*\*kwargs: Additional matplotlib parameters



   .. py:attribute:: duration
      :value: 20.0



   .. py:attribute:: progress_bar
      :value: True



   .. py:attribute:: run_steps


   .. py:attribute:: total_steps


.. py:class:: RasterScanNavigationTask(duration, width = 1.0, height = 1.0, step_size = 0.03, margin = 0.05, speed = 0.15, drift_strength = 15.0, **kwargs)

   Bases: :py:obj:`CustomOpenLoopNavigationTask`


   Preset task for cyclic dual-mode state-aware raster scan exploration.

   Systematically explores the environment using cyclic mode switching:
   1. Horizontal phase: Left-right sweeps moving downward
      → Switches to Vertical when reaching bottom
   2. Vertical phase: Up-down sweeps moving rightward
      → Switches back to Horizontal when reaching right edge
   3. Cycles continuously: H → V → H → V → ...

   This cyclic dual-mode strategy achieves superior coverage by combining
   orthogonal scanning patterns and continuously adapting to avoid walls.

   Performance (200s, 1.0m x 1.0m):
       - Cyclic dual-mode: ~75-80% coverage (continuous cycling)
       - Single horizontal: 54.1% coverage (29 rows)
       - +20-30% improvement over random walk

   :param duration: Simulation duration in seconds
   :param width: Environment width (default: 1.0)
   :param height: Environment height (default: 1.0)
   :param step_size: Scan density - smaller = denser scanning (default: 0.03)
   :param margin: Wall detection margin (default: 0.05)
   :param speed: Movement speed in m/s (default: 0.15)
   :param drift_strength: Drift control strength (default: 15.0)
   :param \*\*kwargs: Additional arguments passed to OpenLoopNavigationTask

   .. rubric:: Example

   ```python
   # High coverage dual-mode exploration
   task = RasterScanNavigationTask(
       duration=200,
       width=1.0,
       height=1.0,
       step_size=0.03,  # Dense scanning in both directions
       speed=0.15  # Movement speed
   )
   task.get_data()
   task.show_trajectory_analysis()
   ```


.. py:class:: StateAwareRasterScanPolicy(width, height, margin = 0.05, step_size = 0.03, speed = 0.15, drift_strength = 15.0)

   Bases: :py:obj:`ActionPolicy`


   State-aware raster scan policy with cyclic dual-mode exploration.

   Scanning strategy (循环扫描):
   1. Horizontal mode: Left-right sweeps moving downward
      → When reaching bottom: switch to Vertical mode
   2. Vertical mode: Up-down sweeps moving rightward
      → When reaching right edge: switch back to Horizontal mode
   3. Cycles continuously: H → V → H → V → ... (避免撞墙)

   This cyclic dual-mode approach achieves comprehensive coverage by combining
   orthogonal scanning patterns and avoiding wall collisions.

   Tested performance (200s, 1.0m x 1.0m environment):
       - Cyclic dual-mode: ~75-80%+ coverage (continuous cycling)
       - Single horizontal: 54.1% coverage (29 rows)

   :param width: Environment width in meters
   :param height: Environment height in meters
   :param margin: Distance from wall to trigger turn (default: 0.05)
   :param step_size: Movement per turn in perpendicular direction (default: 0.03)
   :param speed: Movement speed (default: 0.15)
   :param drift_strength: Drift-to-random ratio for agent.update() (default: 15.0)

   .. rubric:: Example

   ```python
   policy = StateAwareRasterScanPolicy(
       width=1.0, height=1.0,
       step_size=0.03,  # Dense scanning for high coverage
       drift_strength=15.0
   )
   task = CustomOpenLoopNavigationTask(
       duration=200,
       action_policy=policy,
       width=1.0,
       height=1.0,
       start_pos=(0.05, 0.95)  # Start at top-left
   )
   ```


   .. py:method:: compute_action(step_idx, agent)

      Compute next action based on current agent position and scanning mode.

      Implements cyclic dual-mode scanning:
      - Horizontal mode: Left-right sweeps moving downward
      - Vertical mode: Up-down sweeps moving rightward
      - Auto-switches between modes to avoid walls and maintain coverage



   .. py:attribute:: current_direction
      :value: 1.0



   .. py:attribute:: drift_strength
      :value: 15.0



   .. py:attribute:: height


   .. py:attribute:: is_turning
      :value: False



   .. py:attribute:: margin
      :value: 0.05



   .. py:attribute:: mode
      :value: 'horizontal'



   .. py:attribute:: speed
      :value: 0.15



   .. py:attribute:: step_size
      :value: 0.03



   .. py:attribute:: turn_steps_remaining
      :value: 0



   .. py:attribute:: width


.. py:class:: TMazeOpenLoopNavigationTask(w=0.3, l_s=1.0, l_arm=0.75, t=0.3, start_pos=(0.0, 0.15), duration=20.0, dt=None, **kwargs)

   Bases: :py:obj:`OpenLoopNavigationTask`


   Open-loop navigation task in a T-maze environment.

   This subclass configures the environment with a T-maze boundary, which is useful
   for studying decision-making and spatial navigation in a controlled setting.

   Initialize T-maze open-loop navigation task.

   :param w: Width of the corridor (default: 0.3)
   :param l_s: Length of the stem (default: 1.0)
   :param l_arm: Length of each arm (default: 0.75)
   :param t: Thickness of the walls (default: 0.3)
   :param start_pos: Starting position of the agent (default: (0.0, 0.15))
   :param duration: Duration of the trajectory in seconds (default: 20.0)
   :param dt: Time step (default: None, uses bm.get_dt())
   :param \*\*kwargs: Additional keyword arguments passed to OpenLoopNavigationTask


.. py:class:: TMazeRecessOpenLoopNavigationTask(w=0.3, l_s=1.0, l_arm=0.75, t=0.3, recess_width=None, recess_depth=None, start_pos=(0.0, 0.15), duration=20.0, dt=None, **kwargs)

   Bases: :py:obj:`TMazeOpenLoopNavigationTask`


   Open-loop navigation task in a T-maze environment with recesses at stem-arm junctions.

   This variant adds small rectangular indentations at the T-junction, creating
   additional spatial features that may be useful for studying spatial navigation
   and decision-making.

   Initialize T-maze with recesses open-loop navigation task.

   :param w: Width of the corridor (default: 0.3)
   :param l_s: Length of the stem (default: 1.0)
   :param l_arm: Length of each arm (default: 0.75)
   :param t: Thickness of the walls (default: 0.3)
   :param recess_width: Width of recesses at stem-arm junctions (default: t/4)
   :param recess_depth: Depth of recesses extending downward (default: t/4)
   :param start_pos: Starting position of the agent (default: (0.0, 0.15))
   :param duration: Duration of the trajectory in seconds (default: 20.0)
   :param dt: Time step (default: None, uses bm.get_dt())
   :param \*\*kwargs: Additional keyword arguments passed to OpenLoopNavigationTask


.. py:function:: map2pi(a)

