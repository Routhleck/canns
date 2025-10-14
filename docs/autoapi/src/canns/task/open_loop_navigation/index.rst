src.canns.task.open_loop_navigation
===================================

.. py:module:: src.canns.task.open_loop_navigation


Classes
-------

.. autoapisummary::

   src.canns.task.open_loop_navigation.OpenLoopNavigationTask


Functions
---------

.. autoapisummary::

   src.canns.task.open_loop_navigation.map2pi


Module Contents
---------------

.. py:class:: OpenLoopNavigationTask(duration=20.0, start_pos=(2.5, 2.5), initial_head_direction=None, progress_bar=True, width=5, height=5, dimensionality='2D', boundary_conditions='solid', scale=None, dx=0.01, boundary=None, walls=None, holes=None, objects=None, dt=None, speed_mean=0.04, speed_std=0.016, speed_coherence_time=0.7, rotational_velocity_coherence_time=0.08, rotational_velocity_std=120 * np.pi / 180, head_direction_smoothing_timescale=0.15, thigmotaxis=0.5, wall_repel_distance=0.1, wall_repel_strength=1.0)

   Bases: :py:obj:`src.canns.task._base.Task`


   Open-loop spatial navigation task that synthesises trajectories without
   incorporating real-time feedback from a controller.

   Initializes the Task instance.

   :param data_class: A dataclass type for structured data.
                      If provided, the task will use this
                      class to structure the loaded or
                      generated data.
   :type data_class: type, optional


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



   .. py:method:: show_data(show=True, save_path=None)

      Displays the trajectory of the agent in the environment.



   .. py:method:: show_trajectory_analysis(show = True, save_path = None, figsize = (12, 3), smooth_window = 50, **kwargs)

      Display comprehensive trajectory analysis including position, speed, and direction changes.

      :param show: Whether to display the plot
      :param save_path: Path to save the figure
      :param figsize: Figure size (width, height)
      :param smooth_window: Window size for smoothing speed and direction plots (set to 0 to disable smoothing)
      :param \*\*kwargs: Additional matplotlib parameters



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



   .. py:attribute:: duration
      :value: 20.0



   .. py:attribute:: dx
      :value: 0.01



   .. py:attribute:: env


   .. py:attribute:: env_params


   .. py:attribute:: head_direction_smoothing_timescale
      :value: 0.15



   .. py:attribute:: height
      :value: 5



   .. py:attribute:: holes


   .. py:attribute:: initial_head_direction
      :value: None



   .. py:attribute:: objects


   .. py:attribute:: progress_bar
      :value: True



   .. py:attribute:: rotational_velocity_coherence_time
      :value: 0.08



   .. py:attribute:: rotational_velocity_std


   .. py:attribute:: run_steps


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


   .. py:attribute:: wall_repel_distance
      :value: 0.1



   .. py:attribute:: wall_repel_strength
      :value: 1.0



   .. py:attribute:: walls


   .. py:attribute:: width
      :value: 5



.. py:function:: map2pi(a)

