src.canns.models.basic.theta_sweep_model
========================================

.. py:module:: src.canns.models.basic.theta_sweep_model


Classes
-------

.. autoapisummary::

   src.canns.models.basic.theta_sweep_model.DirectionCellNetwork
   src.canns.models.basic.theta_sweep_model.GridCellNetwork


Functions
---------

.. autoapisummary::

   src.canns.models.basic.theta_sweep_model.calculate_theta_modulation


Module Contents
---------------

.. py:class:: DirectionCellNetwork(num, tau = 10.0, tau_v = 100.0, noise_strength = 0.1, k = 0.2, adaptation_strength = 15.0, a = 0.7, A = 3.0, J0 = 1.0, g = 1.0, z_min = -u.math.pi, z_max = u.math.pi, conn_noise = 0.0)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   1D continuous-attractor direction cell network
   .. rubric:: References

   Ji, Z., Lomi, E., Jeffery, K., Mitchell, A. S., & Burgess, N. (2025). Phase Precession Relative to Turning Angle in Theta‐Modulated Head Direction Cells. Hippocampus, 35(2), e70008.


   .. py:method:: calculate_dist(d)


   .. py:method:: get_bump_center(r, x)
      :staticmethod:



   .. py:method:: handle_periodic_condition(A)
      :staticmethod:



   .. py:method:: init_state(*args, **kwargs)

      State initialization function.



   .. py:method:: input_bump(head_direction)


   .. py:method:: make_connection()


   .. py:method:: update(head_direction, theta_input)

      The function to specify the updating rule.



   .. py:attribute:: A
      :value: 3.0



   .. py:attribute:: J0
      :value: 1.0



   .. py:attribute:: a
      :value: 0.7



   .. py:attribute:: adaptation_strength
      :value: 15.0



   .. py:attribute:: conn_mat


   .. py:attribute:: conn_noise
      :value: 0.0



   .. py:attribute:: g
      :value: 1.0



   .. py:attribute:: k
      :value: 0.2



   .. py:attribute:: m
      :value: 1.5



   .. py:attribute:: noise_strength
      :value: 0.1



   .. py:attribute:: num


   .. py:attribute:: tau
      :value: 10.0



   .. py:attribute:: tau_v
      :value: 100.0



   .. py:attribute:: x


   .. py:attribute:: z_max


   .. py:attribute:: z_min


   .. py:attribute:: z_range


.. py:class:: GridCellNetwork(num_dc = 100, num_gc_x = 100, tau = 10.0, tau_v = 100.0, noise_strength = 0.1, conn_noise = 0.0, k = 1.0, adaptation_strength = 15.0, a = 0.8, A = 3.0, J0 = 5.0, g = 1000.0, mapping_ratio = 1, phase_offset = 1.0 / 20)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   2D continuous-attractor grid cell network
   .. rubric:: References

   Ji, Z., Chu, T., Wu, S., & Burgess, N. (2025). A systems model of alternating theta sweeps via firing rate adaptation. Current Biology, 35(4), 709-722.


   .. py:method:: calculate_dist(d)

      d: (..., 2) displacement in original (x,y).
      Return Euclidean distance after transform (hex/rect).



   .. py:method:: calculate_input_from_conjgc(animal_pos, direction_activity, theta_modulation)


   .. py:method:: get_unique_activity_bump(network_activity, animal_posistion)

      Estimate a unique bump (activity peak) from the current network state,
      given the animal's actual position.

      :returns:

                (2,) array
                    Phase coordinates of bump center on the manifold.
                center_position : (2,) array
                    Real-space position of the bump (nearest candidate).
                bump : (N,) array
                    Gaussian bump template centered at center_position.
      :rtype: center_phase



   .. py:method:: handle_periodic_condition(d)


   .. py:method:: init_state(*args, **kwargs)

      State initialization function.



   .. py:method:: make_candidate_centers(Lambda)


   .. py:method:: make_connection()


   .. py:method:: position2phase(position)

      map position->phase; phase is wrapped to [-pi, pi] per-axis



   .. py:method:: update(animal_posistion, direction_activity, theta_modulation)

      The function to specify the updating rule.



   .. py:attribute:: A
      :value: 3.0



   .. py:attribute:: J0
      :value: 5.0



   .. py:attribute:: Lambda


   .. py:attribute:: a
      :value: 0.8



   .. py:attribute:: adaptation_strength
      :value: 15.0



   .. py:attribute:: candidate_centers


   .. py:attribute:: conn_mat


   .. py:attribute:: conn_noise
      :value: 0.0



   .. py:attribute:: coor_transform


   .. py:attribute:: coor_transform_inv


   .. py:attribute:: g
      :value: 1000.0



   .. py:attribute:: k
      :value: 1.0



   .. py:attribute:: m
      :value: 1.5



   .. py:attribute:: mapping_ratio
      :value: 1



   .. py:attribute:: noise_strength
      :value: 0.1



   .. py:attribute:: num
      :value: 10000



   .. py:attribute:: num_dc
      :value: 100



   .. py:attribute:: num_gc_1side
      :value: 100



   .. py:attribute:: phase_offset
      :value: 0.05



   .. py:attribute:: tau
      :value: 10.0



   .. py:attribute:: tau_v
      :value: 100.0



   .. py:attribute:: value_bump


   .. py:attribute:: value_grid


   .. py:attribute:: x_grid


   .. py:attribute:: y_grid


.. py:function:: calculate_theta_modulation(time_step, linear_gain, ang_gain, theta_strength_hd = 0.0, theta_strength_gc = 0.0, theta_cycle_len = 100.0, dt = None)

   Calculate theta oscillation phase and modulation factors for direction and grid cell networks.

   :param time_step: Current time step index
   :param linear_gain: Normalized linear speed gain [0,1]
   :param ang_gain: Normalized angular speed gain [-1,1]
   :param theta_strength_hd: Theta modulation strength for head direction cells
   :param theta_strength_gc: Theta modulation strength for grid cells
   :param theta_cycle_len: Length of theta cycle in time units
   :param dt: Time step size (if None, uses brainstate.environ.get_dt())

   :returns:

             (theta_phase, theta_modulation_hd, theta_modulation_gc)
                 - theta_phase: Current theta phase [-π, π]
                 - theta_modulation_hd: Theta modulation for direction cells
                 - theta_modulation_gc: Theta modulation for grid cells
   :rtype: tuple


