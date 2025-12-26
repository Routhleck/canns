src.canns.models.basic.grid_cell
================================

.. py:module:: src.canns.models.basic.grid_cell

.. autoapi-nested-parse::

   Simplified 2D grid cell network with hexagonal lattice structure.

   This module implements a standard grid cell model,
   suitable for basic spatial navigation and path integration tasks.



Classes
-------

.. autoapisummary::

   src.canns.models.basic.grid_cell.GridCell2D
   src.canns.models.basic.grid_cell.GridCell2D_SFA


Module Contents
---------------

.. py:class:: GridCell2D(length = 30, tau = 10.0, k = 1.0, a = 0.8, A = 3.0, J0 = 5.0, mapping_ratio = 1.5, noise_strength = 0.1, conn_noise = 0.0, g = 1.0)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   Simplified 2D continuous-attractor grid cell network with hexagonal lattice structure.

   This network implements a twisted torus topology that generates grid cell-like
   spatial representations with hexagonal periodicity.

   The network operates in a transformed coordinate system where grid cells form
   a hexagonal lattice, enabling realistic grid field spacing and orientation.

   :param length: Number of grid cells along one dimension (total = length^2). Default: 30
   :param tau: Membrane time constant (ms). Default: 10.0
   :param k: Global inhibition strength for divisive normalization. Default: 1.0
   :param a: Width of connectivity kernel. Determines bump width. Default: 0.8
   :param A: Amplitude of external input. Default: 3.0
   :param J0: Peak recurrent connection strength. Default: 5.0
   :param mapping_ratio: Controls grid spacing (larger = smaller spacing).
                         Grid spacing λ = 2π / mapping_ratio. Default: 1.5
   :param noise_strength: Standard deviation of activity noise. Default: 0.1
   :param conn_noise: Standard deviation of connectivity noise. Default: 0.0
   :param g: Firing rate gain factor (scales to biological range). Default: 1.0

   .. attribute:: num

      Total number of grid cells (length^2)

      :type: int

   .. attribute:: x_grid, y_grid

      Grid cell preferred phases in [-π, π]

      :type: Array

   .. attribute:: value_grid

      Neuron positions in phase space, shape (num, 2)

      :type: Array

   .. attribute:: Lambda

      Grid spacing in real space

      :type: float

   .. attribute:: coor_transform

      Hexagonal to rectangular coordinate transform

      :type: Array

   .. attribute:: coor_transform_inv

      Rectangular to hexagonal coordinate transform

      :type: Array

   .. attribute:: conn_mat

      Recurrent connectivity matrix

      :type: Array

   .. attribute:: candidate_centers

      Grid of candidate bump centers for decoding

      :type: Array

   .. attribute:: r

      Firing rates (shape: num)

      :type: Variable

   .. attribute:: u

      Membrane potentials (shape: num)

      :type: Variable

   .. attribute:: center_phase

      Decoded bump center in phase space (shape: 2)

      :type: Variable

   .. attribute:: center_position

      Decoded position in real space (shape: 2)

      :type: Variable

   .. attribute:: inp

      External input for tracking (shape: num)

      :type: Variable

   .. attribute:: gc_bump

      Grid cell bump activity pattern (shape: num)

      :type: Variable

   .. rubric:: Example

   >>> import brainpy.math as bm
   >>> from canns.models.basic import GridCell2D
   >>>
   >>> bm.set_dt(1.0)
   >>> model = GridCell2D(length=30, mapping_ratio=1.5)
   >>>
   >>> # Update with 2D position
   >>> position = [0.5, 0.3]
   >>> model.update(position)
   >>>
   >>> # Access decoded position
   >>> decoded_pos = model.center_position.value
   >>> print(f"Decoded position: {decoded_pos}")

   .. rubric:: References

   Burak, Y., & Fiete, I. R. (2009).
   Accurate path integration in continuous attractor network models of grid cells.
   PLoS Computational Biology, 5(2), e1000291.

   Initialize the simplified grid cell network.


   .. py:method:: calculate_dist(d)

      Calculate Euclidean distance after hexagonal coordinate transformation.

      Applies periodic boundary conditions and transforms displacement vectors
      from phase space to hexagonal lattice coordinates.

      :param d: Displacement vectors in phase space, shape (..., 2)

      :returns: Euclidean distances in hexagonal space
      :rtype: Array of shape (...,)



   .. py:method:: get_stimulus_by_pos(position)

      Generate Gaussian stimulus centered at given position.

      Useful for previewing input patterns without calling update.

      :param position: 2D position [x, y] in real space

      :returns: External input pattern
      :rtype: Array of shape (num,)



   .. py:method:: get_unique_activity_bump(network_activity, animal_position)

      Decode unique bump location from network activity and animal position.

      Estimates the activity bump center in phase space using population vector
      decoding, then maps it to real space and snaps to the nearest candidate
      center to resolve periodic ambiguity.

      :param network_activity: Grid cell firing rates, shape (num,)
      :param animal_position: Current animal position for disambiguation, shape (2,)

      :returns: Phase coordinates of bump center, shape (2,)
                center_position: Real-space position of bump (nearest candidate), shape (2,)
                bump: Gaussian bump template centered at center_position, shape (num,)
      :rtype: center_phase



   .. py:method:: handle_periodic_condition(d)

      Apply periodic boundary conditions to wrap phases into [-π, π].

      :param d: Phase values (any shape with last dimension = 2)

      :returns: Wrapped phase values in [-π, π]



   .. py:method:: make_candidate_centers(Lambda)

      Generate grid of candidate bump centers for decoding.

      Creates a regular lattice of potential activity bump locations
      used for disambiguating position from grid cell phases.

      :param Lambda: Grid spacing in real space

      :returns: Candidate centers in transformed coordinates
      :rtype: Array of shape (N_c*N_c, 2)



   .. py:method:: make_connection()

      Generate recurrent connectivity matrix with 2D Gaussian kernel.

      Uses hexagonal lattice geometry via coordinate transformation.
      Connection strength decays with distance in transformed space.

      :returns: Recurrent connectivity matrix
      :rtype: Array of shape (num, num)



   .. py:method:: position2phase(position)

      Convert real-space position to grid cell phase coordinates.

      Applies coordinate transformation and wraps to periodic boundaries.
      Each grid cell's preferred phase is determined by the animal's position
      on the hexagonal lattice.

      :param position: Real-space coordinates, shape (2,) or (2, N)

      :returns: Phase coordinates in [-π, π] per axis
      :rtype: Array of shape (2,) or (2, N)



   .. py:method:: update(position)

      Single time-step update of grid cell network dynamics.

      Updates network activity using continuous attractor dynamics with
      direct position-based external input. No adaptation or theta modulation.

      :param position: Current 2D position [x, y] in real space, shape (2,)



   .. py:attribute:: A
      :value: 3.0



   .. py:attribute:: J0
      :value: 5.0



   .. py:attribute:: Lambda


   .. py:attribute:: a
      :value: 0.8



   .. py:attribute:: candidate_centers


   .. py:attribute:: center_phase


   .. py:attribute:: center_position


   .. py:attribute:: conn_mat


   .. py:attribute:: conn_noise
      :value: 0.0



   .. py:attribute:: coor_transform


   .. py:attribute:: coor_transform_inv


   .. py:attribute:: g
      :value: 1.0



   .. py:attribute:: gc_bump


   .. py:attribute:: inp


   .. py:attribute:: k
      :value: 1.0



   .. py:attribute:: length
      :value: 30



   .. py:attribute:: mapping_ratio
      :value: 1.5



   .. py:attribute:: noise_strength
      :value: 0.1



   .. py:attribute:: num
      :value: 900



   .. py:attribute:: r


   .. py:attribute:: tau
      :value: 10.0



   .. py:attribute:: u


   .. py:attribute:: value_bump


   .. py:attribute:: value_grid


   .. py:attribute:: x_grid


   .. py:attribute:: y_grid


.. py:class:: GridCell2D_SFA(length = 30, tau = 10.0, k = 1.0, a = 0.8, A = 3.0, J0 = 5.0, mapping_ratio = 1.5, noise_strength = 0.1, conn_noise = 0.0, g = 1.0, tau_v = 50.0, m = 0.3)

   Bases: :py:obj:`GridCell2D`


   GridCell2D with Spike-Frequency Adaptation (SFA).

   Extends GridCell2D with slow negative feedback adaptation for
   anticipative tracking behavior.

   :param All GridCell2D parameters:
   :param plus:
   :param tau_v: Adaptation time constant (much slower than tau). Default: 50.0
   :param m: Coupling strength from membrane potential to adaptation. Default: 0.3

   Additional Attributes:
       v (Variable): Adaptation variable (shape: num)

   .. rubric:: Example

   >>> import brainpy.math as bm
   >>> from canns.models.basic import GridCell2D_SFA
   >>>
   >>> bm.set_dt(1.0)
   >>> model = GridCell2D_SFA(length=30, mapping_ratio=1.5)
   >>>
   >>> # Same interface as GridCell2D
   >>> position = [0.5, 0.3]
   >>> model.update(position)

   .. rubric:: References

   Adaptation mechanism enables anticipative tracking in moving
   input scenarios.

   Initialize GridCell2D with SFA adaptation.


   .. py:method:: update(position)

      Single time-step update with SFA adaptation.

      Same as GridCell2D.update() but with slow negative feedback
      from adaptation variable v.

      :param position: Current 2D position [x, y] in real space, shape (2,)



   .. py:attribute:: m
      :value: 0.3



   .. py:attribute:: tau_v
      :value: 50.0



   .. py:attribute:: v


