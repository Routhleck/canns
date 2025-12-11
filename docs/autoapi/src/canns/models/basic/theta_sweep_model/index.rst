src.canns.models.basic.theta_sweep_model
========================================

.. py:module:: src.canns.models.basic.theta_sweep_model


Classes
-------

.. autoapisummary::

   src.canns.models.basic.theta_sweep_model.DirectionCellNetwork
   src.canns.models.basic.theta_sweep_model.GridCellNetwork
   src.canns.models.basic.theta_sweep_model.PlaceCellNetwork


Functions
---------

.. autoapisummary::

   src.canns.models.basic.theta_sweep_model.calculate_theta_modulation


Module Contents
---------------

.. py:class:: DirectionCellNetwork(num, tau = 10.0, tau_v = 100.0, noise_strength = 0.1, k = 0.2, adaptation_strength = 15.0, a = 0.7, A = 3.0, J0 = 1.0, g = 1.0, z_min = -bm.pi, z_max = bm.pi, conn_noise = 0.0)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   1D continuous-attractor direction (head direction) cell network.

   This network implements a ring attractor model for representing head direction
   with theta-modulated dynamics and spike-frequency adaptation (SFA). The model
   exhibits key properties of biological head direction cells including:
   - Persistent activity bumps encoding current heading
   - Theta phase precession relative to turning angle
   - Anticipative tracking through adaptation mechanisms

   The network dynamics include:
   - Membrane potential (u) with recurrent excitation and global inhibition
   - Adaptation variable (v) implementing slow negative feedback
   - Firing rate (r) computed via divisive normalization
   - External input modulated by theta oscillations

   :param num: Number of neurons in the network (resolution of head direction representation)
   :param tau: Membrane time constant (ms). Controls speed of neural dynamics.
   :param tau_v: Adaptation time constant (ms). Larger values = slower adaptation.
   :param noise_strength: Standard deviation of Gaussian noise added to inputs
   :param k: Global inhibition strength for divisive normalization
   :param adaptation_strength: Strength of adaptation coupling (dimensionless)
   :param a: Width of connectivity kernel (radians). Determines bump width.
   :param A: Amplitude of external input bump
   :param J0: Peak recurrent connection strength
   :param g: Gain parameter for firing rate transformation
   :param z_min: Minimum value of feature space (default: -π)
   :param z_max: Maximum value of feature space (default: π)
   :param conn_noise: Standard deviation of Gaussian noise added to connectivity matrix

   .. attribute:: num

      Number of neurons

      :type: int

   .. attribute:: x

      Preferred directions of neurons, uniformly distributed in [z_min, z_max)

      :type: Array

   .. attribute:: conn_mat

      Recurrent connectivity matrix with Gaussian profile

      :type: Array

   .. attribute:: r

      Firing rates of neurons

      :type: HiddenState

   .. attribute:: u

      Membrane potentials

      :type: HiddenState

   .. attribute:: v

      Adaptation variables

      :type: HiddenState

   .. attribute:: center

      Current bump center position

      :type: State

   .. attribute:: m

      Effective adaptation strength (adaptation_strength * tau / tau_v)

      :type: float

   .. rubric:: Example

   >>> import brainpy.math as bm
   >>> from canns.models.basic.theta_sweep_model import DirectionCellNetwork
   >>>
   >>> bm.set_dt(1.)  # 1ms time step
   >>> dc_net = DirectionCellNetwork(num=60)
   >>>
   >>> # Update with head direction and theta modulation
   >>> head_direction = 0.5  # radians
   >>> theta_modulation = 1.2  # theta phase-dependent gain
   >>> dc_net.update(head_direction, theta_modulation)

   .. rubric:: References

   Ji, Z., Lomi, E., Jeffery, K., Mitchell, A. S., & Burgess, N. (2025).
   Phase Precession Relative to Turning Angle in Theta‐Modulated Head Direction Cells.
   Hippocampus, 35(2), e70008.


   .. py:method:: calculate_dist(d)

      Calculate distance on circular feature space with periodic boundary.

      :param d: Raw angular difference

      :returns: Shortest angular distance considering periodicity



   .. py:method:: get_bump_center(r, x)
      :staticmethod:


      Decode bump center from population activity using circular mean.

      :param r: Firing rate vector
      :param x: Preferred direction vector

      :returns: Decoded center position in radians



   .. py:method:: handle_periodic_condition(A)
      :staticmethod:



   .. py:method:: input_bump(head_direction)

      Generate Gaussian-shaped external input centered on target direction.

      :param head_direction: Center of input bump in radians

      :returns: Input vector of shape (num,)



   .. py:method:: make_connection()

      Generate recurrent connectivity matrix with Gaussian profile.

      Creates a circulant connectivity matrix where connection strength
      decreases with distance according to a Gaussian kernel.

      :returns: Connectivity matrix
      :rtype: Array of shape (num, num)



   .. py:method:: update(head_direction, theta_input)

      Single time-step update of network dynamics.

      :param head_direction: Target head direction in radians [-π, π]
      :param theta_input: Theta modulation factor (typically 1.0 ± theta_strength)



   .. py:attribute:: A
      :value: 3.0



   .. py:attribute:: J0
      :value: 1.0



   .. py:attribute:: a
      :value: 0.7



   .. py:attribute:: adaptation_strength
      :value: 15.0



   .. py:attribute:: center


   .. py:attribute:: centerI


   .. py:attribute:: conn_mat

      Initialize network state variables.

      Creates and initializes:
      - r: Firing rates (all zeros)
      - u: Membrane potentials (all zeros)
      - v: Adaptation variables (all zeros)
      - center: Current bump center (zero)
      - centerI: Input bump center (zero)


   .. py:attribute:: conn_noise
      :value: 0.0



   .. py:property:: derivative


   .. py:attribute:: g
      :value: 1.0



   .. py:attribute:: integral


   .. py:attribute:: k
      :value: 0.2



   .. py:attribute:: m
      :value: 1.5



   .. py:attribute:: noise_strength
      :value: 0.1



   .. py:attribute:: num


   .. py:attribute:: r


   .. py:attribute:: tau
      :value: 10.0



   .. py:attribute:: tau_v
      :value: 100.0



   .. py:attribute:: u


   .. py:attribute:: v


   .. py:attribute:: x


   .. py:attribute:: z_max


   .. py:attribute:: z_min


   .. py:attribute:: z_range


.. py:class:: GridCellNetwork(num_dc = 100, num_gc_x = 100, tau = 10.0, tau_v = 100.0, noise_strength = 0.1, conn_noise = 0.0, k = 1.0, adaptation_strength = 15.0, a = 0.8, A = 3.0, J0 = 5.0, g = 1000.0, mapping_ratio = 1, phase_offset = 1.0 / 20)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   2D continuous-attractor grid cell network with hexagonal lattice structure.

   This network implements a twisted torus topology that generates grid cell-like
   spatial representations with hexagonal periodicity. The model combines:
   - 2D continuous attractor dynamics on a twisted manifold
   - Spike-frequency adaptation for theta modulation
   - Integration of direction cell inputs via conjunctive cells
   - Phase offset mechanism for theta sweeps

   The network operates in a transformed coordinate system where grid cells form
   a hexagonal lattice, enabling realistic grid field spacing and orientation.

   :param num_dc: Number of direction cells providing heading input
   :param num_gc_x: Number of grid cells along one dimension (total = num_gc_x^2)
   :param tau: Membrane time constant (ms)
   :param tau_v: Adaptation time constant (ms). Larger = slower adaptation.
   :param noise_strength: Standard deviation of activity noise
   :param conn_noise: Standard deviation of connectivity noise
   :param k: Global inhibition strength for divisive normalization
   :param adaptation_strength: Coupling strength between u and v
   :param a: Width of connectivity kernel. Determines bump width.
   :param A: Amplitude of external input
   :param J0: Peak recurrent connection strength
   :param g: Firing rate gain factor (scales to biological range)
   :param mapping_ratio: Controls grid spacing (larger = smaller spacing).
                         Grid spacing λ = 2π / mapping_ratio
   :param phase_offset: Phase shift for conjunctive input, drives theta sweeps.
                        Expressed as fraction of [-π, π] range (default: 1/20)

   .. attribute:: num

      Total number of grid cells (num_gc_x^2)

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

   .. attribute:: conn_mat

      Recurrent connectivity matrix

      :type: Array

   .. attribute:: candidate_centers

      Grid of candidate bump centers for decoding

      :type: Array

   .. attribute:: r

      Firing rates

      :type: HiddenState

   .. attribute:: u

      Membrane potentials

      :type: HiddenState

   .. attribute:: v

      Adaptation variables

      :type: HiddenState

   .. attribute:: center_phase

      Decoded bump center in phase space

      :type: State

   .. attribute:: center_position

      Decoded position in real space

      :type: State

   .. attribute:: gc_bump

      Grid cell bump activity pattern

      :type: State

   .. rubric:: Example

   >>> import brainpy.math as bm
   >>> from canns.models.basic.theta_sweep_model import GridCellNetwork
   >>>
   >>> bm.set_dt(1.0)
   >>> gc_net = GridCellNetwork(num_dc=60, num_gc_x=30, mapping_ratio=1.5)
   >>>
   >>> # Update with position, direction activity, and theta modulation
   >>> position = [0.5, 0.3]  # animal position
   >>> dir_activity = np.random.rand(60)  # direction cell firing
   >>> theta_mod = 1.2  # theta phase modulation
   >>> gc_net.update(position, dir_activity, theta_mod)

   .. rubric:: References

   Ji, Z., Chu, T., Wu, S., & Burgess, N. (2025).
   A systems model of alternating theta sweeps via firing rate adaptation.
   Current Biology, 35(4), 709-722.


   .. py:method:: calculate_dist(d)

      d: (..., 2) displacement in original (x,y).
      Return Euclidean distance after transform (hex/rect).



   .. py:method:: calculate_input_from_conjgc(animal_pos, direction_activity, theta_modulation)

      Calculate external input to grid cells from conjunctive grid cells.

      Conjunctive cells integrate position and direction to generate grid cell inputs
      with phase offsets. This drives theta sweeps when modulated by theta oscillations.

      :param animal_pos: Current position [x, y]
      :param direction_activity: Direction cell firing rates (shape: num_dc)
      :param theta_modulation: Theta phase-dependent gain factor

      :returns: Weighted conjunctive input to grid cells
      :rtype: Array of shape (num_gc,)



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

      Apply periodic boundary conditions to wrap phases into [-π, π].

      :param d: Phase values (any shape)

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



   .. py:method:: update(animal_posistion, direction_activity, theta_modulation)

      Single time-step update of grid cell network dynamics.

      Integrates conjunctive inputs from direction cells, applies theta modulation,
      and updates grid cell activity via continuous attractor dynamics with adaptation.

      :param animal_posistion: Current position [x, y] for disambiguating grid phase
      :param direction_activity: Direction cell firing rates (shape: num_dc)
      :param theta_modulation: Theta phase-dependent gain factor



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


   .. py:attribute:: center_phase


   .. py:attribute:: center_position


   .. py:attribute:: conj_input


   .. py:attribute:: conn_mat

      Initialize network state variables.

      Creates and initializes:
      - r: Firing rates (shape: num)
      - u: Membrane potentials (shape: num)
      - v: Adaptation variables (shape: num)
      - gc_bump: Grid cell bump pattern (shape: num)
      - conj_input: Conjunctive cell input (shape: num)
      - center_phase: Bump center in phase space (shape: 2)
      - center_position: Decoded position in real space (shape: 2)


   .. py:attribute:: conn_noise
      :value: 0.0



   .. py:attribute:: coor_transform


   .. py:attribute:: coor_transform_inv


   .. py:property:: derivative


   .. py:attribute:: g
      :value: 1000.0



   .. py:attribute:: gc_bump


   .. py:attribute:: integral


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



   .. py:attribute:: r


   .. py:attribute:: tau
      :value: 10.0



   .. py:attribute:: tau_v
      :value: 100.0



   .. py:attribute:: u


   .. py:attribute:: v


   .. py:attribute:: value_bump


   .. py:attribute:: value_grid


   .. py:attribute:: x_grid


   .. py:attribute:: y_grid


.. py:class:: PlaceCellNetwork(geodesic_result, tau = 10.0, tau_v = 100.0, noise_strength = 0.0, k = 0.2, m = 3.0, a = 0.2, A = 5.0, J0 = 1.0, g = 1.0, conn_noise = 0.0)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   Graph-based continuous-attractor place cell network using environment geodesic distances.

   This network implements a place cell representation where neurons are tuned to discrete
   locations in a navigation environment. Connectivity is based on geodesic (shortest path)
   distances within the environment, allowing the network to adapt to complex non-convex spaces
   with obstacles.

   Key features:
   - Connectivity matrix based on geodesic distances (not Euclidean)
   - Replaces NetworkX graph representation with grid-based geodesic computation
   - Uses GeodesicDistanceResult for environment definition and distance computation
   - Continuous attractor dynamics with spike-frequency adaptation
   - Supports arbitrary environment shapes (rectangular, T-maze, complex polygons with holes/walls)

   :param geodesic_result: Geodesic distance computation result from navigation task
   :param tau: Membrane time constant (ms). Controls speed of neural dynamics.
   :param tau_v: Adaptation time constant (ms). Larger values = slower adaptation.
   :param noise_strength: Standard deviation of Gaussian noise added to inputs
   :param k: Global inhibition strength for divisive normalization
   :param m: Strength of adaptation coupling (dimensionless)
   :param a: Width of connectivity kernel. Determines bump width in grid units.
   :param A: Amplitude of external input bump
   :param J0: Peak recurrent connection strength
   :param g: Gain parameter for firing rate transformation
   :param conn_noise: Standard deviation of Gaussian noise added to connectivity matrix

   .. attribute:: geodesic_result

      Geodesic distance computation result

      :type: GeodesicDistanceResult

   .. attribute:: cell_num

      Number of place cells (= number of accessible grid cells)

      :type: int

   .. attribute:: D

      Geodesic distance matrix of shape (cell_num, cell_num)

      :type: Array

   .. attribute:: accessible_indices

      Grid indices of accessible cells (cell_num, 2)

      :type: Array

   .. attribute:: cost_grid

      Grid cost information for position lookups

      :type: MovementCostGrid

   .. attribute:: conn_mat

      Recurrent connectivity matrix with Gaussian profile

      :type: Array

   .. attribute:: r

      Firing rates of place cells

      :type: HiddenState

   .. attribute:: u

      Membrane potentials

      :type: HiddenState

   .. attribute:: v

      Adaptation variables

      :type: HiddenState

   .. attribute:: center

      Current decoded bump center

      :type: State

   .. attribute:: m

      Effective adaptation strength (adaptation_strength * tau / tau_v)

      :type: float


   .. py:method:: get_bump_center(r, x)

      Decode bump center from population activity.

      Uses weighted average of cell indices, normalized by total activity.

      :param r: Firing rate vector (cell_num,)

      :returns: Decoded center index (scalar)



   .. py:method:: get_geodesic_index_by_pos(pos)

      Get the geodesic index of the grid cell containing the given position.

      :param pos: (x, y) coordinates of the position

      :returns: Index of the grid cell in the geodesic distance matrix, or None if
                the position is out of bounds or in an impassable cell.



   .. py:method:: input_bump(animal_pos)

      Generate Gaussian bump external input centered on the animal's current position.

      :param animal_pos: Current position (x, y) tuple or array

      :returns: Input vector of shape (cell_num,)



   .. py:method:: make_connection()

      Generate recurrent connectivity matrix with Gaussian profile based on geodesic distance.

      Connection strength between place cells decays with geodesic distance according
      to a normalized Gaussian kernel.

      :returns: Connectivity matrix
      :rtype: Array of shape (cell_num, cell_num)



   .. py:method:: update(animal_pos, theta_input)

      Single time-step update of network dynamics.

      :param animal_pos: Current position (x, y) tuple or array
      :param theta_input: Theta modulation factor (typically 1.0 ± theta_strength)



   .. py:attribute:: A
      :value: 5.0



   .. py:attribute:: D


   .. py:attribute:: J0
      :value: 1.0



   .. py:attribute:: a
      :value: 0.2



   .. py:attribute:: accessible_indices


   .. py:attribute:: cell_num


   .. py:attribute:: center


   .. py:attribute:: conn_mat

      Initialize network state variables.

      Creates and initializes:
      - r: Firing rates (all zeros)
      - u: Membrane potentials (all zeros)
      - v: Adaptation variables (all zeros)
      - center: Current bump center (zero)


   .. py:attribute:: conn_noise
      :value: 0.0



   .. py:attribute:: cost_grid


   .. py:property:: derivative


   .. py:attribute:: g
      :value: 1.0



   .. py:attribute:: geodesic_result


   .. py:attribute:: integral


   .. py:attribute:: k
      :value: 0.2



   .. py:attribute:: m
      :value: 3.0



   .. py:attribute:: noise_strength
      :value: 0.0



   .. py:attribute:: r


   .. py:attribute:: tau
      :value: 10.0



   .. py:attribute:: tau_v
      :value: 100.0



   .. py:attribute:: u


   .. py:attribute:: v


   .. py:attribute:: x


.. py:function:: calculate_theta_modulation(time_step, linear_gain, ang_gain, theta_strength_hd = 0.0, theta_strength_gc = 0.0, theta_cycle_len = 100.0, dt = None)

   Calculate theta oscillation phase and modulation factors for direction and grid cell networks.

   :param time_step: Current time step index
   :param linear_gain: Normalized linear speed gain [0,1]
   :param ang_gain: Normalized angular speed gain [-1,1]
   :param theta_strength_hd: Theta modulation strength for head direction cells
   :param theta_strength_gc: Theta modulation strength for grid cells
   :param theta_cycle_len: Length of theta cycle in time units
   :param dt: Time step size (if None, uses bm.get_dt())

   :returns:

             (theta_phase, theta_modulation_hd, theta_modulation_gc)
                 - theta_phase: Current theta phase [-π, π]
                 - theta_modulation_hd: Theta modulation for direction cells
                 - theta_modulation_gc: Theta modulation for grid cells
   :rtype: tuple


