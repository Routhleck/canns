src.canns.models.basic
======================

.. py:module:: src.canns.models.basic


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/models/basic/cann/index
   /autoapi/src/canns/models/basic/grid_cell/index
   /autoapi/src/canns/models/basic/hierarchical_model/index
   /autoapi/src/canns/models/basic/theta_sweep_model/index


Classes
-------

.. autoapisummary::

   src.canns.models.basic.CANN1D
   src.canns.models.basic.CANN1D_SFA
   src.canns.models.basic.CANN2D
   src.canns.models.basic.CANN2D_SFA
   src.canns.models.basic.GridCell2D
   src.canns.models.basic.GridCell2D_SFA
   src.canns.models.basic.HierarchicalNetwork


Package Contents
----------------

.. py:class:: CANN1D(*args, **kwargs)

   Bases: :py:obj:`BaseCANN1D`


   A standard 1D Continuous Attractor Neural Network (CANN) model.
   This model implements the core dynamics where a localized "bump" of activity
   can be sustained and moved by external inputs.

   Reference:
       Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
       Neural computation, 20(4), 994-1025.

   Initializes the 1D CANN model.

   :param (Parameters are inherited from BaseCANN1D):


   .. py:method:: update(inp)

      The main update function, defining the dynamics of the network for one time step.

      :param inp: The external input for the current time step.
      :type inp: Array



   .. py:attribute:: inp


   .. py:attribute:: r


   .. py:attribute:: u


.. py:class:: CANN1D_SFA(num, tau = 1.0, tau_v = 50.0, k = 8.1, a = 0.3, A = 0.2, J0 = 1.0, z_min = -bm.pi, z_max = bm.pi, m = 0.3, **kwargs)

   Bases: :py:obj:`BaseCANN1D`


   A 1D CANN model that incorporates Spike-Frequency Adaptation (SFA).
   SFA is a slow negative feedback mechanism that causes neurons to fire less
   over time for a sustained input, which can induce anticipative tracking behavior.

   Reference:
       Mi, Y., Fung, C. C., Wong, K. Y., & Wu, S. (2014). Spike frequency adaptation
       implements anticipative tracking in continuous attractor neural networks.
       Advances in neural information processing systems, 27.

   Initializes the 1D CANN model with SFA.

   :param tau_v: The time constant for the adaptation variable 'v'. A larger value means slower adaptation.
   :type tau_v: float
   :param m: The strength of the adaptation, coupling the membrane potential 'u' to the adaptation variable 'v'.
   :type m: float
   :param (Other parameters are inherited from BaseCANN1D):


   .. py:method:: update(inp)

      The main update function for the SFA model. It includes dynamics for both
      the membrane potential and the adaptation variable.

      :param inp: The external input for the current time step.
      :type inp: Array



   .. py:attribute:: inp


   .. py:attribute:: m
      :value: 0.3



   .. py:attribute:: r


   .. py:attribute:: tau_v
      :value: 50.0



   .. py:attribute:: u


   .. py:attribute:: v


.. py:class:: CANN2D(*args, **kwargs)

   Bases: :py:obj:`BaseCANN2D`


   A 2D Continuous Attractor Neural Network (CANN) model.
   This model extends the base CANN2D class to include specific dynamics
   and properties for a 2D neural network.

   Reference:
       Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
       Neural computation, 20(4), 994-1025.

   Initializes the 2D CANN model.

   :param (Parameters are inherited from BaseCANN2D):


   .. py:method:: update(inp)

      The main update function, defining the dynamics of the network for one time step.

      :param inp: The external input to the network, which can be a stimulus or other driving force.
      :type inp: Array



   .. py:attribute:: inp


   .. py:attribute:: r


   .. py:attribute:: u


.. py:class:: CANN2D_SFA(length, tau = 1.0, tau_v = 50.0, k = 8.1, a = 0.3, A = 0.2, J0 = 1.0, z_min = -bm.pi, z_max = bm.pi, m = 0.3, **kwargs)

   Bases: :py:obj:`BaseCANN2D`


   A 2D Continuous Attractor Neural Network (CANN) model with a specific
   implementation of the Synaptic Firing Activity (SFA) dynamics.
   This model extends the base CANN2D class to include SFA-specific dynamics.

   Initializes the 2D CANN model with SFA dynamics.


   .. py:method:: update(inp)

      The main update function for the SFA model. It includes dynamics for both
      the membrane potential and the adaptation variable.

      :param inp: The external input for the current time step.
      :type inp: Array



   .. py:attribute:: inp


   .. py:attribute:: m
      :value: 0.3



   .. py:attribute:: r


   .. py:attribute:: tau_v
      :value: 50.0



   .. py:attribute:: u


   .. py:attribute:: v


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


.. py:class:: HierarchicalNetwork(num_module, num_place, spacing_min=2.0, spacing_max=5.0, module_angle=0.0, band_size=180, band_noise=0.0, band_w_L2S=0.2, band_w_S2L=1.0, band_gain=0.2, grid_num=20, grid_tau=0.1, grid_tau_v=10.0, grid_k=0.005, grid_a=bm.pi / 9, grid_A=1.0, grid_J0=1.0, grid_mbar=1.0, gauss_tau=1.0, gauss_J0=1.1, gauss_k=0.0005, gauss_a=2 / 9 * bm.pi, nonrec_tau=0.1)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModelGroup`


   A full hierarchical network composed of multiple grid modules.

   This class creates and manages a collection of `HierarchicalPathIntegrationModel`
   modules, each with a different grid spacing. By combining the outputs of these
   modules, the network can represent position unambiguously over a large area.
   The final output is a population of place cells whose activities are used to
   decode the animal's estimated position.

   .. attribute:: num_module

      The number of grid modules in the network.

      :type: int

   .. attribute:: num_place

      The number of place cells in the output layer.

      :type: int

   .. attribute:: place_center

      The center locations of the place cells.

      :type: bm.math.ndarray

   .. attribute:: MEC_model_list

      A list containing all the `HierarchicalPathIntegrationModel` instances.

      :type: list

   .. attribute:: grid_fr

      The firing rates of the grid cell population.

      :type: bm.Variable

   .. attribute:: band_x_fr

      The firing rates of the x-oriented band cell population.

      :type: bm.Variable

   .. attribute:: band_y_fr

      The firing rates of the y-oriented band cell population.

      :type: bm.Variable

   .. attribute:: place_fr

      The firing rates of the place cell population.

      :type: bm.Variable

   .. attribute:: decoded_pos

      The final decoded 2D position.

      :type: bm.Variable

   .. rubric:: References

   Anonymous Author(s) "Unfolding the Black Box of Recurrent Neural Networks for Path Integration" (under review).

   Initializes the HierarchicalNetwork.

   :param num_module: The number of grid modules to create.
   :type num_module: int
   :param num_place: The number of place cells along one dimension of a square grid.
   :type num_place: int
   :param spacing_min: Minimum spacing for grid modules. Defaults to 2.0.
   :type spacing_min: float, optional
   :param spacing_max: Maximum spacing for grid modules. Defaults to 5.0.
   :type spacing_max: float, optional
   :param module_angle: Base orientation angle for all modules. Defaults to 0.0.
   :type module_angle: float, optional
   :param band_size: Number of neurons in each BandCell group. Defaults to 180.
   :type band_size: int, optional
   :param band_noise: Noise level for BandCells. Defaults to 0.0.
   :type band_noise: float, optional
   :param band_w_L2S: Weight from band cells to shifter units. Defaults to 0.2.
   :type band_w_L2S: float, optional
   :param band_w_S2L: Weight from shifter units to band cells. Defaults to 1.0.
   :type band_w_S2L: float, optional
   :param band_gain: Gain factor for velocity signal in BandCells. Defaults to 0.2.
   :type band_gain: float, optional
   :param grid_num: Number of neurons per dimension for GridCell. Defaults to 20.
   :type grid_num: int, optional
   :param grid_tau: Synaptic time constant for GridCell. Defaults to 0.1.
   :type grid_tau: float, optional
   :param grid_tau_v: Adaptation time constant for GridCell. Defaults to 10.0.
   :type grid_tau_v: float, optional
   :param grid_k: Global inhibition strength for GridCell. Defaults to 5e-3.
   :type grid_k: float, optional
   :param grid_a: Connection width for GridCell. Defaults to pi/9.
   :type grid_a: float, optional
   :param grid_A: External input magnitude for GridCell. Defaults to 1.0.
   :type grid_A: float, optional
   :param grid_J0: Maximum connection strength for GridCell. Defaults to 1.0.
   :type grid_J0: float, optional
   :param grid_mbar: Base adaptation strength for GridCell. Defaults to 1.0.
   :type grid_mbar: float, optional
   :param gauss_tau: Time constant for GaussRecUnits in BandCells. Defaults to 1.0.
   :type gauss_tau: float, optional
   :param gauss_J0: Connection strength scaling for GaussRecUnits. Defaults to 1.1.
   :type gauss_J0: float, optional
   :param gauss_k: Global inhibition for GaussRecUnits. Defaults to 5e-4.
   :type gauss_k: float, optional
   :param gauss_a: Connection width for GaussRecUnits. Defaults to 2/9*pi.
   :type gauss_a: float, optional
   :param nonrec_tau: Time constant for NonRecUnits in BandCells. Defaults to 0.1.
   :type nonrec_tau: float, optional


   .. py:method:: update(velocity, loc, loc_input_stre=0.0)

      Step function of a network.

      In this update function, the update functions in children systems are
      iteratively called.



   .. py:attribute:: MEC_model_list
      :value: []



   .. py:attribute:: band_x_fr


   .. py:attribute:: band_y_fr


   .. py:attribute:: decoded_pos


   .. py:attribute:: grid_fr


   .. py:attribute:: num_module


   .. py:attribute:: num_place


   .. py:attribute:: place_center


   .. py:attribute:: place_fr


