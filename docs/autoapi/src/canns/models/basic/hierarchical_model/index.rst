src.canns.models.basic.hierarchical_model
=========================================

.. py:module:: src.canns.models.basic.hierarchical_model


Classes
-------

.. autoapisummary::

   src.canns.models.basic.hierarchical_model.BandCell
   src.canns.models.basic.hierarchical_model.GaussRecUnits
   src.canns.models.basic.hierarchical_model.GridCell
   src.canns.models.basic.hierarchical_model.HierarchicalNetwork
   src.canns.models.basic.hierarchical_model.HierarchicalPathIntegrationModel
   src.canns.models.basic.hierarchical_model.NonRecUnits


Module Contents
---------------

.. py:class:: BandCell(angle, spacing, size=180, z_min=-bm.pi, z_max=bm.pi, noise=2.0, w_L2S=0.2, w_S2L=1.0, gain=0.2, gauss_tau=1.0, gauss_J0=1.1, gauss_k=0.0005, gauss_a=2 / 9 * bm.pi, nonrec_tau=0.1, **kwargs)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   A model of a band cell module for path integration.

   This model represents a set of neurons whose receptive fields form parallel bands
   across a 2D space. It is composed of a central `GaussRecUnits` attractor network
   (the band cells proper) that represents a 1D phase, and two `NonRecUnits`
   populations (left and right) that help shift the activity in the attractor
   network based on velocity input. This mechanism allows the module to integrate
   the component of velocity along its preferred direction.

   .. attribute:: size

      The number of neurons in each sub-population.

      :type: int

   .. attribute:: spacing

      The spacing between the bands in the 2D environment.

      :type: float

   .. attribute:: angle

      The orientation angle of the bands.

      :type: float

   .. attribute:: proj_k

      The projection vector for converting 2D position/velocity to 1D phase.

      :type: bm.math.ndarray

   .. attribute:: band_cells

      The core recurrent network representing the phase.

      :type: GaussRecUnits

   .. attribute:: left

      A population of non-recurrent units for positive shifts.

      :type: NonRecUnits

   .. attribute:: right

      A population of non-recurrent units for negative shifts.

      :type: NonRecUnits

   .. attribute:: w_L2S

      Connection weight from band cells to left/right units.

      :type: float

   .. attribute:: w_S2L

      Connection weight from left/right units to band cells.

      :type: float

   .. attribute:: gain

      A gain factor for velocity-modulated input.

      :type: float

   .. attribute:: center_ideal

      The ideal, noise-free center based on velocity integration.

      :type: bm.Variable

   .. attribute:: center

      The actual decoded center of the band cell activity bump.

      :type: bm.Variable

   Initializes the BandCell model.

   :param angle: The orientation angle of the bands.
   :type angle: float
   :param spacing: The spacing between the bands.
   :type spacing: float
   :param size: The number of neurons in each group. Defaults to 180.
   :type size: int, optional
   :param z_min: The minimum value of the feature space (phase). Defaults to -pi.
   :type z_min: float, optional
   :param z_max: The maximum value of the feature space (phase). Defaults to pi.
   :type z_max: float, optional
   :param noise: The noise level for the neuron groups. Defaults to 2.0.
   :type noise: float, optional
   :param w_L2S: Weight from band cells to shifter units. Defaults to 0.2.
   :type w_L2S: float, optional
   :param w_S2L: Weight from shifter units to band cells. Defaults to 1.0.
   :type w_S2L: float, optional
   :param gain: A gain factor for the velocity signal. Defaults to 0.2.
   :type gain: float, optional
   :param gauss_tau: Time constant for GaussRecUnits. Defaults to 1.0.
   :type gauss_tau: float, optional
   :param gauss_J0: Connection strength scaling factor for GaussRecUnits. Defaults to 1.1.
   :type gauss_J0: float, optional
   :param gauss_k: Global inhibition strength for GaussRecUnits. Defaults to 5e-4.
   :type gauss_k: float, optional
   :param gauss_a: Gaussian connection width for GaussRecUnits. Defaults to 2/9*pi.
   :type gauss_a: float, optional
   :param nonrec_tau: Time constant for NonRecUnits. Defaults to 0.1.
   :type nonrec_tau: float, optional
   :param \*\*kwargs: Additional keyword arguments for the base class.


   .. py:method:: Postophase(pos)

      Projects a 2D position to a 1D phase.

      This function converts a 2D coordinate in the environment into a 1D phase
      value based on the band cell's preferred angle and spacing.

      :param pos: The 2D position vector.
      :type pos: Array

      :returns: The corresponding 1D phase.
      :rtype: float



   .. py:method:: dist(d)

      Calculates the periodic distance in the feature space.

      :param d: The array of distances.
      :type d: Array

      :returns: The wrapped distances.
      :rtype: Array



   .. py:method:: get_center()

      Decodes and updates the current center of the band cell activity.



   .. py:method:: get_stimulus_by_pos(pos)

      Generates a stimulus input based on a 2D position.

      This creates a Gaussian bump of input centered on the phase corresponding
      to the given position, which can be used to anchor the network's activity.

      :param pos: The 2D position vector.
      :type pos: Array

      :returns: The stimulus input vector for the band cells.
      :rtype: Array



   .. py:method:: make_conn(shift)

      Creates a shifted Gaussian connection profile.

      This is used to create the connections from the left/right shifter units
      to the band cells, which implements the bump-shifting mechanism.

      :param shift: The amount to shift the connection profile by.
      :type shift: float

      :returns: The shifted connection matrix.
      :rtype: Array



   .. py:method:: move_heading(shift)

      Manually shifts the activity bump in the band cells.

      This is a utility function for testing purposes.

      :param shift: The number of neurons to roll the activity by.
      :type shift: int



   .. py:method:: reset()

      Resets the synaptic inputs of the left and right shifter units.



   .. py:method:: synapses()

      Defines the synaptic connections between the neuron groups.

      This method sets up the shifted connections from the left/right shifter
      populations to the central band cell attractor network, as well as the
      one-to-one connections from the band cells to the shifters.



   .. py:method:: update(velocity, loc, loc_input_stre)

      Updates the BandCell module for one time step.

      It integrates the component of `velocity` along the module's preferred
      direction to update the phase representation. The activity bump is shifted
      by modulating the inputs from the left/right shifter populations. It can
      also incorporate a direct location-based input.

      :param velocity: The 2D velocity vector.
      :type velocity: Array
      :param loc: The current 2D location.
      :type loc: Array
      :param loc_input_stre: The strength of the location-based input.
      :type loc_input_stre: float



   .. py:attribute:: angle


   .. py:attribute:: band_cells


   .. py:attribute:: center


   .. py:attribute:: center_ideal


   .. py:attribute:: dx


   .. py:attribute:: gain
      :value: 0.2



   .. py:attribute:: left


   .. py:attribute:: phase_shift


   .. py:attribute:: proj_k


   .. py:attribute:: rho


   .. py:attribute:: right


   .. py:attribute:: size
      :value: 180



   .. py:attribute:: spacing


   .. py:attribute:: w_L2S
      :value: 0.2



   .. py:attribute:: w_S2L
      :value: 1.0



   .. py:attribute:: x


   .. py:attribute:: z_max


   .. py:attribute:: z_min


   .. py:attribute:: z_range


.. py:class:: GaussRecUnits(size, tau = 1.0, J0 = 1.1, k = 0.0005, a = 2 / 9 * bm.pi, z_min = -bm.pi, z_max = bm.pi, noise = 2.0)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   A model of recurrently connected units with Gaussian connectivity.

   This class implements a 1D continuous attractor neural network (CANN). The network
   maintains a stable "bump" of activity that can represent a continuous variable,
   such as heading direction. The connectivity between neurons is Gaussian, and the
   network dynamics include divisive normalization.

   .. attribute:: size

      The number of neurons in the network.

      :type: int

   .. attribute:: tau

      The time constant for the synaptic input `u`.

      :type: float

   .. attribute:: k

      The inhibition strength for divisive normalization.

      :type: float

   .. attribute:: a

      The width of the Gaussian connection profile.

      :type: float

   .. attribute:: noise_0

      The standard deviation of the Gaussian noise added to the system.

      :type: float

   .. attribute:: z_min

      The minimum value of the encoded feature space.

      :type: float

   .. attribute:: z_max

      The maximum value of the encoded feature space.

      :type: float

   .. attribute:: z_range

      The range of the feature space (z_max - z_min).

      :type: float

   .. attribute:: x

      The preferred feature values for each neuron.

      :type: bm.math.ndarray

   .. attribute:: rho

      The neural density (number of neurons per unit of feature space).

      :type: float

   .. attribute:: dx

      The stimulus density (feature space range per neuron).

      :type: float

   .. attribute:: J

      The final connection strength, scaled by J0.

      :type: float

   .. attribute:: conn_mat

      The connection matrix.

      :type: bm.math.ndarray

   .. attribute:: r

      The firing rates of the neurons.

      :type: bm.Variable

   .. attribute:: u

      The synaptic inputs to the neurons.

      :type: bm.Variable

   .. attribute:: center

      The decoded center of the activity bump.

      :type: bm.Variable

   .. attribute:: input

      The external input to the network.

      :type: bm.Variable

   Initializes the GaussRecUnits model.

   :param size: The number of neurons in the network.
   :type size: int
   :param tau: The time constant of the neurons. Defaults to 1.0.
   :type tau: float, optional
   :param J0: A scaling factor for the critical connection strength. Defaults to 1.1.
   :type J0: float, optional
   :param k: The strength of the global inhibition. Defaults to 5e-4.
   :type k: float, optional
   :param a: The width of the Gaussian connection profile. Defaults to 2/9*pi.
   :type a: float, optional
   :param z_min: The minimum value of the feature space. Defaults to -pi.
   :type z_min: float, optional
   :param z_max: The maximum value of the feature space. Defaults to pi.
   :type z_max: float, optional
   :param noise: The level of noise in the system. Defaults to 2.0.
   :type noise: float, optional


   .. py:method:: Jc()

      Calculates the critical connection strength.

      This is the minimum connection strength required to sustain a stable
      activity bump in the attractor network.



   .. py:method:: decode(r, axis=0)

      Decodes the center of the activity bump.

      This method uses a population vector average to compute the center of the
      neural activity bump from the firing rates.

      :param r: The firing rates of the neurons.
      :type r: Array
      :param axis: The axis along which to perform the decoding. Defaults to 0.
      :type axis: int, optional

      :returns: The angle representing the decoded center of the bump.
      :rtype: float



   .. py:method:: dist(d)

      Calculates the periodic distance in the feature space.

      This function wraps distances to ensure they fall within the periodic
      boundaries of the feature space, i.e., [-z_range/2, z_range/2].

      :param d: The array of distances.
      :type d: bm.math.ndarray



   .. py:method:: make_conn()

      Constructs the periodic Gaussian connection matrix.

      The connection strength between two neurons depends on the periodic distance
      between their preferred feature values, following a Gaussian profile.



   .. py:method:: update(input)

      The function to specify the updating rule.




   .. py:attribute:: J


   .. py:attribute:: a


   .. py:attribute:: center


   .. py:attribute:: conn_mat


   .. py:attribute:: dx


   .. py:attribute:: input


   .. py:attribute:: k
      :value: 0.0005



   .. py:attribute:: noise_0
      :value: 2.0



   .. py:attribute:: r


   .. py:attribute:: rho


   .. py:attribute:: size


   .. py:attribute:: tau
      :value: 1.0



   .. py:attribute:: u


   .. py:attribute:: x


   .. py:attribute:: z_max


   .. py:attribute:: z_min


   .. py:attribute:: z_range


.. py:class:: GridCell(num, angle, spacing, tau=0.1, tau_v=10.0, k=0.005, a=bm.pi / 9, A=1.0, J0=1.0, mbar=1.0)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   A model of a grid cell module using a 2D continuous attractor network.

   This class implements a 2D continuous attractor network on a toroidal manifold
   to model the firing patterns of grid cells. The network dynamics include
   synaptic depression or adaptation, which helps stabilize the activity bumps.
   The connectivity is defined on a hexagonal grid structure.

   .. attribute:: num

      The total number of neurons (num_side x num_side).

      :type: int

   .. attribute:: tau

      The synaptic time constant for `u`.

      :type: float

   .. attribute:: tau_v

      The time constant for the adaptation variable `v`.

      :type: float

   .. attribute:: k

      The degree of rescaled inhibition.

      :type: float

   .. attribute:: a

      The half-width of the excitatory connection range.

      :type: float

   .. attribute:: A

      The magnitude of the external input.

      :type: float

   .. attribute:: J0

      The maximum connection value.

      :type: float

   .. attribute:: m

      The strength of the adaptation.

      :type: float

   .. attribute:: angle

      The orientation of the grid.

      :type: float

   .. attribute:: value_grid

      The (x, y) preferred phase coordinates for each neuron.

      :type: bm.math.ndarray

   .. attribute:: conn_mat

      The connection matrix.

      :type: bm.math.ndarray

   .. attribute:: r

      The firing rates of the neurons.

      :type: bm.Variable

   .. attribute:: u

      The synaptic inputs to the neurons.

      :type: bm.Variable

   .. attribute:: v

      The adaptation variables for the neurons.

      :type: bm.Variable

   .. attribute:: center

      The decoded 2D center of the activity bump.

      :type: bm.Variable

   Initializes the GridCell model.

   :param num: The number of neurons along one dimension of the square grid.
   :type num: int
   :param angle: The orientation angle of the grid pattern.
   :type angle: float
   :param spacing: The spacing of the grid pattern.
   :type spacing: float
   :param tau: The synaptic time constant. Defaults to 0.1.
   :type tau: float, optional
   :param tau_v: The adaptation time constant. Defaults to 10.0.
   :type tau_v: float, optional
   :param k: The strength of global inhibition. Defaults to 5e-3.
   :type k: float, optional
   :param a: The width of the connection profile. Defaults to pi/9.
   :type a: float, optional
   :param A: The magnitude of external input. Defaults to 1.0.
   :type A: float, optional
   :param J0: The maximum connection strength. Defaults to 1.0.
   :type J0: float, optional
   :param mbar: The base strength of adaptation. Defaults to 1.0.
   :type mbar: float, optional


   .. py:method:: circle_period(d)

      Wraps values into the periodic range [-pi, pi].

      :param d: The input values.
      :type d: Array

      :returns: The wrapped values.
      :rtype: Array



   .. py:method:: dist(d)

      Calculates the distance on the hexagonal grid.

      It first maps the periodic difference vector `d` into a Cartesian
      coordinate system that reflects the hexagonal lattice structure and then
      computes the Euclidean distance.

      :param d: An array of difference vectors in the phase space.
      :type d: Array

      :returns: The corresponding distances on the hexagonal lattice.
      :rtype: Array



   .. py:method:: get_center()

      Decodes and updates the 2D center of the activity bump.

      It uses a population vector average for both the x and y dimensions of the
      phase space.



   .. py:method:: make_conn()

      Constructs the connection matrix for the 2D attractor network.

      The connection strength between two neurons is a Gaussian function of the
      hexagonal distance between their preferred phases.

      :returns: The connection matrix (num x num).
      :rtype: Array



   .. py:method:: reset_state(*args, **kwargs)

      Resets the state variables of the model to zeros.



   .. py:method:: update(input)

      The function to specify the updating rule.




   .. py:attribute:: A
      :value: 1.0



   .. py:attribute:: J0
      :value: 1.0



   .. py:attribute:: a


   .. py:attribute:: angle


   .. py:attribute:: center


   .. py:attribute:: conn_mat


   .. py:attribute:: coor_transform


   .. py:property:: derivative


   .. py:attribute:: dxy


   .. py:attribute:: input


   .. py:attribute:: integral


   .. py:attribute:: k
      :value: 0.005



   .. py:attribute:: m
      :value: 0.01



   .. py:attribute:: num


   .. py:attribute:: r


   .. py:attribute:: ratio


   .. py:attribute:: rho


   .. py:attribute:: rot


   .. py:attribute:: tau
      :value: 0.1



   .. py:attribute:: tau_v
      :value: 10.0



   .. py:attribute:: u


   .. py:attribute:: v


   .. py:attribute:: value_grid


   .. py:attribute:: x


   .. py:attribute:: x_grid


   .. py:attribute:: x_range


   .. py:attribute:: y_grid


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


.. py:class:: HierarchicalPathIntegrationModel(spacing, angle, place_center=None, band_size=180, band_noise=0.0, band_w_L2S=0.2, band_w_S2L=1.0, band_gain=0.2, grid_num=20, grid_tau=0.1, grid_tau_v=10.0, grid_k=0.005, grid_a=bm.pi / 9, grid_A=1.0, grid_J0=1.0, grid_mbar=1.0, gauss_tau=1.0, gauss_J0=1.1, gauss_k=0.0005, gauss_a=2 / 9 * bm.pi, nonrec_tau=0.1)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModelGroup`


   A hierarchical model combining band cells and grid cells for path integration.

   This model forms a single grid module. It consists of three `BandCell` modules,
   each with a different preferred orientation (separated by 60 degrees), and one
   `GridCell` module. The band cells integrate velocity along their respective
   directions, and their combined outputs provide the input to the `GridCell`
   network, effectively driving the grid cell's activity bump. The model can
   also project its grid cell activity to a population of place cells.

   .. attribute:: band_cell_x

      The first band cell module (orientation `angle`).

      :type: BandCell

   .. attribute:: band_cell_y

      The second band cell module (orientation `angle` + 60 deg).

      :type: BandCell

   .. attribute:: band_cell_z

      The third band cell module (orientation `angle` + 120 deg).

      :type: BandCell

   .. attribute:: grid_cell

      The grid cell module driven by the band cells.

      :type: GridCell

   .. attribute:: place_center

      The center locations of the target place cells.

      :type: bm.math.ndarray

   .. attribute:: Wg2p

      The connection weights from grid cells to place cells.

      :type: bm.math.ndarray

   .. attribute:: grid_output

      The activity of the place cells.

      :type: bm.Variable

   Initializes the HierarchicalPathIntegrationModel.

   :param spacing: The spacing of the grid pattern for this module.
   :type spacing: float
   :param angle: The base orientation angle for the module.
   :type angle: float
   :param place_center: The center locations of the
                        target place cell population. Defaults to a random distribution.
   :type place_center: bm.math.ndarray, optional
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


   .. py:method:: Postophase(pos)

      Projects a 2D position to the 2D phase space of the grid module.

      :param pos: The 2D position vector.
      :type pos: Array

      :returns: The corresponding 2D phase vector.
      :rtype: Array



   .. py:method:: dist(d)

      Calculates the distance on the hexagonal grid.

      :param d: An array of difference vectors in the phase space.
      :type d: Array

      :returns: The corresponding distances on the hexagonal lattice.
      :rtype: Array



   .. py:method:: get_input(Phase)

      Generates a stimulus input for the grid cell based on a 2D phase.

      :param Phase: The 2D phase vector.
      :type Phase: Array

      :returns: The stimulus input vector for the grid cells.
      :rtype: Array



   .. py:method:: make_Wg2p()

      Creates the connection weights from grid cells to place cells.

      The connection strength is determined by the proximity of a place cell's
      center to a grid cell's firing field, calculated in the phase domain.



   .. py:method:: make_conn()

      Creates the connection matrices from the band cells to the grid cells.

      The connection from a band cell to a grid cell is strong if the grid cell's
      preferred phase along the band cell's direction matches the band cell's
      preferred phase.



   .. py:method:: update(velocity, loc, loc_input_stre=0.0)

      Step function of a network.

      In this update function, the update functions in children systems are
      iteratively called.



   .. py:attribute:: band_cell_x


   .. py:attribute:: band_cell_y


   .. py:attribute:: band_cell_z


   .. py:attribute:: coor_transform


   .. py:attribute:: grid_cell


   .. py:attribute:: grid_output


   .. py:attribute:: num_place


   .. py:attribute:: place_center


   .. py:attribute:: proj_k_x


   .. py:attribute:: proj_k_y


.. py:class:: NonRecUnits(size, tau = 0.1, z_min = -bm.pi, z_max = bm.pi, noise = 2.0)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   A model of non-recurrently connected units.

   This class implements a simple leaky integrator model for a population of
   neurons that do not have recurrent connections among themselves. They respond
   to external inputs and have a non-linear activation function.

   .. attribute:: size

      The number of neurons.

      :type: int

   .. attribute:: noise_0

      The standard deviation of the Gaussian noise.

      :type: float

   .. attribute:: tau

      The time constant for the synaptic input `u`.

      :type: float

   .. attribute:: z_min

      The minimum value of the encoded feature space.

      :type: float

   .. attribute:: z_max

      The maximum value of the encoded feature space.

      :type: float

   .. attribute:: z_range

      The range of the feature space.

      :type: float

   .. attribute:: x

      The preferred feature values for each neuron.

      :type: bm.ndarray

   .. attribute:: rho

      The neural density.

      :type: float

   .. attribute:: dx

      The stimulus density.

      :type: float

   .. attribute:: r

      The firing rates of the neurons.

      :type: bm.Variable

   .. attribute:: u

      The synaptic inputs to the neurons.

      :type: bm.Variable

   .. attribute:: input

      The external input to the neurons.

      :type: bm.Variable

   Initializes the NonRecUnits model.

   :param size: The number of neurons.
   :type size: int
   :param tau: The time constant of the neurons. Defaults to 0.1.
   :type tau: float, optional
   :param z_min: The minimum value of the feature space. Defaults to -pi.
   :type z_min: float, optional
   :param z_max: The maximum value of the feature space. Defaults to pi.
   :type z_max: float, optional
   :param noise: The level of noise in the system. Defaults to 2.0.
   :type noise: float, optional


   .. py:method:: activate(x)

      Applies an activation function to the input.

      :param x: The input to the activation function (e.g., synaptic input `u`).
      :type x: Array

      :returns: The result of the activation function (ReLU).
      :rtype: Array



   .. py:method:: dist(d)

      Calculates the periodic distance in the feature space.

      This function wraps distances to ensure they fall within the periodic
      boundaries of the feature space.

      :param d: The array of distances.
      :type d: Array

      :returns: The wrapped distances.
      :rtype: Array



   .. py:method:: update(input)

      The function to specify the updating rule.




   .. py:attribute:: dx


   .. py:attribute:: input


   .. py:attribute:: noise_0
      :value: 2.0



   .. py:attribute:: r


   .. py:attribute:: rho


   .. py:attribute:: size


   .. py:attribute:: tau
      :value: 0.1



   .. py:attribute:: u


   .. py:attribute:: x


   .. py:attribute:: z_max


   .. py:attribute:: z_min


   .. py:attribute:: z_range


