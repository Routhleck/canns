src.canns.models.basic
======================

.. py:module:: src.canns.models.basic


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/models/basic/cann/index
   /autoapi/src/canns/models/basic/hierarchical_model/index
   /autoapi/src/canns/models/basic/theta_sweep_model/index


Classes
-------

.. autoapisummary::

   src.canns.models.basic.CANN1D
   src.canns.models.basic.CANN1D_SFA
   src.canns.models.basic.CANN2D
   src.canns.models.basic.CANN2D_SFA
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


