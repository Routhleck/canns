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

.. py:class:: CANN1D(num, tau = 1.0, k = 8.1, a = 0.5, A = 10, J0 = 4.0, z_min = -u.math.pi, z_max = u.math.pi, **kwargs)

   Bases: :py:obj:`BaseCANN1D`


   A standard 1D Continuous Attractor Neural Network (CANN) model.
   This model implements the core dynamics where a localized "bump" of activity
   can be sustained and moved by external inputs.

   Reference:
       Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
       Neural computation, 20(4), 994-1025.

   Initializes the base 1D CANN model.

   :param num: The number of neurons in the network.
   :type num: int
   :param tau: The synaptic time constant, controlling how quickly the membrane potential changes.
   :type tau: float
   :param k: A parameter controlling the strength of the global inhibition.
   :type k: float
   :param a: The half-width of the excitatory connection range. It defines the "spread" of local connections.
   :type a: float
   :param A: The magnitude (amplitude) of the external stimulus.
   :type A: float
   :param J0: The maximum connection strength between neurons.
   :type J0: float
   :param z_min: The minimum value of the feature space (e.g., -pi for an angle).
   :type z_min: float
   :param z_max: The maximum value of the feature space (e.g., +pi for an angle).
   :type z_max: float
   :param \*\*kwargs: Additional keyword arguments passed to the parent BasicModel.


   .. py:method:: init_state(*args, **kwargs)

      Initializes the state variables of the model.



   .. py:method:: update(inp)

      The main update function, defining the dynamics of the network for one time step.

      :param inp: The external input for the current time step.
      :type inp: Array



.. py:class:: CANN1D_SFA(num, tau = 1.0, tau_v = 50.0, k = 8.1, a = 0.3, A = 0.2, J0 = 1.0, z_min = -u.math.pi, z_max = u.math.pi, m = 0.3, **kwargs)

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


   .. py:method:: init_state(*args, **kwargs)

      Initializes the state variables of the model, including the adaptation variable.



   .. py:method:: update(inp)

      The main update function for the SFA model. It includes dynamics for both
      the membrane potential and the adaptation variable.

      :param inp: The external input for the current time step.
      :type inp: Array



   .. py:attribute:: m
      :value: 0.3



   .. py:attribute:: tau_v
      :value: 50.0



.. py:class:: CANN2D(length, tau = 1.0, k = 8.1, a = 0.5, A = 10, J0 = 4.0, z_min = -u.math.pi, z_max = u.math.pi, **kwargs)

   Bases: :py:obj:`BaseCANN2D`


   A 2D Continuous Attractor Neural Network (CANN) model.
   This model extends the base CANN2D class to include specific dynamics
   and properties for a 2D neural network.

   Reference:
       Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
       Neural computation, 20(4), 994-1025.

   Initializes the base 2D CANN model.

   :param length: The number of neurons in one dimension of the network (the network is square).
   :type length: int
   :param tau: The synaptic time constant, controlling how quickly the membrane potential changes.
   :type tau: float
   :param k: A parameter controlling the strength of the global inhibition.
   :type k: float
   :param a: The half-width of the excitatory connection range. It defines the "spread" of local connections.
   :type a: float
   :param A: The magnitude (amplitude) of the external stimulus.
   :type A: float
   :param J0: The maximum connection strength between neurons.
   :type J0: float
   :param z_min: The minimum value of the feature space (e.g., -pi for an angle).
   :type z_min: float
   :param z_max: The maximum value of the feature space (e.g., +pi for an angle).
   :type z_max: float
   :param \*\*kwargs: Additional keyword arguments passed to the parent BasicModel.


   .. py:method:: init_state(*args, **kwargs)

      Initializes the state variables of the model.



   .. py:method:: update(inp)

      The main update function, defining the dynamics of the network for one time step.

      :param inp: The external input to the network, which can be a stimulus or other driving force.
      :type inp: Array



.. py:class:: CANN2D_SFA(length, tau = 1.0, tau_v = 50.0, k = 8.1, a = 0.3, A = 0.2, J0 = 1.0, z_min = -u.math.pi, z_max = u.math.pi, m = 0.3, **kwargs)

   Bases: :py:obj:`BaseCANN2D`


   A 2D Continuous Attractor Neural Network (CANN) model with a specific
   implementation of the Synaptic Firing Activity (SFA) dynamics.
   This model extends the base CANN2D class to include SFA-specific dynamics.

   Initializes the 2D CANN model with SFA dynamics.


   .. py:method:: init_state(*args, **kwargs)

      Initializes the state variables of the model, including the adaptation variable.



   .. py:method:: update(inp)

      The main update function for the SFA model. It includes dynamics for both
      the membrane potential and the adaptation variable.

      :param inp: The external input for the current time step.
      :type inp: Array



   .. py:attribute:: m
      :value: 0.3



   .. py:attribute:: tau_v
      :value: 50.0



.. py:class:: HierarchicalNetwork(num_module, num_place)

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

      :type: brainunit.math.ndarray

   .. attribute:: MEC_model_list

      A list containing all the `HierarchicalPathIntegrationModel` instances.

      :type: list

   .. attribute:: grid_fr

      The firing rates of the grid cell population.

      :type: brainstate.HiddenState

   .. attribute:: band_x_fr

      The firing rates of the x-oriented band cell population.

      :type: brainstate.HiddenState

   .. attribute:: band_y_fr

      The firing rates of the y-oriented band cell population.

      :type: brainstate.HiddenState

   .. attribute:: place_fr

      The firing rates of the place cell population.

      :type: brainstate.HiddenState

   .. attribute:: decoded_pos

      The final decoded 2D position.

      :type: brainstate.State

   .. rubric:: References

   Anonymous Author(s) "Unfolding the Black Box of Recurrent Neural Networks for Path Integration" (under review).

   Initializes the HierarchicalNetwork.

   :param num_module: The number of grid modules to create.
   :type num_module: int
   :param num_place: The number of place cells along one dimension of a square grid.
   :type num_place: int


   .. py:method:: init_state(*args, **kwargs)

      State initialization function.



   .. py:method:: update(velocity, loc, loc_input_stre=0.0)

      Update function of a network.

      In this update function, the update functions in children systems are iteratively called.



   .. py:attribute:: MEC_model_list
      :value: []



   .. py:attribute:: num_module


   .. py:attribute:: num_place


   .. py:attribute:: place_center


