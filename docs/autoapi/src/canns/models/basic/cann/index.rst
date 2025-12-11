src.canns.models.basic.cann
===========================

.. py:module:: src.canns.models.basic.cann


Classes
-------

.. autoapisummary::

   src.canns.models.basic.cann.BaseCANN
   src.canns.models.basic.cann.BaseCANN1D
   src.canns.models.basic.cann.BaseCANN2D
   src.canns.models.basic.cann.CANN1D
   src.canns.models.basic.cann.CANN1D_SFA
   src.canns.models.basic.cann.CANN2D
   src.canns.models.basic.cann.CANN2D_SFA


Module Contents
---------------

.. py:class:: BaseCANN(shape, **kwargs)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   Base class for Continuous Attractor Neural Network (CANN) models.
   This class sets up the fundamental properties of the network, including
   neuronal properties, feature space, and the connectivity matrix, which
   are shared by different CANN model variations.

   Initializes the base CANN model.

   :param shape: The number of neurons in the network. If an int is provided,
                 it will be converted to a single-element tuple. If a tuple is provided,
                 it defines the shape of the network (e.g., (length, length) for 2D).
                 Internally, shape is always stored as a tuple.
   :type shape: int or tuple
   :param \*\*kwargs: Additional keyword arguments passed to the parent BasicModel.


   .. py:method:: get_stimulus_by_pos(pos)
      :abstractmethod:


      Generates an external stimulus based on a given position in the feature space.
      This method should be implemented in subclasses to define how the stimulus is shaped.

      :param pos: The position in the feature space where the stimulus is centered.
      :type pos: float or Array

      :returns: An array of stimulus values for each neuron.
      :rtype: Array



   .. py:method:: make_conn()
      :abstractmethod:


      Constructs the connectivity matrix for the CANN model.
      This method should be implemented in subclasses to define how neurons
      are connected based on their feature preferences.

      :returns: A connectivity matrix defining the synaptic strengths between neurons.
      :rtype: Array



.. py:class:: BaseCANN1D(num, tau = 1.0, k = 8.1, a = 0.5, A = 10, J0 = 4.0, z_min = -bm.pi, z_max = bm.pi, **kwargs)

   Bases: :py:obj:`BaseCANN`


   Base class for 1D Continuous Attractor Neural Network (CANN) models.
   This class sets up the fundamental properties of the network, including
   neuronal properties, feature space, and the connectivity matrix, which
   are shared by different CANN model variations.

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


   .. py:method:: dist(d)

      Calculates the shortest distance between two points in a circular feature space
      with periodic boundary conditions.

      :param d: The difference between two positions.
      :type d: Array

      :returns: The shortest distance, wrapped around the periodic boundary.
      :rtype: Array



   .. py:method:: get_stimulus_by_pos(pos)

      Generates a Gaussian-shaped external stimulus centered at a given position.

      :param pos: The center position of the stimulus in the feature space.
      :type pos: float

      :returns: An array of stimulus values for each neuron.
      :rtype: Array



   .. py:method:: make_conn()

      Constructs the connectivity matrix based on a Gaussian-like profile.
      The connection strength between two neurons depends on the distance
      between their preferred feature values in the circular space.

      :returns: A (num x num) connectivity matrix.
      :rtype: Array



   .. py:attribute:: A
      :value: 10



   .. py:attribute:: J0
      :value: 4.0



   .. py:attribute:: a
      :value: 0.5



   .. py:attribute:: conn_mat


   .. py:attribute:: dx


   .. py:attribute:: k
      :value: 8.1



   .. py:attribute:: rho


   .. py:attribute:: tau
      :value: 1.0



   .. py:attribute:: x


   .. py:attribute:: z_max


   .. py:attribute:: z_min


   .. py:attribute:: z_range


.. py:class:: BaseCANN2D(length, tau = 1.0, k = 8.1, a = 0.5, A = 10, J0 = 4.0, z_min = -bm.pi, z_max = bm.pi, **kwargs)

   Bases: :py:obj:`BaseCANN`


   Base class for 2D Continuous Attractor Neural Network (CANN) models.
   This class sets up the fundamental properties of the network, including
   neuronal properties, feature space, and the connectivity matrix, which
   are shared by different CANN model variations.

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


   .. py:method:: dist(d)

      Calculates the shortest distance vector between two points in a 2D feature space
      with periodic boundary conditions (a torus).

      :param d: The difference vector between two positions, e.g., [dx, dy].
      :type d: Array

      :returns:

                The shortest distance vector, with each component wrapped around
                       the periodic boundary.
      :rtype: Array



   .. py:method:: get_stimulus_by_pos(pos)

      Generates a Gaussian-shaped external stimulus centered at a given
      coordinate on the 2D neural grid.

      :param pos: The center coordinate [x, y] of the stimulus
                  in the feature space.
      :type pos: Array, tuple

      :returns: A 2D array (grid) of stimulus values for each neuron.
      :rtype: Array



   .. py:method:: make_conn()

      Constructs the connectivity matrix for a 2D grid of neurons based on a
      Gaussian profile. The connection strength between two neurons depends on the
      Euclidean distance between their preferred feature coordinates in a 2D
      toroidal space (space with periodic boundaries in both dimensions).

      :returns: A ((num*num) x (num*num)) connectivity matrix.
      :rtype: Array



   .. py:method:: show_conn()

      Displays the connectivity matrix as an image.
      This method visualizes the connection strengths between neurons in the 2D feature space.



   .. py:attribute:: A
      :value: 10



   .. py:attribute:: J0
      :value: 4.0



   .. py:attribute:: a
      :value: 0.5



   .. py:attribute:: conn_mat


   .. py:attribute:: dx


   .. py:attribute:: k
      :value: 8.1



   .. py:attribute:: length


   .. py:attribute:: rho


   .. py:attribute:: tau
      :value: 1.0



   .. py:attribute:: x


   .. py:attribute:: z_max


   .. py:attribute:: z_min


   .. py:attribute:: z_range


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


