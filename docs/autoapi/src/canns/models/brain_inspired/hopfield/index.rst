src.canns.models.brain_inspired.hopfield
========================================

.. py:module:: src.canns.models.brain_inspired.hopfield


Classes
-------

.. autoapisummary::

   src.canns.models.brain_inspired.hopfield.AmariHopfieldNetwork


Module Contents
---------------

.. py:class:: AmariHopfieldNetwork(num_neurons, asyn = False, threshold = 0.0, activation = 'sign', temperature = 1.0, **kwargs)

   Bases: :py:obj:`src.canns.models.brain_inspired._base.BrainInspiredModel`


   Amari-Hopfield Network implementation supporting both discrete and continuous dynamics.

   This class implements Hopfield networks with flexible activation functions,
   supporting both discrete binary states and continuous dynamics. The network
   performs pattern completion through energy minimization using asynchronous
   or synchronous updates.

   The network energy function:
   E = -0.5 * Σ_ij W_ij * s_i * s_j

   Where s_i can be discrete {-1, +1} or continuous depending on activation function.

   Reference:
       Amari, S. (1977). Neural theory of association and concept-formation.
       Biological Cybernetics, 26(3), 175-185.

       Hopfield, J. J. (1982). Neural networks and physical systems with
       emergent collective computational abilities. Proceedings of the
       National Academy of Sciences of the USA, 79(8), 2554-2558.

   Initialize the Amari-Hopfield Network.

   :param num_neurons: Number of neurons in the network
   :param asyn: Whether to run asynchronously or synchronously
   :param threshold: Threshold for activation function
   :param activation: Activation function type ("sign", "tanh", "sigmoid")
   :param temperature: Temperature parameter for continuous activations
   :param \*\*kwargs: Additional arguments passed to parent class


   .. py:method:: compute_overlap(pattern1, pattern2)

      Compute overlap between two binary patterns.

      :param pattern1: Binary patterns to compare
      :param pattern2: Binary patterns to compare

      :returns: Overlap value (1 for identical, 0 for orthogonal, -1 for opposite)



   .. py:method:: resize(num_neurons, preserve_submatrix = True)

      Resize the network dimension and state/weights.

      :param num_neurons: New neuron count (N)
      :param preserve_submatrix: If True, copy the top-left min(old, N) block of W into
                                 the new matrix; otherwise reinitialize W with zeros.



   .. py:method:: update(e_old)

      Update network state for one time step.



   .. py:attribute:: W


   .. py:attribute:: activation


   .. py:attribute:: asyn
      :value: False



   .. py:property:: energy

      Compute the energy of the network state.


   .. py:attribute:: num_neurons


   .. py:attribute:: s


   .. py:property:: storage_capacity

      Get theoretical storage capacity.

      :returns: Theoretical storage capacity (approximately N/(4*ln(N)))


   .. py:attribute:: temperature
      :value: 1.0



   .. py:attribute:: threshold
      :value: 0.0



