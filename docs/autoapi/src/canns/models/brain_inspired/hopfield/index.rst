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


   Amari-Hopfield network with discrete or continuous dynamics.

   The model performs pattern completion by iteratively updating the state
   vector ``s`` to reduce energy:
       E = -0.5 * sum_ij W_ij * s_i * s_j

   .. rubric:: Examples

   >>> import jax.numpy as jnp
   >>> from canns.models.brain_inspired import AmariHopfieldNetwork
   >>>
   >>> model = AmariHopfieldNetwork(num_neurons=3, activation="sign")
   >>> pattern = jnp.array([1.0, -1.0, 1.0], dtype=jnp.float32)
   >>> weights = jnp.outer(pattern, pattern)
   >>> weights = weights - jnp.diag(jnp.diag(weights))  # zero diagonal
   >>> model.W.value = weights
   >>> model.s.value = jnp.array([1.0, 1.0, -1.0], dtype=jnp.float32)
   >>> model.update(None)
   >>> model.s.value.shape
   (3,)

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

      :param e_old: Unused placeholder for trainer compatibility.

      :returns: None



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



