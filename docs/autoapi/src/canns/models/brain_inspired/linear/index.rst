src.canns.models.brain_inspired.linear
======================================

.. py:module:: src.canns.models.brain_inspired.linear

.. autoapi-nested-parse::

   Generic linear layer for brain-inspired learning algorithms.



Classes
-------

.. autoapisummary::

   src.canns.models.brain_inspired.linear.LinearLayer


Module Contents
---------------

.. py:class:: LinearLayer(input_size, output_size, use_bcm_threshold = False, threshold_tau = 100.0, **kwargs)

   Bases: :py:obj:`src.canns.models.brain_inspired._base.BrainInspiredModel`


   Generic linear feedforward layer supporting multiple brain-inspired learning rules.

   This model provides a simple linear transformation with optional sliding threshold
   for BCM-style plasticity. It can be used with various trainers:
   - OjaTrainer: Normalized Hebbian learning for PCA
   - BCMTrainer: Sliding threshold plasticity (requires use_bcm_threshold=True)
   - HebbianTrainer: Standard Hebbian learning

   Computation:
       y = W @ x

   where W is the weight matrix, x is the input, and y is the output.

   For BCM learning, an optional sliding threshold θ tracks output activity:
       θ ← θ + (1/τ) * (y² - θ)

   .. rubric:: References

   - Oja (1982): Simplified neuron model as a principal component analyzer
   - Bienenstock et al. (1982): Theory for the development of neuron selectivity

   Initialize the linear layer.

   :param input_size: Dimensionality of input vectors
   :param output_size: Number of output neurons (features to extract)
   :param use_bcm_threshold: Whether to maintain sliding threshold for BCM learning
   :param threshold_tau: Time constant for threshold sliding average (only used if use_bcm_threshold=True)
   :param \*\*kwargs: Additional arguments passed to parent class


   .. py:method:: forward(x)

      Forward pass through the layer.

      :param x: Input vector of shape (input_size,)

      :returns: Output vector of shape (output_size,)



   .. py:method:: resize(input_size, output_size = None, preserve_submatrix = True)

      Resize layer dimensions.

      :param input_size: New input dimension
      :param output_size: New output dimension (if None, keep current)
      :param preserve_submatrix: Whether to preserve existing weights



   .. py:method:: update(prev_energy)

      Update method for trainer compatibility (no-op for feedforward layer).



   .. py:method:: update_threshold()

      Update the sliding threshold based on recent activity (BCM only).

      This method should be called by BCMTrainer after each forward pass.
      Updates θ using: θ ← θ + (1/τ) * (y² - θ)



   .. py:attribute:: W


   .. py:property:: energy
      :type: float


      Energy for trainer compatibility (0 for feedforward layer).


   .. py:attribute:: input_size


   .. py:attribute:: output_size


   .. py:property:: predict_state_attr
      :type: str


      Name of output state for prediction.


   .. py:attribute:: threshold_tau
      :value: 100.0



   .. py:attribute:: use_bcm_threshold
      :value: False



   .. py:property:: weight_attr
      :type: str


      Name of weight parameter for generic training.


   .. py:attribute:: x


   .. py:attribute:: y


