src.canns.models.brain_inspired
===============================

.. py:module:: src.canns.models.brain_inspired

.. autoapi-nested-parse::

   Brain-inspired neural network models.

   This module contains biologically plausible neural network models that incorporate
   principles from neuroscience and cognitive science, including associative memory,
   Hebbian learning, and other brain-inspired mechanisms.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/models/brain_inspired/hopfield/index
   /autoapi/src/canns/models/brain_inspired/linear/index
   /autoapi/src/canns/models/brain_inspired/spiking/index


Classes
-------

.. autoapisummary::

   src.canns.models.brain_inspired.AmariHopfieldNetwork
   src.canns.models.brain_inspired.BrainInspiredModel
   src.canns.models.brain_inspired.BrainInspiredModelGroup
   src.canns.models.brain_inspired.LinearLayer
   src.canns.models.brain_inspired.SpikingLayer


Package Contents
----------------

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



.. py:class:: BrainInspiredModel(name = None, mode = None)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   Base class for brain-inspired models.

   Trainer compatibility notes
   - If a model wants to support generic Hebbian training, expose a weight parameter
     attribute with a ``.value`` array of shape (N, N) (commonly a
     ``bm.Variable``). The recommended attribute name is ``W``.
   - Override ``weight_attr`` to declare a different attribute name if needed. Models
     that use standard backprop may omit this entirely.
   - Implementing ``apply_hebbian_learning`` is optional; prefer letting the trainer
     handle the generic rule when applicable. Implement this only when you need
     model-specific behavior.

   Notes on Predict compatibility
   - For the trainer's generic prediction path, models typically expose:
     1) an ``update(prev_energy)`` method to advance one step (optional; not all models
        require energy-driven updates),
     2) an ``energy`` property to compute current energy (scalar-like),
     3) a state vector attribute (default ``s``) with ``.value`` as 1D array used as
        the prediction state; override ``predict_state_attr`` to change the name.

   Optional resizing
   - Models may implement ``resize(num_neurons: int, preserve_submatrix: bool = True)`` to
     allow trainers to change neuron dimensionality on the fly (e.g., when training with
     patterns of a different length). When implemented, the trainer will call this to
     align dimensions before training/prediction.


   .. py:method:: apply_hebbian_learning(train_data)
      :abstractmethod:


      Optional model-specific Hebbian learning implementation.

      The generic ``HebbianTrainer`` can update ``W`` directly without requiring this
      method. Only implement when custom behavior deviates from the generic rule.



   .. py:method:: predict(pattern)
      :abstractmethod:



   .. py:method:: resize(num_neurons, preserve_submatrix = True)
      :abstractmethod:


      Optional method to resize model state/parameters to ``num_neurons``.

      Default implementation is a stub. Subclasses may override to support dynamic
      dimensionality changes.



   .. py:property:: energy
      :type: float

      :abstractmethod:


      Current energy of the model state (used for convergence checks in prediction).

      Implementations may return a float or a 0-dim array; the trainer treats it as a scalar.


   .. py:property:: predict_state_attr
      :type: str


      Name of the state vector attribute used by generic prediction.

      Override in subclasses if the prediction state is not stored in ``s``.


   .. py:property:: weight_attr
      :type: str


      Name of the connection weight attribute used by generic training.

      Override in subclasses if the weight parameter is not named ``W``.


.. py:class:: BrainInspiredModelGroup(*children_as_tuple, name = None, mode = None, child_type = DynamicalSystem, **children_as_dict)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModelGroup`


   Base class for groups of brain-inspired models.

   This class manages collections of brain-inspired models and provides
   coordinated learning and dynamics across multiple model instances.


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


.. py:class:: SpikingLayer(input_size, output_size, threshold = 1.0, v_reset = 0.0, leak = 0.9, trace_decay = 0.95, dt = 1.0, **kwargs)

   Bases: :py:obj:`src.canns.models.brain_inspired._base.BrainInspiredModel`


   Simple Leaky Integrate-and-Fire (LIF) spiking neuron layer.

   This model provides a minimal spiking neuron implementation for demonstrating
   spike-timing-dependent plasticity (STDP). It features:
   - Leaky integration of input currents
   - Threshold-based spike generation
   - Reset mechanism after spiking
   - Exponential spike traces for STDP learning

   Dynamics:
       v[t+1] = leak * v[t] + W @ x[t]
       spike = 1 if v >= threshold else 0
       v = v_reset if spike else v
       trace = decay * trace + spike

   .. rubric:: References

   - Gerstner & Kistler (2002): Spiking Neuron Models
   - Morrison et al. (2008): Phenomenological models of synaptic plasticity

   Initialize the spiking layer.

   :param input_size: Number of input neurons
   :param output_size: Number of output neurons
   :param threshold: Spike threshold for membrane potential
   :param v_reset: Reset potential after spike
   :param leak: Membrane leak factor (0-1, closer to 1 = less leaky)
   :param trace_decay: Decay factor for spike traces (used in STDP)
   :param dt: Time step size
   :param \*\*kwargs: Additional arguments passed to parent class


   .. py:method:: forward(x)

      Forward pass through the spiking layer.

      :param x: Input spikes of shape (input_size,) with binary values (0 or 1)

      :returns: Output spikes of shape (output_size,) with binary values (0 or 1)



   .. py:method:: reset_state()

      Reset membrane potentials and spike traces.



   .. py:method:: update(prev_energy)

      Update method for trainer compatibility (no-op for spiking layer).



   .. py:attribute:: W


   .. py:attribute:: dt
      :value: 1.0



   .. py:property:: energy
      :type: float


      Energy for trainer compatibility (0 for spiking layer).


   .. py:attribute:: input_size


   .. py:attribute:: leak
      :value: 0.9



   .. py:attribute:: output_size


   .. py:property:: predict_state_attr
      :type: str


      Name of output state for prediction.


   .. py:attribute:: spike


   .. py:attribute:: threshold
      :value: 1.0



   .. py:attribute:: trace_decay
      :value: 0.95



   .. py:attribute:: trace_post


   .. py:attribute:: trace_pre


   .. py:attribute:: v


   .. py:attribute:: v_reset
      :value: 0.0



   .. py:property:: weight_attr
      :type: str


      Name of weight parameter for generic training.


   .. py:attribute:: x


