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


Classes
-------

.. autoapisummary::

   src.canns.models.brain_inspired.AmariHopfieldNetwork
   src.canns.models.brain_inspired.BrainInspiredModel
   src.canns.models.brain_inspired.BrainInspiredModelGroup


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



   .. py:method:: init_state()

      Initialize network state variables.



   .. py:method:: resize(num_neurons, preserve_submatrix = True)

      Resize the network dimension and state/weights.

      :param num_neurons: New neuron count (N)
      :param preserve_submatrix: If True, copy the top-left min(old, N) block of W into
                                 the new matrix; otherwise reinitialize W with zeros.



   .. py:method:: update(e_old)

      Update network state for one time step.



   .. py:attribute:: activation


   .. py:attribute:: asyn
      :value: False



   .. py:property:: energy

      Compute the energy of the network state.


   .. py:attribute:: num_neurons


   .. py:property:: storage_capacity

      Get theoretical storage capacity.

      :returns: Theoretical storage capacity (approximately N/(4*ln(N)))


   .. py:attribute:: temperature
      :value: 1.0



   .. py:attribute:: threshold
      :value: 0.0



.. py:class:: BrainInspiredModel(in_size, name = None)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModel`


   Base class for brain-inspired models.

   Trainer compatibility notes
   - If a model wants to support generic Hebbian training, expose a weight parameter
     attribute with a ``.value`` array of shape (N, N) (commonly a
     ``brainstate.ParamState``). The recommended attribute name is ``W``.
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


.. py:class:: BrainInspiredModelGroup(*children_as_tuple, **children_as_dict)

   Bases: :py:obj:`src.canns.models.basic._base.BasicModelGroup`


   Base class for groups of brain-inspired models.

   This class manages collections of brain-inspired models and provides
   coordinated learning and dynamics across multiple model instances.


