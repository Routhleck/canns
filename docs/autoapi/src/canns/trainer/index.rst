src.canns.trainer
=================

.. py:module:: src.canns.trainer

.. autoapi-nested-parse::

   Training utilities for CANNs models.

   The module exposes the abstract ``Trainer`` base class and concrete implementations
   of classic brain-inspired learning algorithms: ``HebbianTrainer``, ``AntiHebbianTrainer``,
   ``OjaTrainer``, ``BCMTrainer``, ``SangerTrainer``, and ``STDPTrainer``.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/trainer/bcm/index
   /autoapi/src/canns/trainer/hebbian/index
   /autoapi/src/canns/trainer/oja/index
   /autoapi/src/canns/trainer/sanger/index
   /autoapi/src/canns/trainer/stdp/index
   /autoapi/src/canns/trainer/utils/index


Classes
-------

.. autoapisummary::

   src.canns.trainer.AntiHebbianTrainer
   src.canns.trainer.BCMTrainer
   src.canns.trainer.HebbianTrainer
   src.canns.trainer.OjaTrainer
   src.canns.trainer.STDPTrainer
   src.canns.trainer.SangerTrainer
   src.canns.trainer.Trainer


Package Contents
----------------

.. py:class:: AntiHebbianTrainer(model, **kwargs)

   Bases: :py:obj:`HebbianTrainer`


   Anti-Hebbian trainer for pattern decorrelation and unlearning.

   Overview
   - Implements anti-Hebbian learning rule: "Neurons that fire together, wire apart"
   - Uses negative weight updates: ``W <- W - Σ (t t^T)`` instead of positive
   - Inherits all functionality from HebbianTrainer (predict, predict_batch, etc.)

   Applications
   - Sparse coding and independent component analysis
   - Competitive learning networks
   - Decorrelation and whitening of feature representations
   - Lateral inhibition modeling
   - Selective forgetting / pattern unlearning

   Learning Rule
   - For patterns ``x``, compute optional mean activity ``rho`` and update:
     ``W <- W - sum_i (x_i - rho)(x_i - rho)^T`` (note the minus sign)
   - If ``subtract_mean=True``, patterns are centered by mean: ``t = x - rho``
   - If ``normalize_by_patterns=True``, divide by number of patterns
   - All options from HebbianTrainer apply (subtract_mean, zero_diagonal, etc.)

   Example
       >>> model = AmariHopfieldNetwork(num_neurons=100, activation="tanh")
       >>> # Train with Hebbian first
       >>> hebb_trainer = HebbianTrainer(model)
       >>> hebb_trainer.train(all_patterns)
       >>> # Then apply anti-Hebbian to unlearn specific pattern
       >>> anti_trainer = AntiHebbianTrainer(model, subtract_mean=False)
       >>> anti_trainer.train([pattern_to_forget])

   Initialize Anti-Hebbian trainer.

   :param model: The model to train
   :param \*\*kwargs: Additional arguments passed to HebbianTrainer


.. py:class:: BCMTrainer(model, learning_rate = 0.01, weight_attr = 'W', compiled = True, **kwargs)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   BCM (Bienenstock-Cooper-Munro) sliding-threshold plasticity trainer.

   The BCM rule uses a dynamic postsynaptic threshold to switch between
   potentiation and depression based on recent activity, yielding stable
   receptive-field development and experience-dependent refinement.

   Learning Rule:
       ΔW_ij = η * y_i * (y_i - θ_i) * x_j

   where:
       - W_ij is the weight from input j to neuron i
       - x_j is the presynaptic activity
       - y_i is the postsynaptic activity
       - θ_i is the modification threshold for neuron i

   The threshold θ evolves as a sliding average:
       θ_i = <y_i^2>

   This creates two regimes:
       - If y > θ: potentiation (LTP, strengthen synapses)
       - If y < θ: depression (LTD, weaken synapses)

   Reference:
       Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982).
       Theory for the development of neuron selectivity. Journal of Neuroscience, 2(1), 32-48.

   Initialize BCM trainer.

   :param model: The model to train (typically LinearLayer with use_bcm_threshold=True)
   :param learning_rate: Learning rate η for weight updates
   :param weight_attr: Name of model attribute holding the connection weights
   :param compiled: Whether to use JIT-compiled training loop (default: True)
   :param \*\*kwargs: Additional arguments passed to parent Trainer


   .. py:method:: predict(pattern, *args, **kwargs)

      Predict output for a single input pattern.

      :param pattern: Input pattern of shape (input_size,)

      :returns: Output pattern of shape (output_size,)



   .. py:method:: train(train_data)

      Train the model using BCM rule.

      :param train_data: Iterable of input patterns (each of shape (input_size,))



   .. py:attribute:: compiled
      :value: True



   .. py:attribute:: learning_rate
      :value: 0.01



   .. py:attribute:: weight_attr
      :value: 'W'



.. py:class:: HebbianTrainer(model, show_iteration_progress = False, compiled_prediction = True, *, weight_attr = 'W', subtract_mean = True, zero_diagonal = True, normalize_by_patterns = True, prefer_generic = True, state_attr = None, prefer_generic_predict = True, preserve_on_resize = True)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   Generic Hebbian trainer with progress reporting.

   Overview
   - Uses a model-exposed weight parameter (default attribute name: ``W``) to apply a
     standard Hebbian update. If unavailable, falls back to the model's
     ``apply_hebbian_learning``.
   - Works with models that expose a parameter object with a ``.value`` ndarray of shape
     (N, N) (e.g., ``bm.Variable``).

   Generic rule
   - For patterns ``x`` (shape: (N,)), compute optional mean activity ``rho`` and update
     ``W <- W + sum_i (x_i - rho)(x_i - rho)^T``.
   - Options allow zeroing the diagonal and normalizing by number of patterns.

   Key options
   - ``weight_attr``: Name of the weight attribute on the model (default: "W").
   - ``subtract_mean``: Whether to center patterns by mean activity ``rho``.
   - ``zero_diagonal``: Whether to set diagonal of ``W`` to zero after update.
   - ``normalize_by_patterns``: Divide accumulated outer-products by number of patterns.
   - ``prefer_generic``: Prefer the generic Hebbian rule over model-specific method.
   - ``state_attr``: Name of the state vector attribute for prediction (default: ``s``; or
     model-provided ``predict_state_attr``).
   - ``prefer_generic_predict``: Prefer the trainer's generic predict loop over the
     model's ``predict`` implementation (falls back automatically when unsupported).

   Initialize Hebbian trainer.

   :param model: The model to train
   :param show_iteration_progress: Whether to show progress for individual pattern convergence
   :param compiled_prediction: Whether to use compiled prediction by default (faster but no iteration progress)
   :param weight_attr: Name of model attribute holding the connection weights (default: "W").
   :param subtract_mean: Subtract dataset mean activity (rho) before outer-product.
   :param zero_diagonal: Force zero self-connections after update.
   :param normalize_by_patterns: Divide accumulated outer-products by number of patterns.
   :param prefer_generic: If True, use trainer's generic Hebbian rule when possible; otherwise
                          call the model's own implementation if available.


   .. py:method:: predict(pattern, num_iter = 20, compiled = None, show_progress = None, convergence_threshold = 1e-10, progress_callback = None)

      Predict a single pattern.

      :param pattern: Input pattern to predict
      :param num_iter: Maximum number of iterations
      :param compiled: Override default compiled setting
      :param show_progress: Override default progress setting
      :param convergence_threshold: Energy change threshold for convergence

      :returns: Predicted pattern



   .. py:method:: predict_batch(patterns, num_iter = 20, compiled = None, show_sample_progress = True, show_iteration_progress = None, convergence_threshold = 1e-10)

      Predict multiple patterns with progress reporting.

      :param patterns: List of input patterns to predict
      :param num_iter: Maximum number of iterations per pattern
      :param compiled: Override default compiled setting
      :param show_sample_progress: Whether to show progress across samples
      :param show_iteration_progress: Override default iteration progress setting
      :param convergence_threshold: Energy change threshold for convergence

      :returns: List of predicted patterns



   .. py:method:: train(train_data)

      Train the model using Hebbian learning.

      Behavior
      - Preferred path: apply a generic Hebbian update directly to ``model.<weight_attr>``.
      - Fallback path: call ``model.apply_hebbian_learning(train_data)`` if generic path
        is unavailable.

      Requirements for generic path
      - Model must expose ``model.<weight_attr>`` with a ``.value`` array of shape (N, N).
      - Optionally, models can declare ``weight_attr`` property to specify the
        attribute name, allowing ``HebbianTrainer(..., weight_attr=None)``.



   .. py:attribute:: normalize_by_patterns
      :value: True



   .. py:attribute:: prefer_generic
      :value: True



   .. py:attribute:: prefer_generic_predict
      :value: True



   .. py:attribute:: preserve_on_resize
      :value: True



   .. py:attribute:: state_attr
      :value: None



   .. py:attribute:: subtract_mean
      :value: True



   .. py:attribute:: weight_attr
      :value: 'W'



   .. py:attribute:: zero_diagonal
      :value: True



.. py:class:: OjaTrainer(model, learning_rate = 0.01, normalize_weights = True, weight_attr = 'W', compiled = True, **kwargs)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   Oja's normalized Hebbian learning trainer.

   Oja's rule stabilizes pure Hebbian growth by introducing a weight-dependent
   normalization term, enabling single-neuron principal component extraction
   without unbounded weight magnitudes.

   Learning Rule:
       ΔW_ij = η * (y_i * x_j - y_i^2 * W_ij)

   where:
       - W_ij is the weight from input j to output i
       - x_j is the input activity
       - y_i is the output activity (y = W @ x)
       - η is the learning rate

   The rule can be rewritten as:
       ΔW = η * (y @ x^T - diag(y^2) @ W)

   This naturally leads to weight normalization and PCA extraction.

   Reference:
       Oja, E. (1982). Simplified neuron model as a principal component analyzer.
       Journal of Mathematical Biology, 15(3), 267-273.

   Initialize Oja trainer.

   :param model: The model to train (typically LinearLayer)
   :param learning_rate: Learning rate η for weight updates
   :param normalize_weights: Whether to normalize weights to unit norm after each update
   :param weight_attr: Name of model attribute holding the connection weights
   :param compiled: Whether to use JIT-compiled training loop (default: True)
   :param \*\*kwargs: Additional arguments passed to parent Trainer


   .. py:method:: predict(pattern, *args, **kwargs)

      Predict output for a single input pattern.

      :param pattern: Input pattern of shape (input_size,)

      :returns: Output pattern of shape (output_size,)



   .. py:method:: train(train_data)

      Train the model using Oja's rule.

      :param train_data: Iterable of input patterns (each of shape (input_size,))



   .. py:attribute:: compiled
      :value: True



   .. py:attribute:: learning_rate
      :value: 0.01



   .. py:attribute:: normalize_weights
      :value: True



   .. py:attribute:: weight_attr
      :value: 'W'



.. py:class:: STDPTrainer(model, learning_rate = 0.01, A_plus = 0.005, A_minus = 0.00525, weight_attr = 'W', w_min = 0.0, w_max = 1.0, compiled = True, **kwargs)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   STDP (Spike-Timing-Dependent Plasticity) trainer.

   STDP is a biologically-inspired learning rule that adjusts synaptic weights
   based on the precise timing of pre- and post-synaptic spikes. Synapses are
   strengthened when pre-synaptic spikes precede post-synaptic spikes (LTP),
   and weakened when the order is reversed (LTD).

   Trace-based Learning Rule:
       ΔW_ij = A_plus * trace_pre[j] * spike_post[i] - A_minus * trace_post[i] * spike_pre[j]

   where:
       - W_ij is the weight from input j to neuron i
       - spike_pre[j] is the presynaptic spike (0 or 1)
       - spike_post[i] is the postsynaptic spike (0 or 1)
       - trace_pre[j] is the exponential trace of presynaptic spikes
       - trace_post[i] is the exponential trace of postsynaptic spikes
       - A_plus controls LTP (long-term potentiation) magnitude
       - A_minus controls LTD (long-term depression) magnitude

   The spike traces evolve as:
       trace = decay * trace + spike

   This provides a temporal window for spike-timing correlations.

   .. rubric:: References

   - Gerstner & Kistler (2002): Spiking Neuron Models
   - Morrison et al. (2008): Phenomenological models of synaptic plasticity
   - Bi & Poo (1998): Synaptic modifications in cultured hippocampal neurons

   Initialize STDP trainer.

   :param model: The spiking model to train (typically SpikingLayer)
   :param learning_rate: Global learning rate multiplier (default: 0.01)
   :param A_plus: LTP magnitude (default: 0.005)
   :param A_minus: LTD magnitude (default: 0.00525, slightly > A_plus for stability)
   :param weight_attr: Name of model attribute holding the connection weights
   :param w_min: Minimum weight value (default: 0.0 for excitatory synapses)
   :param w_max: Maximum weight value (default: 1.0)
   :param compiled: Whether to use JIT-compiled training loop (default: True)
   :param \*\*kwargs: Additional arguments passed to parent Trainer


   .. py:method:: predict(pattern, *args, **kwargs)

      Predict output spikes for a single input spike pattern.

      :param pattern: Input spike pattern of shape (input_size,)

      :returns: Output spike pattern of shape (output_size,) with binary values (0 or 1)



   .. py:method:: train(train_data)

      Train the model using STDP rule.

      :param train_data: Iterable of input spike patterns (each of shape (input_size,))
                         Each pattern should contain binary values (0 or 1)



   .. py:attribute:: A_minus
      :value: 0.00525



   .. py:attribute:: A_plus
      :value: 0.005



   .. py:attribute:: compiled
      :value: True



   .. py:attribute:: learning_rate
      :value: 0.01



   .. py:attribute:: w_max
      :value: 1.0



   .. py:attribute:: w_min
      :value: 0.0



   .. py:attribute:: weight_attr
      :value: 'W'



.. py:class:: SangerTrainer(model, learning_rate = 0.01, normalize_weights = True, weight_attr = 'W', compiled = True, **kwargs)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   Sanger's rule (Generalized Hebbian Algorithm) for multiple PC extraction.

   Extends Oja's rule with Gram-Schmidt orthogonalization to extract multiple
   principal components. Each neuron learns to be orthogonal to all previous ones.

   Learning Rule:
       ΔW_i = η * (y_i * x - y_i * Σ_{j≤i} y_j * W_j)

   where:
       - W_i is the i-th neuron's weight vector
       - y = W @ x is the output vector
       - The sum enforces orthogonality (Gram-Schmidt process)

   This allows sequential extraction of orthogonal principal components,
   with neuron i converging to the i-th principal component.

   Reference:
       Sanger, T. D. (1989). Optimal unsupervised learning in a single-layer
       linear feedforward neural network. Neural Networks, 2(6), 459-473.

   Initialize Sanger trainer.

   :param model: The model to train (typically LinearLayer)
   :param learning_rate: Learning rate η for weight updates
   :param normalize_weights: Whether to normalize weights to unit norm after each update
   :param weight_attr: Name of model attribute holding the connection weights
   :param compiled: Whether to use JIT-compiled training loop (default: True)
   :param \*\*kwargs: Additional arguments passed to parent Trainer


   .. py:method:: predict(pattern, *args, **kwargs)

      Predict output for a single input pattern.

      :param pattern: Input pattern of shape (input_size,)

      :returns: Output pattern of shape (output_size,)



   .. py:method:: train(train_data)

      Train the model using Sanger's rule.

      :param train_data: Iterable of input patterns (each of shape (input_size,))



   .. py:attribute:: compiled
      :value: True



   .. py:attribute:: learning_rate
      :value: 0.01



   .. py:attribute:: normalize_weights
      :value: True



   .. py:attribute:: weight_attr
      :value: 'W'



.. py:class:: Trainer(model = None, *, show_iteration_progress = False, compiled_prediction = True)

   Bases: :py:obj:`abc.ABC`


   Abstract base class for training utilities in CANNs.


   .. py:method:: configure_progress(*, show_iteration_progress = None, compiled_prediction = None)

      Update progress-related flags for derived trainers.



   .. py:method:: predict(pattern, *args, **kwargs)
      :abstractmethod:


      Predict an output for a single pattern.



   .. py:method:: predict_batch(patterns, *args, **kwargs)

      Predict outputs for multiple patterns using ``predict``.



   .. py:method:: train(train_data)
      :abstractmethod:


      Train the associated model with the provided dataset.



   .. py:attribute:: compiled_prediction
      :value: True



   .. py:attribute:: model
      :value: None



   .. py:attribute:: show_iteration_progress
      :value: False



