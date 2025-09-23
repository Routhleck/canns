src.canns.trainer
=================

.. py:module:: src.canns.trainer

.. autoapi-nested-parse::

   Training utilities for CANNS models.

   The module exposes the abstract ``Trainer`` base class and concrete implementations
   such as ``HebbianTrainer``.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/trainer/hebbian/index


Classes
-------

.. autoapisummary::

   src.canns.trainer.HebbianTrainer
   src.canns.trainer.Trainer


Package Contents
----------------

.. py:class:: HebbianTrainer(model, show_iteration_progress = False, compiled_prediction = True, *, weight_attr = 'W', subtract_mean = True, zero_diagonal = True, normalize_by_patterns = True, prefer_generic = True, state_attr = None, prefer_generic_predict = True, preserve_on_resize = True)

   Bases: :py:obj:`src.canns.trainer._base.Trainer`


   Generic Hebbian trainer with progress reporting.

   Overview
   - Uses a model-exposed weight parameter (default attribute name: ``W``) to apply a
     standard Hebbian update. If unavailable, falls back to the model's
     ``apply_hebbian_learning``.
   - Works with models that expose a parameter object with a ``.value`` ndarray of shape
     (N, N) (e.g., ``brainstate.ParamState``).

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



.. py:class:: Trainer(model = None, *, show_iteration_progress = False, compiled_prediction = True)

   Bases: :py:obj:`abc.ABC`


   Abstract base class for training utilities in CANNS.


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



