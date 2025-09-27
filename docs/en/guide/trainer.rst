Hebbian Memory Examples
=======================

The ``examples/brain_inspired/`` scripts highlight how to drive
:class:`~src.canns.trainer.HebbianTrainer` with real data. They also illustrate common
pre-processing steps for pattern memories.

hopfield_train.py
-----------------

- **Location**: ``examples/brain_inspired/hopfield_train.py``
- **Scenario**: Convert a handful of ``skimage`` images into pattern memories, train a Hopfield network, then recover noisy inputs.
- **Key steps**:

  1. ``preprocess_image`` resizes and thresholds an image to a 128×128 vector in ``{-1, +1}``.
  2. Instantiate :class:`~src.canns.models.brain_inspired.hopfield.AmariHopfieldNetwork`
     (synchronous updates, ``sign`` activation).
  3. Initialise ``HebbianTrainer(model)`` and call ``trainer.train(data_list)``.
  4. Corrupt each pattern by flipping ~30% of the pixels and run ``trainer.predict_batch``.
  5. Use Matplotlib to compare train/input/output panels (saved as ``discrete_hopfield_train.png``).
- **Extensions**:

  - Toggle ``asyn=True`` to observe asynchronous convergence.
  - Set ``normalize_by_patterns=False`` to keep the magnitude of original patterns.

hopfield_train_mnist.py
-----------------------

- **Location**: ``examples/brain_inspired/hopfield_train_mnist.py``
- **Scenario**: Load a small selection of MNIST digits (with a series of fallbacks) and store them in a Hopfield network.
- **Key steps**:

  1. ``_load_mnist()`` attempts Hugging Face ``datasets``, TorchVision, Keras, and finally scikit-learn digits.
  2. Choose one exemplar per digit class and convert with ``_threshold_to_pm1``.
  3. Train with ``trainer.train(patterns)``.
  4. Run ``trainer.predict`` on clean held-out samples to confirm retrieval.
  5. Plot the train/input/output triads via ``plt.subplots``.
- **Extensions**:

  - Introduce noisy test images to measure robustness.
  - Enable ``trainer.configure_progress(show_iteration_progress=True)`` to log energy convergence.

Custom models
-------------

- To plug in your own model, follow the :class:`~src.canns.models.brain_inspired.BrainInspiredModel` interface:
  expose ``W`` and ``s`` states (or override ``weight_attr`` / ``predict_state_attr``) and provide ``update``/``energy``.
- Experiment with :class:`~src.canns.models.brain_inspired.hopfield.AmariHopfieldNetwork` parameters
  such as ``activation`` or ``temperature`` to study the effect on energy descent.
