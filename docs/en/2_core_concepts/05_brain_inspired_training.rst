========================
Brain-Inspired Training
========================

This document explains the brain-inspired learning mechanisms and the Trainer framework in the CANNs library.

Overview
========

The trainer module (``canns.trainer``) provides a unified interface for training brain-inspired models using biologically plausible learning rules. Unlike conventional deep learning with backpropagation, these methods rely on local, activity-dependent plasticity.

Core Concept: Activity-Dependent Plasticity
============================================

The unifying principle behind brain-inspired learning is that **synaptic modifications depend on neural activity patterns** rather than explicit error signals.

Key Characteristics
-------------------

**Local Information Only**
   Weight changes depend on:

   * Pre-synaptic neuron activity
   * Post-synaptic neuron activity
   * Possibly local neuromodulatory signals

   No global error signal propagates through the network.

**Correlation-Based Learning**
   Synapses strengthen when pre- and post-synaptic neurons are co-active. This captures statistical regularities in input patterns.

**Self-Organization**
   Network structure emerges from experience. Attractor patterns form naturally from repeated exposure to similar inputs.

Learning Rules
==============

The library supports several classic learning rules, each capturing different aspects of biological synaptic plasticity.

Hebbian Learning
----------------

**The foundational principle: "Neurons that fire together wire together."** :cite:p:`hebb2005organization`

Mechanism
~~~~~~~~~

When two connected neurons are simultaneously active, the synapse between them strengthens. Mathematically:

.. math::

   \Delta W_{ij} \propto r_i \times r_j

Where :math:`r_i` is pre-synaptic activity and :math:`r_j` is post-synaptic activity.

Use Cases
~~~~~~~~~

* Pattern storage in associative memories
* Unsupervised feature learning
* Self-organizing maps

STDP (Spike-Timing Dependent Plasticity)
-----------------------------------------

Weight changes depend on the precise timing of pre- and post-synaptic spikes :cite:p:`bi1998synaptic`.

Mechanism
~~~~~~~~~

* **Pre before Post**: Potentiation (strengthen synapse)
* **Post before Pre**: Depression (weaken synapse)
* Magnitude depends on time difference

Use Cases
~~~~~~~~~

* Temporal sequence learning
* Causal relationship detection
* Input timing-sensitive tasks

BCM (Bienenstock-Cooper-Munro) Rule
------------------------------------

Weight changes depend on post-synaptic activity relative to a sliding threshold :cite:p:`bienenstock1982theory`.

Mechanism
~~~~~~~~~

* **Above threshold**: Potentiation
* **Below threshold**: Depression
* Threshold adapts based on average activity

Use Cases
~~~~~~~~~

* Selectivity development
* Preventing runaway excitation
* Stable learning with homeostasis

Trainer Framework
=================

Design Rationale
----------------

The Trainer class is separated from Model classes for several reasons:

**Separation of Concerns**
   Models define dynamics. Trainers define learning. This separation allows:

   * Same model architecture with different learning rules
   * Same learning rule applied to different models
   * Cleaner code organization

**Swappable Learning Rules**
   Easily experiment with different plasticity mechanisms:

   * Hebbian for some experiments
   * STDP for temporal tasks
   * Custom rules for specific hypotheses

**Unified API**
   All trainers follow the same interface:

   * ``train(train_data)`` : Main training loop
   * ``predict(pattern)`` : Single pattern inference
   * Configuration methods for progress display

Implementing Custom Trainers
-----------------------------

To create a new trainer, inherit from ``canns.trainer.Trainer`` and implement:

**Constructor**
   Store target model reference and configuration:

   * Model instance to be trained
   * Learning rate parameters
   * Progress display settings

**train(self, train_data)**
   Define parameter update strategy:

   * Iterate over training patterns
   * Apply learning rule to modify weights
   * Track progress and convergence

**predict(self, pattern, \*args, \*\*kwargs)**
   Define single-sample inference:

   * Present pattern to network
   * Allow dynamics to evolve
   * Return final state or readout

**predict_batch(self, patterns)**
   Optional batch inference wrapper:

   * Call ``predict()`` for each pattern
   * Aggregate results
   * Efficient evaluation of test sets

**configure_progress(self, show_progress, use_compile)**
   Standard progress configuration:

   * Enable/disable progress bars
   * Toggle JIT compilation for speed
   * User-friendly training monitoring

Model-Trainer Interaction
--------------------------

Trainers interact with models through agreed-upon attributes:

Weight Access
~~~~~~~~~~~~~

Trainers expect models to expose weights as ``ParamState``:

* Default attribute: ``model.W``
* Custom attribute via ``model.weight_attr`` property
* Direct modification during learning

State Access
~~~~~~~~~~~~

Trainers read network states:

* ``model.s`` for state vectors
* ``model.energy`` for convergence monitoring
* Model-specific diagnostic quantities

Initialization
~~~~~~~~~~~~~~

Trainers may call:

* ``model.init_state()`` to reset before each pattern
* ``model.update()`` during dynamics evolution
* Model methods for specialized operations

Training Workflow
=================

Typical Usage
-------------

1. **Create Model**

   * Instantiate brain-inspired model
   * Initialize state and weights

2. **Create Trainer**

   * Instantiate appropriate trainer
   * Configure learning parameters

3. **Prepare Training Data**

   * Format patterns as arrays
   * Ensure compatibility with model size

4. **Train**

   * Call ``trainer.train(patterns)``
   * Monitor progress and energy

5. **Evaluate**

   * Test pattern completion
   * Measure attractor quality
   * Assess storage capacity

Progress Monitoring
-------------------

Trainers provide feedback:

* Current training iteration
* Network energy evolution
* Convergence indicators

Optional compilation:

* JIT compile for faster training
* Disable for debugging
* User-controlled trade-off

Advanced Topics
===============

Custom Learning Rules
---------------------

Beyond standard rules, users can implement:

* Homeostatic mechanisms
* Competition-based learning
* Modulatory gating

Simply override the weight update logic in custom Trainer subclass.

Capacity and Generalization
----------------------------

Brain-inspired training raises questions:

* How many patterns can be stored?
* What determines retrieval accuracy?
* How does noise affect performance?

The trainer framework supports systematic studies of these properties.

Integration with Analysis
--------------------------

After training:

* Visualize learned weight matrices
* Analyze attractor basins
* Compare with theoretical predictions

Analysis tools work seamlessly with trained brain-inspired models.

Summary
=======

The brain-inspired training module provides:

1. **Activity-Dependent Plasticity** - Local learning based on neural correlations
2. **Multiple Learning Rules** - Hebbian, STDP, BCM implementations
3. **Unified Trainer Interface** - Consistent API for all learning methods
4. **Model-Trainer Separation** - Clean architecture enabling flexibility

This framework enables research into biologically plausible learning mechanisms, associative memory formation, and self-organizing neural systems—all within the CANN paradigm.
