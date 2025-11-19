========================
Brain-Inspired Training
========================

This document explains the brain-inspired learning mechanisms and the Trainer framework in the CANNs library.

Overview
========

The trainer module (``canns.trainer``) provides a unified interface for training brain-inspired models using biologically plausible learning rules. Unlike conventional deep learning with backpropagation, these methods rely on local, activity-dependent plasticity.

Core Concept: Activity-Dependent Plasticity
============================================

.. important::

   The unifying principle behind brain-inspired learning is that **synaptic modifications depend on neural activity patterns** rather than explicit error signals.

Key Characteristics
-------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🎯 Local Information Only
      :class-header: bg-primary text-white text-center

      Weight changes depend on:

      * Pre-synaptic neuron activity
      * Post-synaptic neuron activity
      * Possibly local neuromodulatory signals

      **No global error signal propagates through the network**

   .. grid-item-card:: 📊 Correlation-Based Learning
      :class-header: bg-success text-white text-center

      Synapses strengthen when pre- and post-synaptic neurons are co-active. This captures statistical regularities in input patterns.

   .. grid-item-card:: 🌱 Self-Organization
      :class-header: bg-info text-white text-center
      :columns: 12

      Network structure emerges from experience. Attractor patterns form naturally from repeated exposure to similar inputs.

Learning Rules
==============

The library supports several classic learning rules, each capturing different aspects of biological synaptic plasticity.

.. tab-set::

   .. tab-item:: 🧠 Hebbian Learning

      **The foundational principle: "Neurons that fire together wire together."**

      .. grid:: 2

         .. grid-item::

            **Mechanism**

            When two connected neurons are simultaneously active, the synapse between them strengthens.

            Mathematically::

               ΔW_ij ∝ r_i × r_j

            Where ``r_i`` is pre-synaptic activity and ``r_j`` is post-synaptic activity.

         .. grid-item::

            **Use Cases**

            * Pattern storage in associative memories
            * Unsupervised feature learning
            * Self-organizing maps

   .. tab-item:: ⚡ STDP

      **Spike-Timing Dependent Plasticity**

      Weight changes depend on the precise timing of pre- and post-synaptic spikes.

      .. grid:: 2

         .. grid-item::

            **Mechanism**

            * **Pre before Post**: Potentiation (strengthen synapse)
            * **Post before Pre**: Depression (weaken synapse)
            * Magnitude depends on time difference

         .. grid-item::

            **Use Cases**

            * Temporal sequence learning
            * Causal relationship detection
            * Input timing-sensitive tasks

   .. tab-item:: 📈 BCM

      **Bienenstock-Cooper-Munro Rule**

      Weight changes depend on post-synaptic activity relative to a sliding threshold.

      .. grid:: 2

         .. grid-item::

            **Mechanism**

            * **Above threshold**: Potentiation
            * **Below threshold**: Depression
            * Threshold adapts based on average activity

         .. grid-item::

            **Use Cases**

            * Selectivity development
            * Preventing runaway excitation
            * Stable learning with homeostasis

Trainer Framework
=================

Design Rationale
----------------

.. note::

   The Trainer class is separated from Model classes for several reasons:

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: 🔧 Separation of Concerns
      :class-header: bg-light

      Models define dynamics. Trainers define learning. This separation allows:

      * Same model architecture with different learning rules
      * Same learning rule applied to different models
      * Cleaner code organization

   .. grid-item-card:: 🔄 Swappable Learning Rules
      :class-header: bg-light

      Easily experiment with different plasticity mechanisms:

      * Hebbian for some experiments
      * STDP for temporal tasks
      * Custom rules for specific hypotheses

   .. grid-item-card:: 🎯 Unified API
      :class-header: bg-light

      All trainers follow the same interface:

      * ``train(train_data)``: Main training loop
      * ``predict(pattern)``: Single pattern inference
      * Configuration methods for progress display

Implementing Custom Trainers
-----------------------------

To create a new trainer, inherit from ``canns.trainer.Trainer`` and implement:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - **Constructor**
     - Store target model reference and configuration (model instance, learning rate, progress settings)
   * - ``train(self, train_data)``
     - Define parameter update strategy: iterate over training patterns, apply learning rule, track progress
   * - ``predict(self, pattern, *args, **kwargs)``
     - Define single-sample inference: present pattern, allow dynamics to evolve, return final state
   * - ``predict_batch(self, patterns)``
     - Optional batch inference wrapper: call predict() for each pattern, aggregate results
   * - ``configure_progress(...)``
     - Standard progress configuration: enable/disable progress bars, toggle JIT compilation

Model-Trainer Interaction
--------------------------

.. important::

   Trainers interact with models through agreed-upon attributes:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔗 Weight Access
      :class-header: bg-light text-center

      Trainers expect models to expose weights as ``ParamState``:

      * Default attribute: ``model.W``
      * Custom attribute via ``model.weight_attr`` property
      * Direct modification during learning

   .. grid-item-card:: 💾 State Access
      :class-header: bg-light text-center

      Trainers read network states:

      * ``model.s`` for state vectors
      * ``model.energy`` for convergence monitoring
      * Model-specific diagnostic quantities

   .. grid-item-card:: ⚙️ Initialization
      :class-header: bg-light text-center
      :columns: 12

      Trainers may call:

      * ``model.init_state()`` to reset before each pattern
      * ``model.update()`` during dynamics evolution
      * Model methods for specialized operations

Training Workflow
=================

Typical Usage
-------------

.. admonition:: Standard Training Steps
   :class: tip

   1. 🏗️ **Create Model**

      * Instantiate brain-inspired model
      * Initialize state and weights

   2. 🎓 **Create Trainer**

      * Instantiate appropriate trainer
      * Configure learning parameters

   3. 📊 **Prepare Training Data**

      * Format patterns as arrays
      * Ensure compatibility with model size

   4. ⚡ **Train**

      * Call ``trainer.train(patterns)``
      * Monitor progress and energy

   5. 📈 **Evaluate**

      * Test pattern completion
      * Measure attractor quality
      * Assess storage capacity

Progress Monitoring
-------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📊 Feedback
      :class-header: bg-info text-white text-center

      Trainers provide:

      * Current training iteration
      * Network energy evolution
      * Convergence indicators

   .. grid-item-card:: ⚡ Optional Compilation
      :class-header: bg-warning text-dark text-center

      User-controlled trade-off:

      * JIT compile for faster training
      * Disable for debugging
      * Configurable via ``configure_progress``

Advanced Topics
===============

.. tab-set::

   .. tab-item:: 🔬 Custom Learning Rules

      Beyond standard rules, users can implement:

      * Homeostatic mechanisms
      * Competition-based learning
      * Modulatory gating

      Simply override the weight update logic in custom Trainer subclass.

   .. tab-item:: 📊 Capacity and Generalization

      Brain-inspired training raises questions:

      * How many patterns can be stored?
      * What determines retrieval accuracy?
      * How does noise affect performance?

      The trainer framework supports systematic studies of these properties.

   .. tab-item:: 🔗 Integration with Analysis

      After training:

      * Visualize learned weight matrices
      * Analyze attractor basins
      * Compare with theoretical predictions

      Analysis tools work seamlessly with trained brain-inspired models.

Summary
=======

The brain-inspired training module provides:

.. grid:: 2 2 2 4
   :gutter: 2

   .. grid-item-card::
      :class-header: bg-primary text-white text-center

      1️⃣
      ^^^
      **Activity-Dependent Plasticity**: Local learning based on neural correlations

   .. grid-item-card::
      :class-header: bg-success text-white text-center

      2️⃣
      ^^^
      **Multiple Learning Rules**: Hebbian, STDP, BCM implementations

   .. grid-item-card::
      :class-header: bg-info text-white text-center

      3️⃣
      ^^^
      **Unified Trainer Interface**: Consistent API for all learning methods

   .. grid-item-card::
      :class-header: bg-warning text-dark text-center

      4️⃣
      ^^^
      **Model-Trainer Separation**: Clean architecture enabling flexibility

This framework enables research into biologically plausible learning mechanisms, associative memory formation, and self-organizing neural systems within the CANN paradigm.
