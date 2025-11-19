==================
Model Collections
==================

This document explains the different categories of models in the CANNs library and how to extend them.

Overview
========

The models module (``canns.models``) implements various CANN architectures and their variants. Models are organized into three categories:

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: 🏗️ Basic Models
      :class-header: bg-primary text-white text-center

      ``canns.models.basic``

      Standard CANN implementations and variants

   .. grid-item-card:: 🧠 Brain-Inspired Models
      :class-header: bg-success text-white text-center

      ``canns.models.brain_inspired``

      Models with biological learning mechanisms

   .. grid-item-card:: 🔗 Hybrid Models
      :class-header: bg-info text-white text-center

      ``canns.models.hybrid``

      Combinations of CANN with artificial neural networks

All models are built on BrainState's dynamics framework, which provides state management, time stepping, and JIT compilation capabilities.

Basic Models
============

Basic models implement the core CANN dynamics as described in theoretical neuroscience literature. They use predefined connectivity patterns (typically Gaussian kernels) and fixed parameters.

Available Basic Models
----------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔵 CANN1D
      :class-header: bg-light text-center

      ``canns.models.basic.CANN1D``

      +++

      **One-dimensional continuous attractor network**

      * Default 512 neurons arranged on a ring
      * Gaussian recurrent connections
      * Suitable for head direction encoding, angular variables

   .. grid-item-card:: 🔄 CANN1D_SFA
      :class-header: bg-light text-center

      ``canns.models.basic.CANN1D_SFA``

      +++

      **CANN1D with Spike Frequency Adaptation**

      * Adds activity-dependent negative feedback
      * Enables self-sustained wave propagation
      * Useful for modeling intrinsic dynamics

   .. grid-item-card:: 🟦 CANN2D
      :class-header: bg-light text-center

      ``canns.models.basic.CANN2D``

      +++

      **Two-dimensional continuous attractor network**

      * Neurons arranged on a torus
      * Suitable for place field encoding, spatial variables

   .. grid-item-card:: 🔃 CANN2D_SFA
      :class-header: bg-light text-center

      ``canns.models.basic.CANN2D_SFA``

      +++

      **CANN2D with Spike Frequency Adaptation**

      * Supports 2D traveling waves

   .. grid-item-card:: 🌐 Hierarchical Models
      :class-header: bg-light text-center
      :columns: 12

      **Grid Cells, Place Cells**

      +++

      * Multi-scale attractor networks
      * Based on CANN principles but with more complex connectivity
      * Model spatial cognition hierarchies

Implementing Basic Models
--------------------------

Every basic model inherits from ``canns.models.basic.BasicModel`` or ``canns.models.basic.BasicModelGroup``. Required implementations:

.. important:: Constructor Setup

   Call the parent constructor with the total neuron count::

      super().__init__(math.prod(shape), **kwargs)

   Store shape information in ``self.shape`` and ``self.varshape`` for proper dimensional handling.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🔗 Connection Matrix
      :class-header: bg-light

      **``make_conn()``**

      Generate the recurrent connection matrix. Typical implementation uses Gaussian kernels:

      * Compute pairwise distances between neurons
      * Apply Gaussian function with specified width
      * Store result in ``self.conn_mat``

      See ``src/canns/models/basic/cann.py`` for reference Gaussian kernel implementations.

   .. grid-item-card:: 📍 Stimulus Generation
      :class-header: bg-light

      **``get_stimulus_by_pos(pos)``**

      Convert feature space positions into external input patterns. This method is called by task modules to generate neural inputs. Implementation typically:

      * Takes position coordinates as input
      * Returns a stimulus vector matching network size
      * Uses Gaussian bump or similar localized pattern

   .. grid-item-card:: 🔧 State Initialization
      :class-header: bg-light

      **``init_state()``**

      Register state variables using BrainState containers:

      * ``self.u``: Membrane potential (``brainstate.HiddenState``)
      * ``self.r``: Firing rate (``brainstate.HiddenState``)
      * ``self.inp``: External input (``brainstate.State``)

      Additional states for variants (e.g., ``self.v`` for SFA).

   .. grid-item-card:: ⚡ Update Dynamics
      :class-header: bg-light

      **``update(inputs)``**

      Define single-step state evolution. Key points:

      * Read current states
      * Compute derivatives based on CANN equations
      * Apply time step: ``new_state = old_state + derivative * brainstate.environ.get_dt()``
      * Write updated states

.. tip:: Diagnostic Properties

   Expose useful information for analysis:

   * ``self.x``: Feature space coordinates
   * ``self.rho``: Neuron density
   * Peak detection methods for bump tracking

Brain-Inspired Models
=====================

Brain-inspired models feature biologically plausible learning mechanisms. Unlike basic models with fixed weights, these networks modify their connectivity through local, activity-dependent plasticity.

Key Characteristics
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Characteristic
     - Description
   * - 🎯 Local Learning Rules
     - Weight updates depend only on pre- and post-synaptic activity
   * - 🚫 No Error Backpropagation
     - Learning happens without explicit error signals
   * - ⚖️ Energy-Based Dynamics
     - Network states evolve to minimize an energy function
   * - 🧲 Attractor Formation
     - Stored patterns become fixed points of dynamics

Available Brain-Inspired Models
--------------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: 🧠 HopfieldNetwork
      :class-header: bg-success text-white text-center

      ``canns.models.brain_inspired.hopfield.HopfieldNetwork``

      +++

      **Classic associative memory model**

      * Binary pattern storage
      * Hebbian learning for weight formation
      * Content-addressable memory

Implementing Brain-Inspired Models
-----------------------------------

Inherit from ``canns.models.brain_inspired.BrainInspiredModel`` or ``canns.models.brain_inspired.BrainInspiredModelGroup``. Required implementations:

.. note:: State and Weight Initialization (``init_state()``)

   Register both state variables and trainable weights:

   * ``self.s``: State vector (commonly ``brainstate.HiddenState``)
   * ``self.W``: Connection weights (commonly ``brainstate.ParamState``)

   Using ``ParamState`` for weights allows trainers to modify them during learning.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🏷️ Weight Attribute
      :class-header: bg-light

      **``weight_attr`` property**

      If weights are stored under a different name, override this property to help trainers locate the weight matrix::

         @property
         def weight_attr(self):
             return 'W'  # or custom attribute name

   .. grid-item-card:: 🔄 Update Dynamics
      :class-header: bg-light

      **``update(...)``**

      Define state evolution under current weights. Typically involves matrix-vector multiplication and activation function.

   .. grid-item-card:: ⚡ Energy Function
      :class-header: bg-light

      **``energy`` property**

      Return scalar energy value for current state. Trainers use this to monitor convergence::

         @property
         def energy(self):
             return -0.5 * state @ weights @ state

   .. grid-item-card:: 🧠 Hebbian Learning
      :class-header: bg-light

      **``apply_hebbian_learning(patterns)``**

      Optional custom implementation of weight updates. If not provided, trainer uses default outer product rule::

         W += learning_rate * patterns.T @ patterns

.. tip:: Dynamic Resizing

   Optional support for changing network size while preserving learned structure: ``resize(num_neurons, preserve_submatrix)``

   See ``src/canns/models/brain_inspired/hopfield.py`` for reference implementation.

Hybrid Models
=============

.. admonition:: Under Development
   :class: note

   Hybrid models combine CANN dynamics with other neural network architectures (under development). The vision includes:

   * CANN modules embedded in larger artificial neural networks
   * Differentiable CANN layers for end-to-end training
   * Integration of attractor dynamics with feedforward processing
   * Bridging biological plausibility with deep learning capabilities

   Current status: Placeholder module structure exists in ``canns.models.hybrid`` for future implementations.

BrainState Foundation
=====================

All models leverage BrainState's infrastructure:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⚙️ Dynamics Abstraction
      :class-header: bg-light text-center

      ``brainstate.nn.Dynamics`` provides:

      * Automatic state tracking
      * JIT compilation support
      * Composable submodules

   .. grid-item-card:: 💾 State Containers
      :class-header: bg-light text-center

      * ``brainstate.State``: Mutable during simulation
      * ``brainstate.HiddenState``: Internal network states
      * ``brainstate.ParamState``: Learnable parameters

      These containers enable transparent JAX transformations while maintaining intuitive object-oriented syntax.

   .. grid-item-card:: ⏱️ Time Management
      :class-header: bg-light text-center

      ``brainstate.environ`` provides global configuration:

      * ``brainstate.environ.set(dt=0.1)``: Set simulation time step
      * ``brainstate.environ.get_dt()``: Retrieve current time step

      This ensures consistency across models, tasks, and trainers.

   .. grid-item-card:: ⚡ Compiled Simulation
      :class-header: bg-light text-center

      ``brainstate.compile.for_loop`` enables efficient simulation:

      * JIT compilation for GPU/TPU acceleration
      * Automatic differentiation support
      * Progress tracking integration

Summary
=======

The CANNs model collection provides:

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card::
      :class-header: bg-primary text-white text-center

      1️⃣ Basic Models
      ^^^
      Standard CANN implementations for immediate use

   .. grid-item-card::
      :class-header: bg-success text-white text-center

      2️⃣ Brain-Inspired Models
      ^^^
      Networks with local learning capabilities

   .. grid-item-card::
      :class-header: bg-info text-white text-center

      3️⃣ Hybrid Models
      ^^^
      Future integration with deep learning (in development)

Each category follows consistent patterns through base class inheritance, making the library both powerful and extensible. The BrainState foundation handles complexity, allowing users to focus on defining neural dynamics rather than implementation details.
