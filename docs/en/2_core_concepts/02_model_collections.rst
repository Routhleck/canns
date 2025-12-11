==================
Model Collections
==================

This document explains the different categories of models in the CANNs library and how to extend them.

Overview
========

The models module ( ``canns.models`` ) implements various CANN architectures and their variants. Models are organized into three categories:

**Basic Models** ( ``canns.models.basic`` )
   Standard CANN implementations and variants

**Brain-Inspired Models** ( ``canns.models.brain_inspired`` )
   Models with biological learning mechanisms

**Hybrid Models** ( ``canns.models.hybrid`` )
   Combinations of CANN with artificial neural networks

All models are built on BrainPy's dynamics framework, which provides state management, time stepping, and JIT compilation capabilities.

Basic Models
============

Basic models implement the core CANN dynamics as described in theoretical neuroscience literature. They use predefined connectivity patterns (typically Gaussian kernels) and fixed parameters.

Available Basic Models
----------------------

Models are organized by module files in ``canns.models.basic``:

Origin CANN (cann.py)
~~~~~~~~~~~~~~~~~~~~~

Core continuous attractor neural network implementations.

``CANN1D``
   One-dimensional continuous attractor network. Default 512 neurons arranged on a ring. Gaussian recurrent connections. Suitable for head direction encoding, angular variables.

``CANN1D_SFA``
   CANN1D with Spike Frequency Adaptation. Adds activity-dependent negative feedback, enables self-sustained wave propagation. Useful for modeling intrinsic dynamics.

``CANN2D``
   Two-dimensional continuous attractor network. Neurons arranged on a torus. Suitable for place field encoding, spatial variables.

``CANN2D_SFA``
   CANN2D with Spike Frequency Adaptation. Supports 2D traveling waves.

Hierarchical Path Integration Model (hierarchical_model.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hierarchical models combining multiple cell types for spatial cognition.

``GaussRecUnits``
   Recurrent units with Gaussian connectivity.

``NonRecUnits``
   Non-recurrent units for comparison.

``BandCell``
   Band cell for 1D path integration.

``GridCell``
   Single grid cell module with multiple scales.

``HierarchicalPathIntegrationModel``
   Full path integration system with grid and place cells.

``HierarchicalNetwork``
   Combines multiple cell types for spatial cognition.

Theta Sweep Model (theta_sweep_model.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models designed for theta rhythm analysis and spatial navigation studies.

``DirectionCellNetwork``
   Head direction cell network.

``GridCellNetwork``
   Network of grid cell modules.

``PlaceCellNetwork``
   Place cell network based on grid cell inputs.

Implementing Basic Models
--------------------------

Every basic model inherits from ``canns.models.basic.BasicModel`` or ``canns.models.basic.BasicModelGroup``.

Constructor Setup
~~~~~~~~~~~~~~~~~

Call the parent constructor with the total neuron count::

   super().__init__(math.prod(shape), **kwargs)

Store shape information in ``self.shape`` and ``self.varshape`` for proper dimensional handling.

Required Methods
~~~~~~~~~~~~~~~~

**Connection Matrix** ( ``make_conn()`` )
   Generate the recurrent connection matrix. Typical implementation uses Gaussian kernels:

   - Compute pairwise distances between neurons
   - Apply Gaussian function with specified width
   - Store result in ``self.conn_mat``

   See ``src/canns/models/basic/cann.py`` for reference implementations.

**Stimulus Generation** ( ``get_stimulus_by_pos(pos)`` )
   Convert feature space positions into external input patterns. Called by task modules to generate neural inputs:

   - Takes position coordinates as input
   - Returns a stimulus vector matching network size
   - Uses Gaussian bump or similar localized pattern

**State Initialization** ( ``init_state()`` )
   Register state variables using BrainPy containers:

   - ``self.u`` : Membrane potential ( ``bm.Variable`` )
   - ``self.r`` : Firing rate ( ``bm.Variable`` )
   - ``self.inp`` : External input ( ``bm.Variable`` )

   Additional states for variants (e.g., ``self.v`` for SFA).

**Update Dynamics** ( ``update(inputs)`` )
   Define single-step state evolution:

   - Read current states
   - Compute derivatives based on CANN equations
   - Apply time step: ``new_state = old_state + derivative * bm.get_dt()``
   - Write updated states

**Diagnostic Properties**
   Expose useful information for analysis:

   - ``self.x`` : Feature space coordinates
   - ``self.rho`` : Neuron density
   - Peak detection methods for bump tracking

Brain-Inspired Models
=====================

Brain-inspired models feature biologically plausible learning mechanisms. Unlike basic models with fixed weights, these networks modify their connectivity through local, activity-dependent plasticity.

Key Characteristics
-------------------

**Local Learning Rules**
   Weight updates depend only on pre- and post-synaptic activity

**No Error Backpropagation**
   Learning happens without explicit error signals

**Energy-Based Dynamics**
   Network states evolve to minimize an energy function

**Attractor Formation**
   Stored patterns become fixed points of dynamics

Available Brain-Inspired Models
--------------------------------

``AmariHopfieldNetwork``
   Classic associative memory model with binary pattern storage. Hebbian learning for weight formation. Content-addressable memory.

``LinearLayer``
   Linear layer with learnable weights for comparison and testing.

``SpikingLayer``
   Spiking neural network layer with biologically realistic spike dynamics.

Implementing Brain-Inspired Models
-----------------------------------

Inherit from ``canns.models.brain_inspired.BrainInspiredModel`` or ``canns.models.brain_inspired.BrainInspiredModelGroup``.

State and Weight Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register both state variables and trainable weights in ``init_state()`` :

- ``self.s`` : State vector ( ``bm.Variable`` )
- ``self.W`` : Connection weights ( ``bm.Variable`` )

All state and weight variables use ``bm.Variable`` in BrainPy.

Weight Attribute
~~~~~~~~~~~~~~~~

If weights are stored under a different name, override the ``weight_attr`` property::

   @property
   def weight_attr(self):
       return 'W'  # or custom attribute name

Update Dynamics
~~~~~~~~~~~~~~~

Define state evolution under current weights in ``update(...)`` . Typically involves matrix-vector multiplication and activation function.

Energy Function
~~~~~~~~~~~~~~~

Return scalar energy value for current state. Trainers use this to monitor convergence::

   @property
   def energy(self):
       return -0.5 * state @ weights @ state

Hebbian Learning
~~~~~~~~~~~~~~~~

Optional custom implementation of weight updates in ``apply_hebbian_learning(patterns)`` . If not provided, trainer uses default outer product rule::

   W += learning_rate * patterns.T @ patterns

Dynamic Resizing
~~~~~~~~~~~~~~~~

Optional support for changing network size while preserving learned structure: ``resize(num_neurons, preserve_submatrix)``

See ``src/canns/models/brain_inspired/hopfield.py`` for reference implementation.

Hybrid Models
=============

.. note::

   Hybrid models combine CANN dynamics with other neural network architectures (under development). The vision includes:

   - CANN modules embedded in larger artificial neural networks
   - Differentiable CANN layers for end-to-end training
   - Integration of attractor dynamics with feedforward processing
   - Bridging biological plausibility with deep learning capabilities

   Current status: Placeholder module structure exists in ``canns.models.hybrid`` for future implementations.

BrainPy Foundation
==================

All models leverage BrainPy's infrastructure:

Dynamics Abstraction
--------------------

``bp.DynamicalSystem`` provides:

- Automatic state tracking
- JIT compilation support
- Composable submodules

State Containers
----------------

``bm.Variable``
   Universal container for all state variables (mutable, internal, or learnable parameters)

These containers enable transparent JAX transformations while maintaining intuitive object-oriented syntax.

Time Management
---------------

``brainpy.math`` provides time step management:

- ``bm.set_dt(0.1)`` : Set simulation time step
- ``bm.get_dt()`` : Retrieve current time step

This ensures consistency across models, tasks, and trainers.

Compiled Simulation
-------------------

``bm.for_loop`` enables efficient simulation:

- JIT compilation for GPU/TPU acceleration
- Automatic differentiation support
- Progress tracking integration

Summary
=======

The CANNs model collection provides:

1. **Basic Models** - Standard CANN implementations for immediate use
2. **Brain-Inspired Models** - Networks with local learning capabilities
3. **Hybrid Models** - Future integration with deep learning (in development)

Each category follows consistent patterns through base class inheritance, making the library both powerful and extensible. The BrainPy foundation handles complexity, allowing users to focus on defining neural dynamics rather than implementation details.
