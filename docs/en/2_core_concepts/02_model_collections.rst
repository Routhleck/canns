==================
Model Collections
==================

This document explains the different categories of models in the CANNs library and how to extend them.

Overview
========

The models module (``canns.models``) implements various CANN architectures and their variants. Models are organized into three categories:

**Basic Models** (``canns.models.basic``)
   Standard CANN implementations and variants

**Brain-Inspired Models** (``canns.models.brain_inspired``)
   Models with biological learning mechanisms

**Hybrid Models** (``canns.models.hybrid``)
   Combinations of CANN with artificial neural networks

All models are built on BrainPy's dynamics framework, which provides state management, time stepping, and JIT compilation capabilities.

Basic Models
============

Basic models implement the mathematically tractable and canonical continuous attractor neural network called the **Wu-Amari-Wong (WAW)** model :cite:p:`amari1977dynamics,wu2008dynamics,fung2010moving,wu2016continuous` as described in theoretical neuroscience literature. They use predefined connectivity patterns (typically Gaussian kernels) and fixed parameters.

Available Basic Models
----------------------

Models are organized by module files in ``canns.models.basic``:

Origin CANN (cann.py)
~~~~~~~~~~~~~~~~~~~~~

Core continuous attractor neural network implementations.

``CANN1D``
   One-dimensional continuous attractor network. Defaults to 512 neurons arranged on a ring with Gaussian recurrent connections. Suitable for head direction encoding :cite:p:`taube1990head` and angular variables.

``CANN1D_SFA``
   CANN1D with Spike Frequency Adaptation. It adds activity-dependent negative feedback and enables self-sustained wave propagation. Useful for modeling intrinsic dynamics.

``CANN2D``
   Two-dimensional continuous attractor network with neurons arranged on a torus. Suitable for place field encoding :cite:p:`o1971hippocampus` and spatial variables.

``CANN2D_SFA``
   CANN2D with Spike Frequency Adaptation. Supports 2D traveling waves.

Hierarchical Path Integration Model (hierarchical_model.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hierarchical models :cite:p:`chu2025localized` combining multiple cell types for spatial cognition.

``GaussRecUnits``
   Recurrent units with Gaussian connectivity.

``NonRecUnits``
   Non-recurrent units for comparison.

``BandCell``
   Band cell for 1D path integration.

``GridCell``
   Single grid cell :cite:p:`hafting2005microstructure` module with multiple scales.

``HierarchicalPathIntegrationModel``
   Full path integration :cite:p:`mcnaughton2006path` system with grid and place cells.

``HierarchicalNetwork``
   Combines multiple cell types for spatial cognition.

Theta Sweep Model (theta_sweep_model.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models designed for theta :cite:p:`chu2024firing,ji2025systems` rhythm analysis and spatial navigation studies :cite:p:`mi2014spike,li2025dynamics,ji2025systems`.

``DirectionCellNetwork`` :cite:p:`ji2025phase`
   Head direction cell network.

``GridCellNetwork`` :cite:p:`ji2025systems`
   Network of grid cell modules.

``PlaceCellNetwork`` :cite:p:`chu2024firing`
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

**Connection Matrix** (``make_conn()``)
   Generate the recurrent connection matrix. Typical implementation uses Gaussian kernels:

   - Compute pairwise distances between neurons
   - Apply Gaussian function with specified width
   - Store result in ``self.conn_mat``

   See ``src/canns/models/basic/cann.py`` for reference implementations.

**Stimulus Generation** (``get_stimulus_by_pos(pos)``)
   Convert feature space positions into external input patterns. Called by task modules to generate neural inputs:

   - Takes position coordinates as input
   - Returns a stimulus vector matching network size
   - Uses Gaussian bump or similar localized pattern

**Update Dynamics** (``update(inputs)``)
   Define single-step state evolution:

   - Read current states
   - Compute derivatives based on CANN equations
   - Apply time step: ``new_state = old_state + derivative * bm.get_dt()``
   - Write updated states

**Diagnostic Properties**
   Expose useful information for analysis:

   - ``self.x``: Feature space coordinates
   - ``self.rho``: Neuron density
   - Peak detection methods for bump tracking

Brain-Inspired Models
=====================

Brain-inspired models use biologically plausible learning mechanisms. Unlike basic models with fixed weights, these networks modify their connectivity through local, activity-dependent plasticity.

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
   Classic associative memory model :cite:p:`amari1977neural,hopfield1982neural` with binary pattern storage. Hebbian learning :cite:p:`hebb2005organization` for weight formation. Content-addressable memory.

``LinearLayer``
   Linear layer with learnable weights for comparison and testing. Supports various unsupervised learning rules including Oja's rule :cite:p:`oja1982simplified` for principal component extraction and Sanger's rule :cite:p:`sanger1989optimal` for multiple principal components.

``SpikingLayer``
   Spiking neural network layer with biologically realistic spike dynamics.

Implementing Brain-Inspired Models
-----------------------------------

Inherit from ``canns.models.brain_inspired.BrainInspiredModel`` or ``canns.models.brain_inspired.BrainInspiredModelGroup``.

State and Weight Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define state variables and trainable weights:

- ``self.s``: State vector (``bm.Variable``)
- ``self.W``: Connection weights (``bm.Variable``)

All state and weight variables use ``bm.Variable`` in BrainPy.

Weight Attribute
~~~~~~~~~~~~~~~~

If weights are stored under a different name, override the ``weight_attr`` property::

   @property
   def weight_attr(self):
       return 'W'  # or custom attribute name

Update Dynamics
~~~~~~~~~~~~~~~

Define state evolution under current weights in ``update(...)``. Typically involves matrix-vector multiplication and activation function.

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

All models leverage BrainPy's :cite:p:`wang2023brainpy` infrastructure:

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

These containers enable transparent JAX :cite:p:`jax2018github` transformations while maintaining intuitive object-oriented syntax.

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
