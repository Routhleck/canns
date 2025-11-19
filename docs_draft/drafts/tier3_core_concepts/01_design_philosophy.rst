=====================================================
Design Philosophy & Architecture Overview
=====================================================

This document explains the core design principles and module organization of the CANNs library.

Core Design Principles
======================

The CANNs library is built around two fundamental principles that guide its architecture and implementation.

Separation of Concerns
----------------------

The library strictly separates different functional responsibilities into independent modules:

:🏗️ **Models** (``canns.models``):
   Define neural network dynamics and state evolution

:📊 **Tasks** (``canns.task``):
   Generate experimental paradigms and input data

:📈 **Analyzers** (``canns.analyzer``):
   Visualize and analyze simulation results

:🧠 **Trainers** (``canns.trainer``):
   Implement learning rules for brain-inspired models

:🔗 **Pipeline** (``canns.pipeline``):
   Orchestrate complete experimental workflows

Each module focuses on a single responsibility. Models do not generate their own input data. Tasks do not analyze results. Analyzers do not modify model parameters. This separation makes the codebase maintainable, testable, and extensible.

Extensibility Through Base Classes
-----------------------------------

Every major component inherits from abstract base classes that define standard interfaces:

* ``canns.models.basic.BasicModel`` for basic CANN models
* ``canns.models.brain_inspired.BrainInspiredModel`` for brain-inspired models
* ``canns.trainer.Trainer`` for training algorithms

These base classes establish contracts that ensure all implementations work seamlessly with the rest of the library. Users can create custom models, tasks, or trainers by inheriting from these bases and implementing the required methods.

Module Architecture
===================

Four Core Application Scenarios
--------------------------------

The CANNs library supports four distinct workflows, each addressing different research needs. These scenarios demonstrate the modular design and flexibility of the architecture.

.. figure:: ../../_static/canns_scenarios_custom.png
   :align: center
   :width: 100%

   CANNs Four Core Application Scenarios

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔬 Scenario 1: CANN Modeling and Simulation
      :class-header: bg-light text-center

      **The most common workflow for studying continuous attractor dynamics**

      Model Building → Task Data Generation → Simulation Experiment → Model Analysis

      +++

      **Steps:**

      1. **Model Building**: Create and configure CANN (``CANN1D()``, ``CANN2D()``)
      2. **Task Data Generation**: Generate inputs (``SmoothTracking1D``)
      3. **Simulation Experiment**: Run dynamics (``brainstate.compile.for_loop``)
      4. **Model Analysis**: Visualize (``animate_dynamics()``, ``energy_landscape_1d()``)

      **Use cases**: Testing CANN theories, parameter exploration, understanding attractor dynamics

   .. grid-item-card:: 📊 Scenario 2: Data Analysis
      :class-header: bg-light text-center

      **Analyzing experimental or virtual neural recordings**

      Real/Virtual Exp. Data → Data Analysis → Attractor/Dynamics Analysis → Results

      +++

      **Steps:**

      1. **Real/Virtual Exp. Data**: Load spike trains or firing rate data
      2. **Data Analysis**: Process with Data Analyzer tools
      3. **Attractor/Dynamics Analysis**: Apply bump fitting, decoding, TDA methods
      4. **Results**: Extract encoded variables, characterize population activity

      **Use cases**: Validating models against neural recordings, decoding spatial variables

   .. grid-item-card:: 🧠 Scenario 3: Brain-Inspired Learning
      :class-header: bg-light text-center

      **Training networks with biologically plausible learning rules**

      Task Dataset → Brain-Inspired Modeling → Brain-Inspired Training → Evaluation

      +++

      **Steps:**

      1. **Task Dataset**: Prepare training patterns
      2. **Brain-Inspired Modeling**: Create models with learnable weights (``HopfieldNetwork``)
      3. **Brain-Inspired Training**: Apply Hebbian/STDP/BCM learning via subclasses of ``Trainer``
      4. **Evaluation**: Test pattern completion, measure storage capacity

      **Use cases**: Studying associative memory, exploring local learning rules, self-organization

   .. grid-item-card:: 🔗 Scenario 4: End-to-End Pipeline
      :class-header: bg-light text-center

      **Automated experimental workflows from configuration to results**

      Input Config → Pipeline Orchestration → Auto Execution → Output Reports

      +++

      **Steps:**

      1. **Input Config**: Specify parameters via configuration files
      2. **Pipeline Orchestration**: ``canns.pipeline`` coordinates all modules
      3. **Auto Execution**: Automatic model creation, simulation, and analysis
      4. **Output Reports**: Generate comprehensive result summaries

      **Use cases**: Systematic parameter sweeps, reproducible experiments, batch processing

Module Interaction Pattern
---------------------------

.. admonition:: Data Flow Pattern
   :class: note

   Across all scenarios, modules interact following the separation of concerns principle:

   * 🟡 **Input Stage**: Data enters the system (models, datasets, configs)
   * ⚫ **Processing Stage**: Core computations (simulation, training, analysis)
   * 🟢 **Output Stage**: Results visualization and interpretation

   This consistent structure makes the library intuitive while supporting diverse research workflows.

BrainState Integration
----------------------

The CANNs library builds on BrainState (``brainstate``), the core framework from the Brain Simulation Ecosystem. BrainState provides:

:⚙️ **Dynamics Abstraction**:
   ``brainstate.nn.Dynamics`` base class for neural systems

:💾 **State Management**:
   ``brainstate.State``, ``brainstate.HiddenState``, and ``brainstate.ParamState`` containers

:⏱️ **Time Step Control**:
   ``brainstate.environ.set(dt=...)`` for unified temporal management

:⚡ **JIT Compilation**:
   ``brainstate.compile.for_loop`` for high-performance simulation

:🎲 **Random Number Management**:
   ``brainstate.random`` for reproducible stochasticity

With BrainState, CANN models only need to define variables and update equations. Time stepping, parallelization, and compilation are handled automatically, significantly reducing implementation complexity.

Module Relationships
====================

How Modules Interact
--------------------

Model ↔ Task Coupling
~~~~~~~~~~~~~~~~~~~~~~

Some tasks require a model instance to access stimulus generation methods. For example, ``SmoothTracking1D`` needs access to ``model.get_stimulus_by_pos()`` to convert position coordinates into neural input patterns. This coupling is intentional for user convenience but is limited to tracking tasks.

Model ↔ Analyzer Independence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzers work with model outputs (firing rates, membrane potentials) but do not modify model state. They accept simulation results as NumPy arrays and produce visualizations. This independence allows the same analyzer to work with any model that produces compatible outputs.

Model ↔ Trainer Collaboration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trainers modify model parameters (specifically connection weights) according to learning rules. They interact with models through agreed-upon attributes like ``model.W`` for weights and ``model.s`` for state vectors. The trainer framework is designed for brain-inspired models that use local, activity-dependent plasticity.

Pipeline Orchestration
~~~~~~~~~~~~~~~~~~~~~~~

The ``canns.pipeline`` module coordinates all other modules into complete experimental workflows. It manages the full cycle from model setup through task execution to result analysis, providing a high-level interface for common use cases.

Design Trade-offs
=================

Flexibility vs. Convenience
---------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔧 Flexibility
      :class-header: bg-primary text-white

      Advanced users can override any component or create custom implementations

   .. grid-item-card:: 🚀 Convenience
      :class-header: bg-success text-white

      Standard workflows should require minimal boilerplate

This balance is achieved through sensible defaults combined with extensive customization options. For example, ``CANN1D()`` uses default parameters that work for most cases, but every parameter can be explicitly specified.

Performance vs. Simplicity
--------------------------

The library achieves high performance through a multi-layered strategy:

Python Layer (BrainState/JAX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX-based compilation provides GPU/TPU acceleration but requires functional programming patterns. The library abstracts this complexity by:

* Encapsulating JIT compilation in BrainState's ``for_loop``
* Managing state through explicit containers
* Providing utility functions that handle common patterns

Users benefit from GPU/TPU acceleration without writing JAX-specific code directly.

Native Layer (canns-lib)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   For performance-critical operations where Python overhead is significant, the library provides optional Rust-powered backends through ``canns-lib``:

   * **Ripser Module**: Topological Data Analysis with 1.13x average speedup (up to 1.82x) vs. pure Python
   * **Spatial Navigation**: Accelerated RatInABox environments with ~700x speedup for long trajectory integration
   * **Future Modules**: Planned support for approximate nearest neighbors, dynamics computation

   The canns-lib integration follows the same principle: expose simple Python APIs while leveraging native performance for bottleneck operations. Users can opt into these accelerations without changing their code structure.

Extending the Library
=====================

Creating Custom Models
----------------------

To add a new model, inherit from the appropriate base class and implement required methods.

.. tab-set::

   .. tab-item:: Basic Models

      Required methods:

      * ``make_conn()``: Generate connection matrix
      * ``get_stimulus_by_pos()``: Convert positions to input patterns
      * ``init_state()``: Register state variables
      * ``update()``: Define single-step dynamics

   .. tab-item:: Brain-Inspired Models

      Required methods:

      * ``init_state()``: Register state and weight parameters
      * ``update()``: Define state evolution
      * ``energy``: Property returning network energy

Creating Custom Tasks
---------------------

.. tip::

   Tasks should generate input sequences compatible with model expectations. Key considerations:

   * Use ``brainstate.environ.get_dt()`` for time step consistency
   * Return data in formats expected by models
   * Provide trajectory information for analysis

Creating Custom Trainers
-------------------------

.. note::

   Trainers inherit from ``canns.trainer.Trainer`` and implement:

   * ``train()``: Parameter update strategy
   * ``predict()``: Single-sample inference
   * Standard progress and compilation configuration

Summary
=======

The CANNs library achieves its goals through careful architectural choices:

.. grid:: 2 2 2 4
   :gutter: 2

   .. grid-item-card::
      :class-header: bg-light text-center

      1️⃣
      ^^^
      **Separation of concerns** keeps modules focused and independent

   .. grid-item-card::
      :class-header: bg-light text-center

      2️⃣
      ^^^
      **Base class inheritance** ensures consistent interfaces

   .. grid-item-card::
      :class-header: bg-light text-center

      3️⃣
      ^^^
      **BrainState integration** provides performance without complexity

   .. grid-item-card::
      :class-header: bg-light text-center

      4️⃣
      ^^^
      **Flexible coupling** balances convenience with modularity

These principles enable both rapid prototyping and rigorous research while maintaining code quality and extensibility.
