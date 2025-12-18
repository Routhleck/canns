================
Task Generators
================

This document explains the task generation philosophy and available paradigms in the CANNs library.

Overview
========

The task module (``canns.task``) generates experimental data for CANN simulations with support for saving, loading, importing, and visualization. It provides standardized paradigms that abstract common experimental scenarios, ensuring reproducibility and convenience.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📊 Generate Input Sequences
      :class-header: bg-primary text-white text-center

      Create time-varying external inputs that drive network dynamics

   .. grid-item-card:: 🎯 Provide Ground Truth
      :class-header: bg-success text-white text-center

      Supply trajectory information for analysis and comparison

Task Categories
===============

Tasks are organized into two main categories based on the cognitive function they model.

.. tab-set::

   .. tab-item:: 📍 Tracking Tasks

      **Tracking tasks simulate scenarios where the network follows an external signal.** The bump of activity in the CANN tracks a moving stimulus position.

      .. grid:: 1
         :gutter: 2

         .. grid-item-card:: 🔵 Population Coding
            :class-header: bg-light

            Network receives static input at a fixed location. Tests basic attractor stability and population representation accuracy.

         .. grid-item-card:: 🎯 Template Matching
            :class-header: bg-light

            Network receives brief, possibly noisy inputs. Tests pattern completion and recognition capabilities.

         .. grid-item-card:: 🔄 Smooth Tracking
            :class-header: bg-light

            **Most common paradigm**

            Network receives continuously moving input signals. Tests dynamic tracking ability with varying speeds and directions.

            **Available implementations:**

            * ``SmoothTracking1D``: One-dimensional tracking for ring networks
            * ``SmoothTracking2D``: Two-dimensional tracking for torus networks (under development)

   .. tab-item:: 🧭 Navigation Tasks

      **Navigation tasks simulate spatial movement scenarios** :cite:p:`mcnaughton2006path` where the network receives velocity or heading information rather than direct position inputs through path integration :cite:p:`etienne2004path,samsonovich1997path`.

      .. grid:: 1
         :gutter: 2

         .. grid-item-card:: 🔁 Closed-Loop Navigation
            :class-header: bg-light

            Network updates its internal representation based on self-motion signals. Feedback from the environment can correct errors.

         .. grid-item-card:: ➡️ Open-Loop Navigation
            :class-header: bg-light

            Network integrates velocity inputs without external feedback. Tests path integration capabilities and accumulation of errors over time.

      .. note::

         Navigation tasks do not require direct model coupling because they provide richer data (velocity, angles, etc.) that users interpret based on their specific application.

Model-Task Coupling
===================

Why Coupling Exists
-------------------

Tracking tasks require a CANN model instance to be passed during construction::

   task = SmoothTracking1D(cann_instance=cann, ...)

.. important::

   This coupling exists because tracking tasks need access to ``cann.get_stimulus_by_pos()``. This method converts abstract position coordinates into concrete neural input patterns that match the network's encoding scheme.

   **The coupling provides user convenience:**

   * Automatic stimulus generation matching network topology
   * Consistent encoding between task and model
   * Reduced boilerplate for common use cases

When Coupling Is Required
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Task Type
     - Requires ``cann_instance``
     - Data Provided
   * - **Tracking tasks**
     - ✅ Yes
     - Input patterns in neural space
   * - | Population Coding
       | Template Matching
       | Smooth Tracking
     - ✅ Yes
     - Use ``model.get_stimulus_by_pos()`` internally
   * - **Navigation tasks**
     - ❌ No
     - Velocity, heading, position data
   * - | Closed-Loop Navigation
       | Open-Loop Navigation
     - ❌ No
     - Users decide how to convert to neural inputs

.. admonition:: Design Rationale
   :class: note

   This distinction reflects the different nature of these paradigms. Tracking involves direct sensory input to the network, while navigation involves internal state updates based on self-motion.

Task Components
===============

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⚙️ Task Configuration
      :class-header: bg-light text-center

      Tasks are configured through constructor parameters:

      * **Target positions**: Where the stimulus appears or moves to
      * **Durations**: How long each segment lasts
      * **Time step**: Temporal resolution (from ``bm.get_dt()``)
      * **Additional parameters**: Speed profiles, noise levels, initial conditions

   .. grid-item-card:: 📊 Data Generation
      :class-header: bg-light text-center

      The ``get_data()`` method returns:

      * **Input sequence**: Array of neural inputs over time (for tracking tasks)
      * **Trajectory information**: Position, velocity, time stamps
      * **Metadata**: Task parameters for documentation

   .. grid-item-card:: 💾 Data Persistence
      :class-header: bg-light text-center

      Tasks support saving and loading:

      * ``save(filename)``: Store task data for reproducibility
      * ``load(filename)``: Reload previously generated tasks
      * Standard formats ensure compatibility

   .. grid-item-card:: 📥 Trajectory Import
      :class-header: bg-light text-center

      **Feature under development**

      The library supports importing external trajectories from experimental recordings. This enables:

      * Replay of real animal movement paths
      * Validation against experimental data
      * Comparison of model predictions with neural recordings

Task Usage Patterns
====================

Standard Workflow
-----------------

.. admonition:: Typical Usage Steps
   :class: tip

   1. **Create model instance**
   2. **Configure task** with positions and durations
   3. **Generate data** using ``get_data()``
   4. **Run simulation** feeding task inputs to model
   5. **Analyze results** comparing model output to task trajectory

Multiple Trial Generation
--------------------------

Tasks support generating multiple trials with:

* Same paradigm, different random seeds
* Systematic parameter variations
* Batch processing capabilities

Parameter Sweeps
----------------

Combine tasks with analysis pipelines to:

* Test model robustness across conditions
* Find optimal parameter ranges
* Characterize attractor properties

Design Considerations
=====================

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⏱️ Time Step Consistency
      :class-header: bg-warning text-dark text-center

      Tasks use ``bm.get_dt()`` to ensure temporal resolution matches the simulation environment.

      **Always set the global time step before creating tasks:**

      .. code-block:: python

         bm.set_dt(0.1)
         task = SmoothTracking1D(...)

   .. grid-item-card:: 🎯 Position Encoding
      :class-header: bg-info text-white text-center

      Tasks operate in abstract feature space (angles, coordinates). The conversion to neural activity patterns is handled by:

      * ``model.get_stimulus_by_pos()`` for direct coupling
      * User-defined encoding for decoupled scenarios

   .. grid-item-card:: 🔧 Extensibility
      :class-header: bg-success text-white text-center
      :columns: 12

      Custom tasks can be created by:

      * Inheriting from base task classes
      * Implementing required data generation methods
      * Following conventions for output formats

Summary
=======

The task module provides:

.. grid:: 2 2 2 4
   :gutter: 2

   .. grid-item-card::
      :class-header: bg-light text-center

      1️⃣
      ^^^
      **Tracking Tasks**: Direct stimulus following (population coding, template matching, smooth tracking)

   .. grid-item-card::
      :class-header: bg-light text-center

      2️⃣
      ^^^
      **Navigation Tasks**: Self-motion integration (closed-loop, open-loop navigation)

   .. grid-item-card::
      :class-header: bg-light text-center

      3️⃣
      ^^^
      **Model Coupling**: Automatic stimulus generation for tracking tasks

   .. grid-item-card::
      :class-header: bg-light text-center

      4️⃣
      ^^^
      **Flexibility**: Navigation tasks allow user-defined input interpretation

Tasks abstract experimental paradigms into reusable components—enabling systematic study of CANN dynamics across standardized conditions. The coupling design balances convenience for common cases with flexibility for specialized applications.
