================
Analysis Methods
================

This document explains the analysis and visualization tools in the CANNs library.

Overview
========

The analyzer module (``canns.analyzer``) provides tools for visualizing and interpreting both simulation outputs and experimental data. It organizes into distinct components based on data source and analysis type:

Module Structure
----------------

.. admonition:: New Organization (v2.0+)
   :class: note

   The analyzer module is organized by function:

   * **metrics/** - Computational analysis (no matplotlib dependency)

     * ``spatial_metrics`` - Spatial metrics computation
     * ``utils`` - Spike train conversion utilities
     * ``experimental/`` - CANN1D/2D experimental data analysis

   * **visualization/** - Plotting and animation (matplotlib-based)

     * ``config`` - PlotConfig unified configuration system
     * ``spatial_plots`` - Spatial visualizations
     * ``energy_plots`` - Energy landscape visualizations
     * ``spike_plots`` - Raster plots and firing rate plots
     * ``tuning_plots`` - Tuning curve visualizations
     * ``experimental/`` - Experimental data visualizations

   * **slow_points/** - Fixed point analysis
   * **model_specific/** - Specialized model analyzers

.. grid:: 2 2 2 4
   :gutter: 3

   .. grid-item-card:: 📊 Model Analyzer
      :class-header: bg-primary text-white text-center

      Analyze CANN simulation outputs

   .. grid-item-card:: 📈 Data Analyzer
      :class-header: bg-success text-white text-center

      Analyze experimental neural recordings

   .. grid-item-card:: 🔬 RNN Dynamics Analysis
      :class-header: bg-info text-white text-center

      Study fixed points and slow manifolds

   .. grid-item-card:: 🌐 Topological Data Analysis
      :class-header: bg-warning text-dark text-center

      Detect geometric structures in neural activity

Model Analyzer
==============

The Model Analyzer visualizes the outputs of CANN simulations, focusing on network activity patterns and their evolution over time.

Core Capabilities
-----------------

.. tab-set::

   .. tab-item:: 📹 Activity Visualization

      .. list-table::
         :widths: 40 60

         * - ``animate_dynamics()``
           - Animate firing rate evolution over time
         * - ``plot_network_state()``
           - Snapshot of current activity pattern
         * - ``plot_bump_trajectory()``
           - Track bump center position

   .. tab-item:: ⚡ Energy Landscape

      .. list-table::
         :widths: 40 60

         * - ``energy_landscape_1d()``
           - Visualize attractor basin structure
         * - ``energy_landscape_2d()``
           - Two-dimensional energy surface
         * - **Purpose**
           - Show how different states relate to attractor minima

   .. tab-item:: 🔗 Connectivity

      .. list-table::
         :widths: 40 60

         * - ``plot_weight_matrix()``
           - Visualize recurrent connections
         * - ``plot_connection_profile()``
           - Single neuron's connectivity pattern
         * - **Purpose**
           - Reveal Mexican-hat or other kernel structures

Design Philosophy
-----------------

.. important::

   Model Analyzer functions receive simulation results as arrays rather than model objects. This independence means:

   * Same visualizations work across different model types
   * Results can be saved and analyzed later
   * No dependency on model internal structure during analysis

   **Functions accept standardized formats:**

   * Firing rates as ``(time, neurons)`` arrays
   * Membrane potentials as ``(time, neurons)`` arrays
   * Spatial coordinates for bump localization

PlotConfig System
-----------------

.. admonition:: Configuration Pattern
   :class: tip

   The library uses ``PlotConfig`` dataclasses for visualization configuration:

   **Benefits:**

   * ✅ **Reusability**: Same configuration applies to multiple plots
   * ✅ **Type Safety**: Parameters validated at construction
   * ✅ **Sharing**: Pass configuration objects between functions

   **Common configuration includes:**

   * ``figsize``: Figure dimensions
   * ``interval``: Animation speed
   * ``colormap``: Color scheme selection
   * ``show_colorbar``: Toggle color legend

   While PlotConfig provides convenience, direct parameter passing remains supported for backward compatibility.

Data Analyzer
=============

The Data Analyzer processes experimental neural recordings, typically spike trains or firing rate estimates.

Key Differences from Model Analyzer
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Model Analyzer
     - Data Analyzer
   * - **Input Data**
     - Clean simulation outputs
     - Spike trains—sparse, discrete events—and firing rate estimates
   * - **Focus**
     - Visualize CANN dynamics
     - Decode neural activity, fit parametric models
   * - **Noise**
     - Minimal (simulation)
     - Potentially noisy or incomplete recordings

Capabilities
------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📊 Population Activity Analysis
      :class-header: bg-light text-center

      * Estimate bump position from neural population
      * Fit Gaussian profiles to activity patterns
      * Track decoded position over time

   .. grid-item-card:: 🔬 Virtual Data Generation
      :class-header: bg-light text-center

      * Create synthetic spike trains for algorithm testing
      * Generate ground truth scenarios
      * Validate analysis pipelines

   .. grid-item-card:: 📈 Statistical Tools
      :class-header: bg-light text-center

      * Tuning curve estimation
      * Circular statistics for angular variables
      * Error quantification metrics

   .. grid-item-card:: 🎯 Use Cases
      :class-header: bg-light text-center

      * Validate CANN models against experimental recordings
      * Develop decoding algorithms for neural data
      * Test theoretical predictions with simulated experiments

RNN Dynamics Analysis
=====================

This component analyzes recurrent neural networks as dynamical systems :cite:p:`sussillo2013opening`, finding fixed points :cite:p:`golub2018fixedpointfinder` and characterizing the phase space structure.

Purpose
-------

.. note::

   CANN models are continuous-time dynamical systems. Understanding their behavior requires:

   * Identifying stable fixed points (attractors)
   * Finding unstable fixed points (saddles, repellers)
   * Mapping slow manifolds where dynamics concentrate

Methods
-------

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: 📍 Fixed Point Finding
      :class-header: bg-primary text-white

      Locate states where dynamics vanish (du/dt = 0):

      * Numerical root finding
      * Multiple initial conditions for thorough search
      * Classification by stability (eigenvalue analysis)

   .. grid-item-card:: 📊 Stability Analysis
      :class-header: bg-success text-white

      Characterize dynamics near fixed points:

      * Jacobian computation
      * Eigenvalue decomposition
      * Attractor vs. saddle vs. repeller classification

   .. grid-item-card:: 🌀 Slow Manifold Identification
      :class-header: bg-info text-white

      Find low-dimensional structures in state space:

      * Dimensionality reduction
      * Identify directions of slow dynamics
      * Visualize state space organization

Current Scope
-------------

.. admonition:: Implementation Status
   :class: note

   Currently focused on analyzing RNN models (including CANNs as a special case). Provides tools for:

   * Understanding intrinsic network dynamics
   * Characterizing attractor landscapes
   * Studying bifurcations under parameter changes

Topological Data Analysis (TDA)
================================

TDA tools :cite:p:`carlsson2009topology` detect geometric and topological structures in high-dimensional neural activity data using persistent homology :cite:p:`edelsbrunner2010computational`.

Why TDA for CANNs
-----------------

.. important::

   CANN activity patterns often live on low-dimensional manifolds:

   * **Ring attractors**: Activity on a circle (1D torus)
   * **Torus attractors**: Activity on a 2D torus (grid cells)
   * **Sphere attractors**: Activity on a sphere

   Traditional methods may miss these structures. TDA provides mathematically rigorous detection.

Available Tools
---------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔬 Persistent Homology
      :class-header: bg-light text-center

      * Accelerated ripser implementation
      * Detects topological features (loops, voids)
      * Persistence diagrams quantify feature significance

   .. grid-item-card:: 📉 Dimensionality Reduction
      :class-header: bg-light text-center

      * User applies external tools (UMAP, PCA, etc.)
      * Library provides preprocessing utilities
      * Visualization of reduced representations

Use Cases
---------

.. tab-set::

   .. tab-item:: 🧭 Grid Cell Analysis

      Grid cells encode position on a torus. TDA can:

      * ✅ Verify toroidal structure :cite:p:`carlsson2009topology,edelsbrunner2010computational` in neural recordings
      * ✅ Quantify how well activity matches theoretical prediction
      * ✅ Detect deviations from ideal topology

   .. tab-item:: 🔍 Attractor Structure Discovery

      For unknown networks:

      * ✅ Infer attractor geometry from activity patterns
      * ✅ Test hypotheses about encoding manifolds
      * ✅ Compare experimental data with model predictions

Implementation Notes
---------------------

.. admonition:: Technical Details
   :class: tip

   * **Ripser integration** for fast persistent homology :cite:p:`carlsson2009topology,edelsbrunner2010computational`
   * **External dependencies** for some advanced methods
   * **Focus on tools** relevant to attractor network research

   See the ``canns-lib`` Ripser module for performance details (1.13x average speedup, up to 1.82x).

Summary
=======

The analysis module provides comprehensive tools for:

.. grid:: 2 2 2 4
   :gutter: 2

   .. grid-item-card::
      :class-header: bg-primary text-white text-center

      1️⃣
      ^^^
      **Model Analyzer**: Visualize CANN simulation outputs with standardized functions

   .. grid-item-card::
      :class-header: bg-success text-white text-center

      2️⃣
      ^^^
      **Data Analyzer**: Process experimental recordings and synthetic neural data

   .. grid-item-card::
      :class-header: bg-info text-white text-center

      3️⃣
      ^^^
      **RNN Dynamics**: Study fixed points and phase space structure

   .. grid-item-card::
      :class-header: bg-warning text-dark text-center

      4️⃣
      ^^^
      **TDA**: Detect topological properties of neural representations

These tools enable both forward modeling (simulation analysis) and reverse engineering (experimental data interpretation)—supporting the full research cycle from theory to validation.
