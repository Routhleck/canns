Scenario 2: Data Analysis and Neural Decoding
============================================

A comprehensive tutorial on experimental neural data analysis, topological decoding, and RNN dynamics.

Tutorial List
-------------

.. toctree::
   :maxdepth: 1
   :caption: Experimental Data Analysis

   01_asa_pipeline
   02_cann1d_bump_fit
   04_cell_classification

.. toctree::
   :maxdepth: 1
   :caption: RNN Dynamics Analysis

   03_flipflop_tutorial

Tutorial Overview
-----------------

**ASA Pipeline Tutorial**

Covers the complete pipeline from ``spike/x/y/t`` input to TDA, decoding, CohoMap/CohoSpace/PathCompare, and FR/FRM, with corresponding example scripts in the repository.

**1D CANN ROI Bump Fitting Tutorial**

Demonstrates how to use ``roi_bump_fits`` to extract bump parameters and generate animations for analyzing ring attractor dynamics.

**Cell Classification Tutorial**

Illustrates a cell classification workflow based on GridScore and autocorrelation features, including examples of single-cell scoring and grid module segmentation.

**RNN Fixed-Point Analysis Tutorial (FlipFlop Task)**

This tutorial provides a detailed guide on using the ``FixedPointFinder`` tool to analyze the dynamical properties of recurrent neural networks (RNNs):

- **Theoretical Foundation**: Understanding the concept of fixed points in dynamical systems
- **FlipFlop Task**: Training an RNN to perform a multi-channel memory task
- **Fixed-Point Identification**: Using optimization methods to locate stable and unstable fixed points
- **Visualization Analysis**: Displaying fixed-point distributions in state space via PCA dimensionality reduction
- **Multi-Configuration Comparison**: Comparing fixed-point structures across 2-bit, 3-bit, and 4-bit tasks

**Key Finding**: For an N-bit FlipFlop task, a successfully trained RNN learns to create 2^N stable fixed points—each corresponding to a unique combination of memory states.

Example Code
------------

- ``examples/experimental_data_analysis``: Scripts related to the ASA pipeline (TDA/decoding/CohoMap/CohoSpace/PathCompare/FR, etc.)
- ``examples/cell_classification``: Examples for cell classification and related analyses

Related Resources
-----------------

You may find these resources helpful:

- :doc:`../01_cann_modeling/index` — Understanding the CANN model
- :doc:`../04_pipeline/index` — End-to-end research workflow
- Core Concept Documentation — Detailed analytical methodologies