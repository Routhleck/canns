Scenario 2: Data Analysis and Neural Decoding
==============================================

Comprehensive tutorials on analyzing neural network dynamics, identifying fixed points, and understanding recurrent computations.

.. note::
   **Current Status**

   - ✅ **Completed**: RNN Fixed Point Analysis Tutorial (FlipFlop Task)
   - 🚧 **In Progress**: Additional tutorials on experimental neural data analysis, spatial decoding, and CANN data analysis are being prepared. Stay tuned!

Tutorial List
-------------

.. toctree::
   :maxdepth: 1
   :caption: RNN Dynamics Analysis

   flipflop_tutorial

Tutorial Overview
-----------------

**RNN Fixed Point Analysis Tutorial (FlipFlop Task)**

This tutorial provides a detailed guide on using the ``FixedPointFinder`` tool to analyze the dynamical properties of recurrent neural networks (RNNs):

- **Theoretical Foundation**: Understanding fixed points in dynamical systems
- **FlipFlop Task**: Training RNNs to complete multi-channel memory tasks
- **Fixed Point Finding**: Using optimization methods to identify stable and unstable fixed points
- **Visualization Analysis**: Displaying fixed point distribution in state space through PCA dimensionality reduction
- **Multi-Configuration Comparison**: Comparing fixed point structures across 2-bit, 3-bit, and 4-bit tasks

**Key Finding**: For N-bit FlipFlop tasks, successfully trained RNNs learn to create 2^N stable fixed points—each corresponding to a unique memory state combination.

Learning Path
-------------

**Recommended Order**:

1. Start with the fixed point analysis tutorial to understand RNN internal computational mechanisms
2. Learn how to apply the same analysis methods to your own RNN models
3. Explore dynamical structures under different tasks

Prerequisites
-------------

- Basic understanding of recurrent neural networks
- Familiarity with Python programming and JAX
- Knowledge of basic concepts in dynamical systems

Related Resources
-----------------

You may find these resources helpful:

- :doc:`../01_cann_modeling/index`—Understanding CANN models
- :doc:`../04_pipeline/index`—End-to-end research workflows
- Core Concepts documentation—Detailed analysis methods
