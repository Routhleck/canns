Tutorials
=========

.. warning::

   ⚠️ **Important Notice**: Parts of this documentation are still under development and verification and may be incomplete. It is recommended for reference only. Please confirm the completeness of relevant features with the development team before important projects.

Welcome to the CANNs Tutorials! This tutorial adopts a **scenario-driven** approach to help you quickly find relevant content based on your practical needs.

How to Use This Tutorial
------------------------

Unlike traditional "module introduction" approaches, our tutorials are organized around **the tasks you want to accomplish**:

- **Want to analyze CANN dynamics?** → :doc:`cann_dynamics/index`
- **Want to model spatial navigation?** → :doc:`spatial_navigation/index`
- **Want to train memory networks?** → :doc:`memory_networks/index`
- **Want to learn unsupervised algorithms?** → :doc:`unsupervised_learning/index`

Each scenario includes:

1. **Scenario Description** - What problem you will solve
2. **Complete Example** - Ready-to-run code
3. **Step-by-Step Explanation** - How the code works
4. **Results Analysis** - How to interpret the output
5. **Extension Directions** - What to learn next

Tutorial Scenarios
------------------

.. toctree::
   :maxdepth: 1
   :caption: Choose Your Scenario

   cann_dynamics/index
   spatial_navigation/index
   memory_networks/index
   unsupervised_learning/index
   receptive_fields/index
   temporal_learning/index
   experimental_analysis/index
   advanced_workflows/index

Scenario Overview
-----------------

1. CANN Dynamics Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Understand how CANN responds to different inputs and analyze bump dynamics.

**Keywords**: Tracking, tuning curves, oscillations, visualization

**Suitable for**: Beginners, researchers who need to understand basic models

→ :doc:`cann_dynamics/index`

2. Spatial Navigation Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Simulate the brain's spatial navigation system (grid cells, place cells, theta rhythm).

**Keywords**: Path integration, hierarchical networks, theta sweep, hippocampus

**Suitable for**: Neuroscience researchers, spatial cognition researchers

→ :doc:`spatial_navigation/index`

3. Memory Network Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Implement associative memory and pattern storage using Hopfield networks.

**Keywords**: Hebbian, pattern completion, energy function, capacity analysis

**Suitable for**: Students learning neural network fundamentals, memory researchers

→ :doc:`memory_networks/index`

4. Unsupervised Learning
~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Extract principal components through Oja and Sanger rules.

**Keywords**: PCA, Hebbian, weight normalization, dimensionality reduction

**Suitable for**: Researchers interested in bio-inspired learning

→ :doc:`unsupervised_learning/index`

5. Receptive Field Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Train orientation-selective neurons using the BCM rule.

**Keywords**: BCM, sliding threshold, receptive field, orientation selectivity

**Suitable for**: Visual neuroscience researchers, developmental plasticity researchers

→ :doc:`receptive_fields/index`

6. Temporal Pattern Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Train spiking neural networks to learn temporal patterns using STDP.

**Keywords**: STDP, spike timing, causality, LTP/LTD

**Suitable for**: Spiking neural network researchers, temporal coding researchers

→ :doc:`temporal_learning/index`

7. Experimental Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Analyze real neural recording data and fit bump activity.

**Keywords**: Bump fitting, ROI data, calcium imaging, preprocessing

**Suitable for**: Experimental neuroscientists, data analysts

→ :doc:`experimental_analysis/index`

8. Advanced Workflows
~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Build end-to-end pipelines and implement complex automated workflows.

**Keywords**: Pipeline, batch processing, parameter sweep, automation

**Suitable for**: Researchers and engineers who need efficient workflows

→ :doc:`advanced_workflows/index`

Getting Help
------------

- **Example Code**: :doc:`../examples/README` - All runnable examples
- **API Documentation**: :doc:`../../autoapi/index` - Detailed API reference
- **Community Support**: `GitHub Issues <https://github.com/your-org/canns/issues>`_
