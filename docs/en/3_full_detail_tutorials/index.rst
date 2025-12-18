Full Detail Tutorials
=====================

Comprehensive tutorials covering all aspects of the CANNs library, organized by research scenarios.

Overview
--------

These tutorials provide in-depth, hands-on guidance for using the CANNs library across different research scenarios. Each tutorial is a complete Jupyter notebook that you can run and modify.

Scenarios
---------

.. toctree::
   :maxdepth: 1
   :caption: Tutorial Scenarios:

   01_cann_modeling/index
   02_data_analysis/index
   03_brain_inspired/index
   04_pipeline/index

Scenario Descriptions
---------------------

**Scenario 1: CANN Modeling and Simulation** (7 tutorials)
   Build, simulate, and analyze continuous attractor neural networks from scratch. Learn network dynamics, parameter effects, and advanced architectures.

   - Foundation: Basic models, tasks, visualization, parameters
   - Advanced: Hierarchical networks, theta sweeps, complex environments

**Scenario 2: Data Analysis and Neural Decoding** (Coming Soon)
   Analyze experimental neural recordings, decode spatial representations, and validate model predictions against real data.

**Scenario 3: Brain-Inspired Learning** (1 tutorial)
   Implement biologically-inspired learning rules including Hebbian plasticity and associative memory mechanisms.

   - Pattern storage and recall with Hopfield networks

**Scenario 4: End-to-End Research Workflows** (1 tutorial)
   Use high-level pipelines for complete analyses without detailed implementation knowledge. Perfect for experimental neuroscientists.

   - Theta sweep pipeline for trajectory analysis

Learning Paths
--------------

**For Computational Neuroscientists**:

1. Start with Scenario 1 (CANN Modeling)—Learn the foundations
2. Explore Scenario 3 (Brain-Inspired Learning)—Understand learning mechanisms
3. Use Scenario 4 (Pipelines) for rapid analysis

**For Experimental Neuroscientists**:

1. Begin with Scenario 4 (Pipelines)—Quick analysis of your data
2. Optionally explore Scenario 1—Understand what's happening under the hood
3. Explore Scenario 3 for learning-based models

**For Method Developers**:

1. Master Scenario 1 (CANN Modeling)—Deep understanding of models
2. Study Scenario 3 (Brain-Inspired Learning)—Extend learning rules
3. Use Scenario 4 code as reference for creating new pipelines

Prerequisites
-------------

- **Programming**: Basic Python knowledge (NumPy, matplotlib)
- **Math**: Linear algebra, differential equations (helpful but not required)
- **Neuroscience**: Basic understanding of neural coding (recommended)

Each scenario has specific prerequisites listed in its index page.

Time Commitment
---------------

- **Scenario 1**: 5 hours (7 tutorials)
- **Scenario 2**: Coming soon
- **Scenario 3**: 35 minutes (1 tutorial)
- **Scenario 4**: 60 minutes (1 tutorial)

Total estimated time: 6.5 hours for all available tutorials

Getting Started
---------------

**New to CANNs?**

1. Complete the :doc:`../1_quick_starts/index` first
2. Read :doc:`../2_core_concepts/index` for background
3. Then dive into these detailed tutorials

**Have experience?**

Jump directly to the scenario that matches your needs.

Running the Tutorials
---------------------

All tutorials are provided as Jupyter notebooks (`.ipynb` files).

**To run locally**:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/routhleck/canns.git
   cd canns

   # Install dependencies
   pip install -e .[dev]

   # Launch Jupyter
   jupyter notebook docs/en/3_full_detail_tutorials/

**Online**:

- Open notebooks directly on GitHub
- Use Google Colab (upload the notebook)
- Use Binder (link coming soon)

Support and Feedback
--------------------

- **Documentation**: Full API reference available
- **Examples**: Additional examples in the `examples/` directory
- **Issues**: Report problems on `GitHub Issues <https://github.com/routhleck/canns/issues>`_
- **Discussions**: Ask questions in GitHub Discussions

Contributing
------------

We welcome contributions! If you:

- Find errors or improvements for existing tutorials
- Want to add new tutorials
- Have suggestions for better explanations

Please submit a pull request or open an issue.

Next Steps
----------

Choose a scenario above and start learning! Each scenario page provides detailed information about its content and learning objectives.
