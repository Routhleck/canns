==============
Core Concepts
==============

Deep dive into library design, architecture, and theoretical foundations.

This section explains the CANNs library's design principles, module organization, and conceptual foundations. These documents focus on the "why" and "when" rather than the "how"—helping you understand the library's architecture and make informed decisions about using its components.

.. toctree::
   :maxdepth: 1
   :caption: Topics:

   01_design_philosophy
   02_model_collections
   03_task_generators
   04_analysis_methods
   05_brain_inspired_training

Overview
--------

:doc:`01_design_philosophy`
   Understand the library's architecture, core design principles, and four application scenarios. Learn about separation of concerns, extensibility, BrainPy :cite:p:`wang2023brainpy` integration, and performance strategies.

:doc:`02_model_collections`
   Explore three model categories: Basic CANN models, Brain-Inspired models with learning mechanisms, and Hybrid models combining CANNs with ANNs. Understand the BrainPy foundation and how to implement custom models.

:doc:`03_task_generators`
   Task generation philosophy and available paradigms. Learn about tracking tasks (population coding, template matching, smooth tracking) and navigation tasks (closed-loop, open-loop). Understand model-task coupling and design considerations.

:doc:`04_analysis_methods`
   Comprehensive analysis tools: Model Analyzer for simulations, Data Analyzer for experimental recordings, RNN Dynamics Analysis for fixed points, and Topological Data Analysis :cite:p:`carlsson2009topology,edelsbrunner2010computational` for geometric structures.

:doc:`05_brain_inspired_training`
   Brain-inspired learning mechanisms and the Trainer framework. Understand activity-dependent plasticity, learning rules (Hebbian, STDP :cite:p:`bi1998synaptic`, BCM), and how to implement custom trainers for biologically plausible learning.
