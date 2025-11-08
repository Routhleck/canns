Receptive Field Development
============================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scenario Description
--------------------

How do receptive fields (such as orientation selectivity) of visual cortex neurons develop through experience? The BCM (Bienenstock-Cooper-Munro) rule provides an elegant answer. This series of tutorials will teach you:

- Principles of the BCM sliding threshold mechanism
- How selectivity emerges from random connections
- Formation and analysis of orientation tuning curves
- Receptive field visualization methods

What You Will Learn
-------------------

1. Mathematical principles and biological significance of the BCM rule
2. Training networks with oriented grating stimuli
3. The emergence process of orientation selectivity
4. Analysis and visualization of tuning curves
5. The relationship between threshold dynamics and weight evolution

Tutorial List
-------------

.. toctree::
   :maxdepth: 1

   bcm_sliding_threshold
   orientation_selectivity

Target Audience
---------------

- Neuroscientists researching visual system development
- Machine learning researchers interested in bio-inspired learning
- Students studying theories of plasticity

Prerequisites
-------------

- Basic neuroscience knowledge
- Introductory understanding of the visual system
- Python and image processing fundamentals

Core Algorithm
--------------

**BCM Rule**:

.. math::

   \Delta W = \eta \cdot \phi(y, \theta) \cdot x^T

Where:

- :math:`\phi(y, \theta) = y(y - \theta)` is the BCM modulation function
- :math:`\theta = \langle y^2 \rangle` is the sliding threshold

**Key Features**:

- **LTP Region** (y > θ): Synaptic potentiation
- **LTD Region** (y < θ): Synaptic depression
- **Homeostasis**: Threshold adapts autonomously based on neuronal activity

Practical Applications
----------------------

- **Computer Vision**: Self-organized learning of feature detectors
- **Developmental Neuroscience**: Understanding critical period plasticity
- **Neuropathology**: Modeling visual deprivation effects
- **Artificial Intelligence**: Bio-inspired feature learning

Biological Background
---------------------

BCM theory successfully explains:

- **Critical Period**: Time-dependency of visual experience on development
- **Monocular Deprivation**: Formation of ocular dominance columns
- **Contrast Adaptation**: Dynamic adjustment of thresholds
- **Selectivity Development**: Transition from non-selective to selective

Start Learning
---------------

Begin with :doc:`bcm_sliding_threshold` to see how selectivity emerges!
