Temporal Pattern Learning
=========================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scenario Description
--------------------

STDP (Spike-Timing-Dependent Plasticity) is the first learning rule that considers precise spike timing. It implements causal relationship learning in spiking neural networks. This tutorial series will teach you:

- The biological principles and mathematical models of STDP
- Implementation of spiking neural networks
- Learning mechanisms of temporal patterns
- Time windows of LTP and LTD

What You Will Learn
-------------------

1. Mathematical formulas and implementation of STDP rules
2. LIF (Leaky Integrate-and-Fire) neuron model
3. Usage of spike traces
4. Encoding and learning of temporal patterns
5. Neural representation of causal relationships

Tutorial List
-------------

.. toctree::
   :maxdepth: 1

   stdp_spike_timing

Applicable Audience
-------------------

- Researchers studying spiking neural networks
- Students interested in temporal learning
- Engineers developing neuromorphic computing
- Neuroscientists researching synaptic plasticity

Prerequisite Knowledge
----------------------

- Fundamentals of spiking neuron models
- Basic differential equations
- Event-driven programming concepts

Core Algorithm
--------------

**STDP Rule**:

.. math::

   \Delta W_{ij} = A_{plus} \cdot trace_{pre}[j] \cdot spike_{post}[i] - A_{minus} \cdot trace_{post}[i] \cdot spike_{pre}[j]

Where:

- **LTP**: Pre-spike before post-spike (causal relationship)
- **LTD**: Post-spike before pre-spike (anti-causal relationship)

**Spike Trace Update**:

.. math::

   trace = decay \cdot trace + spike

The time window is typically 20-40ms (from biological data).

Practical Applications
----------------------

- **Sequence Learning**: Learning patterns in temporal sequences
- **Predictive Coding**: Early signals predicting subsequent events
- **Motor Control**: Neural encoding of action sequences
- **Speech Recognition**: Learning of temporal acoustic features
- **Neuromorphic Chips**: Efficient learning in hardware implementations

Biological Discoveries
----------------------

**Landmark experiments by Bi & Poo (1998)**:

- Pre→Post (+10ms): ~40% synaptic potentiation (LTP)
- Post→Pre (-10ms): ~30% synaptic depression (LTD)
- Outside ±40ms window: No significant changes

**Brain Region Evidence**:

- Hippocampus: Sequence memory and place cell sequences
- Auditory Cortex: Temporal learning of speech and music
- Motor Cortex: Action sequence planning
- Cerebellum: Precise temporal control

Differences from Rate Coding
-----------------------------

========== ========================= =========================
Feature    Rate Coding (Hebbian)     STDP (Temporal Coding)
========== ========================= =========================
Time Resolution  Coarse (~100ms window)    Fine (~1ms precision)
Information  Average firing rate     Precise spike time
Causality  No temporal order        Pre→Post order matters
Biological Plausibility  Simplified abstraction      Closer to biology
Computation  Continuous values       Discrete events
========== ========================= =========================

Start Learning
---------------

Begin with :doc:`stdp_spike_timing` to explore the mysteries of temporal learning!