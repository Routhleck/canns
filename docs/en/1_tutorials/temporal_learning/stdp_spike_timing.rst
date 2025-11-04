STDP: Spike-Timing-Dependent Plasticity
========================================

Scenario Description
--------------------

You want to understand STDP (Spike-Timing-Dependent Plasticity), one of the most important learning rules in the brain. Unlike traditional Hebbian learning which only focuses on "two neurons being active simultaneously," STDP focuses on "which neuron fires first." This implements learning of causal relationships.

What You Will Learn
--------------------

- Mathematical principles and temporal dependence of STDP rules
- Implementation of spiking neuron (LIF) models
- Role of spike traces
- Mechanisms of LTP (long-term potentiation) and LTD (long-term depression)
- Demonstration of temporal pattern learning

Complete Example
----------------

.. literalinclude:: ../../../../examples/brain_inspired/stdp_temporal_learning.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **Understanding STDP's Asymmetry**

   Standard Hebbian rule is symmetric: simultaneous activity → enhancement

   .. code-block:: text

      Pre-syn spike:  X           X
      Post-syn spike:     X           X
      Result:        Enhancement (Hebbian)  Enhancement (Hebbian)

   STDP is asymmetric: order matters!

   .. code-block:: text

      Timeline: t=0      t=10ms      t=20ms

      Case 1 (LTP):
      Pre-syn:   ⚡       •          •       (spike at t=0)
      Post-syn:  •        ⚡         •       (spike at t=10)
      Result:    LTP (pre leads post, enhancement)

      Case 2 (LTD):
      Pre-syn:   •        ⚡         •       (spike at t=10)
      Post-syn:  ⚡       •          •       (spike at t=0)
      Result:    LTD (post leads pre, depression)

2. **Initializing Spiking Neural Network**

   .. code-block:: python

      import brainstate
      import jax.numpy as jnp
      import numpy as np
      from canns.models.brain_inspired import SpikingLayer
      from canns.trainer import STDPTrainer

      np.random.seed(42)
      brainstate.random.seed(42)

      # Create spiking neuron layer (LIF model)
      model = SpikingLayer(
          input_size=20,           # 20 input neurons
          output_size=1,           # 1 output neuron
          threshold=0.5,           # Spike threshold
          v_reset=0.0,             # Reset potential
          leak=0.9,                # Leak constant
          trace_decay=0.90         # Spike trace decay
      )
      model.init_state()

   **Explanation**:
   - ``threshold=0.5``: Spike generated when membrane potential reaches 0.5
   - ``leak=0.9``: Membrane potential decays to 90% per step (leak conductance)
   - ``trace_decay=0.90``: Exponential decay of spike trace

3. **Creating Temporal Input Patterns**

   .. code-block:: python

      # Generate temporal input patterns
      # First 4 inputs activate early, then subsequent inputs activate
      temporal_patterns = []

      # Pattern 1: Early activation (should enhance)
      pattern1 = np.zeros(20)
      pattern1[[0, 1, 2, 3]] = 1.0  # First 4 neurons activate
      temporal_patterns.append(pattern1)

      # Pattern 2: Delayed activation (should depress)
      pattern2 = np.zeros(20)
      pattern2[[10, 11, 12, 13]] = 1.0  # Later neurons activate
      temporal_patterns.append(pattern2)

      # Mix multiple time points
      for _ in range(3):
          temporal_patterns.append(pattern1)  # Repeat early activation
      for _ in range(3):
          temporal_patterns.append(pattern2)  # Repeat late activation

   **Explanation**:
   - First 4 inputs (indices 0-3) represent the "cause" in causal relationships
   - Last 4 inputs (indices 10-13) represent "unrelated" signals
   - STDP should strengthen connections with the first 4 inputs

4. **Configuring and Running STDP Training**

   .. code-block:: python

      # Create STDP trainer
      trainer = STDPTrainer(
          model,
          learning_rate=0.02,      # Global learning rate
          A_plus=0.005,            # LTP amplitude
          A_minus=0.00525,         # LTD amplitude (slightly larger than A_plus to prevent explosion)
          w_min=0.0,               # Weight lower bound (non-negativity)
          w_max=1.0,               # Weight upper bound
          compiled=False            # False for easier debug output
      )

      # Record weight changes
      W_init = model.W.value.copy()

      # Train for multiple epochs
      print("Starting STDP training...")
      for epoch in range(50):
          model.reset_state()  # Reset membrane potential and trace
          trainer.train(temporal_patterns)

          if (epoch + 1) % 10 == 0:
              W_current = model.W.value
              weight_change = np.linalg.norm(W_current - W_init)
              print(f"Epoch {epoch+1}: weight_change={weight_change:.4f}")

      W_final = model.W.value.copy()

   **Explanation**:
   - Each epoch processes all temporal patterns
   - ``reset_state()`` resets membrane potential and trace (fresh start)
   - Weights should gradually change (learning)

5. **Analyzing Learning Results**

   .. code-block:: python

      import matplotlib.pyplot as plt

      # Analyze weight changes
      weight_change = W_final - W_init

      # Plot weight change heatmaps
      fig, axes = plt.subplots(1, 3, figsize=(15, 4))

      # Initial weights
      ax = axes[0]
      im1 = ax.imshow(W_init, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
      ax.set_title("Initial Weights (Random)")
      ax.set_xlabel("Input Neurons")
      ax.set_ylabel("Output Neurons")
      plt.colorbar(im1, ax=ax)

      # Final weights
      ax = axes[1]
      im2 = ax.imshow(W_final, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
      ax.set_title("Final Weights (After Learning)")
      ax.set_xlabel("Input Neurons")
      plt.colorbar(im2, ax=ax)

      # Weight change
      ax = axes[2]
      im3 = ax.imshow(weight_change, aspect='auto', cmap='RdBu_r')
      ax.set_title("Weight Change ΔW")
      ax.set_xlabel("Input Neurons")
      plt.colorbar(im3, ax=ax)

      plt.tight_layout()
      plt.savefig('stdp_weight_analysis.png')
      plt.show()

      # Analyze weight changes grouped by input
      print("\n=== Weight Change Analysis ===")
      print(f"Weight change for early inputs (0-4): {weight_change[0, 0:4].mean():.4f}")
      print(f"Weight change for late inputs (10-14): {weight_change[0, 10:14].mean():.4f}")
      print(f"Weight change for other inputs: {weight_change[0, 14:].mean():.4f}")

Running Results
---------------

**Expected Weight Change Pattern**

.. code-block:: text

Weight change for early inputs (0-4): +0.12    ✅ Enhancement (LTP)
Weight change for late inputs (10-14): -0.05  ✅ Depression (LTD)
Weight change for other inputs: +0.01         ✅ Minor change

**Explanation**:

- **Early inputs**: These inputs activate before the output spike
  - Spike trace is high
  - Strong LTP triggered when output spike occurs
  - Result: Weights enhanced ✅

- **Late inputs**: These inputs activate after the output spike
  - Post-spike trace is still high
  - LTD triggered when these inputs activate
  - Result: Weights depressed ✅

Key Concepts
------------

**STDP Time Window**

Biological evidence (Bi & Poo, 1998):

.. code-block:: text

     LTP region ↓
      |  /
      | /
 ΔW  +|/___________
      |   \
      |    \
      |     \ ← LTD region
      |______\_____ Δt (ms)
     -40  0  +40

- Pre-spike leads by 0-20ms: Strong LTP
- Pre-spike leads by 20-40ms: Weak LTP
- Post-spike leads by 0-40ms: LTD
- Beyond ±40ms: No change

**Spike Trace Mathematics**

.. math::

   \text{trace} = \text{decay} \times \text{trace} + \text{spike}

For example, with decay=0.9:

.. code-block:: text

   Time:  0    1    2    3    4    5    6
   Spike: 1    0    0    0    0    0    0
   Trace: 1   0.9  0.81 0.73 0.66 0.59 0.53

- Spike immediately sets trace to 1
- Trace decays exponentially
- Decay constant determines the "memory" time window

**STDP and Causality**

STDP implements learning of "prediction":

.. code-block:: text

Scenario A (Causal):
  Early signal X → 1 second later → Reward signal Y
  Learning: Strengthen X→output connections
  Adaptation: X predicts Y

Scenario B (Non-causal):
  Reward signal Y → 1 second later → Signal X
  Learning: Weaken X→output connections
  Adaptation: X is unrelated to Y

Performance and Parameters
--------------------------

**Learning Rate Selection**

=== ============== ==============
Learning Rate  Convergence Speed  Stability
=== ============== ==============
0.001  Very Slow    Very Stable
0.01   Slow         Stable
0.02   Moderate     Quite Stable ✓
0.1    Fast         Possible Oscillation
=== ============== ==============

**Time Window (via trace_decay)**

.. code-block:: python

   # Narrow time window (fast decay)
   model = SpikingLayer(trace_decay=0.95)

   # Wide time window (slow decay)
   model = SpikingLayer(trace_decay=0.99)

**LTP/LTD Balance**

.. code-block:: python

   # A_minus > A_plus: Biased toward LTD (inhibitory)
   trainer = STDPTrainer(model, A_plus=0.005, A_minus=0.01)

   # A_minus = A_plus: Balanced
   trainer = STDPTrainer(model, A_plus=0.005, A_minus=0.005)

   # A_minus < A_plus: Biased toward LTP (excitatory)
   trainer = STDPTrainer(model, A_plus=0.01, A_minus=0.005)

Experimental Variations
-----------------------

**1. Changing Input Timing**

.. code-block:: python

   # Tighter temporal relationship
   pattern1[0:2] = 1.0      # Very early
   pattern2[18:20] = 1.0    # Very late

**2. Multiple Output Neurons Competing**

.. code-block:: python

   model = SpikingLayer(
       input_size=20,
       output_size=3,  # 3 output neurons competing
   )

   # Different neurons should learn different temporal relationships

**3. Changing Spike Patterns**

.. code-block:: python

   # Use more complex temporal sequences
   # Such as: ABC pattern → predict D

Related API
-----------

- :class:`~src.canns.models.brain_inspired.SpikingLayer` - LIF spiking neurons
- :class:`~src.canns.trainer.STDPTrainer` - STDP trainer
- :class:`~src.canns.trainer.STDPTrainer.predict` - Spike prediction

Biological Applications
-----------------------

**Hippocampus**

- Temporal sequence memory (place cell sequences)
- Theta phase precession (temporal coding in theta rhythm)

**Auditory Cortex**

- Sound sequence recognition
- Speech processing

**Motor Cortex**

- Action sequence planning
- Skill learning

**Cerebellum**

- Precise temporal control
- Eye tracking (vestibulo-ocular reflex)

Frequently Asked Questions
--------------------------

**Q: Why is spike trace necessary?**

A: Direct calculation of time difference Δt=t_post-t_pre is inefficient and energy-intensive. Spike trace provides a "time stamp" through exponential decay, which is both biologically plausible and computationally efficient.

**Q: Why A_minus > A_plus?**

A: This is a trade-off for stability:
   - If A_plus > A_minus: Weights will explode exponentially
   - If A_minus > A_plus: Weights gradually weaken, naturally stable
   - A_minus ≈ A_plus + 5%: Provides competition and stability

**Q: How to handle "noisy" spikes?**

A: STDP has built-in noise resistance:
   - Random spikes won't have consistent temporal relationships
   - Only meaningful patterns will be strengthened
   - Automatically learns causal relationships while ignoring noise

Next Steps
----------

1. Try the experimental variations above
3. Explore :doc:`causal_learning` to understand causal relationship learning
4. Study :doc:`ltp_ltd_mechanisms` to understand molecular mechanisms of LTP/LTD

Reference Resources
-------------------

- **Original Discovery**: Bi, G. Q., & Poo, M. M. (1998). Synaptic Modifications in Cultured Hippocampal Neurons. Journal of Neuroscience, 18(24), 10464-10472.
- **Theory Review**: Gerstner, W., & Kistler, W. M. (2002). Spiking Neuron Models. Cambridge University Press.
- **Application Review**: Morrison, A., Diesmann, M., & Gerstner, W. (2008). Phenomenological Models of Synaptic Plasticity. Biological Cybernetics, 98(6), 459-478.
