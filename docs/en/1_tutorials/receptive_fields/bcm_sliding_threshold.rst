BCM Rule: Receptive Field Development with Sliding Threshold
=============================================================

Scenario Description
--------------------

You want to understand how the BCM (Bienenstock, Cooper, Munro) rule implements adaptive synaptic plasticity through dynamic threshold adjustment to form selective receptive fields. The BCM rule demonstrates how complex feature learning emerges from local information.

What You Will Learn
--------------------

- Mathematical formulation of the BCM rule
- Mechanism of sliding threshold
- Formation of receptive fields
- M-curve and nonlinearity of plasticity
- Biological evidence

Complete Example
----------------

.. literalinclude:: ../../../../examples/brain_inspired/bcm_receptive_fields.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **Core of the BCM Rule**

   .. code-block:: python

      # BCM rule:
      # ΔW = η · y · (y - θ) · x
      #
      # Where:
      # - y: output
      # - θ: modification threshold (sliding threshold)
      # - x: input

      def bcm_update(x, y, w, theta, learning_rate=0.01):
          """Single step update of BCM rule"""
          delta_w = learning_rate * y * (y - theta) * x
          w_new = w + delta_w
          return w_new

2. **Sliding Threshold**

   .. code-block:: python

      # Threshold tracks the second moment of output
      # θ = E[y²]

      # Online estimation:
      theta = 0.99 * theta + 0.01 * y**2

      # Meaning:
      # - If y > θ: strengthen weights (LTP)
      # - If y < θ: weaken weights (LTD)
      # - θ automatically adjusts to balance learning

3. **S-shaped Nonlinearity of M-curve**

   .. code-block:: python

      import numpy as np
      import matplotlib.pyplot as plt

      # M-curve: plasticity vs output
      y_values = np.linspace(0, 2, 100)
      theta = 1.0  # threshold

      learning_curve = y_values * (y_values - theta)

      plt.plot(y_values, learning_curve)
      plt.axhline(0, color='k', linestyle='-', alpha=0.3)
      plt.axvline(theta, color='r', linestyle='--', label=f'θ={theta}')
      plt.xlabel('Output y')
      plt.ylabel('Learning signal y(y-θ)')
      plt.title('M-curve of BCM')

4. **Feature Learning**

   .. code-block:: python

      # Training on natural images
      from canns.trainer import BCMTrainer

      trainer = BCMTrainer(
          input_size=784,      # 28×28 image
          output_size=100,     # 100 learned features
          learning_rate=0.001,
          theta_learning_rate=0.01
      )

      # Training
      for epoch in range(10):
          for batch in image_batches:
              output = trainer.train(batch)

      # Learned features should be orientation, color, texture, etc.

Key Concepts
------------

**M-curve and Plasticity**

.. code-block:: text

   Learning signal
        ↑
        │        ╱╲
     LTP│       ╱  ╲
        │      ╱    ╲
      0 │─────╱──────╲─────→ Output y
        │    ╱ θ      ╲
     LTD│   ╱          ╲
        │  ╱            ╲

   Properties:
   - S-shaped nonlinearity (Sigmoid)
   - Symmetric about threshold θ
   - Saturation regions on both sides

**Stability Analysis**

Convergence of BCM:

- Weights are always stable (no explosion)
- Automatic normalization (through second moment)
- Selective features eventually form

**Comparison with Hebbian**

=============== ============ ==============
Feature         Hebbian      BCM
=============== ============ ==============
Rule            y·x          y·(y-θ)·x
Stability       Requires norm Auto-stable
Threshold       Fixed/absent Adaptive/sliding
Learning        Unconditional Conditional
=============== ============ ==============

Experimental Variations
-----------------------

**1. Different Initial Thresholds**

.. code-block:: python

   for theta_init in [0.5, 1.0, 2.0, 4.0]:
       trainer = BCMTrainer(theta_init=theta_init)
       learned_features = trainer.train(data)
       # Observe how learned features vary

**2. Learning Rate Effects**

.. code-block:: python

   for lr in [0.0001, 0.001, 0.01, 0.1]:
       trainer = BCMTrainer(learning_rate=lr)
       convergence_time = measure_convergence(trainer)

**3. Input Statistics**

.. code-block:: python

   # Gaussian input
   X_gaussian = np.random.randn(1000, 100)

   # Natural images (with statistical structure)
   X_natural = load_natural_images()

   # Compare learned features

Related API
-----------

- :class:`~src.canns.trainer.BCMTrainer`
- :class:`~src.canns.models.brain_inspired.BCMNeuron`

Biological Applications
-----------------------

**Visual Cortex**

- Development of orientation selectivity
- Ocular dominance and critical period
- Desensitization in dark rearing

**Auditory Cortex**

- Formation of frequency tuning
- Tone selectivity

Additional Resources
--------------------

- :doc:`orientation_selectivity` - Orientation selectivity
- :doc:`tuning_visualization` - Tuning curve visualization
- :doc:`../temporal_learning/stdp_spike_timing` - Comparison with STDP

Frequently Asked Questions
---------------------------

**Q: What is the difference between BCM and STDP?**

A: BCM is rate coding (frequency), STDP is spike-time coding (precise timing). BCM is used for feature learning, STDP is used for temporal learning.

**Q: Why is the M-curve necessary?**

A: The S-shaped nonlinearity ensures:
   1. Stability (no explosion)
   2. Bidirectional plasticity (LTP and LTD)
   3. Competition (strong outputs strengthen, weak outputs weaken)

**Q: How to choose the learning rate?**

A: BCM is sensitive to learning rate. Typically:
   - Weight learning rate: 0.001-0.01
   - Threshold learning rate: 0.01-0.1

Next Steps
----------

1. Train on real images and analyze learned features
2. Compare features learned by BCM and Oja
3. Investigate the effects of critical period
4. Read biological evidence
