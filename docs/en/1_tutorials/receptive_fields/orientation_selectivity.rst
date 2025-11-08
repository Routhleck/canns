Development of Orientation Selectivity
========================================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scene Description
-----------------

Neurons in the visual cortex respond strongly to stimuli of specific orientations. You want to understand how this orientation selectivity naturally emerges from unstructured initial connections through learning mechanisms such as BCM.

What You Will Learn
--------------------

- Neurobiological foundations of orientation selectivity
- Modeling and learning orientation preferences
- Formation of cortical maps
- The role of competition and cooperation

Complete Example
----------------

Orientation selectivity development based on BCM:

.. code-block:: python

   from canns.trainer import BCMTrainer
   import numpy as np
   import matplotlib.pyplot as plt

   # Generate orientation stimuli
   def create_oriented_gratings(num_orientations=8, size=32):
       """Create sinusoidal grating stimuli of different orientations"""
       stimuli = []
       for orientation in np.linspace(0, np.pi, num_orientations):
           grating = np.sin(np.arange(size) * np.cos(orientation) +
                           np.arange(size)[:, None] * np.sin(orientation))
           grating = (grating - grating.min()) / (grating.max() - grating.min())
           stimuli.append(grating.flatten())
       return np.array(stimuli)

   # Training
   stimuli = create_oriented_gratings()
   trainer = BCMTrainer(input_size=1024, output_size=100)

   for epoch in range(100):
       for stimulus in stimuli:
           trainer.train(stimulus.reshape(1, -1))

   # Visualize learned filters
   filters = trainer.model.W.value  # [100, 1024]

   fig, axes = plt.subplots(10, 10, figsize=(10, 10))
   for i, ax in enumerate(axes.flat):
       filter_2d = filters[i].reshape(32, 32)
       ax.imshow(filter_2d, cmap='RdBu_r')
       ax.set_xticks([])
       ax.set_yticks([])

   plt.suptitle('Learned Orientation Selectivity Filters')
   plt.tight_layout()
   plt.savefig('orientation_filters.png')

Key Concepts
------------

**Orientation Tuning Curve**

.. code-block:: text

   Response Strength
       ↑
       │    ╱╲
       │   ╱  ╲
       │  ╱    ╲
       │ ╱      ╲
       └─────────→ Orientation (0-180°)

   Preferred Orientation: ~45°
   Tuning Width: ~30°

**Self-Organizing Map**

Adjacent neurons learn similar orientations:

.. code-block:: text

   Cortical Surface:

   0°  15°  30°  45°
   15° 30°  45°  60°
   30° 45°  60°  75°
   45° 60°  75°  90°

   Orientation Columns: Along cortical depth, orientation changes smoothly

Experimental Variations
-----------------------

**1. Varying Stimulus Complexity**

.. code-block:: python

   # Simple: Pure orientation variation
   # Moderate: Orientation + spatial frequency
   # Complex: Natural images

**2. Competition Mechanisms**

.. code-block:: python

   # With lateral inhibition
   # Without lateral inhibition

   # Observe: How inhibition improves selectivity

**3. Critical Period**

.. code-block:: python

   # Early training (during development)
   # Late training (after maturation)

   # Observe: When preferences cannot be changed

Related API
-----------

- :class:`~src.canns.trainer.BCMTrainer`

Next Steps
----------

- :doc:`tuning_visualization` - Visualizing tuning properties
- :doc:`../unsupervised_learning/algorithm_comparison` - Comparing other learning rules
