Hierarchical Navigation Network Architecture
=============================================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scene Description
-----------------

You want to understand how to organize multiple scales of neural coding (grid cells, band cells, place cells) into a unified navigation system. Hierarchical network structure is the true organizational principle of the entorhinal cortex and hippocampus in the brain.

What You Will Learn
--------------------

- Architecture design of hierarchical networks
- Functional roles of different neuron types
- Integration mechanisms across scales
- Network initialization and state management
- Advantages of multi-module encoding

Complete Example
----------------

.. literalinclude:: ../../../../examples/cann/hierarchical_path_integration.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **Components of the Hierarchical Network**

   .. code-block:: python

      from canns.models.basic import HierarchicalNetwork

      network = HierarchicalNetwork(
          num_module=5,           # 5 grid modules
          num_place=30,          # 30 place cells
      )
      network.init_state()

   **Network Structure**:

   .. code-block:: text

      Motion Input (velocity, position)
              │
              ↓
         Band Cell Layer
        (X and Y directions)
              │
              ↓
       ┌──────┴──────┬────────┬────────┐
       ↓             ↓        ↓        ↓
     Module 1    Module 2  Module 3  Module 5
    (coarse)     (medium)  (fine)    (ultra-fine)
     Grid cells   Grid cells Grid cells Grid cells
       │            │        │        │
       └────────────┴────────┴────────┘
              │
              ↓
         Place Cell Layer
              │
              ↓
          Position Representation

2. **Band Cell Layer (Motion Encoding)**

   .. code-block:: python

      # Band cells encode motion in X and Y directions
      band_x_response = network.band_x_fr.value  # X direction encoding
      band_y_response = network.band_y_fr.value  # Y direction encoding

      # Motion input is encoded through Gaussian encoding
      # v_x = 0.02 m/s → X direction band cells activated
      # v_y = 0.01 m/s → Y direction band cells activated

   **Explanation**:
   - Band cells have the same direction selectivity as motion
   - Provide continuous motion information
   - Drive updates of all grid modules

3. **Multi-Scale Grid Modules**

   .. code-block:: python

      # Each module represents a different spatial scale
      module_activities = [
          network.grid_fr.value  # Aggregated grid activity
      ]

      # Scale ratios between modules
      scale_ratios = [1.0, 2.4, 5.76, 13.8, 33.1]  # 2.4x scaling

      # Grid spacing of each module
      grid_spacings = {
          'module_1': 50,   # cm
          'module_2': 120,  # cm
          'module_3': 290,  # cm
          'module_4': 700,  # cm
          'module_5': 1700  # cm
      }

   **Explanation**:
   - Adjacent modules have approximately 2.4x spacing relationship
   - Small modules provide fine-grained localization
   - Large modules provide broad navigation
   - Multi-scale encoding avoids ambiguity

4. **Place Cell Layer (Integration Layer)**

   .. code-block:: python

      # Place cells integrate all grid modules
      place_activity = network.place_fr.value

      # Activity of individual place cells forms "place fields"
      # Based on the Chinese Remainder Theorem principle:
      #   Combination of periods from different modules produces unique position representation

   **Mathematical Principle** (Chinese Remainder Theorem):

   .. code-block:: text

      Assume 3 modules with periods 3, 5, 7 respectively

      Module 1 (period 3): 0 1 2 0 1 2 0 1 2 ...
      Module 2 (period 5): 0 1 2 3 4 0 1 2 3 ...
      Module 3 (period 7): 0 1 2 3 4 5 6 0 1 ...

      Combination: (0,0,0) → Position 0 (unique)
                  (1,1,1) → Position 1 (unique)
                  (2,2,2) → Position 2 (unique)
                  ...
                  (1,3,5) → Position 36 (unique)

      With 5 modules, you can cover an enormous representational space!

5. **Network Learning Mechanism**

   .. code-block:: python

      # Localization input strength controls learning
      network(
          velocity=velocity_input,
          loc=actual_position,           # Teacher signal
          loc_input_stre=input_strength  # 0 = no learning, 100 = strong learning
      )

      # When loc_input_stre=100: use position signal to calibrate network
      # When loc_input_stre=0: pure path integration (open-loop)

Running Results
---------------

Running the hierarchical network generates:

1. **Trajectory and Activity Data**

   - Animal trajectory: movement path in the environment
   - Band cell activity: motion direction encoding
   - Grid activity: grid patterns of all 5 modules
   - Place cell activity: position encoding

2. **Activation Heatmaps**

   .. code-block:: text

      Grid Module 1 (coarse scale):
      ○   ○   ○   ○
       ○   ○   ○   ○    Large hexagons

      Grid Module 5 (fine scale):
      ○○○○○○○○○○
      ○○○○○○○○○○    Small hexagons, high resolution
      ○○○○○○○○○○

      Place Cells:
      █     █       █    Each place cell represents a specific location

3. **Performance Metrics**

   - Simulation time: ~20 seconds
   - Network size: ~5000 neurons
   - Memory usage: ~500 MB
   - Trajectory length: ~500,000 time steps

Key Concepts
------------

**Scale Invariance and Aliasing Problem**

Why do we need multiple scales?

.. code-block:: text

   Problem with a single grid module:
   ○───○───○───○───○───○
   At position (0,0): pattern A
   At position (6,0): also pattern A (because period is 6)
   → Cannot distinguish!

   Solution with multiple modules:
   Module 1 (period 6): position 0 → A,  position 6 → A (ambiguous)
   Module 2 (period 7): position 0 → B,  position 6 → C (distinguishable!)
   → Combination (A, B) uniquely maps to position 0
   → Combination (A, C) uniquely maps to position 6

**Module Independence and Redundancy**

- Each module performs path integration independently
- Error in one module does not affect others
- Provides robustness and error correction capability

**Scale-Invariant Navigation**

Multi-module structure enables navigation to scale across different environments:

.. code-block:: python

   # In a mouse environment
   grid_spacings = [50, 120, 290, 700, 1700]  # cm

   # In an elephant environment
   grid_spacings = [5, 12, 29, 70, 170]  # m (scaled)

Experimental Variations
-----------------------

**1. Change the Number of Modules**

.. code-block:: python

   # Fewer modules (low precision)
   network = HierarchicalNetwork(num_module=2, num_place=10)

   # More modules (high precision)
   network = HierarchicalNetwork(num_module=8, num_place=50)

**2. Analyze Grid Spacing of Individual Modules**

.. code-block:: python

   from canns.analyzer.spatial import compute_firing_field

   # Compute heatmap for each module
   for module_idx in range(5):
       heatmap = compute_firing_field(
           module_activities[module_idx],
           animal_trajectory
       )
       # Extract grid spacing from heatmap

**3. Test Position Decoding Accuracy**

.. code-block:: python

   def decode_position_from_place_cells(place_activity):
       """Decode position from place cell activity"""
       # Use the cell with maximum activity as position estimate
       active_cell = np.argmax(place_activity)
       estimated_position = cell_to_position_map[active_cell]
       return estimated_position

   # Evaluate decoding accuracy
   decoded_positions = [decode_position_from_place_cells(p)
                       for p in place_activity]
   position_errors = np.linalg.norm(
       np.array(decoded_positions) - animal_trajectory,
       axis=1
   )

**4. Study Phase Relationships Between Modules**

.. code-block:: python

   # Activity periods of grid modules
   for module_idx in range(5):
       # Compute autocorrelation of activity
       autocorr = compute_autocorrelation(module_activities[module_idx])
       period = find_peak_distance(autocorr)
       print(f"Module {module_idx} period: {period:.1f}cm")

Related API
-----------

- :class:`~src.canns.models.basic.HierarchicalNetwork` - Hierarchical Navigation Network
- :class:`~src.canns.models.basic.CANN2D` - Basic 2D CANN (for individual modules)
- :func:`~src.canns.analyzer.spatial.compute_firing_field` - Heatmap Computation

Biological Applications
-----------------------

**Advantages of Multi-Scale Navigation**

1. **Precise Localization**: Fine-scale modules provide centimeter-level precision
2. **Broad Coverage**: Coarse-scale modules cover kilometers of range
3. **Error Correction**: Large-scale modules detect small-scale drift
4. **Energy Efficiency**: Sparse coding reduces neural activity

**Evolution and Development**

- Larger animals have larger grid spacing
- Different species use different numbers of modules
- Young animals may have different grid cell scales

More Resources
--------------

- :doc:`path_integration` - Understanding basic path integration
- :doc:`grid_place_cells` - Interaction between grid and place cells
- :doc:`complex_environments` - Navigation in complex environments

Frequently Asked Questions
--------------------------

**Q: Why are place cells active at multiple locations?**

A: Because they are combinations of multiple grid modules. Each grid module has periodic patterns, and their combination produces multiple place fields. Through learning, the network typically reinforces a main place field.

**Q: How many modules are "sufficient"?**

A: Depends on environment size and required precision. Mathematically, 5 modules can cover arbitrarily large space with a 2.4x scaling ratio. Most rodents use 7-8 scales.

**Q: How do modules synchronize?**

A: Through localization input. Periodic external localization inputs (visual landmarks, etc.) reset all modules to a consistent state, preventing accumulation of drift.

Next Steps
----------

After completing this tutorial, we recommend:

1. Change the number of modules and observe precision changes
2. Analyze the learning process under visual constraints
3. Study adaptation to different environment sizes
4. Read :doc:`grid_place_cells` to understand interactions between cells