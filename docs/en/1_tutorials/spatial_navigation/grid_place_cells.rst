Interaction Between Grid Cells and Place Cells
===============================================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scene Description
-----------------

You want to understand how Grid Cells and Place Cells collaborate, and how they form spatial representations in the brain. These two cell types represent different levels of spatial encoding.

What You Will Learn
--------------------

- Characteristics and comparison of grid cells and place cells
- Their neuroanatomical locations
- Mutual connections and information flow
- How to generate place fields from grid inputs
- Integration of multi-scale representations

Complete Example
----------------

Grid-place cell analysis based on hierarchical network model:

.. code-block:: python

   import numpy as np
   from canns.models.basic import HierarchicalNetwork
   from canns.task.open_loop_navigation import OpenLoopNavigationTask
   from canns.analyzer.spatial import compute_firing_field

   # Create navigation environment and network
   task = OpenLoopNavigationTask(width=5, height=5, duration=50000)
   task.get_data()

   network = HierarchicalNetwork(num_module=5, num_place=30)
   network.init_state()

   # Run network
   def run_step(t, velocity, position):
       network(velocity=velocity, loc=position, loc_input_stre=0)
       return (network.grid_fr.value, network.place_fr.value)

   grid_activity, place_activity = brainstate.compile.for_loop(
       run_step,
       time_indices,
       task.data.velocity,
       task.data.position
   )

   # Analyze firing fields of individual cells
   for grid_cell_idx in range(grid_activity.shape[1]):
       grid_heatmap = compute_firing_field(
           grid_activity[:, grid_cell_idx].reshape(-1, 1),
           task.data.position
       )
       # Should display regular hexagonal pattern

   for place_cell_idx in range(place_activity.shape[1]):
       place_heatmap = compute_firing_field(
           place_activity[:, place_cell_idx].reshape(-1, 1),
           task.data.position
       )
       # Should display single Gaussian place field

Step-by-Step Explanation
-------------------------

1. **Characteristics of Grid Cells**

   .. code-block:: python

      # Grid cell characteristics
      grid_cell_properties = {
          'location': 'medial entorhinal cortex (MEC)',
          'firing_pattern': 'regular hexagonal grid',
          'tuning_width': '30-60 cm',
          'number_of_place_fields': 'multiple periodic fields',
          'function': 'provides metric'
      }

   **Grid Pattern**:

   .. code-block:: text

      Activation heatmap of a grid cell:

         █ █ █ █ █
        █ █ █ █ █ █
       █ █ █ █ █ █ █
        █ █ █ █ █ █
         █ █ █ █ █

      - Multiple distributed place fields
      - Regular hexagonal symmetry
      - Covers entire environment

2. **Characteristics of Place Cells**

   .. code-block:: python

      # Place cell characteristics
      place_cell_properties = {
          'location': 'hippocampus CA1',
          'firing_pattern': 'single place field',
          'tuning_width': '20-40 cm',
          'number_of_place_fields': 'typically 1 (can be multiple)',
          'function': 'encodes animal position'
      }

   **Place Field Shape**:

   .. code-block:: text

      Activation heatmap of a place cell:

         ░░░░░░░░░░
         ░░░░░░░░░░
         ░░░████░░░
         ░░░████░░░
         ░░░░░░░░░░

      - Single, localized activity region
      - Gaussian-shaped profile
      - Corresponds to specific location in environment

3. **Grid-Place Cell Integration Mechanism**

   .. code-block:: python

      # How do place cells form from grid cells?
      # Through linear combination and nonlinear integration

      position_cell_activity = 0
      for grid_cell_idx in range(num_grid_cells):
          weight = synaptic_strength[grid_cell_idx]
          position_cell_activity += weight * grid_cell_activity[grid_cell_idx]

      # Nonlinearity: activation only when input exceeds threshold
      position_cell_output = relu(position_cell_activity - threshold)

   **Key Principles**:
   - Place cells are linear combinations of multiple grid cells
   - Combination coefficients are determined through learning
   - Implements mapping from multiple grids to single location

4. **"Convergence" and "Divergence"**

   .. code-block:: text

      Grid cells → Place cells: Convergence
      ┌─────────────────────┐
      │  Place cell (single field) │
      └─────────────────────┘
             ▲  ▲  ▲  ▲  ▲
             │  │  │  │  │
      ┌──────┴──┴──┴──┴──┴──┐
      │ Grid cells 1 2 3 4 5... │
      └───────────────────────┘

      Place cells → Grid cells: Divergence
      ┌─────────────────────┐
      │ Place cell (single location) │
      └─────────────────────┘
             ▼  ▼  ▼  ▼  ▼
             │  │  │  │  │
      ┌──────┴──┴──┴──┴──┴──┐
      │ Grid cells 1 2 3 4 5... │
      └───────────────────────┘

Key Concepts
------------

**Brain Implementation of Chinese Remainder Theorem**

Combination of multiple grid modules produces unique location:

.. code-block:: python

   # 5 modules, each with different period
   periods = [6, 10, 15, 25, 40]  # centimeters

   # Location 300cm:
   location % 6 = 0    → phase A of grid module 1
   location % 10 = 0   → phase B of grid module 2
   location % 15 = 0   → phase C of grid module 3
   location % 25 = 0   → phase D of grid module 4
   location % 40 = 20  → phase E of grid module 5

   combination (A,B,C,D,E) → unique location

**Metric Distance**

Grid cells encode distance, not just position:

.. code-block:: text

   Distance between two locations can be inferred from grid activity:
   - Same-phase grid cells → distance is multiple of period
   - Similar grid patterns between location A and B → distance is closer
   - Different grid patterns between location A and C → distance is farther

Experimental Variations
------------------------

**1. Analyze grid-place encoding weights**

.. code-block:: python

   # Compute dependence of place cells on each grid cell
   from sklearn.linear_model import LinearRegression

   X = grid_activity  # grid cell activity
   y = place_activity  # place cell activity

   model = LinearRegression().fit(X, y)
   weights = model.coef_  # [num_place_cells, num_grid_cells]

   # Plot weight matrix
   plt.imshow(weights, aspect='auto', cmap='RdBu_r')
   plt.xlabel('Grid cells')
   plt.ylabel('Place cells')
   plt.title('Connection weights from grid to place')

**2. Test decoding accuracy**

.. code-block:: python

   # Decode position from grid activity (without place cells)
   grid_decoder = LinearRegression().fit(grid_activity, position)
   grid_predicted_pos = grid_decoder.predict(grid_activity_test)

   # Decode position from place cell activity
   place_decoder = LinearRegression().fit(place_activity, position)
   place_predicted_pos = place_decoder.predict(place_activity_test)

   # Compare accuracy
   grid_error = np.mean(np.linalg.norm(grid_predicted_pos - true_position, axis=1))
   place_error = np.mean(np.linalg.norm(place_predicted_pos - true_position, axis=1))

**3. Lesion one grid module and observe effects**

.. code-block:: python

   # Simulate damage to one grid module
   damaged_grid_activity = grid_activity.copy()
   damaged_grid_activity[:, :, 0] = 0  # damage module 0

   # Retrain place cell decoder
   damaged_decoder = LinearRegression().fit(
       damaged_grid_activity, position
   )
   damaged_error = np.mean(...)

   # Compare with intact system
   print(f"Intact system error: {normal_error:.2f} cm")
   print(f"Error after damage: {damaged_error:.2f} cm")
   print(f"Error increase: {(damaged_error/normal_error - 1)*100:.1f}%")

Related API
-----------

- :class:`~src.canns.models.basic.HierarchicalNetwork` - complete grid-place network
- :func:`~src.canns.analyzer.spatial.compute_firing_field` - activation heatmap
- :class:`~src.canns.models.basic.CANN2D` - single grid module

Biological Applications
-----------------------

**Multi-level Representation of Navigation**

1. **Grid Cells** (MEC):
   - Provide stable, invariant coordinate system
   - Maintain same spacing across different environments
   - Support path integration

2. **Place Cells** (Hippocampus):
   - Environment-specific
   - Support contextual memory
   - Participate in formation of cognitive maps

3. **Head Direction Cells** (MEC):
   - Encode heading direction
   - Global polarization
   - Work in concert with place and grid cells

**Clinical Significance**

- Dementia and grid cell abnormalities
- Traumatic brain injury and spatial disorientation
- Loss of navigation ability may predict neurodegenerative disease

More Resources
--------------

- :doc:`path_integration` - Understanding path integration
- :doc:`hierarchical_network` - Hierarchical structure

Frequently Asked Questions
---------------------------

**Q: Why do place cells have only one place field while grid cells have multiple?**

A: Because place cells are nonlinear combinations of multiple grid modules. By selecting appropriate input weights and thresholds, multiple periodic grid patterns can be combined to produce a single, localized place field.

**Q: Is the grid spacing in the brain fixed?**

A: Essentially yes, but may be adjusted during learning. More importantly, different animal species have different baseline spacings (related to body size).

**Q: How does place cell learning occur?**

A: Through supervised or reinforcement learning. External signals (visual landmarks, rewards) reinforce certain grid-location associations, gradually shaping the tuning curves of place cells.

Next Steps
----------

1. Analyze rapid formation of place fields in new environments
2. Study information flow between grid and place cells
3. Compare encoding strategies across species
4. Read :doc:`complex_environments` for more complex scenarios