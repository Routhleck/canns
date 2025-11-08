Path Integration and Location Encoding
=======================================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scene Description
-----------------

You want to understand how animals maintain a continuous representation of their own position by integrating motion information (velocity and direction). This is a fundamental mechanism of spatial navigation, known as "path integration" or "dead reckoning".

What You Will Learn
--------------------

- Mathematical principles of path integration
- Mechanisms for implementing path integration using CANN
- The role of grid cells and place cells
- How motor inputs drive updates to location representation
- Error accumulation and correction mechanisms

Complete Example
----------------

.. literalinclude:: ../../../../examples/cann/hierarchical_path_integration.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **Create an Open-Loop Navigation Task**

   .. code-block:: python

      from canns.task.open_loop_navigation import OpenLoopNavigationTask

      task = OpenLoopNavigationTask(
          width=5,
          height=5,
          speed_mean=0.04,      # Average speed
          speed_std=0.016,      # Speed variability
          duration=50000.0,     # Duration (milliseconds)
          dt=0.05,              # Time step
          start_pos=(2.5, 2.5), # Starting position
          progress_bar=True
      )

   **Explanation**:
   - Simulates random walk in a 5×5 environment
   - Speed fluctuates over time (biological realism)
   - Returns velocity and position trajectories

2. **Initialize the Hierarchical Network**

   .. code-block:: python

      from canns.models.basic import HierarchicalNetwork

      network = HierarchicalNetwork(
          num_module=5,      # 5 grid cell modules
          num_place=30,      # 30 place cells
      )
      network.init_state()

   **Explanation**:
   - Contains grid cell modules at multiple scales
   - Place cells are formed by integrating grid inputs
   - Simulates realistic navigation systems in mammals

3. **Run Path Integration**

   .. code-block:: python

      def run_step(t, velocity, position):
          network(
              velocity=velocity,     # Velocity input
              loc=position,          # Current position (for learning)
              loc_input_stre=0.      # Localization input strength
          )
          return (
              network.band_x_fr.value,
              network.band_y_fr.value,
              network.grid_fr.value,
              network.place_fr.value
          )

      # Compile and run
      results = brainstate.compile.for_loop(
          run_step,
          time_indices,
          velocity_data,
          position_data
      )

   **Explanation**:
   - Band cells encode movement direction (X and Y)
   - Grid cells form periodic grid patterns
   - Place cells encode specific locations

4. **Analyze Location Encoding**

   .. code-block:: python

      from canns.analyzer.spatial import compute_firing_field

      # Compute firing heatmap for each cell
      grid_heatmaps = compute_firing_field(
          grid_activity,
          animal_trajectory,
          width=5,
          height=5
      )

   **Explanation**:
   - Heatmaps show activity patterns of cells in the environment
   - Grid cells should display regular hexagonal patterns
   - Place cells should display specific "place fields"

Results
-------

Running this script will generate:

1. **Trajectory Plot** (`trajectory_graph.png`)

   - Shows the animal's movement trajectory in the environment
   - Helps understand the complexity of input data

2. **Neural Activity Heatmaps**

   - **Grid cell heatmaps**: Regular hexagonal patterns
   - **Band cell heatmaps**: Representation of direction and movement
   - **Place cell heatmaps**: Single or multiple place fields

3. **Performance Metrics**

   - Simulation time: ~10-30 seconds
   - Memory usage: ~500 MB
   - Generated files: ~50 heatmaps

Key Concepts
------------

**Mathematical Model of Path Integration**

Position updates follow a simple integration equation:

.. math::

   \vec{p}(t+\Delta t) = \vec{p}(t) + \vec{v}(t) \cdot \Delta t

Where:
- p(t): current position
- v(t): velocity
- Δt: time step

**Hexagonal Encoding of Grid Cells**

Grid cells form regular hexagonal grids:

.. code-block:: text

   Top view:
   ○───○───○───○
    ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
   ○───○───○───○
    ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
   ○───○───○───○

   - Each grid cell corresponds to a hexagon center
   - Modules at multiple scales encode different resolutions
   - Discovered by Moser & Moser (2005, Nobel Prize)

**Multi-Module Integration**

Advantages of hierarchical structure:

.. code-block:: text

   Large-scale grid module (>1m)
        ↓
   Medium-scale grid module
        ↓
   Small-scale grid module (<10cm)
        ↓
   Place cells

   - Each module performs path integration independently
   - Scale ratio between different modules is 2-3 fold
   - Combination produces precise location representation

Experimental Variations
-----------------------

**1. Change Environment Size**

.. code-block:: python

   # Large environment (more place cell activity)
   task = OpenLoopNavigationTask(width=10, height=10)

   # Small environment (simpler representation)
   task = OpenLoopNavigationTask(width=2, height=2)

**2. Analyze Error Accumulation**

.. code-block:: python

   # Compare network-predicted positions with actual positions
   predicted_positions = decode_from_place_cells(place_activity)
   actual_positions = task.data.position

   position_error = np.linalg.norm(
       predicted_positions - actual_positions,
       axis=1
   )

   print(f"Average position error: {np.mean(position_error):.3f}m")
   print(f"Maximum position error: {np.max(position_error):.3f}m")

**3. Change Speed Statistics**

.. code-block:: python

   # Fast movement (difficult to track)
   task = OpenLoopNavigationTask(
       speed_mean=0.08,
       speed_std=0.032
   )

   # Slow movement (easy to track precisely)
   task = OpenLoopNavigationTask(
       speed_mean=0.01,
       speed_std=0.004
   )

**4. Analyze Grid Spacing of Grid Cells**

.. code-block:: python

   import numpy as np
   from scipy.fft import fft2

   # Compute spectrum of heatmaps
   for module_idx, heatmap in enumerate(grid_heatmaps):
       fft_result = np.abs(fft2(heatmap))
       # Extract grid spacing from spectrum
       spacing = analyze_grid_spacing(fft_result)
       print(f"Module {module_idx} grid spacing: {spacing:.2f}cm")

Related APIs
-----------

- :class:`~src.canns.models.basic.HierarchicalNetwork` - Hierarchical navigation network
- :class:`~src.canns.task.open_loop_navigation.OpenLoopNavigationTask` - Open-loop navigation task
- :func:`~src.canns.analyzer.spatial.compute_firing_field` - Firing field heatmap computation

Biological Background
---------------------

**Entorhinal Cortex**

The medial entorhinal cortex (MEC) contains:

1. **Grid cells**: Account for 5-10% of approximately 60,000 MEC cells
   - Regular hexagonal grid patterns
   - Different spacing in modules at multiple scales
   - Provide a metric for path integration

2. **Head direction cells**: Encode the animal's head direction
   - Tuning curves are bell-shaped
   - Globally polarized, aligned with the environment

**Hippocampus**

- **Place cells**: Strong responses at specific locations
- Formed by integrating grid and direction information
- Support spatial memory and planning

**Biological Path Integration Abilities**

- Desert ants can return to their nest without visual cues
- Honeybees perform path integration during waggle dances in the hive
- Rats can construct location maps even in complete darkness

More Resources
--------------

- :doc:`hierarchical_network` - Understanding hierarchical structure
- :doc:`theta_modulation` - The role of theta oscillations in navigation
- :doc:`grid_place_cells` - Relationships between grid cells and place cells

Frequently Asked Questions
--------------------------

**Q: Why use grid cells?**

A: Grid cells provide efficient spatial encoding:
   - Hexagonal encoding is more efficient than Cartesian coordinates
   - Multi-module structure allows arbitrary precision
   - Matches biological observations

**Q: Does position error grow infinitely?**

A: Yes, in open-loop path integration without correction, errors accumulate. However, the brain has mechanisms to correct them:
   - Visual input (landmarks) resets position
   - Olfactory cues (odor)
   - Tactile feedback
   - Proprioceptive information

**Q: How to decode position from cell activity?**

A: Maximum likelihood estimation can be used:

   .. code-block:: python

      def decode_position(place_cell_activity):
          # Assume each place cell is Gaussian tuned
          likelihood = np.ones(num_positions)
          for cell_idx, activity in enumerate(place_cell_activity):
              likelihood *= likelihood_given_activity(
                  activity,
                  place_field_of_cell[cell_idx]
              )
          return positions[np.argmax(likelihood)]

Next Steps
----------

After completing this tutorial, it is recommended to:

1. Analyze the roles of different grid modules
2. Test learning with visual cues
3. Compare navigation abilities of different animals
4. Read :doc:`grid_place_cells` to understand interactions between cells