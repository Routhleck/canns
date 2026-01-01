src.canns.analyzer.metrics.systematic_ratemap
=============================================

.. py:module:: src.canns.analyzer.metrics.systematic_ratemap

.. autoapi-nested-parse::

   Systematic rate map sampling for grid cell analysis.

   This module provides functions for computing neural rate maps through systematic
   spatial sampling instead of trajectory-based methods. This approach ensures:
   - 100% spatial coverage (no gaps)
   - Uniform sampling density across all locations
   - Stable continuous attractor dynamics preservation
   - Higher quality grid scores for grid cell analysis

   The systematic sampling method samples the entire 2D spatial grid in a structured way:
   1. Horizontal sweep: Move from left to right, saving network state at each position
   2. Vertical sampling: From each horizontal position state, scan all vertical positions
   3. Assembly: Combine responses into complete (resolution × resolution) rate map

   This matches the approach used in the reference Burak & Fiete (2009) implementation
   and produces much cleaner grid cell firing fields compared to trajectory-based methods.

   .. rubric:: References

   Burak, Y., & Fiete, I. R. (2009). Accurate path integration in continuous
   attractor network models of grid cells. PLoS Computational Biology, 5(2), e1000291.

   .. rubric:: Example

   >>> from canns.models.basic import GridCell2DVelocity
   >>> from canns.analyzer.metrics.systematic_ratemap import compute_systematic_ratemap
   >>>
   >>> # Initialize and heal network
   >>> model = GridCell2DVelocity(length=40)
   >>> model.heal_network(num_healing_steps=5000)
   >>>
   >>> # Compute systematic rate map
   >>> ratemap = compute_systematic_ratemap(
   ...     model,
   ...     box_width=2.2,
   ...     box_height=2.2,
   ...     resolution=30,
   ...     speed=0.3,
   ... )
   >>> ratemap.shape  # (resolution, resolution, num_neurons)
   (30, 30, 1600)



Functions
---------

.. autoapisummary::

   src.canns.analyzer.metrics.systematic_ratemap.compute_systematic_ratemap


Module Contents
---------------

.. py:function:: compute_systematic_ratemap(model, box_width = 2.2, box_height = 2.2, resolution = 100, speed = 0.5, num_batches = 10, verbose = True)

   Compute rate maps by systematically sampling the entire spatial grid.

   This function implements the systematic spatial sampling approach used in
   the reference Burak & Fiete (2009) grid cell model. Instead of relying on
   trajectory coverage, it systematically samples every point in the spatial
   grid to ensure complete and uniform coverage.

   The algorithm works in three steps:
   1. **Horizontal sweep**: Move from left to right along the bottom edge,
      saving the network state at each horizontal position
   2. **Vertical sampling**: From each saved horizontal state, move upward
      sampling neural activity at each vertical position
   3. **Assembly**: Combine all sampled activities into a complete rate map

   IMPORTANT - Sampling Method:
   This is NOT a continuous trajectory. Instead, it uses state restoration:
   - Horizontal sweep (Step 1): Continuous movement via velocity input ✓
   - Vertical sampling (Step 2): For each x-position, the network state is
     RESTORED (jumped) to the saved state, then moved continuously upward

   Why use state restoration instead of continuous scanning?
   - Ensures uniform sampling density across all spatial locations
   - Computationally efficient (avoids redundant back-and-forth movement)
   - Maintains continuous attractor dynamics within each vertical sweep

   This is a standard technique for spatial analysis of grid cells, trading
   strict continuity for uniform spatial coverage. The bump activity remains
   stable because each restored state comes from continuous movement.

   CRITICAL: This function assumes the model has already been healed via
   `model.heal_network()` to establish a stable attractor state. The healed
   state is automatically preserved - DO NOT call `model.reset()` before or
   after this function as it will destroy the healed state.

   :param model: Grid cell model (must have been healed via model.heal_network())
   :param box_width: Arena width in meters (default: 2.2)
   :param box_height: Arena height in meters (default: 2.2)
   :param resolution: Number of spatial bins (creates resolution×resolution grid)
                      (default: 100)
   :param speed: Movement speed in m/s (default: 0.5)
   :param num_batches: Number of batches to split computation for memory efficiency
                       (default: 10)
   :param verbose: Print progress information (default: True)

   :returns:

             Rate map array of shape (resolution, resolution, num_neurons)
                 containing the neural activity at each spatial location
   :rtype: ratemap

   .. rubric:: Example

   >>> from canns.models.basic import GridCell2DVelocity
   >>> import brainpy.math as bm
   >>>
   >>> # Setup
   >>> bm.set_dt(5e-4)
   >>> model = GridCell2DVelocity(length=40, alpha=0.1, lambda_net=17.0)
   >>>
   >>> # CRITICAL: Heal network first to establish stable attractor
   >>> model.heal_network(num_healing_steps=5000)
   >>>
   >>> # Compute systematic rate map
   >>> ratemap = compute_systematic_ratemap(
   ...     model,
   ...     box_width=2.2,
   ...     box_height=2.2,
   ...     resolution=30,
   ...     speed=0.3,
   ...     num_batches=5,
   ... )
   >>>
   >>> # Verify shape and coverage
   >>> ratemap.shape
   (30, 30, 1600)
   >>> np.sum(np.isnan(ratemap))  # Should be 0 (100% coverage)
   0

   .. rubric:: Notes

   - Expected grid scores: 0.4-0.8 (much higher than trajectory-based methods)
   - Computation time scales with resolution^2 and num_neurons
   - For faster computation with slight quality tradeoff, reduce resolution
   - The healed state is automatically preserved; no need to save/restore it

   .. rubric:: References

   Burak & Fiete (2009). PLoS Computational Biology, 5(2), e1000291.
   See `.ref/grid_cells_burak_fiete/evaluate_grid.ipynb` for reference implementation.


