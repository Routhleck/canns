Tasks and Navigation Examples
=============================

The scripts in ``examples/cann/`` and ``examples/pipeline/`` show how to construct trajectory
inputs with :mod:`~src.canns.task`, import external recordings, and drive more elaborate models.

import_external_trajectory.py
-----------------------------

- **Location**: ``examples/cann/import_external_trajectory.py``
- **Goal**: Replace the default random walk in :class:`~src.canns.task.spatial_navigation.SpatialNavigationTask`
  with external position samples.
- **Workflow**:

  1. Generate (or load) a noisy random walk trajectory.
  2. Initialise :class:`~src.canns.task.spatial_navigation.SpatialNavigationTask` and call
     ``import_data(position_data=..., times=...)``.
  3. Run ``calculate_theta_sweep_data()`` to compute linear and angular speed gains for later theta-sweep analysis.
  4. Produce summary figures with ``show_trajectory_analysis`` and optional matplotlib overlays.
- **Output**: ``import_external_trajectory.png``, ``our_data_comparison.png`` and console diagnostics.
- **Extensions**:

  - Swap ``positions`` for recorded experiments; include ``head_direction`` if you already have orientation data.
  - Persist the dataset via ``snt.save_data(...)`` so downstream scripts can reuse it.

hierarchical_path_integration.py
--------------------------------

- **Location**: ``examples/cann/hierarchical_path_integration.py``
- **Goal**: Demonstrate the hierarchical path-integration network coupled to ``SpatialNavigationTask``.
- **Workflow**:

  1. Simulate a long navigation session (``duration=1000``) and store it as ``trajectory_test.npz``.
  2. Build :class:`~src.canns.models.basic.hierarchical_model.HierarchicalNetwork`,
     which stacks band, grid, and place-cell populations.
  3. Use ``brainstate.compile.for_loop`` to prime the network (``loc_input_stre`` warm-up) and then run the full trajectory.
  4. Compare compiled performance with :func:`~src.canns.misc.benchmark.benchmark`.
- **Output**: ``trajectory_graph.png`` and ``band_grid_place_activity.npz`` (optional).
- **Extensions**:

  - Combine with :doc:`models` to explore how connection parameters influence integration accuracy.
  - Replace the random walk with ``SpatialNavigationTask.import_data`` to replay experimental paths.

Tips
----

- ``SpatialNavigationTask`` depends on ``Ratinabox`` and will create the default environment on first run.
  You can customise layouts by passing ``walls`` or ``objects``.
- For batch simulations, loop over ``task.get_data()`` and write each dataset to disk—the pipeline examples
  happily consume cached trajectories.
