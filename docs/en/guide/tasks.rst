Tasks and Navigation Examples
=============================

The scripts in ``examples/cann/`` and ``examples/pipeline/`` show how to construct trajectory
inputs with :mod:`~src.canns.task`, import external recordings, and drive more elaborate models.

import_external_trajectory.py
-----------------------------

- **Location**: ``examples/cann/import_external_trajectory.py``
- **Goal**: Replace the default random walk in :class:`~src.canns.task.open_loop_navigation.OpenLoopNavigationTask`
  with external position samples.
- **Workflow**:

  1. Generate (or load) a noisy random walk trajectory.
  2. Initialise :class:`~src.canns.task.open_loop_navigation.OpenLoopNavigationTask` and call
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
- **Goal**: Demonstrate the hierarchical path-integration network coupled to ``OpenLoopNavigationTask``.
- **Workflow**:

  1. Simulate a long navigation session (``duration=1000``) and store it as ``trajectory_test.npz``.
  2. Build :class:`~src.canns.models.basic.hierarchical_model.HierarchicalNetwork`,
     which stacks band, grid, and place-cell populations.
  3. Use ``brainstate.compile.for_loop`` to prime the network (``loc_input_stre`` warm-up) and then run the full trajectory.
  4. Compare compiled performance with :func:`~src.canns.utils.benchmark.benchmark`.
- **Output**: ``trajectory_graph.png`` and ``band_grid_place_activity.npz`` (optional).
- **Extensions**:

  - Combine with :doc:`models` to explore how connection parameters influence integration accuracy.
  - Replace the random walk with ``OpenLoopNavigationTask.import_data`` to replay experimental paths.


Tips
----

- ``OpenLoopNavigationTask`` relies on the ``canns_lib`` spatial backend for agent/environment
  simulation and will create the default environment on first run. You can customise layouts by
  passing ``walls`` or ``objects``.
- For batch simulations, loop over ``task.get_data()`` and write each dataset to disk—the pipeline examples
  happily consume cached trajectories.


Closed-loop navigation utilities
--------------------------------

- **Location**: ``src/canns/task/closed_loop_navigation.py``
- **Goal**: Provide environment-aware movement planning tools, including cost-grid generation and
  geodesic visualisations, on top of the ``canns_lib`` closed-loop agent implementation.
- **Workflow**:

  1. Instantiate :class:`~src.canns.task.closed_loop_navigation.ClosedLoopNavigationTask`
     (or the :class:`~src.canns.task.closed_loop_navigation.TMazeClosedLoopNavigationTask` helper).
  2. Call :meth:`~src.canns.task.closed_loop_navigation.ClosedLoopNavigationTask.build_movement_cost_grid`
     with your desired ``dx``/``dy`` resolution to obtain a labelled grid map where blocked cells carry
     ``INT32_MAX`` weight.
  3. Overlay the grid on the agent trajectory via ``show_data(overlay_movement_cost=True, cost_grid=...)``
     to inspect obstacles, or render the pairwise shortest-path distances with
     :meth:`~src.canns.task.closed_loop_navigation.ClosedLoopNavigationTask.show_geodesic_distance_matrix`.
- **Output**: Annotated matplotlib figures that highlight traversable cells vs. walls/holes and a dense
  geodesic distance matrix for custom planners.
- **Extensions**:

  - Feed the returned :class:`~src.canns.task.closed_loop_navigation.MovementCostGrid` into other planners or
    export it to disk for debugging.
  - Use the accompanying pytest in ``tests/task/closed_loop_navigation`` as a template for custom maze
    regression tests.
  - Render a more complex environment by running ``uv run python examples/cann/closed_loop_complex_environment.py``,
    which saves both the movement-cost overlay and geodesic heatmap to ``figures/closed_loop_complex``.
