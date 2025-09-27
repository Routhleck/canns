Theta Sweep Pipeline Examples
=============================

:class:`~src.canns.pipeline.theta_sweep.ThetaSweepPipeline` bundles navigation tasks, direction/grid-cell models,
and visualisation utilities into an end-to-end workflow. The scripts below illustrate a minimal run and a fully customised setup.

theta_sweep_from_external_data.py
---------------------------------

- **Location**: ``examples/pipeline/theta_sweep_from_external_data.py``
- **Scenario**: Generate a smooth closed-loop trajectory (or load a recorded path) and run the complete theta-sweep pipeline.
- **Workflow**:

  1. Create ``times`` and ``positions`` arrays—either synthetic Catmull–Rom samples or an imported dataset.
  2. Instantiate :class:`~src.canns.pipeline.theta_sweep.ThetaSweepPipeline` with the default model and theta parameters.
  3. Call ``pipeline.run(output_dir="theta_sweep_results")`` to render animations and summary plots.
- **Output**:

  - ``theta_sweep_results/`` containing GIF/MP4 animations, population-activity heat maps, and trajectory diagnostics.
  - Console summary with duration and save paths.
- **Extensions**:

  - Combine with :doc:`tasks` (``import_external_trajectory.py``) to replay experimental trajectories.
  - Adjust ``env_size``/``dt`` or runtime options ``animation_fps`` and ``animation_dpi`` for quality/performance trade-offs.

advanced_theta_sweep_pipeline.py
--------------------------------

- **Location**: ``examples/pipeline/advanced_theta_sweep_pipeline.py``
- **Scenario**: Expose every configuration knob—network sizes, theta parameters, output settings, and verbose reports.
- **Workflow**:

  1. Build a deterministic L-shaped trajectory with controlled perturbations.
  2. Pass custom ``direction_cell_params`` / ``grid_cell_params`` / ``theta_params`` / ``spatial_nav_params`` when constructing the pipeline.
  3. Execute ``run(..., save_animation=True, save_plots=True, verbose=True)`` and inspect the returned ``results`` dictionary.
  4. Extract arrays such as ``gc_activity`` and ``theta_phase`` from ``results["data"]`` for additional analysis.
- **Output**: ``advanced_theta_sweep_results/`` containing animations, figures, and cached simulation tensors.
- **Extensions**:

  - Experiment with ``theta_strength_hd/gc`` or ``theta_cycle_len`` to compare rhythm settings.
  - Export ``grid_activity`` / ``dc_activity`` as ``.npz`` files for comparison with experimental recordings.

Usage tips
----------

- Both scripts depend on :mod:`~src.canns.task.spatial_navigation` and :mod:`~src.canns.analyzer.theta_sweep`.
  Revisit :doc:`tasks` or :doc:`models` if you need implementation details.
- Animation rendering can take a few minutes—watch the progress bar. On headless machines keep ``show=False`` or install ``imageio[ffmpeg]``.
