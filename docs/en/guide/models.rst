CANN Network Examples
=====================

This chapter covers the scripts under ``examples/cann/`` that drive CANN models
directly—1D/2D bump tracking, tuning curves, and theta-modulated visualisations.
Each section lists the APIs involved, the typical output, and ideas for extending the demo.

cann1d_oscillatory_tracking.py
------------------------------

- **Location**: ``examples/cann/cann1d_oscillatory_tracking.py``
- **Goal**: Run a 1D CANN under :class:`~src.canns.task.tracking.SmoothTracking1D` input and export an energy-landscape animation.
- **Key APIs**:

  - :class:`~src.canns.models.basic.CANN1D`
  - :class:`~src.canns.task.tracking.SmoothTracking1D`
  - :func:`~src.canns.analyzer.plotting.energy_landscape_1d_animation`
- **Workflow**:

  1. Configure ``brainstate.environ.set(dt=0.1)`` and initialise a 512-neuron CANN.
  2. Define ``Iext`` and ``duration`` sequences, then call ``task.get_data()`` to build the stimulus buffer.
  3. Use ``brainstate.compile.for_loop`` to advance the network and collect state/history arrays.
  4. Render the animation via :func:`~src.canns.analyzer.plotting.energy_landscape_1d_animation`; presets from ``PlotConfigs.energy_landscape_1d_animation`` still apply.
- **Output**: produces ``test_smooth_tracking_1d.gif`` (or the path you supply).
- **Try next**:

  - Swap in :class:`~src.canns.models.basic.CANN1D_SFA` to compare adaptation behaviour.
  - Feed a more complex stimulus sequence and pair the run with the tuning-curve utility below.

cann1d_tuning_curve.py
----------------------

- **Location**: ``examples/cann/cann1d_tuning_curve.py``
- **Goal**: Sample firing-rate statistics from selected neurons and plot their tuning curves.
- **Key APIs**:

  - :class:`~src.canns.models.basic.CANN1D`
  - :class:`~src.canns.task.tracking.SmoothTracking1D`
  - :func:`~src.canns.analyzer.plotting.tuning_curve`
- **Workflow**:

  1. Build a 1D CANN spanning ``[-π, π]`` with several stimulus transitions.
  2. Run the simulation, recording both firing rates and external inputs.
  3. Use ``PlotConfigs.tuning_curve`` to configure neuron indices, bin count, and styling.
  4. Call :func:`~src.canns.analyzer.plotting.tuning_curve`; set ``save_path`` in the config to persist the plot.
- **Output**: interactive matplotlib figure or PNG/SVG (depending on the config).
- **Try next**:

  - Expand ``neuron_indices_to_plot`` or align the preferred-stimulus axis with ``pref_stim=cann.x``.
  - Combine with :meth:`~src.canns.trainer.HebbianTrainer.predict` to compare tuning before and after learning.

cann2d_tracking.py
------------------

- **Location**: ``examples/cann/cann2d_tracking.py``
- **Goal**: Demonstrate a 2D CANN following a sequence of targets and generate a heat-map animation.
- **Key APIs**:

  - :class:`~src.canns.models.basic.CANN2D`
  - :class:`~src.canns.task.tracking.SmoothTracking2D`
  - :func:`~src.canns.analyzer.plotting.energy_landscape_2d_animation`
- **Workflow**:

  1. Instantiate a ``length=100`` CANN and initialise its state.
  2. Provide a list of two-dimensional ``Iext`` targets with matching ``duration`` values.
  3. Execute ``brainstate.compile.for_loop`` to step the network and collect ``u/r/inp`` tensors.
  4. Render the animation with ``PlotConfigs.energy_landscape_2d_animation``.
- **Output**: ``CANN2D_encoding.gif`` by default.
- **Try next**:

  - Tune ``length`` or ``time_steps_per_second`` to balance resolution and runtime.
  - Reuse the 1D scripts to compare dimensionality effects.

theta_sweep_grid_cell_network.py
--------------------------------

- **Location**: ``examples/cann/theta_sweep_grid_cell_network.py``
- **Goal**: Run the theta-modulated direction/grid-cell pair and showcase diagnostic plots and animations.
- **Key APIs**:

  - :class:`~src.canns.models.basic.theta_sweep_model.DirectionCellNetwork`
  - :class:`~src.canns.models.basic.theta_sweep_model.GridCellNetwork`
  - :mod:`~src.canns.analyzer.theta_sweep`
- **Workflow**:

  1. Build a :class:`~src.canns.task.open_loop_navigation.OpenLoopNavigationTask` to generate trajectories and theta gains.
  2. Step the networks with ``calculate_theta_modulation`` driving the oscillatory envelopes.
  3. Visualise the results using
     :func:`~src.canns.analyzer.theta_sweep.plot_population_activity_with_theta`,
     :func:`~src.canns.analyzer.theta_sweep.plot_grid_cell_manifold`, and
     :func:`~src.canns.analyzer.theta_sweep.create_theta_sweep_animation`.
- **Output**: ``theta_sweep_animation.gif`` plus intermediate figures.
- **Try next**:

  - Sweep ``mapping_ratio`` or ``theta_strength_*`` to study modulation depth.
  - Feed the generated data into :class:`~src.canns.pipeline.theta_sweep.ThetaSweepPipeline` for automated processing.

More scripts
------------

- :doc:`tasks` points to ``hierarchical_path_integration.py`` and ``import_external_trajectory.py``, which integrate the navigation task layer.
- If you only need a quick visual check, browse the pre-rendered GIF/PNG assets in the repository root—the files beginning with ``test_`` match the examples above.
