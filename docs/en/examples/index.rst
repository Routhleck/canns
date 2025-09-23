Examples Gallery
================

Below is a curated list of ready-to-run scripts and notebooks that demonstrate
common CANN workflows. All scripts live under the repository's ``examples/``
directory and can be launched locally or through the Binder / Colab badges on
the project README.

.. list-table:: Featured Examples
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Description
   * - ``examples/brain_inspired/hopfield_train.py``
     - Trains an ``AmariHopfieldNetwork`` with the unified ``HebbianTrainer`` and
       performs pattern completion on corrupted images.
   * - ``examples/brain_inspired/hopfield_train_mnist.py``
     - Stores MNIST exemplars in a Hopfield network, showing the same training
       pipeline on real datasets.
   * - ``examples/cann/cann1d_oscillatory_tracking.py``
     - Runs oscillatory tracking in a 1D CANN and generates energy-landscape
       animations with the plotting helpers.
   * - ``examples/cann/cann2d_tracking.py``
     - Demonstrates smooth tracking in a 2D CANN and exports animated energy
       landscapes using configuration-based plotting.
   * - ``examples/experimental_cann1d_analysis.py``
     - Loads ROI activity and fits 1D bumps via the experimental-data analyzers,
       producing GIFs of bump evolution.
   * - ``examples/experimental_cann2d_analysis.py``
     - Applies spike embedding, UMAP, and TDA utilities to 2D experimental data,
       concluding with torus visualisations.
   * - ``examples/pipeline/theta_sweep_from_external_data.py``
     - Imports external trajectories and executes the high-level
       ``ThetaSweepPipeline`` for direction/grid-cell analysis.
   * - ``examples/pipeline/advanced_theta_sweep_pipeline.py``
     - Shows full parameter customisation of the theta-sweep pipeline for power
       users.

For additional scripts and notebooks, explore the `examples folder on GitHub
<https://github.com/routhleck/canns/tree/master/examples>`_.
