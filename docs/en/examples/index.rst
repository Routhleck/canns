Examples Overview
==================

The following lists commonly used example scripts and notebooks in the repository. All examples are located in
``examples/`` under the project root directory, can be run locally, or experienced online directly through the Binder / Colab
buttons provided in the README.

.. list-table:: Featured Examples
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Description
   * - ``examples/brain_inspired/hopfield_train.py``
     - Train ``AmariHopfieldNetwork`` using the unified ``HebbianTrainer``, performing pattern recovery on noisy images.
   * - ``examples/brain_inspired/hopfield_train_mnist.py``
     - Store MNIST characters in a Hopfield network, demonstrating the performance of the same training process on real datasets.
   * - ``examples/cann/cann1d_oscillatory_tracking.py``
     - Run oscillatory tracking in a 1D CANN and use plotting tools to generate energy landscape animations.
   * - ``examples/cann/cann2d_tracking.py``
     - Demonstrate smooth tracking of a 2D CANN and export energy landscape animations through configuration-based plotting.
   * - ``examples/experimental_cann1d_analysis.py``
     - Load ROI activity, call the experimental data analyzer to fit 1D bumps, and export frame-by-frame GIFs.
   * - ``examples/experimental_cann2d_analysis.py``
     - Perform spike embedding, UMAP and TDA analysis on 2D experimental data, and generate torus visualizations.
   * - ``examples/pipeline/theta_sweep_from_external_data.py``
     - Import external trajectories and run the advanced ``ThetaSweepPipeline`` for direction/grid cell analysis.
   * - ``examples/pipeline/advanced_theta_sweep_pipeline.py``
     - Demonstrate complete parameter configuration of the theta-sweep pipeline, suitable for advanced user reference.

More scripts can be found in the `GitHub examples directory
<https://github.com/routhleck/canns/tree/master/examples>`_.
