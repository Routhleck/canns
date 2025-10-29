Experimental Analysis Examples
==============================

When you need to process real neural recordings or produce richer visualisations, start with the
``examples/experimental_*`` scripts. They show how to combine
:mod:`~src.canns.analyzer.experimental_data`, :class:`~src.canns.analyzer.plotting.config.PlotConfig`, and
third-party tooling (UMAP, TDA, Numba, …) to build a full analysis pipeline.

experimental_cann1d_analysis.py
-------------------------------

- **Location**: ``examples/experimental_cann1d_analysis.py``
- **Data source**: ``load_roi_data()`` fetches ROI traces from the Hugging Face cache.
- **Analysis steps**:

  1. Run ``bump_fits`` (MCMC) to extract bump parameters frame by frame.
  2. Configure ``CANN1DPlotConfig.for_bump_animation`` for title, frame rate, and brightness limits.
  3. Call ``create_1d_bump_animation`` to generate the GIF; ``nframes`` and progress-bar settings are optional tweaks.
- **Output**: ``bump_analysis_demo.gif`` and summary statistics in the console.
- **Extensions**:

  - Tune ``n_steps`` / ``n_roi`` to match your dataset.
  - Without Numba the script falls back to NumPy—watch the startup message if you care about runtime.

experimental_cann2d_analysis.py
-------------------------------

- **Location**: ``examples/experimental_cann2d_analysis.py``
- **Data source**: ``load_grid_data()`` downloads spike and position data for grid cells.
- **Analysis steps**:

  1. Configure ``SpikeEmbeddingConfig`` (smoothing, speed filters) and run ``embed_spike_trains``.
  2. Reduce dimensionality with ``umap.UMAP`` and visualise using ``plot_projection``.
  3. Compute persistence with ``tda_vis`` + ``TDAConfig`` to validate torus topology.
  4. Decode phases via ``decode_circular_coordinates`` and animate with ``plot_3d_bump_on_torus``.
- **Output**: ``experimental_cann2d_analysis_torus.gif`` and supporting plots.
- **Extensions**:

  - Set ``do_shuffle`` / ``num_shuffles`` in ``tda_config`` for statistical tests.
  - Use ``save_path`` arguments to archive projections and barcodes.

Tools & dependencies
--------------------

- On first run the helpers create ``~/.canns/data`` and cache downloads automatically.
- Additional libraries you may need: ``umap-learn``, ``canns_lib.ripser`` (via ``canns-lib``), ``numba``, ``matplotlib``. Install them with:

  .. code-block:: bash

     uv add umap-learn canns-lib numba

- ``PlotConfig`` and ``CANN2DPlotConfig`` accept ``show=False`` so you can render figures on headless servers without
  changing the code.
