canns.analyzer.data.asa.cohomap
===============================

.. py:module:: canns.analyzer.data.asa.cohomap

.. autoapi-nested-parse::

   CohoMap (Ecoho-style) computation and plotting.



Functions
---------

.. autoapisummary::

   canns.analyzer.data.asa.cohomap.cohomap
   canns.analyzer.data.asa.cohomap.fit_cohomap_stripes
   canns.analyzer.data.asa.cohomap.plot_cohomap


Module Contents
---------------

.. py:function:: cohomap(decoding_result, position_data, *, coords_key = None, bins = 101, margin_frac = 0.0025, smooth_sigma = 1.0, fill_nan = True, fill_sigma = None, fill_min_weight = 0.001, align_torus = True, align_trim = 25, align_grid_size = None, align_min_valid_frac = None, align_max_fit_error = None)

   Compute EcohoMap phase maps using circular-mean binning.

   This mirrors GridCellTorus get_ang_hist: bin spatial positions and compute the
   circular mean of each decoded angle within spatial bins, then smooth in sin/cos
   space. Optional toroidal alignment follows the GridCellTorus stripe fit + rotation.
   You can gate alignment by valid fraction or fit error thresholds.


.. py:function:: fit_cohomap_stripes(phase_map, *, grid_size = 151, trim = 25, angle_grid = 10, phase_grid = 10, spacing_grid = 10, spacing_range = (1.0, 6.0))

   Fit a cosine stripe model to a phase map, mirroring GridCellTorus fit_sine_wave.


.. py:function:: plot_cohomap(cohomap_result, *, config = None, save_path = None, show = False, figsize = (10, 4), cmap = 'viridis', mode = 'cos')

   Plot EcohoMap phase maps (two panels: phase_map1/phase_map2).

   mode:
       "phase" to show raw phase (radians),
       "cos" or "sin" to show cosine/sine of phase like GridCellTorus.


