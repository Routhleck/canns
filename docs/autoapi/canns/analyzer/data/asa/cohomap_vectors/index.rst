canns.analyzer.data.asa.cohomap_vectors
=======================================

.. py:module:: canns.analyzer.data.asa.cohomap_vectors

.. autoapi-nested-parse::

   Stripe vectors and diagnostics for CohoMap.



Functions
---------

.. autoapisummary::

   canns.analyzer.data.asa.cohomap_vectors.cohomap_vectors
   canns.analyzer.data.asa.cohomap_vectors.plot_cohomap_stripes
   canns.analyzer.data.asa.cohomap_vectors.plot_cohomap_vectors


Module Contents
---------------

.. py:function:: cohomap_vectors(cohomap_result, *, grid_size = 151, trim = 25, angle_grid = 10, phase_grid = 10, spacing_grid = 10, spacing_range = (1.0, 6.0))

   Fit CohoMap stripe parameters and compute parallelogram vectors (v, w).

   Returns a dict containing the stripe fit, rotated parameters, vector components,
   and angle (deg) following GridCellTorus conventions.


.. py:function:: plot_cohomap_stripes(cohomap_result, *, cohomap_vectors_result = None, grid_size = 151, trim = 25, angle_grid = 10, phase_grid = 10, spacing_grid = 10, spacing_range = (1.0, 6.0), config = None, save_path = None, show = False, figsize = (10, 6), cmap = 'viridis')

   Plot stripe fit diagnostics for CohoMap (observed vs fitted stripes).


.. py:function:: plot_cohomap_vectors(cohomap_vectors_result, *, config = None, save_path = None, show = False, figsize = (5, 5), color = '#f28e2b')

   Plot v/w vectors and the parallelogram in spatial coordinates.


