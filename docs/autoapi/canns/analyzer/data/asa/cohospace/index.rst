canns.analyzer.data.asa.cohospace
=================================

.. py:module:: canns.analyzer.data.asa.cohospace


Functions
---------

.. autoapisummary::

   canns.analyzer.data.asa.cohospace.cohospace
   canns.analyzer.data.asa.cohospace.plot_cohospace
   canns.analyzer.data.asa.cohospace.plot_cohospace_skewed


Module Contents
---------------

.. py:function:: cohospace(coords, spikes, *, times = None, coords_key = None, bins = 51, coords_in_unit = False, smooth_sigma = 0.0)

   Compute EcohoSpace rate maps and phase centers.

   Mirrors GridCellTorus get_ratemaps: mean activity in coho-space bins and
   a circular-mean center for each neuron. Optionally smooths the rate maps.


.. py:function:: plot_cohospace(cohospace_result, *, neuron_id = 0, config = None, save_path = None, show = False, figsize = (5, 5), cmap = 'viridis')

   Plot a single-neuron EcohoSpace rate map.


.. py:function:: plot_cohospace_skewed(cohospace_result, *, neuron_id = 0, config = None, save_path = None, show = False, figsize = (5, 5), cmap = 'viridis', show_grid = True)

   Plot a single-neuron EcohoSpace rate map in skewed torus coordinates.


