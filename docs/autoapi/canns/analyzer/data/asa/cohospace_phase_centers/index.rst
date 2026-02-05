canns.analyzer.data.asa.cohospace_phase_centers
===============================================

.. py:module:: canns.analyzer.data.asa.cohospace_phase_centers

.. autoapi-nested-parse::

   Phase-center utilities for CohoSpace (skewed coordinates).



Functions
---------

.. autoapisummary::

   canns.analyzer.data.asa.cohospace_phase_centers.cohospace_phase_centers
   canns.analyzer.data.asa.cohospace_phase_centers.plot_cohospace_phase_centers


Module Contents
---------------

.. py:function:: cohospace_phase_centers(cohospace_result)

   Compute per-neuron CohoSpace phase centers and their skewed coordinates.

   Input
   -----
   cohospace_result : dict
       Output from `data.cohospace(...)` (must include `centers`).


.. py:function:: plot_cohospace_phase_centers(cohospace_result, *, neuron_id = None, show_all = False, config = None, save_path = None, show = False, figsize = (5, 5), all_color = 'tab:blue', highlight_color = 'tab:red', alpha = 0.7, s = 12)

   Plot CohoSpace phase centers on the skewed torus domain.

   If neuron_id is None, plot all neurons. If neuron_id is provided, show_all controls
   whether all neurons are drawn lightly or only the selected neuron is shown.


