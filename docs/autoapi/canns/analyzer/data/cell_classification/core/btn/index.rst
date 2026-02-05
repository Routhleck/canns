canns.analyzer.data.cell_classification.core.btn
================================================

.. py:module:: canns.analyzer.data.cell_classification.core.btn

.. autoapi-nested-parse::

   BTN (Bursty/Theta/Non-bursty) classification.

   Workflow:
   1) Compute ISI autocorrelograms from spike times.
   2) Normalize and smooth each autocorr curve.
   3) Compute cosine distance between curves.
   4) Cluster with Tomato using a manual kNN graph and density weights.



Attributes
----------

.. autoapisummary::

   canns.analyzer.data.cell_classification.core.btn.Tomato


Classes
-------

.. autoapisummary::

   canns.analyzer.data.cell_classification.core.btn.BTNAnalyzer
   canns.analyzer.data.cell_classification.core.btn.BTNConfig
   canns.analyzer.data.cell_classification.core.btn.BTNResult


Module Contents
---------------

.. py:class:: BTNAnalyzer(config = None)

   Analyzer that clusters neurons into BTN groups using Tomato.


   .. py:method:: classify_btn(spike_data, *, mapping = None, return_intermediates = False, plot_diagram = False)

      Cluster neurons into BTN classes using ISI autocorr + Tomato.

      :param spike_data: ASA-style dict with keys ``spike`` and ``t`` (and optionally x/y).
      :type spike_data: dict
      :param mapping: Optional mapping from cluster id to BTN label string.
      :type mapping: dict, optional
      :param return_intermediates: If True, include intermediate arrays in the result.
      :type return_intermediates: bool
      :param plot_diagram: If True, call Tomato.plot_diagram() for visual inspection.
      :type plot_diagram: bool



   .. py:attribute:: config


.. py:class:: BTNConfig

   Configuration for BTN clustering.


   .. py:attribute:: b_log
      :type:  bool
      :value: False



   .. py:attribute:: b_one
      :type:  bool
      :value: True



   .. py:attribute:: maxt
      :type:  float
      :value: 0.2



   .. py:attribute:: metric
      :type:  str
      :value: 'cosine'



   .. py:attribute:: n_clusters
      :type:  int
      :value: 4



   .. py:attribute:: nbs
      :type:  int
      :value: 80



   .. py:attribute:: res
      :type:  float
      :value: 0.001



   .. py:attribute:: smooth_sigma
      :type:  float
      :value: 4.0



.. py:class:: BTNResult

   Result container for BTN clustering.


   .. py:attribute:: btn_labels
      :type:  numpy.ndarray | None


   .. py:attribute:: cluster_sizes
      :type:  numpy.ndarray


   .. py:attribute:: intermediates
      :type:  dict[str, numpy.ndarray] | None


   .. py:attribute:: labels
      :type:  numpy.ndarray


   .. py:attribute:: mapping
      :type:  dict[int, str] | None


   .. py:attribute:: n_clusters
      :type:  int


.. py:data:: Tomato
   :value: None


