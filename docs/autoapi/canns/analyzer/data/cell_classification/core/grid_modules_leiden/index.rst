canns.analyzer.data.cell_classification.core.grid_modules_leiden
================================================================

.. py:module:: canns.analyzer.data.cell_classification.core.grid_modules_leiden


Functions
---------

.. autoapisummary::

   canns.analyzer.data.cell_classification.core.grid_modules_leiden.identify_grid_modules_and_stats


Module Contents
---------------

.. py:function:: identify_grid_modules_and_stats(autocorrs, *, gridness_analyzer, center_bins = 2, k = 30, resolution = 1.0, score_thr = 0.3, consistency_thr = 0.5, min_cells = 10, merge_corr_thr = 0.7, metric = 'manhattan')

   Identify grid modules with Leiden clustering on autocorrelogram point cloud.

   :param autocorrs: Array of shape (N, H, W).
   :type autocorrs: np.ndarray
   :param gridness_analyzer: An instance that provides compute_gridness_score(autocorr)->GridnessResult.
   :param center_bins: Radius (in bins) to mask around center peak.
   :type center_bins: int
   :param k: Neighbors for kNN graph.
   :type k: int
   :param resolution: Leiden resolution parameter.
   :type resolution: float
   :param score_thr: Module acceptance and merging thresholds.
   :param consistency_thr: Module acceptance and merging thresholds.
   :param min_cells: Module acceptance and merging thresholds.
   :param merge_corr_thr: Module acceptance and merging thresholds.

   :returns: module_id (N,), cluster_id (N,), modules (list of dict), params
   :rtype: dict with keys


