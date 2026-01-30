canns.analyzer.metrics.spatial_metrics
======================================

.. py:module:: canns.analyzer.metrics.spatial_metrics

.. autoapi-nested-parse::

   Spatial analysis utilities for neural activity data.

   This module provides functions for analyzing spatial patterns in neural data,
   particularly for computing firing fields and spatial smoothing operations.
   Includes specialized functions for grid cell analysis such as spatial
   autocorrelation, grid scores, and spacing measurements.



Functions
---------

.. autoapisummary::

   canns.analyzer.metrics.spatial_metrics.compute_firing_field
   canns.analyzer.metrics.spatial_metrics.compute_grid_score
   canns.analyzer.metrics.spatial_metrics.compute_spatial_autocorrelation
   canns.analyzer.metrics.spatial_metrics.find_grid_spacing
   canns.analyzer.metrics.spatial_metrics.gaussian_smooth_heatmaps


Module Contents
---------------

.. py:function:: compute_firing_field(A, positions, width, height, M, K)

   Compute spatial firing fields for neural population activity.

   This function bins neural activity into a 2D spatial grid based on
   (x, y) positions. The input shapes match the usage patterns in analyzer
   tests: activity is ``(T, N)`` and positions is ``(T, 2)``.

   :param A: Neural activity of shape ``(T, N)``.
   :type A: np.ndarray
   :param positions: Positions of shape ``(T, 2)``.
   :type positions: np.ndarray
   :param width: Environment width.
   :type width: float
   :param height: Environment height.
   :type height: float
   :param M: Number of bins along width.
   :type M: int
   :param K: Number of bins along height.
   :type K: int

   :returns: Heatmaps of shape ``(N, M, K)``.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.metrics.spatial_metrics import compute_firing_field
   >>>
   >>> # Dummy inputs (T timesteps, N neurons)
   >>> activity = np.random.rand(100, 3)
   >>> positions = np.column_stack(
   ...     [np.linspace(0, 1.0, 100), np.linspace(0, 1.0, 100)]
   ... )
   >>>
   >>> heatmaps = compute_firing_field(activity, positions, 1.0, 1.0, 10, 10)
   >>> print(heatmaps.shape)
   (3, 10, 10)


.. py:function:: compute_grid_score(autocorr, annulus_inner = 0.3, annulus_outer = 0.7)

   Compute grid score from spatial autocorrelation.

   :param autocorr: 2D spatial autocorrelation map.
   :type autocorr: np.ndarray
   :param annulus_inner: Inner radius of annulus (fraction of map size).
   :type annulus_inner: float
   :param annulus_outer: Outer radius of annulus (fraction of map size).
   :type annulus_outer: float

   :returns: ``(grid_score, rotated_corrs)`` where ``rotated_corrs`` maps
             angles ``{30, 60, 90, 120, 150}`` to correlation values.
   :rtype: tuple

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.metrics.spatial_metrics import compute_grid_score
   >>>
   >>> autocorr = np.random.rand(15, 15)
   >>> grid_score, rotated_corrs = compute_grid_score(autocorr)
   >>> print(sorted(rotated_corrs.keys()))
   [30, 60, 90, 120, 150]


.. py:function:: compute_spatial_autocorrelation(rate_map, max_lag = None)

   Compute 2D spatial autocorrelation of a firing rate map.

   :param rate_map: 2D firing rate map of shape ``(M, K)``.
   :type rate_map: np.ndarray
   :param max_lag: Optional max lag for cropping around the center.
   :type max_lag: int | None

   :returns: Autocorrelation map normalized to ``[-1, 1]``.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.metrics.spatial_metrics import compute_spatial_autocorrelation
   >>>
   >>> rate_map = np.random.rand(10, 10)
   >>> autocorr = compute_spatial_autocorrelation(rate_map)
   >>> print(autocorr.shape)
   (10, 10)


.. py:function:: find_grid_spacing(autocorr, bin_size = None)

   Estimate grid spacing from spatial autocorrelation.

   :param autocorr: 2D autocorrelation map.
   :type autocorr: np.ndarray
   :param bin_size: Spatial bin size in real units. If provided,
                    the function also returns spacing in real units.
   :type bin_size: float | None

   :returns: ``(spacing_bins, spacing_real)``.
   :rtype: tuple

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.metrics.spatial_metrics import find_grid_spacing
   >>>
   >>> autocorr = np.random.rand(20, 20)
   >>> spacing_bins, spacing_m = find_grid_spacing(autocorr, bin_size=0.05)
   >>> print(spacing_m is not None)
   True


.. py:function:: gaussian_smooth_heatmaps(heatmaps, sigma = 1.0)

   Apply Gaussian smoothing to spatial heatmaps without mixing channels.

   :param heatmaps: Array of shape ``(N, M, K)``.
   :type heatmaps: np.ndarray
   :param sigma: Gaussian kernel width. Defaults to ``1.0``.
   :type sigma: float, optional

   :returns: Smoothed heatmaps with the same shape as input.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> import numpy as np
   >>> from canns.analyzer.metrics.spatial_metrics import gaussian_smooth_heatmaps
   >>>
   >>> heatmaps = np.random.rand(2, 5, 5)
   >>> smoothed = gaussian_smooth_heatmaps(heatmaps, sigma=1.0)
   >>> print(smoothed.shape)
   (2, 5, 5)


