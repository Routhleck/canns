src.canns.analyzer.metrics.spatial_metrics
==========================================

.. py:module:: src.canns.analyzer.metrics.spatial_metrics

.. autoapi-nested-parse::

   Spatial analysis utilities for neural activity data.

   This module provides functions for analyzing spatial patterns in neural data,
   particularly for computing firing fields and spatial smoothing operations.
   Includes specialized functions for grid cell analysis such as spatial
   autocorrelation, grid scores, and spacing measurements.



Functions
---------

.. autoapisummary::

   src.canns.analyzer.metrics.spatial_metrics.compute_firing_field
   src.canns.analyzer.metrics.spatial_metrics.compute_grid_score
   src.canns.analyzer.metrics.spatial_metrics.compute_spatial_autocorrelation
   src.canns.analyzer.metrics.spatial_metrics.find_grid_spacing
   src.canns.analyzer.metrics.spatial_metrics.gaussian_smooth_heatmaps


Module Contents
---------------

.. py:function:: compute_firing_field(A, positions, width, height, M, K)

   Compute spatial firing fields for neural population activity.

   This function bins neural activity into a 2D spatial grid based on the
   animal's position, creating a heatmap for each neuron showing where it
   fires most strongly. Uses Numba JIT compilation for high performance.

   :param A: Neural activity array of shape (T, N) where T is the
             number of time steps and N is the number of neurons.
   :type A: np.ndarray
   :param positions: Position data of shape (T, 2) containing
                     (x, y) coordinates at each time step.
   :type positions: np.ndarray
   :param width: Width of the spatial environment.
   :type width: float
   :param height: Height of the spatial environment.
   :type height: float
   :param M: Number of bins along the width dimension.
   :type M: int
   :param K: Number of bins along the height dimension.
   :type K: int

   :returns:

             Heatmaps array of shape (N, M, K) containing the average
                 firing rate of each neuron in each spatial bin.
   :rtype: np.ndarray

   .. rubric:: Example

   >>> activity = np.random.rand(1000, 30)  # 1000 timesteps, 30 neurons
   >>> positions = np.random.rand(1000, 2) * 5.0  # Random walk in 5x5 space
   >>> heatmaps = compute_firing_field(activity, positions, 5.0, 5.0, 50, 50)
   >>> heatmaps.shape
   (30, 50, 50)


.. py:function:: compute_grid_score(autocorr, annulus_inner = 0.3, annulus_outer = 0.7)

   Compute grid score from spatial autocorrelation.

   Grid score quantifies hexagonal symmetry by comparing correlations at 60°/120°
   (hexagonal angles) versus 30°/90°/150° (non-hexagonal angles). Developed by
   Sargolini et al. (2006), this is the standard metric for grid cell identification.

   Formula:
       grid_score = min(r60, r120) - max(r30, r90, r150)

   where rX is the Pearson correlation between the original and rotated autocorrelation
   within an annulus region.

   :param autocorr: 2D spatial autocorrelation map.
   :type autocorr: np.ndarray
   :param annulus_inner: Inner radius of annulus as fraction of map size.
                         Defaults to 0.3.
   :type annulus_inner: float
   :param annulus_outer: Outer radius of annulus as fraction of map size.
                         Defaults to 0.7.
   :type annulus_outer: float

   :returns: Grid score value. Values > 0.3 typically indicate grid cells.
             rotated_corrs (dict): Dictionary mapping rotation angles to correlation values.
                 Keys: 30, 60, 90, 120, 150 (degrees).
   :rtype: grid_score (float)

   .. rubric:: Example

   >>> from canns.analyzer.metrics.spatial_metrics import compute_spatial_autocorrelation, compute_grid_score
   >>> autocorr = compute_spatial_autocorrelation(rate_map)
   >>> grid_score, rotated_corrs = compute_grid_score(autocorr)
   >>> print(f"Grid score: {grid_score:.3f}")
   Grid score: 0.456
   >>> if grid_score > 0.3:
   ...     print("This is a grid cell!")

   .. rubric:: References

   Sargolini et al. (2006). Conjunctive representation of position, direction,
   and velocity in entorhinal cortex. Science, 312(5774), 758-762.


.. py:function:: compute_spatial_autocorrelation(rate_map, max_lag = None)

   Compute 2D spatial autocorrelation of a firing rate map.

   For grid cells, spatial autocorrelation reveals hexagonal periodicity patterns.
   This is the gold standard method for identifying grid cells in neuroscience.
   Uses periodic boundary conditions (wrap) appropriate for toroidal grid cell topology.

   :param rate_map: 2D firing rate map of shape (M, K).
   :type rate_map: np.ndarray
   :param max_lag: Maximum lag for output cropping. If None, returns
                   full autocorrelation map. Defaults to None.
   :type max_lag: int | None

   :returns:

             2D autocorrelation map normalized to [-1, 1]. For grid cells,
                 this will show a characteristic hexagonal pattern of peaks.
   :rtype: np.ndarray

   .. rubric:: Example

   >>> from canns.analyzer.metrics.spatial_metrics import compute_firing_field, compute_spatial_autocorrelation
   >>> # After computing firing fields
   >>> rate_map = firing_fields[0]  # First neuron
   >>> autocorr = compute_spatial_autocorrelation(rate_map)
   >>> # For grid cells, autocorr will show hexagonal pattern
   >>> autocorr.shape == rate_map.shape
   True

   .. rubric:: References

   Sargolini et al. (2006). Conjunctive representation of position, direction,
   and velocity in entorhinal cortex. Science, 312(5774), 758-762.


.. py:function:: find_grid_spacing(autocorr, bin_size = None)

   Estimate grid spacing from spatial autocorrelation.

   Finds the distance from center to the first major peak in the autocorrelation,
   which corresponds to the grid field spacing.

   :param autocorr: 2D autocorrelation map.
   :type autocorr: np.ndarray
   :param bin_size: Size of spatial bins in real units (e.g., meters).
                    If provided, returns spacing in real units. Defaults to None.
   :type bin_size: float | None

   :returns: Grid spacing in number of bins.
             spacing_real (float | None): Grid spacing in real units if bin_size provided,
                 otherwise None.
   :rtype: spacing_bins (float)

   .. rubric:: Example

   >>> from canns.analyzer.metrics.spatial_metrics import compute_spatial_autocorrelation, find_grid_spacing
   >>> autocorr = compute_spatial_autocorrelation(rate_map)
   >>> spacing_bins, spacing_m = find_grid_spacing(autocorr, bin_size=0.06)  # 6cm bins
   >>> print(f"Grid spacing: {spacing_bins:.1f} bins = {spacing_m:.3f} m")
   Grid spacing: 52.3 bins = 3.138 m


.. py:function:: gaussian_smooth_heatmaps(heatmaps, sigma = 1.0)

   Apply Gaussian smoothing to spatial heatmaps without mixing channels.

   This function applies Gaussian filtering to each heatmap independently,
   preserving zero values (unvisited spatial bins) and only smoothing regions
   with activity.

   :param heatmaps: Array of shape (N, M, K) where N is the number
                    of neurons/channels and (M, K) is the spatial grid size.
   :type heatmaps: np.ndarray
   :param sigma: Standard deviation for Gaussian kernel.
                 Defaults to 1.0.
   :type sigma: float, optional

   :returns:

             Smoothed heatmaps with the same shape as input. Zero values
                 in the original heatmaps are preserved.
   :rtype: np.ndarray

   .. rubric:: Example

   >>> heatmaps = np.random.rand(30, 50, 50)
   >>> smoothed = gaussian_smooth_heatmaps(heatmaps, sigma=1.5)
   >>> smoothed.shape
   (30, 50, 50)


