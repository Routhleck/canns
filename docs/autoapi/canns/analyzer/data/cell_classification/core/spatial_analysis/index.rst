canns.analyzer.data.cell_classification.core.spatial_analysis
=============================================================

.. py:module:: canns.analyzer.data.cell_classification.core.spatial_analysis

.. autoapi-nested-parse::

   Spatial Analysis Functions

   Functions for computing spatial firing rate maps and related metrics.



Attributes
----------

.. autoapisummary::

   canns.analyzer.data.cell_classification.core.spatial_analysis.time_stamps


Functions
---------

.. autoapisummary::

   canns.analyzer.data.cell_classification.core.spatial_analysis.compute_field_statistics
   canns.analyzer.data.cell_classification.core.spatial_analysis.compute_grid_spacing
   canns.analyzer.data.cell_classification.core.spatial_analysis.compute_rate_map
   canns.analyzer.data.cell_classification.core.spatial_analysis.compute_rate_map_from_binned
   canns.analyzer.data.cell_classification.core.spatial_analysis.compute_spatial_information


Module Contents
---------------

.. py:function:: compute_field_statistics(rate_map, threshold = 0.2, min_area = 9)

   Extract firing field statistics from a rate map.

   :param rate_map: 2D firing rate map (Hz)
   :type rate_map: np.ndarray
   :param threshold: Threshold as fraction of peak rate. Default is 0.2 (20% of peak).
   :type threshold: float, optional
   :param min_area: Minimum field size in pixels. Default is 9.
   :type min_area: int, optional

   :returns: **stats** -- Dictionary with:
             - num_fields: number of detected fields
             - field_sizes: list of field areas
             - field_peaks: list of peak firing rates
             - field_centers: list of field centers (x, y)
   :rtype: dict

   .. rubric:: Examples

   >>> rate_map = np.random.rand(50, 50) * 10
   >>> stats = compute_field_statistics(rate_map)
   >>> print(f"Found {stats['num_fields']} firing fields")


.. py:function:: compute_grid_spacing(rate_map, method = 'autocorr')

   Estimate grid spacing from a rate map.

   :param rate_map: 2D firing rate map
   :type rate_map: np.ndarray
   :param method: Method for estimation: 'autocorr' (default) or 'fft'
   :type method: str, optional

   :returns: **spacing** -- Estimated grid spacing in bins, or None if cannot be determined
   :rtype: float or None

   .. rubric:: Notes

   This is a simplified implementation. For full grid analysis,
   use GridnessAnalyzer.


.. py:function:: compute_rate_map(spike_times, positions, time_stamps, spatial_bins = 20, position_range = None, smoothing_sigma = 2.0, min_occupancy = 0.0)

   Compute 2D spatial firing rate map.

   :param spike_times: Spike times in seconds
   :type spike_times: np.ndarray
   :param positions: Animal positions, shape (N, 2) where columns are (x, y) coordinates
   :type positions: np.ndarray
   :param time_stamps: Time stamps for position samples
   :type time_stamps: np.ndarray
   :param spatial_bins: Number of spatial bins. If int, uses same for both dimensions.
                        Default is 20.
   :type spatial_bins: int or tuple, optional
   :param position_range: (min, max) for position coordinates. If None, inferred from data.
   :type position_range: tuple of float, optional
   :param smoothing_sigma: Standard deviation of Gaussian smoothing kernel. Default is 2.0.
   :type smoothing_sigma: float, optional
   :param min_occupancy: Minimum occupancy (seconds) for valid bins. Default is 0.0.
   :type min_occupancy: float, optional

   :returns: * **rate_map** (*np.ndarray*) -- 2D firing rate map (Hz), shape (spatial_bins, spatial_bins)
             * **occupancy_map** (*np.ndarray*) -- Time spent in each bin (seconds)
             * **x_edges** (*np.ndarray*) -- Bin edges for x coordinate
             * **y_edges** (*np.ndarray*) -- Bin edges for y coordinate

   .. rubric:: Examples

   >>> # Simulate data
   >>> time_stamps = np.linspace(0, 100, 10000)
   >>> positions = np.column_stack([
   ...     np.sin(time_stamps * 0.1),
   ...     np.cos(time_stamps * 0.1)
   ... ])
   >>> spike_times = time_stamps[::50]  # Some spikes
   >>> rate_map, occ, x_edges, y_edges = compute_rate_map(
   ...     spike_times, positions, time_stamps
   ... )


.. py:function:: compute_rate_map_from_binned(x, y, spike_counts, bins = 35, min_occupancy = 0.0)

   Compute a 2D rate map from binned spike counts aligned to positions.

   :param x: X positions aligned to spike_counts (same length).
   :type x: np.ndarray
   :param y: Y positions aligned to spike_counts (same length).
   :type y: np.ndarray
   :param spike_counts: Spike counts per time bin (same length as x/y).
   :type spike_counts: np.ndarray
   :param bins: Number of spatial bins per dimension. Default is 35.
   :type bins: int, optional
   :param min_occupancy: Minimum occupancy count for valid bins. Default is 0.
   :type min_occupancy: float, optional

   :returns: * **rate_map** (*np.ndarray*) -- 2D firing rate map, shape (bins, bins).
             * **occupancy_map** (*np.ndarray*) -- Occupancy counts per bin.
             * **x_edges** (*np.ndarray*) -- Bin edges for x coordinate.
             * **y_edges** (*np.ndarray*) -- Bin edges for y coordinate.


.. py:function:: compute_spatial_information(rate_map, occupancy_map, mean_rate = None)

   Compute spatial information score (bits per spike).

   Spatial information quantifies how much information about the animal's
   location is conveyed by each spike.

   :param rate_map: 2D firing rate map (Hz)
   :type rate_map: np.ndarray
   :param occupancy_map: Time spent in each bin (seconds)
   :type occupancy_map: np.ndarray
   :param mean_rate: Mean firing rate. If None, computed from rate_map and occupancy_map.
   :type mean_rate: float, optional

   :returns: **spatial_info** -- Spatial information in bits per spike
   :rtype: float

   .. rubric:: Examples

   >>> rate_map = np.random.rand(20, 20) * 10
   >>> occupancy_map = np.ones((20, 20))
   >>> info = compute_spatial_information(rate_map, occupancy_map)

   .. rubric:: Notes

   Formula: I = Σ_i p_i * (r_i / r_mean) * log2(r_i / r_mean)
   where:
   - p_i is probability of occupancy in bin i
   - r_i is firing rate in bin i
   - r_mean is mean firing rate

   .. rubric:: References

   Skaggs et al. (1993). "An information-theoretic approach to deciphering
   the hippocampal code." NIPS.


.. py:data:: time_stamps

