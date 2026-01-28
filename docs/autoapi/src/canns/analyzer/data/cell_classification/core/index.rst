src.canns.analyzer.data.cell_classification.core
================================================

.. py:module:: src.canns.analyzer.data.cell_classification.core

.. autoapi-nested-parse::

   Core analysis modules.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/analyzer/data/cell_classification/core/grid_cells/index
   /autoapi/src/canns/analyzer/data/cell_classification/core/grid_modules_leiden/index
   /autoapi/src/canns/analyzer/data/cell_classification/core/head_direction/index
   /autoapi/src/canns/analyzer/data/cell_classification/core/spatial_analysis/index


Classes
-------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.core.GridnessAnalyzer
   src.canns.analyzer.data.cell_classification.core.GridnessResult
   src.canns.analyzer.data.cell_classification.core.HDCellResult
   src.canns.analyzer.data.cell_classification.core.HeadDirectionAnalyzer


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.core.compute_2d_autocorrelation
   src.canns.analyzer.data.cell_classification.core.compute_field_statistics
   src.canns.analyzer.data.cell_classification.core.compute_grid_spacing
   src.canns.analyzer.data.cell_classification.core.compute_rate_map
   src.canns.analyzer.data.cell_classification.core.compute_rate_map_from_binned
   src.canns.analyzer.data.cell_classification.core.compute_spatial_information
   src.canns.analyzer.data.cell_classification.core.identify_grid_modules_and_stats


Package Contents
----------------

.. py:class:: GridnessAnalyzer(threshold = 0.2, min_orientation = 15.0, min_center_radius = 2, num_gridness_radii = 3)

   Analyzer for computing gridness scores from spatial autocorrelograms.

   This implements the rotation-correlation method for quantifying hexagonal
   grid patterns in neural firing rate maps.

   :param threshold: Normalized threshold for contour detection (0-1). Default is 0.2.
   :type threshold: float, optional
   :param min_orientation: Minimum angular difference between fields (degrees). Default is 15.
   :type min_orientation: float, optional
   :param min_center_radius: Minimum center field radius in pixels. Default is 2.
   :type min_center_radius: int, optional
   :param num_gridness_radii: Number of adjacent radii to average for gridness score. Default is 3.
   :type num_gridness_radii: int, optional

   .. rubric:: Examples

   >>> analyzer = GridnessAnalyzer()
   >>> # Assume we have a 2D rate map
   >>> autocorr = compute_2d_autocorrelation(rate_map)
   >>> result = analyzer.compute_gridness_score(autocorr)
   >>> print(f"Gridness score: {result.score:.3f}")
   >>> print(f"Grid spacing: {result.spacing}")

   .. rubric:: Notes

   Based on gridnessScore.m from the MATLAB codebase.

   .. rubric:: References

   The gridness score algorithm computes correlations between the autocorrelogram
   and rotated versions at 30°, 60°, 90°, 120°, and 150°. The score is:
   min(r_60°, r_120°) - max(r_30°, r_90°, r_150°)

   This exploits the 60° rotational symmetry of hexagonal grids.


   .. py:method:: compute_gridness_score(autocorr)

      Compute gridness score from a 2D autocorrelogram.

      :param autocorr: 2D autocorrelogram of a firing rate map
      :type autocorr: np.ndarray

      :returns: **result** -- Complete gridness analysis results
      :rtype: GridnessResult

      :raises ValueError: If autocorr is not 2D or if center field cannot be detected



   .. py:attribute:: min_center_radius
      :value: 2



   .. py:attribute:: min_orientation
      :value: 15.0



   .. py:attribute:: num_gridness_radii
      :value: 3



   .. py:attribute:: threshold
      :value: 0.2



.. py:class:: GridnessResult

   Results from gridness score computation.

   .. attribute:: score

      Gridness score (range -2 to 2, typical grid cells: 0.3-1.3)

      :type: float

   .. attribute:: spacing

      Array of 3 grid field spacings (distances from center)

      :type: np.ndarray

   .. attribute:: orientation

      Array of 3 grid field orientations (angles in degrees)

      :type: np.ndarray

   .. attribute:: ellipse

      Fitted ellipse parameters [cx, cy, rx, ry, theta]

      :type: np.ndarray

   .. attribute:: ellipse_theta_deg

      Ellipse orientation in degrees [0, 180]

      :type: float

   .. attribute:: center_radius

      Radius of the central autocorrelation field

      :type: float

   .. attribute:: optimal_radius

      Radius at which gridness score is maximized

      :type: float

   .. attribute:: peak_locations

      Coordinates of detected grid peaks (N x 2 array)

      :type: np.ndarray


   .. py:attribute:: center_radius
      :type:  float


   .. py:attribute:: ellipse
      :type:  numpy.ndarray


   .. py:attribute:: ellipse_theta_deg
      :type:  float


   .. py:attribute:: optimal_radius
      :type:  float


   .. py:attribute:: orientation
      :type:  numpy.ndarray


   .. py:attribute:: peak_locations
      :type:  numpy.ndarray | None
      :value: None



   .. py:attribute:: score
      :type:  float


   .. py:attribute:: spacing
      :type:  numpy.ndarray


.. py:class:: HDCellResult

   Results from head direction cell classification.

   .. attribute:: is_hd

      Whether the cell is classified as a head direction cell

      :type: bool

   .. attribute:: mvl_hd

      Mean Vector Length for head direction tuning

      :type: float

   .. attribute:: preferred_direction

      Preferred head direction in radians

      :type: float

   .. attribute:: mvl_theta

      Mean Vector Length for theta phase tuning (if provided)

      :type: float or None

   .. attribute:: tuning_curve

      Tuple of (bin_centers, firing_rates)

      :type: tuple

   .. attribute:: rayleigh_p

      P-value from Rayleigh test for non-uniformity

      :type: float


   .. py:attribute:: is_hd
      :type:  bool


   .. py:attribute:: mvl_hd
      :type:  float


   .. py:attribute:: mvl_theta
      :type:  float | None


   .. py:attribute:: preferred_direction
      :type:  float


   .. py:attribute:: rayleigh_p
      :type:  float


   .. py:attribute:: tuning_curve
      :type:  tuple[numpy.ndarray, numpy.ndarray]


.. py:class:: HeadDirectionAnalyzer(mvl_hd_threshold = 0.4, mvl_theta_threshold = 0.3, strict_mode = True, n_bins = 60)

   Analyzer for classifying head direction cells based on directional tuning.

   Head direction cells fire when the animal's head points in a specific direction.
   Classification is based on the strength of directional tuning measured by
   Mean Vector Length (MVL).

   :param mvl_hd_threshold: MVL threshold for head direction. Default is 0.4 (strict).
                            Use 0.2 for looser threshold.
   :type mvl_hd_threshold: float, optional
   :param mvl_theta_threshold: MVL threshold for theta phase modulation. Default is 0.3.
   :type mvl_theta_threshold: float, optional
   :param strict_mode: If True, requires both HD and theta criteria. Default is True.
   :type strict_mode: bool, optional
   :param n_bins: Number of directional bins for tuning curve. Default is 60 (6° bins).
   :type n_bins: int, optional

   .. rubric:: Examples

   >>> analyzer = HeadDirectionAnalyzer(mvl_hd_threshold=0.4, strict_mode=True)
   >>> result = analyzer.classify_hd_cell(spike_times, head_directions, time_stamps)
   >>> print(f"Is HD cell: {result.is_hd}")
   >>> print(f"MVL: {result.mvl_hd:.3f}")
   >>> print(f"Preferred direction: {np.rad2deg(result.preferred_direction):.1f}°")

   .. rubric:: Notes

   Based on MATLAB classification from fig2.m and plotSwsExample.m:
   - Strict: MVL_hd > 0.4 AND MVL_theta > 0.3
   - Loose: MVL_hd > 0.2 AND MVL_theta > 0.3

   .. rubric:: References

   Classification thresholds follow standard conventions in head direction
   cell literature and the CircStat toolbox.


   .. py:method:: classify_hd_cell(spike_times, head_directions, time_stamps, theta_phases = None)

      Classify a cell as head direction cell based on MVL thresholds.

      :param spike_times: Spike times in seconds
      :type spike_times: np.ndarray
      :param head_directions: Head direction at each time point (radians)
      :type head_directions: np.ndarray
      :param time_stamps: Time stamps corresponding to head_directions (seconds)
      :type time_stamps: np.ndarray
      :param theta_phases: Theta phase at each time point (radians). If None, theta
                           criterion is not checked.
      :type theta_phases: np.ndarray, optional

      :returns: **result** -- Classification result with MVL, preferred direction, and tuning curve
      :rtype: HDCellResult

      .. rubric:: Examples

      >>> # Simulate a head direction cell
      >>> time_stamps = np.linspace(0, 100, 10000)
      >>> head_directions = np.linspace(0, 20*np.pi, 10000) % (2*np.pi) - np.pi
      >>> preferred_dir = 0.5
      >>> spike_times = time_stamps[np.abs(head_directions - preferred_dir) < 0.3]
      >>> result = analyzer.classify_hd_cell(spike_times, head_directions, time_stamps)



   .. py:method:: compute_mvl(angles, weights = None)

      Compute Mean Vector Length (MVL).

      The MVL is a measure of circular variance, ranging from 0 (uniform
      distribution) to 1 (concentrated distribution).

      :param angles: Angles in radians
      :type angles: np.ndarray
      :param weights: Weights for each angle (e.g., firing rates). If None, uniform weights.
      :type weights: np.ndarray, optional

      :returns: **mvl** -- Mean vector length
      :rtype: float

      .. rubric:: Examples

      >>> # Concentrated distribution
      >>> angles = np.random.normal(0, 0.1, 100)
      >>> mvl = analyzer.compute_mvl(angles)
      >>> print(f"MVL: {mvl:.3f}")  # Should be close to 1

      >>> # Uniform distribution
      >>> angles = np.random.uniform(-np.pi, np.pi, 100)
      >>> mvl = analyzer.compute_mvl(angles)
      >>> print(f"MVL: {mvl:.3f}")  # Should be close to 0

      .. rubric:: Notes

      Uses the circ_r function from circular statistics utilities.



   .. py:method:: compute_tuning_curve(spike_times, head_directions, time_stamps, n_bins = None)

      Compute directional tuning curve.

      :param spike_times: Spike times in seconds
      :type spike_times: np.ndarray
      :param head_directions: Head direction at each time point (radians)
      :type head_directions: np.ndarray
      :param time_stamps: Time stamps corresponding to head_directions (seconds)
      :type time_stamps: np.ndarray
      :param n_bins: Number of bins. If None, uses self.n_bins.
      :type n_bins: int, optional

      :returns: * **bin_centers** (*np.ndarray*) -- Center of each directional bin (radians)
                * **firing_rates** (*np.ndarray*) -- Firing rate in each bin (Hz)
                * **occupancy** (*np.ndarray*) -- Time spent in each bin (seconds)

      .. rubric:: Examples

      >>> bins, rates, occ = analyzer.compute_tuning_curve(
      ...     spike_times, head_directions, time_stamps
      ... )
      >>> # Plot polar tuning curve
      >>> import matplotlib.pyplot as plt
      >>> ax = plt.subplot(111, projection='polar')
      >>> ax.plot(bins, rates)



   .. py:attribute:: mvl_hd_threshold
      :value: 0.4



   .. py:attribute:: mvl_theta_threshold
      :value: 0.3



   .. py:attribute:: n_bins
      :value: 60



   .. py:attribute:: strict_mode
      :value: True



.. py:function:: compute_2d_autocorrelation(rate_map, overlap = 0.8)

   Compute 2D spatial autocorrelation of a firing rate map.

   This is a convenience wrapper around the autocorrelation function
   from the correlation module.

   :param rate_map: 2D firing rate map
   :type rate_map: np.ndarray
   :param overlap: Overlap percentage (0-1). Default is 0.8.
   :type overlap: float, optional

   :returns: **autocorr** -- 2D autocorrelogram
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> rate_map = np.random.rand(50, 50)
   >>> autocorr = compute_2d_autocorrelation(rate_map)
   >>> print(autocorr.shape)

   .. rubric:: Notes

   Based on autocorrelation.m from the MATLAB codebase.


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


