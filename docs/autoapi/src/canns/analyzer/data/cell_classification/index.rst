src.canns.analyzer.data.cell_classification
===========================================

.. py:module:: src.canns.analyzer.data.cell_classification

.. autoapi-nested-parse::

   Cell Classification Package

   Python implementation of grid cell and head direction cell classification algorithms.

   Based on the MATLAB code from:
   Vollan, Gardner, Moser & Moser (Nature, 2025)
   "Left-right-alternating sweeps in entorhinal-hippocampal maps of space"



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/analyzer/data/cell_classification/core/index
   /autoapi/src/canns/analyzer/data/cell_classification/io/index
   /autoapi/src/canns/analyzer/data/cell_classification/utils/index
   /autoapi/src/canns/analyzer/data/cell_classification/visualization/index


Classes
-------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.GridnessAnalyzer
   src.canns.analyzer.data.cell_classification.GridnessResult
   src.canns.analyzer.data.cell_classification.HDCellResult
   src.canns.analyzer.data.cell_classification.HeadDirectionAnalyzer
   src.canns.analyzer.data.cell_classification.MATFileLoader
   src.canns.analyzer.data.cell_classification.TuningCurve
   src.canns.analyzer.data.cell_classification.Unit


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.autocorrelation_2d
   src.canns.analyzer.data.cell_classification.cart2pol
   src.canns.analyzer.data.cell_classification.circ_dist
   src.canns.analyzer.data.cell_classification.circ_dist2
   src.canns.analyzer.data.cell_classification.circ_mean
   src.canns.analyzer.data.cell_classification.circ_r
   src.canns.analyzer.data.cell_classification.circ_rtest
   src.canns.analyzer.data.cell_classification.circ_std
   src.canns.analyzer.data.cell_classification.compute_2d_autocorrelation
   src.canns.analyzer.data.cell_classification.compute_field_statistics
   src.canns.analyzer.data.cell_classification.compute_grid_spacing
   src.canns.analyzer.data.cell_classification.compute_rate_map
   src.canns.analyzer.data.cell_classification.compute_rate_map_from_binned
   src.canns.analyzer.data.cell_classification.compute_spatial_information
   src.canns.analyzer.data.cell_classification.fit_ellipse
   src.canns.analyzer.data.cell_classification.identify_grid_modules_and_stats
   src.canns.analyzer.data.cell_classification.label_connected_components
   src.canns.analyzer.data.cell_classification.normalized_xcorr2
   src.canns.analyzer.data.cell_classification.pearson_correlation
   src.canns.analyzer.data.cell_classification.plot_autocorrelogram
   src.canns.analyzer.data.cell_classification.plot_grid_score_histogram
   src.canns.analyzer.data.cell_classification.plot_gridness_analysis
   src.canns.analyzer.data.cell_classification.plot_hd_analysis
   src.canns.analyzer.data.cell_classification.plot_polar_tuning
   src.canns.analyzer.data.cell_classification.plot_rate_map
   src.canns.analyzer.data.cell_classification.plot_temporal_autocorr
   src.canns.analyzer.data.cell_classification.pol2cart
   src.canns.analyzer.data.cell_classification.polyarea
   src.canns.analyzer.data.cell_classification.regionprops
   src.canns.analyzer.data.cell_classification.rotate_image
   src.canns.analyzer.data.cell_classification.squared_distance
   src.canns.analyzer.data.cell_classification.wrap_to_pi


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



.. py:class:: MATFileLoader

   Loader for MATLAB .mat files containing neuroscience data.

   Handles both MATLAB v5/v7 files (via scipy.io) and v7.3+ files (via h5py).


   .. py:method:: load(filepath)
      :staticmethod:


      Load a .mat file, automatically detecting the version.

      :param filepath: Path to .mat file
      :type filepath: str

      :returns: **data** -- Dictionary containing the loaded data
      :rtype: dict

      .. rubric:: Examples

      >>> loader = MATFileLoader()
      >>> data = loader.load("example.mat")
      >>> print(data.keys())



   .. py:method:: load_example_cells(filepath)
      :staticmethod:


      Load example cell data from exampleIdCells.mat format.

      Expected structure:
      - res: struct array with fields:
        - recName, id
        - hdTuning, posTuning
        - tempAcorr (temporal autocorrelation)

      :param filepath: Path to example cells .mat file
      :type filepath: str

      :returns: **units** -- List of Unit objects
      :rtype: list of Unit

      .. rubric:: Examples

      >>> loader = MATFileLoader()
      >>> cells = loader.load_example_cells("../results/exampleIdCells.mat")
      >>> print(f"Loaded {len(cells)} example cells")



   .. py:method:: load_unit_data(filepath)
      :staticmethod:


      Load unit data from a .mat file.

      Expected structure (from unit_data_25953.mat):
      - units: struct array with fields:
        - id or spikeInds or spikeTimes
        - rmf.hd, rmf.pos, rmf.theta (tuning structures)
        - isGrid (boolean)

      :param filepath: Path to unit data .mat file
      :type filepath: str

      :returns: **units** -- List of Unit objects
      :rtype: list of Unit

      .. rubric:: Examples

      >>> loader = MATFileLoader()
      >>> units = loader.load_unit_data("../results/unit_data_25953.mat")
      >>> print(f"Loaded {len(units)} units")
      >>> print(f"Grid cells: {sum(u.is_grid for u in units if u.is_grid)}")



.. py:class:: TuningCurve

   Represents a tuning curve (e.g., head direction or spatial tuning).

   .. attribute:: bins

      Bin centers (e.g., angles for HD, positions for spatial)

      :type: np.ndarray

   .. attribute:: rates

      Firing rates in each bin (Hz)

      :type: np.ndarray

   .. attribute:: mvl

      Mean Vector Length (for directional tuning)

      :type: float, optional

   .. attribute:: center_of_mass

      Preferred direction/position

      :type: float, optional

   .. attribute:: peak_rate

      Maximum firing rate

      :type: float, optional


   .. py:method:: __post_init__()

      Compute derived properties.



   .. py:attribute:: bins
      :type:  numpy.ndarray


   .. py:attribute:: center_of_mass
      :type:  float | None
      :value: None



   .. py:attribute:: mvl
      :type:  float | None
      :value: None



   .. py:attribute:: peak_rate
      :type:  float | None
      :value: None



   .. py:attribute:: rates
      :type:  numpy.ndarray


.. py:class:: Unit

   Represents a single neural unit (neuron).

   .. attribute:: unit_id

      Unique identifier for this unit

      :type: int or str

   .. attribute:: spike_times

      Spike times in seconds

      :type: np.ndarray

   .. attribute:: spike_indices

      Indices into session time array

      :type: np.ndarray, optional

   .. attribute:: hd_tuning

      Head direction tuning curve

      :type: TuningCurve, optional

   .. attribute:: pos_tuning

      Spatial position tuning (2D rate map)

      :type: TuningCurve, optional

   .. attribute:: theta_tuning

      Theta phase tuning

      :type: TuningCurve, optional

   .. attribute:: is_grid

      Whether this is a grid cell

      :type: bool, optional

   .. attribute:: is_hd

      Whether this is a head direction cell

      :type: bool, optional

   .. attribute:: gridness_score

      Grid cell score

      :type: float, optional

   .. attribute:: metadata

      Additional metadata

      :type: dict


   .. py:attribute:: gridness_score
      :type:  float | None
      :value: None



   .. py:attribute:: hd_tuning
      :type:  TuningCurve | None
      :value: None



   .. py:attribute:: is_grid
      :type:  bool | None
      :value: None



   .. py:attribute:: is_hd
      :type:  bool | None
      :value: None



   .. py:attribute:: metadata
      :type:  dict[str, Any]


   .. py:attribute:: pos_tuning
      :type:  TuningCurve | None
      :value: None



   .. py:attribute:: spike_indices
      :type:  numpy.ndarray | None
      :value: None



   .. py:attribute:: spike_times
      :type:  numpy.ndarray


   .. py:attribute:: theta_tuning
      :type:  TuningCurve | None
      :value: None



   .. py:attribute:: unit_id
      :type:  Any


.. py:function:: autocorrelation_2d(array, overlap = 0.8, normalize = True)

   Compute 2D autocorrelation of an array.

   This is a convenience function specifically for computing spatial
   autocorrelation of firing rate maps, which is needed for grid cell analysis.

   :param array: 2D array (e.g., firing rate map)
   :type array: np.ndarray
   :param overlap: Percentage of overlap region to keep (0-1). Default is 0.8.
                   The autocorrelogram is cropped to this central region to avoid
                   edge artifacts.
   :type overlap: float, optional
   :param normalize: Whether to normalize the correlation. Default is True.
   :type normalize: bool, optional

   :returns: **autocorr** -- 2D autocorrelation array
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> # Create a simple periodic pattern (grid-like)
   >>> x = np.linspace(0, 4*np.pi, 50)
   >>> xx, yy = np.meshgrid(x, x)
   >>> pattern = np.cos(xx) * np.cos(yy)
   >>> autocorr = autocorrelation_2d(pattern)
   >>> # Autocorr should show hexagonal/grid pattern

   .. rubric:: Notes

   Based on autocorrelation.m from the MATLAB codebase.
   Replaces NaN values with 0 before computing correlation.


.. py:function:: cart2pol(x, y)

   Transform Cartesian coordinates to polar coordinates.

   :param x: X coordinates
   :type x: np.ndarray
   :param y: Y coordinates
   :type y: np.ndarray

   :returns: * **theta** (*np.ndarray*) -- Angle in radians, range [-π, π]
             * **rho** (*np.ndarray*) -- Radius (distance from origin)

   .. rubric:: Examples

   >>> x = np.array([1, 0, -1])
   >>> y = np.array([0, 1, 0])
   >>> theta, rho = cart2pol(x, y)
   >>> print(theta)  # [0, π/2, π]
   >>> print(rho)    # [1, 1, 1]

   .. rubric:: Notes

   Equivalent to MATLAB's cart2pol function.


.. py:function:: circ_dist(x, y)

   Pairwise angular distance between angles (x_i - y_i) around the circle.

   Computes the shortest signed angular distance from y to x, respecting
   circular topology (wrapping at ±π).

   :param x: First set of angles in radians
   :type x: np.ndarray
   :param y: Second set of angles in radians (must be same shape as x, or scalar)
   :type y: np.ndarray

   :returns: **r** -- Angular distances in radians, range [-π, π]
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> x = np.array([0.1, np.pi])
   >>> y = np.array([0.0, -np.pi])  # -π and π are same location
   >>> dist = circ_dist(x, y)
   >>> print(dist)  # [0.1, 0.0]

   >>> # Distance wraps around at ±π
   >>> x = np.array([np.pi - 0.1])
   >>> y = np.array([-np.pi + 0.1])
   >>> dist = circ_dist(x, y)
   >>> print(dist)  # Small value, not 2π - 0.2

   .. rubric:: Notes

   Based on CircStat toolbox circ_dist.m by Philipp Berens (2009)
   References: Biostatistical Analysis, J. H. Zar, p. 651


.. py:function:: circ_dist2(x, y = None)

   All pairwise angular distances (x_i - y_j) around the circle.

   Computes the matrix of all pairwise angular distances between two sets
   of angles, or within one set if y is not provided.

   :param x: First set of angles in radians (will be treated as column vector)
   :type x: np.ndarray
   :param y: Second set of angles in radians (will be treated as column vector).
             If None, computes pairwise distances within x. Default is None.
   :type y: np.ndarray, optional

   :returns: **r** -- Matrix of pairwise angular distances, shape (len(x), len(y))
             Element (i, j) contains the distance from y[j] to x[i]
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> x = np.array([0, np.pi/2, np.pi])
   >>> D = circ_dist2(x)  # All pairwise distances within x
   >>> print(D.shape)  # (3, 3)

   >>> y = np.array([0, np.pi])
   >>> D = circ_dist2(x, y)  # All distances from y to x
   >>> print(D.shape)  # (3, 2)

   .. rubric:: Notes

   Based on CircStat toolbox circ_dist2.m by Philipp Berens (2009)


.. py:function:: circ_mean(alpha, w = None, axis = 0)

   Compute mean direction for circular data.

   :param alpha: Sample of angles in radians
   :type alpha: np.ndarray
   :param w: Weights for each angle (e.g., for binned data). If None, uniform weights assumed.
   :type w: np.ndarray, optional
   :param axis: Compute along this dimension. Default is 0.
   :type axis: int, optional

   :returns: **mu** -- Mean direction in radians, range [-π, π]
   :rtype: float or np.ndarray

   .. rubric:: Examples

   >>> angles = np.array([0, 0.1, 0.2, -0.1, -0.2])
   >>> mean_angle = circ_mean(angles)
   >>> print(f"Mean direction: {mean_angle:.3f} rad")

   >>> # Weighted mean
   >>> angles = np.array([0, np.pi])
   >>> weights = np.array([3, 1])  # 3x more weight on 0
   >>> mean_angle = circ_mean(angles, w=weights)

   .. rubric:: Notes

   Based on CircStat toolbox circ_mean.m by Philipp Berens (2009)


.. py:function:: circ_r(alpha, w = None, d = 0.0, axis = 0)

   Compute mean resultant vector length for circular data.

   This is a measure of circular variance (concentration). Values near 1 indicate
   high concentration, values near 0 indicate uniform distribution.

   :param alpha: Sample of angles in radians
   :type alpha: np.ndarray
   :param w: Weights for each angle (e.g., for binned data). If None, uniform weights assumed.
   :type w: np.ndarray, optional
   :param d: Spacing of bin centers for binned data. If supplied, correction factor is used
             to correct for bias in estimation of r (in radians). Default is 0 (no correction).
   :type d: float, optional
   :param axis: Compute along this dimension. Default is 0.
   :type axis: int, optional

   :returns: **r** -- Mean resultant vector length
   :rtype: float or np.ndarray

   .. rubric:: Examples

   >>> angles = np.array([0, 0.1, 0.2, -0.1, -0.2])  # Concentrated around 0
   >>> r = circ_r(angles)
   >>> print(f"MVL: {r:.3f}")  # Should be close to 1

   >>> angles = np.linspace(0, 2*np.pi, 100)  # Uniform distribution
   >>> r = circ_r(angles)
   >>> print(f"MVL: {r:.3f}")  # Should be close to 0

   .. rubric:: Notes

   Based on CircStat toolbox circ_r.m by Philipp Berens (2009)


.. py:function:: circ_rtest(alpha, w = None)

   Rayleigh test for non-uniformity of circular data.

   H0: The population is uniformly distributed around the circle.
   HA: The population is not uniformly distributed.

   :param alpha: Sample of angles in radians
   :type alpha: np.ndarray
   :param w: Weights for each angle. If None, uniform weights assumed.
   :type w: np.ndarray, optional

   :returns: **pval** -- p-value of Rayleigh test. Small values (< 0.05) indicate
             significant deviation from uniformity.
   :rtype: float

   .. rubric:: Examples

   >>> # Concentrated distribution
   >>> angles = np.random.normal(0, 0.1, 100)
   >>> p = circ_rtest(angles)
   >>> print(f"p-value: {p:.4f}")  # Should be < 0.05

   >>> # Uniform distribution
   >>> angles = np.random.uniform(-np.pi, np.pi, 100)
   >>> p = circ_rtest(angles)
   >>> print(f"p-value: {p:.4f}")  # Should be > 0.05

   .. rubric:: Notes

   Test statistic: Z = n * r^2, where n is sample size and r is MVL
   Approximation for p-value: p ≈ exp(-Z) * (1 + (2*Z - Z^2)/(4*n))

   References: Topics in Circular Statistics, S.R. Jammalamadaka et al., p. 48


.. py:function:: circ_std(alpha, w = None, d = 0.0, axis = 0)

   Compute circular standard deviation for circular data.

   :param alpha: Sample of angles in radians
   :type alpha: np.ndarray
   :param w: Weights for each angle. If None, uniform weights assumed.
   :type w: np.ndarray, optional
   :param d: Spacing of bin centers for binned data (correction factor). Default is 0.
   :type d: float, optional
   :param axis: Compute along this dimension. Default is 0.
   :type axis: int, optional

   :returns: * **s** (*float or np.ndarray*) -- Angular deviation (equation 26.20, Zar)
             * **s0** (*float or np.ndarray*) -- Circular standard deviation (equation 26.21, Zar)

   .. rubric:: Examples

   >>> angles = np.array([0, 0.1, 0.2, -0.1, -0.2])
   >>> s, s0 = circ_std(angles)
   >>> print(f"Angular deviation: {s:.3f} rad")

   .. rubric:: Notes

   Based on CircStat toolbox circ_std.m by Philipp Berens (2009)
   References: Biostatistical Analysis, J. H. Zar


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


.. py:function:: fit_ellipse(x, y)

   Least-squares fit of ellipse to 2D points.

   Implements the Direct Least Squares Fitting algorithm by Fitzgibbon et al. (1999).
   This is a robust method that includes scaling to reduce roundoff error and
   returns geometric parameters rather than quadratic form coefficients.

   :param x: X coordinates of points (1D array)
   :type x: np.ndarray
   :param y: Y coordinates of points (1D array)
   :type y: np.ndarray

   :returns: **params** -- Array of shape (5,) containing:
             [center_x, center_y, radius_x, radius_y, theta_radians]
             where theta is the orientation angle of the major axis
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> # Generate points on an ellipse
   >>> t = np.linspace(0, 2*np.pi, 100)
   >>> cx, cy = 5, 3  # center
   >>> rx, ry = 4, 2  # radii
   >>> angle = np.pi/4  # rotation
   >>> x = cx + rx * np.cos(t) * np.cos(angle) - ry * np.sin(t) * np.sin(angle)
   >>> y = cy + rx * np.cos(t) * np.sin(angle) + ry * np.sin(t) * np.cos(angle)
   >>> params = fit_ellipse(x, y)
   >>> print(f"Fitted center: ({params[0]:.2f}, {params[1]:.2f})")
   >>> print(f"Fitted radii: ({params[2]:.2f}, {params[3]:.2f})")

   .. rubric:: Notes

   Based on fitEllipse.m from the MATLAB codebase.

   .. rubric:: References

   Fitzgibbon, A.W., Pilu, M., and Fisher, R.B. (1999).
   "Direct least-squares fitting of ellipses". IEEE T-PAMI, 21(5):476-480.
   http://research.microsoft.com/en-us/um/people/awf/ellipse/


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


.. py:function:: label_connected_components(binary_image, connectivity = 2)

   Label connected components in a binary image.

   :param binary_image: Binary input image
   :type binary_image: np.ndarray
   :param connectivity: Connectivity for defining neighbors:
                        - 1: 4-connectivity
                        - 2: 8-connectivity (default)
   :type connectivity: int, optional

   :returns: * **labels** (*np.ndarray*) -- Labeled image where each connected component has a unique integer label
             * **num_labels** (*int*) -- Number of connected components found

   .. rubric:: Examples

   >>> binary = (np.random.rand(50, 50) > 0.7)
   >>> labels, n = label_connected_components(binary)
   >>> print(f"Found {n} connected components")

   .. rubric:: Notes

   Based on MATLAB's bwconncomp function.
   Uses skimage.measure.label.


.. py:function:: normalized_xcorr2(template, image, mode = 'same', min_overlap = 0)

   Normalized 2D cross-correlation.

   Computes the normalized cross-correlation of two 2D arrays. Unlike
   scipy.signal.correlate, this function properly handles varying overlap
   regions and works correctly even when template and image are the same size.

   :param template: 2D template array
   :type template: np.ndarray
   :param image: 2D image array
   :type image: np.ndarray
   :param mode: 'full' - full correlation (default for autocorrelation)
                'same' - output size same as image
                'valid' - only where template fully overlaps image
   :type mode: str, optional
   :param min_overlap: Minimum number of overlapping pixels required for valid correlation.
                       Locations with fewer overlapping pixels are set to 0.
                       Default is 0 (no threshold).
   :type min_overlap: int, optional

   :returns: **C** -- Normalized cross-correlation. Values range from -1 to 1.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> # Autocorrelation (template = image)
   >>> image = np.random.rand(50, 50)
   >>> autocorr = normalized_xcorr2(image, image, mode='full')
   >>> # Peak should be at center with value 1.0
   >>> center = np.array(autocorr.shape) // 2
   >>> print(f"Peak value: {autocorr[tuple(center)]:.3f}")

   >>> # Template matching
   >>> image = np.random.rand(100, 100)
   >>> template = image[40:60, 40:60]  # Extract 20x20 patch
   >>> corr = normalized_xcorr2(template, image)
   >>> # Should find the template location

   .. rubric:: Notes

   This is a simplified Python implementation. For the full general version
   (handling all edge cases), see normxcorr2_general.m by Dirk Padfield.

   For most neuroscience applications (autocorrelation of rate maps),
   scipy.signal.correlate with normalization is sufficient.

   .. rubric:: References

   Padfield, D. "Masked FFT registration". CVPR, 2010.
   Lewis, J.P. "Fast Normalized Cross-Correlation". Industrial Light & Magic.


.. py:function:: pearson_correlation(x, y)

   Compute Pearson correlation coefficient between x and each column of y.

   This is an optimized implementation that efficiently handles multiple
   correlations when y has multiple columns.

   :param x: First array, shape (n,) or (n, 1)
   :type x: np.ndarray
   :param y: Second array, shape (n,) or (n, m) where m is number of columns
   :type y: np.ndarray

   :returns: **r** -- Correlation coefficients. If y is 1-D, returns scalar.
             If y is 2-D with m columns, returns array of shape (m,)
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> x = np.array([1, 2, 3, 4, 5])
   >>> y = np.array([2, 4, 6, 8, 10])
   >>> r = pearson_correlation(x, y)
   >>> print(f"Correlation: {r:.3f}")  # Should be 1.0

   >>> # Multiple correlations at once
   >>> y_multi = np.column_stack([
   ...     [2, 4, 6, 8, 10],  # Perfect positive correlation
   ...     [5, 4, 3, 2, 1],   # Perfect negative correlation
   ... ])
   >>> r = pearson_correlation(x, y_multi)
   >>> print(r)  # [1.0, -1.0]

   .. rubric:: Notes

   Based on corrPearson.m from the MATLAB codebase.
   Normalization factor (n-1) omitted since we renormalize anyway.


.. py:function:: plot_autocorrelogram(autocorr, config = None, *, gridness_score = None, center_radius = None, peak_locations = None, title = 'Spatial Autocorrelation', xlabel = 'X Lag (bins)', ylabel = 'Y Lag (bins)', figsize = (6, 6), save_path = None, show = True, ax = None, **kwargs)

   Plot 2D autocorrelogram with optional annotations.


.. py:function:: plot_grid_score_histogram(scores, config = None, *, bins = 30, title = 'Grid Score Distribution', xlabel = 'Grid Score', ylabel = 'Count', figsize = (6, 4), save_path = None, show = True, ax = None, **kwargs)

   Plot histogram of gridness scores.


.. py:function:: plot_gridness_analysis(rate_map, autocorr, result, config = None, *, title = 'Grid Cell Analysis', figsize = (15, 5), save_path = None, show = True)

   Comprehensive grid analysis plot with rate map, autocorr, and statistics.


.. py:function:: plot_hd_analysis(result, time_stamps = None, head_directions = None, spike_times = None, figsize = (15, 5))

   Comprehensive head direction analysis plot.

   :param result: Results from HeadDirectionAnalyzer
   :type result: HDCellResult
   :param time_stamps: Time stamps for plotting trajectory
   :type time_stamps: np.ndarray, optional
   :param head_directions: Head direction time series
   :type head_directions: np.ndarray, optional
   :param spike_times: Spike times
   :type spike_times: np.ndarray, optional
   :param figsize: Figure size
   :type figsize: tuple, optional

   :returns: **fig** -- The figure object
   :rtype: plt.Figure


.. py:function:: plot_polar_tuning(angles, rates, preferred_direction = None, mvl = None, title = 'Directional Tuning', ax = None)

   Plot directional tuning curve in polar coordinates.

   :param angles: Angular bins in radians
   :type angles: np.ndarray
   :param rates: Firing rates for each bin
   :type rates: np.ndarray
   :param preferred_direction: Preferred direction to mark (radians)
   :type preferred_direction: float, optional
   :param mvl: Mean Vector Length to display
   :type mvl: float, optional
   :param title: Plot title
   :type title: str, optional
   :param ax: Polar axes to plot on. If None, creates new figure.
   :type ax: plt.Axes, optional

   :returns: **ax** -- The axes object
   :rtype: plt.Axes


.. py:function:: plot_rate_map(rate_map, config = None, *, title = 'Firing Field (Rate Map)', xlabel = 'X Position (bins)', ylabel = 'Y Position (bins)', figsize = (6, 6), colorbar = True, save_path = None, show = True, ax = None, **kwargs)

   Plot 2D spatial firing rate map.


.. py:function:: plot_temporal_autocorr(lags, acorr, title = 'Temporal Autocorrelation', ax = None)

   Plot temporal autocorrelation as bar plot.

   :param lags: Time lags (ms or bins)
   :type lags: np.ndarray
   :param acorr: Autocorrelation values
   :type acorr: np.ndarray
   :param title: Plot title
   :type title: str, optional
   :param ax: Axes to plot on
   :type ax: plt.Axes, optional

   :returns: **ax** -- The axes object
   :rtype: plt.Axes


.. py:function:: pol2cart(theta, rho)

   Transform polar coordinates to Cartesian coordinates.

   :param theta: Angle in radians
   :type theta: np.ndarray
   :param rho: Radius (distance from origin)
   :type rho: np.ndarray

   :returns: * **x** (*np.ndarray*) -- X coordinates
             * **y** (*np.ndarray*) -- Y coordinates

   .. rubric:: Examples

   >>> theta = np.array([0, np.pi/2, np.pi])
   >>> rho = np.array([1, 1, 1])
   >>> x, y = pol2cart(theta, rho)
   >>> print(x)  # [1, 0, -1]
   >>> print(y)  # [0, 1, 0]

   .. rubric:: Notes

   Equivalent to MATLAB's pol2cart function.


.. py:function:: polyarea(x, y)

   Compute area of a polygon using the shoelace formula.

   :param x: X coordinates of polygon vertices
   :type x: np.ndarray
   :param y: Y coordinates of polygon vertices
   :type y: np.ndarray

   :returns: **area** -- Area of the polygon (always positive)
   :rtype: float

   .. rubric:: Examples

   >>> # Unit square
   >>> x = np.array([0, 1, 1, 0])
   >>> y = np.array([0, 0, 1, 1])
   >>> area = polyarea(x, y)
   >>> print(f"Area: {area}")  # Should be 1.0

   >>> # Triangle
   >>> x = np.array([0, 1, 0.5])
   >>> y = np.array([0, 0, 1])
   >>> area = polyarea(x, y)
   >>> print(f"Area: {area}")  # Should be 0.5

   .. rubric:: Notes

   Based on MATLAB's polyarea function using the shoelace formula.
   The polygon can be specified in either clockwise or counter-clockwise order.


.. py:function:: regionprops(labeled_image, intensity_image = None)

   Measure properties of labeled image regions.

   :param labeled_image: Labeled image (output from label_connected_components)
   :type labeled_image: np.ndarray
   :param intensity_image: Intensity image for computing intensity-based properties
   :type intensity_image: np.ndarray, optional

   :returns: **properties** -- List of region property objects. Each object has attributes like:
             - centroid: (row, col) of region center
             - area: number of pixels in region
             - bbox: bounding box coordinates
             - etc.
   :rtype: list of RegionProperties

   .. rubric:: Examples

   >>> binary = (np.random.rand(50, 50) > 0.7)
   >>> labels, _ = label_connected_components(binary)
   >>> props = regionprops(labels)
   >>> for prop in props:
   ...     print(f"Region at {prop.centroid}, area={prop.area}")

   .. rubric:: Notes

   Based on MATLAB's regionprops function.
   Uses skimage.measure.regionprops.


.. py:function:: rotate_image(image, angle, output_shape = None, method = 'bilinear', preserve_range = True)

   Rotate an image by a given angle.

   :param image: 2D image array to rotate
   :type image: np.ndarray
   :param angle: Rotation angle in degrees. Positive values rotate counter-clockwise.
   :type angle: float
   :param output_shape: Shape of the output image (height, width). If None, uses input shape.
   :type output_shape: tuple of int, optional
   :param method: Interpolation method: 'bilinear' (default), 'nearest', 'cubic'
   :type method: str, optional
   :param preserve_range: Whether to preserve the original value range. Default is True.
   :type preserve_range: bool, optional

   :returns: **rotated** -- Rotated image
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> image = np.random.rand(50, 50)
   >>> rotated = rotate_image(image, 30)  # Rotate 30 degrees CCW
   >>> rotated_90 = rotate_image(image, 90)  # Rotate 90 degrees

   .. rubric:: Notes

   Based on MATLAB's imrotate function. Uses scipy.ndimage.rotate.
   The rotation is performed around the center of the image.


.. py:function:: squared_distance(X, Y)

   Compute squared Euclidean distance matrix between two sets of points.

   Efficiently computes all pairwise squared distances between points in X and Y
   using the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y

   :param X: First set of points, shape (d, n) where d is dimension, n is number of points
   :type X: np.ndarray
   :param Y: Second set of points, shape (d, m) where d is dimension, m is number of points
   :type Y: np.ndarray

   :returns: **D** -- Squared distance matrix, shape (n, m)
             D[i, j] = ||X[:, i] - Y[:, j]||^2
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> # 2D points
   >>> X = np.array([[0, 1, 2], [0, 0, 0]])  # 3 points along x-axis
   >>> Y = np.array([[0, 0], [1, 2]])  # 2 points along y-axis
   >>> D = squared_distance(X, Y)
   >>> print(D)  # Distances from X points to Y points

   .. rubric:: Notes

   Based on sqDistance inline function from gridnessScore.m and findCentreRadius.m.
   Uses bsxfun-style broadcasting for efficiency.


.. py:function:: wrap_to_pi(angles)

   Wrap angles to the range [-π, π].

   :param angles: Angles in radians
   :type angles: np.ndarray

   :returns: **wrapped** -- Angles wrapped to [-π, π]
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> angles = np.array([0, np.pi, -np.pi, 3*np.pi, -3*np.pi])
   >>> wrapped = wrap_to_pi(angles)
   >>> print(wrapped)  # All in [-π, π]

   .. rubric:: Notes

   Equivalent to MATLAB's wrapToPi function.


