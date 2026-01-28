src.canns.analyzer.data.cell_classification.core.grid_cells
===========================================================

.. py:module:: src.canns.analyzer.data.cell_classification.core.grid_cells

.. autoapi-nested-parse::

   Grid Cell Classification

   Implementation of gridness score algorithm for identifying and characterizing grid cells.

   Based on the MATLAB gridnessScore.m implementation.



Attributes
----------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.core.grid_cells.x


Classes
-------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.core.grid_cells.GridnessAnalyzer
   src.canns.analyzer.data.cell_classification.core.grid_cells.GridnessResult


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.core.grid_cells.compute_2d_autocorrelation


Module Contents
---------------

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


.. py:data:: x

