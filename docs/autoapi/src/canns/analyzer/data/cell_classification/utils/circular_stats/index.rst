src.canns.analyzer.data.cell_classification.utils.circular_stats
================================================================

.. py:module:: src.canns.analyzer.data.cell_classification.utils.circular_stats

.. autoapi-nested-parse::

   Circular Statistics Utilities

   Python port of CircStat MATLAB toolbox functions for circular statistics.

   .. rubric:: References

   - Statistical analysis of circular data, N.I. Fisher
   - Topics in circular statistics, S.R. Jammalamadaka et al.
   - Biostatistical Analysis, J. H. Zar
   - CircStat MATLAB toolbox by Philipp Berens (2009)



Attributes
----------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.utils.circular_stats.angles
   src.canns.analyzer.data.cell_classification.utils.circular_stats.angular_distance
   src.canns.analyzer.data.cell_classification.utils.circular_stats.angular_mean
   src.canns.analyzer.data.cell_classification.utils.circular_stats.angular_std
   src.canns.analyzer.data.cell_classification.utils.circular_stats.mvl


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.utils.circular_stats.circ_dist
   src.canns.analyzer.data.cell_classification.utils.circular_stats.circ_dist2
   src.canns.analyzer.data.cell_classification.utils.circular_stats.circ_mean
   src.canns.analyzer.data.cell_classification.utils.circular_stats.circ_r
   src.canns.analyzer.data.cell_classification.utils.circular_stats.circ_rtest
   src.canns.analyzer.data.cell_classification.utils.circular_stats.circ_std


Module Contents
---------------

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


.. py:data:: angles

.. py:data:: angular_distance

.. py:data:: angular_mean

.. py:data:: angular_std

.. py:data:: mvl

