src.canns.analyzer.data.cell_classification.utils
=================================================

.. py:module:: src.canns.analyzer.data.cell_classification.utils

.. autoapi-nested-parse::

   Utility modules.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/analyzer/data/cell_classification/utils/circular_stats/index
   /autoapi/src/canns/analyzer/data/cell_classification/utils/correlation/index
   /autoapi/src/canns/analyzer/data/cell_classification/utils/geometry/index
   /autoapi/src/canns/analyzer/data/cell_classification/utils/image_processing/index


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.utils.autocorrelation_2d
   src.canns.analyzer.data.cell_classification.utils.cart2pol
   src.canns.analyzer.data.cell_classification.utils.circ_dist
   src.canns.analyzer.data.cell_classification.utils.circ_dist2
   src.canns.analyzer.data.cell_classification.utils.circ_mean
   src.canns.analyzer.data.cell_classification.utils.circ_r
   src.canns.analyzer.data.cell_classification.utils.circ_rtest
   src.canns.analyzer.data.cell_classification.utils.circ_std
   src.canns.analyzer.data.cell_classification.utils.dilate_image
   src.canns.analyzer.data.cell_classification.utils.find_contours_at_level
   src.canns.analyzer.data.cell_classification.utils.find_regional_maxima
   src.canns.analyzer.data.cell_classification.utils.fit_ellipse
   src.canns.analyzer.data.cell_classification.utils.gaussian_filter_2d
   src.canns.analyzer.data.cell_classification.utils.label_connected_components
   src.canns.analyzer.data.cell_classification.utils.normalized_xcorr2
   src.canns.analyzer.data.cell_classification.utils.pearson_correlation
   src.canns.analyzer.data.cell_classification.utils.pol2cart
   src.canns.analyzer.data.cell_classification.utils.polyarea
   src.canns.analyzer.data.cell_classification.utils.regionprops
   src.canns.analyzer.data.cell_classification.utils.rotate_image
   src.canns.analyzer.data.cell_classification.utils.squared_distance
   src.canns.analyzer.data.cell_classification.utils.wrap_to_pi


Package Contents
----------------

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


.. py:function:: dilate_image(image, footprint = None, selem_type = 'square', selem_size = 3)

   Perform morphological dilation on a binary image.

   :param image: Binary input image
   :type image: np.ndarray
   :param footprint: Structuring element. If None, uses selem_type and selem_size.
   :type footprint: np.ndarray, optional
   :param selem_type: Type of structuring element: 'square', 'disk', 'diamond'
                      Default is 'square'.
   :type selem_type: str, optional
   :param selem_size: Size of structuring element. Default is 3.
   :type selem_size: int, optional

   :returns: **dilated** -- Dilated image
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> binary_image = (np.random.rand(50, 50) > 0.8)
   >>> dilated = dilate_image(binary_image, selem_type='square', selem_size=3)

   .. rubric:: Notes

   Based on MATLAB's imdilate function.
   Uses skimage.morphology.dilation.


.. py:function:: find_contours_at_level(image, level)

   Find contours in an image at a specific threshold level.

   :param image: 2D input image
   :type image: np.ndarray
   :param level: Threshold level for contour detection
   :type level: float

   :returns: **contours** -- List of contours. Each contour is an (N, 2) array of (row, col) coordinates.
             Note: Returns (row, col) = (y, x), opposite of MATLAB's (x, y) order!
   :rtype: list of np.ndarray

   .. rubric:: Examples

   >>> # Create a simple image with a circular feature
   >>> x = np.linspace(-5, 5, 100)
   >>> xx, yy = np.meshgrid(x, x)
   >>> image = np.exp(-(xx**2 + yy**2))
   >>> contours = find_contours_at_level(image, 0.5)
   >>> print(f"Found {len(contours)} contours")

   .. rubric:: Notes

   Based on MATLAB's contourc function.
   Uses skimage.measure.find_contours.

   CRITICAL: Coordinate order difference!
   - MATLAB contourc: returns [x; y] (column major)
   - Python find_contours: returns (row, col) = (y, x)

   For gridness analysis, this coordinate swap must be handled!


.. py:function:: find_regional_maxima(image, connectivity = 1, allow_diagonal = False)

   Find regional maxima in an image.

   A regional maximum is a connected component of pixels with the same value,
   surrounded by pixels with strictly lower values.

   :param image: 2D input image
   :type image: np.ndarray
   :param connectivity: Connectivity for defining neighbors:
                        - 1: 4-connectivity (default, equivalent to MATLAB connectivity=4)
                        - 2: 8-connectivity (equivalent to MATLAB connectivity=8)
   :type connectivity: int, optional
   :param allow_diagonal: If True, uses 8-connectivity. If False, uses 4-connectivity.
                          Overrides connectivity parameter if specified.
   :type allow_diagonal: bool, optional

   :returns: **maxima** -- Binary image where True indicates regional maxima
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> # Create image with some peaks
   >>> x = np.linspace(-3, 3, 50)
   >>> xx, yy = np.meshgrid(x, x)
   >>> image = np.exp(-(xx**2 + yy**2)) + 0.5 * np.exp(-((xx-1.5)**2 + (yy-1.5)**2))
   >>> maxima = find_regional_maxima(image)
   >>> print(f"Found {np.sum(maxima)} maxima")

   .. rubric:: Notes

   Based on MATLAB's imregionalmax function.

   IMPORTANT: Connectivity mapping differs between MATLAB and Python!
   - MATLAB imregionalmax(image, 4) → Python connectivity=1
   - MATLAB imregionalmax(image, 8) → Python connectivity=2

   Uses skimage.morphology.local_maxima for detection.


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


.. py:function:: gaussian_filter_2d(image, sigma, mode = 'reflect', truncate = 4.0)

   Apply 2D Gaussian filter to an image.

   :param image: 2D input image
   :type image: np.ndarray
   :param sigma: Standard deviation of Gaussian kernel
   :type sigma: float
   :param mode: Boundary handling mode:
                - 'reflect' (default): reflect at boundaries
                - 'constant': pad with zeros
                - 'nearest': replicate edge values
                - 'mirror': mirror at boundaries
                - 'wrap': wrap around
   :type mode: str, optional
   :param truncate: Truncate filter at this many standard deviations. Default is 4.0.
   :type truncate: float, optional

   :returns: **filtered** -- Filtered image
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> image = np.random.rand(100, 100)
   >>> smoothed = gaussian_filter_2d(image, sigma=2.0)

   .. rubric:: Notes

   Based on MATLAB's imgaussfilt function.
   Uses scipy.ndimage.gaussian_filter.


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


