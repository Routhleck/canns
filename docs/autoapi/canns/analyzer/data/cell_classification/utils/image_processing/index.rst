canns.analyzer.data.cell_classification.utils.image_processing
==============================================================

.. py:module:: canns.analyzer.data.cell_classification.utils.image_processing

.. autoapi-nested-parse::

   Image Processing Utilities

   Functions for image manipulation including rotation, filtering, and morphological operations.



Attributes
----------

.. autoapisummary::

   canns.analyzer.data.cell_classification.utils.image_processing.image


Functions
---------

.. autoapisummary::

   canns.analyzer.data.cell_classification.utils.image_processing.dilate_image
   canns.analyzer.data.cell_classification.utils.image_processing.find_contours_at_level
   canns.analyzer.data.cell_classification.utils.image_processing.find_regional_maxima
   canns.analyzer.data.cell_classification.utils.image_processing.gaussian_filter_2d
   canns.analyzer.data.cell_classification.utils.image_processing.label_connected_components
   canns.analyzer.data.cell_classification.utils.image_processing.regionprops
   canns.analyzer.data.cell_classification.utils.image_processing.rotate_image


Module Contents
---------------

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


.. py:data:: image

