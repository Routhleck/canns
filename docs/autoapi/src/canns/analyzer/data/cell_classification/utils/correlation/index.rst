src.canns.analyzer.data.cell_classification.utils.correlation
=============================================================

.. py:module:: src.canns.analyzer.data.cell_classification.utils.correlation

.. autoapi-nested-parse::

   Correlation Utilities

   Functions for computing Pearson correlation and normalized cross-correlation,
   optimized for neuroscience data analysis.



Attributes
----------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.utils.correlation.x


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.cell_classification.utils.correlation.autocorrelation_2d
   src.canns.analyzer.data.cell_classification.utils.correlation.normalized_xcorr2
   src.canns.analyzer.data.cell_classification.utils.correlation.pearson_correlation


Module Contents
---------------

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


.. py:data:: x

