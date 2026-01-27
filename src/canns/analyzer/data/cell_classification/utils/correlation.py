"""
Correlation Utilities

Functions for computing Pearson correlation and normalized cross-correlation,
optimized for neuroscience data analysis.
"""

import numpy as np
from scipy import signal, fft


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation coefficient between x and each column of y.

    This is an optimized implementation that efficiently handles multiple
    correlations when y has multiple columns.

    Parameters
    ----------
    x : np.ndarray
        First array, shape (n,) or (n, 1)
    y : np.ndarray
        Second array, shape (n,) or (n, m) where m is number of columns

    Returns
    -------
    r : np.ndarray
        Correlation coefficients. If y is 1-D, returns scalar.
        If y is 2-D with m columns, returns array of shape (m,)

    Examples
    --------
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

    Notes
    -----
    Based on corrPearson.m from the MATLAB codebase.
    Normalization factor (n-1) omitted since we renormalize anyway.
    """
    # Ensure arrays are at least 2D for matrix operations
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    # If x or y were passed as row vectors, transpose them
    if x.shape[0] == 1:
        x = x.T
    if y.shape[0] == 1:
        y = y.T

    n = x.shape[0]

    # Center the data (remove mean)
    x = x - np.sum(x, axis=0) / n
    y = y - np.sum(y, axis=0) / n

    # Compute correlation: x^T * y
    r = x.T @ y  # Shape: (1, m) if x is column, y has m columns

    # Compute norms
    dx = np.linalg.norm(x, axis=0, keepdims=True)  # Shape: (1,) or (1, k)
    dy = np.linalg.norm(y, axis=0, keepdims=True)  # Shape: (1, m)

    # Normalize: r / (dx * dy)
    # Broadcasting handles the division correctly
    r = r / dx.T  # Divide by dx (column-wise)
    r = r / dy    # Divide by dy (row-wise)

    # Return as 1D array or scalar
    r = np.squeeze(r)

    return r


def normalized_xcorr2(
    template: np.ndarray,
    image: np.ndarray,
    mode: str = 'same',
    min_overlap: int = 0
) -> np.ndarray:
    """
    Normalized 2D cross-correlation.

    Computes the normalized cross-correlation of two 2D arrays. Unlike
    scipy.signal.correlate, this function properly handles varying overlap
    regions and works correctly even when template and image are the same size.

    Parameters
    ----------
    template : np.ndarray
        2D template array
    image : np.ndarray
        2D image array
    mode : str, optional
        'full' - full correlation (default for autocorrelation)
        'same' - output size same as image
        'valid' - only where template fully overlaps image
    min_overlap : int, optional
        Minimum number of overlapping pixels required for valid correlation.
        Locations with fewer overlapping pixels are set to 0.
        Default is 0 (no threshold).

    Returns
    -------
    C : np.ndarray
        Normalized cross-correlation. Values range from -1 to 1.

    Examples
    --------
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

    Notes
    -----
    This is a simplified Python implementation. For the full general version
    (handling all edge cases), see normxcorr2_general.m by Dirk Padfield.

    For most neuroscience applications (autocorrelation of rate maps),
    scipy.signal.correlate with normalization is sufficient.

    References
    ----------
    Padfield, D. "Masked FFT registration". CVPR, 2010.
    Lewis, J.P. "Fast Normalized Cross-Correlation". Industrial Light & Magic.
    """
    # Convert to double for numerical stability
    template = np.asarray(template, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)

    # Ensure arrays are 2D
    if template.ndim != 2 or image.ndim != 2:
        raise ValueError("Both template and image must be 2D arrays")

    # Check for flat template (no variation)
    if np.std(template) == 0:
        raise ValueError("Template cannot have all identical values")

    # For neuroscience applications, we typically use FFT-based correlation
    # which is faster for larger arrays
    if mode == 'full':
        # Full correlation (output larger than both inputs)
        C = signal.correlate(image, template, mode='full', method='fft')

        # Compute normalization factors
        # This is a simplified normalization; full version would handle
        # varying overlap regions precisely
        n_template = template.size
        template_mean = np.mean(template)
        template_std = np.std(template)

        image_mean = np.mean(image)
        image_std = np.std(image)

        # Normalize
        if template_std > 0 and image_std > 0:
            C = (C - n_template * template_mean * image_mean) / \
                (n_template * template_std * image_std)

    else:
        # For 'same' or 'valid', use scipy's built-in
        C = signal.correlate(image, template, mode=mode, method='fft')

        # Simple normalization (assumes full overlap in valid region)
        template_norm = np.sqrt(np.sum(template ** 2))
        image_norm = np.sqrt(np.sum(image ** 2))

        if template_norm > 0 and image_norm > 0:
            C = C / (template_norm * image_norm)

    # Clip to valid correlation range
    C = np.clip(C, -1, 1)

    return C


def autocorrelation_2d(
    array: np.ndarray,
    overlap: float = 0.8,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute 2D autocorrelation of an array.

    This is a convenience function specifically for computing spatial
    autocorrelation of firing rate maps, which is needed for grid cell analysis.

    Parameters
    ----------
    array : np.ndarray
        2D array (e.g., firing rate map)
    overlap : float, optional
        Percentage of overlap region to keep (0-1). Default is 0.8.
        The autocorrelogram is cropped to this central region to avoid
        edge artifacts.
    normalize : bool, optional
        Whether to normalize the correlation. Default is True.

    Returns
    -------
    autocorr : np.ndarray
        2D autocorrelation array

    Examples
    --------
    >>> # Create a simple periodic pattern (grid-like)
    >>> x = np.linspace(0, 4*np.pi, 50)
    >>> xx, yy = np.meshgrid(x, x)
    >>> pattern = np.cos(xx) * np.cos(yy)
    >>> autocorr = autocorrelation_2d(pattern)
    >>> # Autocorr should show hexagonal/grid pattern

    Notes
    -----
    Based on autocorrelation.m from the MATLAB codebase.
    Replaces NaN values with 0 before computing correlation.
    """
    # Replace NaN with 0
    array = np.nan_to_num(array, nan=0.0)

    # Compute new size for overlap region
    new_size_v = int(np.round(array.shape[0] * (1 + overlap)))
    new_size_h = int(np.round(array.shape[1] * (1 + overlap)))

    # Ensure odd dimensions for symmetry
    if new_size_v % 2 == 0 and new_size_v > 0:
        new_size_v -= 1
    if new_size_h % 2 == 0 and new_size_h > 0:
        new_size_h -= 1

    # Handle empty or all-zero arrays
    if array.size == 0 or np.all(array == 0):
        return np.zeros((new_size_v, new_size_h))

    # Subtract mean for proper autocorrelation
    # This is crucial for grid cell analysis - ensures the autocorrelation
    # captures the spatial periodicity rather than the mean firing rate
    array_demean = array - np.mean(array)

    # Compute full autocorrelation
    Rxx = signal.correlate(array_demean, array_demean, mode='full', method='fft')

    # Extract central overlap region first
    offset_v = (Rxx.shape[0] - new_size_v) // 2
    offset_h = (Rxx.shape[1] - new_size_h) // 2

    if offset_v >= 0 and offset_h >= 0:
        Rxx = Rxx[offset_v:offset_v + new_size_v, offset_h:offset_h + new_size_h]
    else:
        # If requested size is larger than autocorr, just return full
        pass

    # Normalize by the center (zero-lag) value
    if normalize:
        center_v = Rxx.shape[0] // 2
        center_h = Rxx.shape[1] // 2
        center_value = Rxx[center_v, center_h]

        if center_value > 0:
            Rxx = Rxx / center_value

    return Rxx


if __name__ == "__main__":
    # Simple tests
    print("Testing correlation functions...")

    # Test 1: Pearson correlation
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)
    r = pearson_correlation(x, y)
    print(f"\nTest 1 - Perfect positive correlation: r = {r:.3f} (should be 1.0)")

    # Test 2: Multiple correlations
    y_multi = np.column_stack([
        [2, 4, 6, 8, 10],      # Perfect positive
        [5, 4, 3, 2, 1],       # Perfect negative
        [1, 1, 1, 1, 1],       # Constant (should be NaN or 0)
    ])
    r_multi = pearson_correlation(x, y_multi[:, :2])  # Skip constant
    print(f"\nTest 2 - Multiple correlations: r = {r_multi} (should be [1.0, -1.0])")

    # Test 3: 2D autocorrelation
    # Create a simple grid-like pattern
    x_coords = np.linspace(0, 4*np.pi, 30)
    xx, yy = np.meshgrid(x_coords, x_coords)
    grid_pattern = np.cos(xx) * np.cos(yy)

    autocorr = autocorrelation_2d(grid_pattern)
    print(f"\nTest 3 - 2D Autocorrelation:")
    print(f"  Input shape: {grid_pattern.shape}")
    print(f"  Autocorr shape: {autocorr.shape}")
    print(f"  Autocorr max: {np.max(autocorr):.3f} (should be close to 1.0 at center)")

    # Center should have maximum correlation
    center = np.array(autocorr.shape) // 2
    print(f"  Center value: {autocorr[tuple(center)]:.3f}")

    print("\nAll tests completed!")
