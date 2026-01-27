"""
Image Processing Utilities

Functions for image manipulation including rotation, filtering, and morphological operations.
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology


def rotate_image(
    image: np.ndarray,
    angle: float,
    output_shape: tuple[int, int] | None = None,
    method: str = "bilinear",
    preserve_range: bool = True,
) -> np.ndarray:
    """
    Rotate an image by a given angle.

    Parameters
    ----------
    image : np.ndarray
        2D image array to rotate
    angle : float
        Rotation angle in degrees. Positive values rotate counter-clockwise.
    output_shape : tuple of int, optional
        Shape of the output image (height, width). If None, uses input shape.
    method : str, optional
        Interpolation method: 'bilinear' (default), 'nearest', 'cubic'
    preserve_range : bool, optional
        Whether to preserve the original value range. Default is True.

    Returns
    -------
    rotated : np.ndarray
        Rotated image

    Examples
    --------
    >>> image = np.random.rand(50, 50)
    >>> rotated = rotate_image(image, 30)  # Rotate 30 degrees CCW
    >>> rotated_90 = rotate_image(image, 90)  # Rotate 90 degrees

    Notes
    -----
    Based on MATLAB's imrotate function. Uses scipy.ndimage.rotate.
    The rotation is performed around the center of the image.
    """
    # Map method names to scipy orders
    order_map = {"nearest": 0, "bilinear": 1, "cubic": 3}
    order = order_map.get(method.lower(), 1)

    # Rotate image (scipy rotates counter-clockwise for positive angles, same as MATLAB)
    rotated = ndimage.rotate(
        image,
        angle,
        order=order,
        reshape=False if output_shape else True,
        mode="constant",
        cval=0.0,
        prefilter=True,
    )

    # Resize to requested output shape if specified
    if output_shape is not None:
        if rotated.shape != output_shape:
            # Simple crop/pad to match output shape
            h, w = rotated.shape
            oh, ow = output_shape

            # Calculate center offsets
            start_h = (h - oh) // 2 if h > oh else 0
            start_w = (w - ow) // 2 if w > ow else 0

            if h >= oh and w >= ow:
                # Crop
                rotated = rotated[start_h : start_h + oh, start_w : start_w + ow]
            elif h <= oh and w <= ow:
                # Pad
                pad_h = (oh - h) // 2
                pad_w = (ow - w) // 2
                rotated = np.pad(
                    rotated,
                    ((pad_h, oh - h - pad_h), (pad_w, ow - w - pad_w)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                # Mixed crop/pad - just use zoom
                from scipy.ndimage import zoom

                zoom_factors = (oh / h, ow / w)
                rotated = zoom(rotated, zoom_factors, order=order)

    return rotated


def find_regional_maxima(
    image: np.ndarray, connectivity: int = 1, allow_diagonal: bool = False
) -> np.ndarray:
    """
    Find regional maxima in an image.

    A regional maximum is a connected component of pixels with the same value,
    surrounded by pixels with strictly lower values.

    Parameters
    ----------
    image : np.ndarray
        2D input image
    connectivity : int, optional
        Connectivity for defining neighbors:
        - 1: 4-connectivity (default, equivalent to MATLAB connectivity=4)
        - 2: 8-connectivity (equivalent to MATLAB connectivity=8)
    allow_diagonal : bool, optional
        If True, uses 8-connectivity. If False, uses 4-connectivity.
        Overrides connectivity parameter if specified.

    Returns
    -------
    maxima : np.ndarray
        Binary image where True indicates regional maxima

    Examples
    --------
    >>> # Create image with some peaks
    >>> x = np.linspace(-3, 3, 50)
    >>> xx, yy = np.meshgrid(x, x)
    >>> image = np.exp(-(xx**2 + yy**2)) + 0.5 * np.exp(-((xx-1.5)**2 + (yy-1.5)**2))
    >>> maxima = find_regional_maxima(image)
    >>> print(f"Found {np.sum(maxima)} maxima")

    Notes
    -----
    Based on MATLAB's imregionalmax function.

    IMPORTANT: Connectivity mapping differs between MATLAB and Python!
    - MATLAB imregionalmax(image, 4) → Python connectivity=1
    - MATLAB imregionalmax(image, 8) → Python connectivity=2

    Uses skimage.morphology.local_maxima for detection.
    """
    # Handle connectivity parameter
    if allow_diagonal:
        connectivity = 2

    # Use skimage to find local maxima
    # Note: skimage uses different connectivity convention than MATLAB
    maxima = morphology.local_maxima(image, connectivity=connectivity)

    return maxima


def find_contours_at_level(image: np.ndarray, level: float) -> list:
    """
    Find contours in an image at a specific threshold level.

    Parameters
    ----------
    image : np.ndarray
        2D input image
    level : float
        Threshold level for contour detection

    Returns
    -------
    contours : list of np.ndarray
        List of contours. Each contour is an (N, 2) array of (row, col) coordinates.
        Note: Returns (row, col) = (y, x), opposite of MATLAB's (x, y) order!

    Examples
    --------
    >>> # Create a simple image with a circular feature
    >>> x = np.linspace(-5, 5, 100)
    >>> xx, yy = np.meshgrid(x, x)
    >>> image = np.exp(-(xx**2 + yy**2))
    >>> contours = find_contours_at_level(image, 0.5)
    >>> print(f"Found {len(contours)} contours")

    Notes
    -----
    Based on MATLAB's contourc function.
    Uses skimage.measure.find_contours.

    CRITICAL: Coordinate order difference!
    - MATLAB contourc: returns [x; y] (column major)
    - Python find_contours: returns (row, col) = (y, x)

    For gridness analysis, this coordinate swap must be handled!
    """
    contours = measure.find_contours(image, level)
    return contours


def gaussian_filter_2d(
    image: np.ndarray, sigma: float, mode: str = "reflect", truncate: float = 4.0
) -> np.ndarray:
    """
    Apply 2D Gaussian filter to an image.

    Parameters
    ----------
    image : np.ndarray
        2D input image
    sigma : float
        Standard deviation of Gaussian kernel
    mode : str, optional
        Boundary handling mode:
        - 'reflect' (default): reflect at boundaries
        - 'constant': pad with zeros
        - 'nearest': replicate edge values
        - 'mirror': mirror at boundaries
        - 'wrap': wrap around
    truncate : float, optional
        Truncate filter at this many standard deviations. Default is 4.0.

    Returns
    -------
    filtered : np.ndarray
        Filtered image

    Examples
    --------
    >>> image = np.random.rand(100, 100)
    >>> smoothed = gaussian_filter_2d(image, sigma=2.0)

    Notes
    -----
    Based on MATLAB's imgaussfilt function.
    Uses scipy.ndimage.gaussian_filter.
    """
    filtered = ndimage.gaussian_filter(image, sigma=sigma, mode=mode, truncate=truncate)
    return filtered


def dilate_image(
    image: np.ndarray,
    footprint: np.ndarray | None = None,
    selem_type: str = "square",
    selem_size: int = 3,
) -> np.ndarray:
    """
    Perform morphological dilation on a binary image.

    Parameters
    ----------
    image : np.ndarray
        Binary input image
    footprint : np.ndarray, optional
        Structuring element. If None, uses selem_type and selem_size.
    selem_type : str, optional
        Type of structuring element: 'square', 'disk', 'diamond'
        Default is 'square'.
    selem_size : int, optional
        Size of structuring element. Default is 3.

    Returns
    -------
    dilated : np.ndarray
        Dilated image

    Examples
    --------
    >>> binary_image = (np.random.rand(50, 50) > 0.8)
    >>> dilated = dilate_image(binary_image, selem_type='square', selem_size=3)

    Notes
    -----
    Based on MATLAB's imdilate function.
    Uses skimage.morphology.dilation.
    """
    if footprint is None:
        if selem_type == "square":
            footprint = morphology.footprint_rectangle((selem_size, selem_size))
        elif selem_type == "disk":
            footprint = morphology.disk(selem_size)
        elif selem_type == "diamond":
            footprint = morphology.diamond(selem_size)
        else:
            raise ValueError(f"Unknown structuring element type: {selem_type}")

    dilated = morphology.binary_dilation(image, footprint=footprint)
    return dilated


def label_connected_components(
    binary_image: np.ndarray, connectivity: int = 2
) -> tuple[np.ndarray, int]:
    """
    Label connected components in a binary image.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary input image
    connectivity : int, optional
        Connectivity for defining neighbors:
        - 1: 4-connectivity
        - 2: 8-connectivity (default)

    Returns
    -------
    labels : np.ndarray
        Labeled image where each connected component has a unique integer label
    num_labels : int
        Number of connected components found

    Examples
    --------
    >>> binary = (np.random.rand(50, 50) > 0.7)
    >>> labels, n = label_connected_components(binary)
    >>> print(f"Found {n} connected components")

    Notes
    -----
    Based on MATLAB's bwconncomp function.
    Uses skimage.measure.label.
    """
    labels = measure.label(binary_image, connectivity=connectivity)
    num_labels = labels.max()
    return labels, num_labels


def regionprops(labeled_image: np.ndarray, intensity_image: np.ndarray | None = None) -> list:
    """
    Measure properties of labeled image regions.

    Parameters
    ----------
    labeled_image : np.ndarray
        Labeled image (output from label_connected_components)
    intensity_image : np.ndarray, optional
        Intensity image for computing intensity-based properties

    Returns
    -------
    properties : list of RegionProperties
        List of region property objects. Each object has attributes like:
        - centroid: (row, col) of region center
        - area: number of pixels in region
        - bbox: bounding box coordinates
        - etc.

    Examples
    --------
    >>> binary = (np.random.rand(50, 50) > 0.7)
    >>> labels, _ = label_connected_components(binary)
    >>> props = regionprops(labels)
    >>> for prop in props:
    ...     print(f"Region at {prop.centroid}, area={prop.area}")

    Notes
    -----
    Based on MATLAB's regionprops function.
    Uses skimage.measure.regionprops.
    """
    props = measure.regionprops(labeled_image, intensity_image=intensity_image)
    return props


if __name__ == "__main__":
    # Simple tests
    print("Testing image processing functions...")

    # Test 1: Image rotation
    print("\nTest 1 - Image rotation:")
    image = np.random.rand(50, 50)
    rotated_30 = rotate_image(image, 30)
    rotated_90 = rotate_image(image, 90)
    print(f"  Original shape: {image.shape}")
    print(f"  Rotated 30° shape: {rotated_30.shape}")
    print(f"  Rotated 90° shape: {rotated_90.shape}")

    # Test 2: Regional maxima
    print("\nTest 2 - Regional maxima:")
    x = np.linspace(-3, 3, 50)
    xx, yy = np.meshgrid(x, x)
    # Two Gaussian peaks
    peaks = np.exp(-(xx**2 + yy**2)) + 0.5 * np.exp(-((xx - 1.5) ** 2 + (yy - 1.5) ** 2))
    maxima = find_regional_maxima(peaks)
    print(f"  Found {np.sum(maxima)} maxima")
    print("  Maxima locations:")
    coords = np.argwhere(maxima)
    for i, (y, x) in enumerate(coords[:5]):  # Show first 5
        print(f"    Peak {i + 1}: ({x}, {y}), value={peaks[y, x]:.3f}")

    # Test 3: Contour detection
    print("\nTest 3 - Contour detection:")
    circle = np.sqrt(xx**2 + yy**2)
    contours = find_contours_at_level(circle, 1.5)
    print(f"  Found {len(contours)} contours at level 1.5")
    if len(contours) > 0:
        print(f"  Largest contour has {len(contours[0])} points")

    # Test 4: Gaussian filtering
    print("\nTest 4 - Gaussian filtering:")
    noisy = image + 0.1 * np.random.randn(*image.shape)
    smoothed = gaussian_filter_2d(noisy, sigma=2.0)
    print(f"  Noisy image std: {np.std(noisy):.4f}")
    print(f"  Smoothed image std: {np.std(smoothed):.4f}")

    # Test 5: Connected components
    print("\nTest 5 - Connected components:")
    binary = np.random.rand(50, 50) > 0.7
    labels, n = label_connected_components(binary)
    print(f"  Binary image has {np.sum(binary)} True pixels")
    print(f"  Found {n} connected components")

    props = regionprops(labels)
    print("  Region properties for first 3 components:")
    for i, prop in enumerate(props[:3]):
        print(f"    Region {i + 1}: centroid={prop.centroid}, area={prop.area}")

    print("\nAll tests completed!")
