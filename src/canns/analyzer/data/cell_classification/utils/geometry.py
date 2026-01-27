"""
Geometry Utilities

Functions for geometric calculations including ellipse fitting,
distance computations, and polygon operations.
"""

import numpy as np
from typing import Tuple


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Least-squares fit of ellipse to 2D points.

    Implements the Direct Least Squares Fitting algorithm by Fitzgibbon et al. (1999).
    This is a robust method that includes scaling to reduce roundoff error and
    returns geometric parameters rather than quadratic form coefficients.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of points (1D array)
    y : np.ndarray
        Y coordinates of points (1D array)

    Returns
    -------
    params : np.ndarray
        Array of shape (5,) containing:
        [center_x, center_y, radius_x, radius_y, theta_radians]
        where theta is the orientation angle of the major axis

    Examples
    --------
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

    Notes
    -----
    Based on fitEllipse.m from the MATLAB codebase.

    References
    ----------
    Fitzgibbon, A.W., Pilu, M., and Fisher, R.B. (1999).
    "Direct least-squares fitting of ellipses". IEEE T-PAMI, 21(5):476-480.
    http://research.microsoft.com/en-us/um/people/awf/ellipse/
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 5 or len(y) < 5:
        raise ValueError("Need at least 5 points to fit an ellipse")

    # Normalize data to reduce roundoff error
    mx = np.mean(x)
    my = np.mean(y)
    sx = (np.max(x) - np.min(x)) / 2
    sy = (np.max(y) - np.min(y)) / 2

    if sx == 0 or sy == 0:
        raise ValueError("Points must have non-zero extent in both dimensions")

    x_norm = (x - mx) / sx
    y_norm = (y - my) / sy

    # Force to column vectors
    x_norm = x_norm.ravel()
    y_norm = y_norm.ravel()

    # Build design matrix
    # D = [x^2, xy, y^2, x, y, 1]
    D = np.column_stack([
        x_norm ** 2,
        x_norm * y_norm,
        y_norm ** 2,
        x_norm,
        y_norm,
        np.ones_like(x_norm)
    ])

    # Build scatter matrix
    S = D.T @ D

    # Build 6x6 constraint matrix for ellipse
    # This constrains the solution to be an ellipse (not hyperbola or parabola)
    C = np.zeros((6, 6))
    C[0, 2] = -2
    C[1, 1] = 1
    C[2, 0] = -2

    # Solve the generalized eigensystem using the stable method
    # Break into blocks (as in the "new way" from the MATLAB code)
    tmpA = S[:3, :3]
    tmpB = S[:3, 3:6]
    tmpC = S[3:6, 3:6]
    tmpD = C[:3, :3]

    # Solve the reduced eigensystem
    try:
        tmpE = np.linalg.inv(tmpC) @ tmpB.T
        tmpM = np.linalg.inv(tmpD) @ (tmpA - tmpB @ tmpE)

        # Find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(tmpM)

        # Find the positive eigenvalue (since det(tmpD) < 0)
        # Look for small positive or negative eigenvalues near zero
        idx = np.where((np.real(eigenvalues) < 1e-8) & ~np.isinf(eigenvalues))[0]

        if len(idx) == 0:
            # Fallback: take the smallest positive eigenvalue
            idx = np.argmin(np.abs(eigenvalues))
        else:
            idx = idx[0]

        # Extract eigenvector
        A_top = np.real(eigenvectors[:, idx])

        # Recover the bottom half
        A_bottom = -tmpE @ A_top
        A = np.concatenate([A_top, A_bottom])

    except np.linalg.LinAlgError:
        raise ValueError("Failed to fit ellipse: singular matrix encountered")

    # Unnormalize the coefficients
    par = np.array([
        A[0] * sy * sy,
        A[1] * sx * sy,
        A[2] * sx * sx,
        -2*A[0]*sy*sy*mx - A[1]*sx*sy*my + A[3]*sx*sy*sy,
        -A[1]*sx*sy*mx - 2*A[2]*sx*sx*my + A[4]*sx*sx*sy,
        A[0]*sy*sy*mx*mx + A[1]*sx*sy*mx*my + A[2]*sx*sx*my*my -
        A[3]*sx*sy*sy*mx - A[4]*sx*sx*sy*my + A[5]*sx*sx*sy*sy
    ])

    # Convert quadratic form to geometric parameters
    theta_rad = 0.5 * np.arctan2(par[1], par[0] - par[2])
    cost = np.cos(theta_rad)
    sint = np.sin(theta_rad)
    sin_squared = sint ** 2
    cos_squared = cost ** 2
    cos_sin = sint * cost

    Ao = par[5]
    Au = par[3] * cost + par[4] * sint
    Av = -par[3] * sint + par[4] * cost
    Auu = par[0] * cos_squared + par[2] * sin_squared + par[1] * cos_sin
    Avv = par[0] * sin_squared + par[2] * cos_squared - par[1] * cos_sin

    # Center in rotated coordinates
    tuCentre = -Au / (2 * Auu)
    tvCentre = -Av / (2 * Avv)
    wCentre = Ao - Auu * tuCentre ** 2 - Avv * tvCentre ** 2

    # Transform back to original coordinates
    uCentre = tuCentre * cost - tvCentre * sint
    vCentre = tuCentre * sint + tvCentre * cost

    # Radii
    Ru = -wCentre / Auu
    Rv = -wCentre / Avv

    Ru = np.sqrt(np.abs(Ru)) * np.sign(Ru)
    Rv = np.sqrt(np.abs(Rv)) * np.sign(Rv)

    # Return geometric parameters
    result = np.array([uCentre, vCentre, Ru, Rv, theta_rad])

    return result


def squared_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distance matrix between two sets of points.

    Efficiently computes all pairwise squared distances between points in X and Y
    using the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y

    Parameters
    ----------
    X : np.ndarray
        First set of points, shape (d, n) where d is dimension, n is number of points
    Y : np.ndarray
        Second set of points, shape (d, m) where d is dimension, m is number of points

    Returns
    -------
    D : np.ndarray
        Squared distance matrix, shape (n, m)
        D[i, j] = ||X[:, i] - Y[:, j]||^2

    Examples
    --------
    >>> # 2D points
    >>> X = np.array([[0, 1, 2], [0, 0, 0]])  # 3 points along x-axis
    >>> Y = np.array([[0, 0], [1, 2]])  # 2 points along y-axis
    >>> D = squared_distance(X, Y)
    >>> print(D)  # Distances from X points to Y points

    Notes
    -----
    Based on sqDistance inline function from gridnessScore.m and findCentreRadius.m.
    Uses bsxfun-style broadcasting for efficiency.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    # ||x||^2 for each column of X
    X_norm_sq = np.sum(X ** 2, axis=0, keepdims=True)  # Shape: (1, n)

    # ||y||^2 for each column of Y
    Y_norm_sq = np.sum(Y ** 2, axis=0, keepdims=True)  # Shape: (1, m)

    # Compute: ||x||^2 + ||y||^2 - 2*x·y using broadcasting
    # X_norm_sq.T: (n, 1)
    # Y_norm_sq: (1, m)
    # X.T @ Y: (n, m)
    D = X_norm_sq.T + Y_norm_sq - 2 * (X.T @ Y)

    # Ensure non-negative (due to floating point errors)
    D = np.maximum(D, 0)

    return D


def polyarea(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute area of a polygon using the shoelace formula.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of polygon vertices
    y : np.ndarray
        Y coordinates of polygon vertices

    Returns
    -------
    area : float
        Area of the polygon (always positive)

    Examples
    --------
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

    Notes
    -----
    Based on MATLAB's polyarea function using the shoelace formula.
    The polygon can be specified in either clockwise or counter-clockwise order.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) < 3:
        return 0.0

    # Shoelace formula: A = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    # Use np.roll to get next elements cyclically
    area = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1)) - np.sum(np.roll(x, -1) * y)
    )

    return float(area)


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """
    Wrap angles to the range [-π, π].

    Parameters
    ----------
    angles : np.ndarray
        Angles in radians

    Returns
    -------
    wrapped : np.ndarray
        Angles wrapped to [-π, π]

    Examples
    --------
    >>> angles = np.array([0, np.pi, -np.pi, 3*np.pi, -3*np.pi])
    >>> wrapped = wrap_to_pi(angles)
    >>> print(wrapped)  # All in [-π, π]

    Notes
    -----
    Equivalent to MATLAB's wrapToPi function.
    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def cart2pol(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates

    Returns
    -------
    theta : np.ndarray
        Angle in radians, range [-π, π]
    rho : np.ndarray
        Radius (distance from origin)

    Examples
    --------
    >>> x = np.array([1, 0, -1])
    >>> y = np.array([0, 1, 0])
    >>> theta, rho = cart2pol(x, y)
    >>> print(theta)  # [0, π/2, π]
    >>> print(rho)    # [1, 1, 1]

    Notes
    -----
    Equivalent to MATLAB's cart2pol function.
    """
    theta = np.arctan2(y, x)
    rho = np.sqrt(x ** 2 + y ** 2)
    return theta, rho


def pol2cart(theta: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    theta : np.ndarray
        Angle in radians
    rho : np.ndarray
        Radius (distance from origin)

    Returns
    -------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates

    Examples
    --------
    >>> theta = np.array([0, np.pi/2, np.pi])
    >>> rho = np.array([1, 1, 1])
    >>> x, y = pol2cart(theta, rho)
    >>> print(x)  # [1, 0, -1]
    >>> print(y)  # [0, 1, 0]

    Notes
    -----
    Equivalent to MATLAB's pol2cart function.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


if __name__ == "__main__":
    # Simple tests
    print("Testing geometry functions...")

    # Test 1: Fit ellipse
    print("\nTest 1 - Ellipse fitting:")
    t = np.linspace(0, 2*np.pi, 100)
    cx, cy = 5, 3  # center
    rx, ry = 4, 2  # radii
    angle = np.pi/6  # rotation

    # Generate points on ellipse
    x = cx + rx * np.cos(t) * np.cos(angle) - ry * np.sin(t) * np.sin(angle)
    y = cy + rx * np.cos(t) * np.sin(angle) + ry * np.sin(t) * np.cos(angle)

    params = fit_ellipse(x, y)
    print(f"  True center: ({cx}, {cy})")
    print(f"  Fitted center: ({params[0]:.2f}, {params[1]:.2f})")
    print(f"  True radii: ({rx}, {ry})")
    print(f"  Fitted radii: ({abs(params[2]):.2f}, {abs(params[3]):.2f})")
    print(f"  True angle: {np.rad2deg(angle):.2f}°")
    print(f"  Fitted angle: {np.rad2deg(params[4]):.2f}°")

    # Test 2: Squared distance
    print("\nTest 2 - Squared distance:")
    X = np.array([[0, 1, 2], [0, 0, 0]])  # 3 points along x-axis
    Y = np.array([[0, 0], [1, 2]])  # 2 points along y-axis
    D = squared_distance(X, Y)
    print(f"  X points: {X.T}")
    print(f"  Y points: {Y.T}")
    print(f"  Squared distances:\n{D}")

    # Test 3: Polygon area
    print("\nTest 3 - Polygon area:")
    # Unit square
    x = np.array([0, 1, 1, 0])
    y = np.array([0, 0, 1, 1])
    area = polyarea(x, y)
    print(f"  Unit square area: {area:.3f} (should be 1.0)")

    # Triangle
    x = np.array([0, 1, 0.5])
    y = np.array([0, 0, 1])
    area = polyarea(x, y)
    print(f"  Triangle area: {area:.3f} (should be 0.5)")

    # Test 4: Coordinate transformations
    print("\nTest 4 - Coordinate transformations:")
    x = np.array([1, 0, -1, 0])
    y = np.array([0, 1, 0, -1])
    theta, rho = cart2pol(x, y)
    print(f"  Cartesian: {np.column_stack([x, y])}")
    print(f"  Polar (θ, ρ): {np.column_stack([np.rad2deg(theta), rho])}")

    x2, y2 = pol2cart(theta, rho)
    print(f"  Back to Cartesian: {np.column_stack([x2, y2])}")
    print(f"  Error: {np.max(np.abs(np.column_stack([x-x2, y-y2]))):.10f}")

    print("\nAll tests completed!")
