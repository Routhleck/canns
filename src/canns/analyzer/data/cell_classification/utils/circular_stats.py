"""
Circular Statistics Utilities

Python port of CircStat MATLAB toolbox functions for circular statistics.

References:
    - Statistical analysis of circular data, N.I. Fisher
    - Topics in circular statistics, S.R. Jammalamadaka et al.
    - Biostatistical Analysis, J. H. Zar
    - CircStat MATLAB toolbox by Philipp Berens (2009)
"""

import numpy as np


def circ_r(
    alpha: np.ndarray, w: np.ndarray | None = None, d: float = 0.0, axis: int = 0
) -> float | np.ndarray:
    """
    Compute mean resultant vector length for circular data.

    This is a measure of circular variance (concentration). Values near 1 indicate
    high concentration, values near 0 indicate uniform distribution.

    Parameters
    ----------
    alpha : np.ndarray
        Sample of angles in radians
    w : np.ndarray, optional
        Weights for each angle (e.g., for binned data). If None, uniform weights assumed.
    d : float, optional
        Spacing of bin centers for binned data. If supplied, correction factor is used
        to correct for bias in estimation of r (in radians). Default is 0 (no correction).
    axis : int, optional
        Compute along this dimension. Default is 0.

    Returns
    -------
    r : float or np.ndarray
        Mean resultant vector length

    Examples
    --------
    >>> angles = np.array([0, 0.1, 0.2, -0.1, -0.2])  # Concentrated around 0
    >>> r = circ_r(angles)
    >>> print(f"MVL: {r:.3f}")  # Should be close to 1

    >>> angles = np.linspace(0, 2*np.pi, 100)  # Uniform distribution
    >>> r = circ_r(angles)
    >>> print(f"MVL: {r:.3f}")  # Should be close to 0

    Notes
    -----
    Based on CircStat toolbox circ_r.m by Philipp Berens (2009)
    """
    if w is None:
        w = np.ones_like(alpha)

    # Compute weighted sum of complex exponentials
    r = np.sum(w * np.exp(1j * alpha), axis=axis)

    # Obtain length normalized by sum of weights
    r = np.abs(r) / np.sum(w, axis=axis)

    # Apply correction factor for binned data if spacing is provided
    if d != 0:
        c = d / (2 * np.sin(d / 2))
        r = c * r

    return r


def circ_mean(alpha: np.ndarray, w: np.ndarray | None = None, axis: int = 0) -> float | np.ndarray:
    """
    Compute mean direction for circular data.

    Parameters
    ----------
    alpha : np.ndarray
        Sample of angles in radians
    w : np.ndarray, optional
        Weights for each angle (e.g., for binned data). If None, uniform weights assumed.
    axis : int, optional
        Compute along this dimension. Default is 0.

    Returns
    -------
    mu : float or np.ndarray
        Mean direction in radians, range [-π, π]

    Examples
    --------
    >>> angles = np.array([0, 0.1, 0.2, -0.1, -0.2])
    >>> mean_angle = circ_mean(angles)
    >>> print(f"Mean direction: {mean_angle:.3f} rad")

    >>> # Weighted mean
    >>> angles = np.array([0, np.pi])
    >>> weights = np.array([3, 1])  # 3x more weight on 0
    >>> mean_angle = circ_mean(angles, w=weights)

    Notes
    -----
    Based on CircStat toolbox circ_mean.m by Philipp Berens (2009)
    """
    if w is None:
        w = np.ones_like(alpha)
    else:
        if alpha.ndim > 0 and w.shape != alpha.shape:
            raise ValueError("Input dimensions do not match")

    # Compute weighted sum of complex exponentials
    r = np.sum(w * np.exp(1j * alpha), axis=axis)

    # Obtain mean angle from complex sum
    mu = np.angle(r)

    return mu


def circ_std(
    alpha: np.ndarray, w: np.ndarray | None = None, d: float = 0.0, axis: int = 0
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Compute circular standard deviation for circular data.

    Parameters
    ----------
    alpha : np.ndarray
        Sample of angles in radians
    w : np.ndarray, optional
        Weights for each angle. If None, uniform weights assumed.
    d : float, optional
        Spacing of bin centers for binned data (correction factor). Default is 0.
    axis : int, optional
        Compute along this dimension. Default is 0.

    Returns
    -------
    s : float or np.ndarray
        Angular deviation (equation 26.20, Zar)
    s0 : float or np.ndarray
        Circular standard deviation (equation 26.21, Zar)

    Examples
    --------
    >>> angles = np.array([0, 0.1, 0.2, -0.1, -0.2])
    >>> s, s0 = circ_std(angles)
    >>> print(f"Angular deviation: {s:.3f} rad")

    Notes
    -----
    Based on CircStat toolbox circ_std.m by Philipp Berens (2009)
    References: Biostatistical Analysis, J. H. Zar
    """
    if w is None:
        w = np.ones_like(alpha)
    else:
        if w.shape != alpha.shape:
            raise ValueError("Input dimensions do not match")

    # Compute mean resultant vector length
    r = circ_r(alpha, w=w, d=d, axis=axis)

    # Compute standard deviations (equations from Zar)
    s = np.sqrt(2 * (1 - r))  # 26.20 - angular deviation
    s0 = np.sqrt(-2 * np.log(r))  # 26.21 - circular standard deviation

    return s, s0


def circ_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pairwise angular distance between angles (x_i - y_i) around the circle.

    Computes the shortest signed angular distance from y to x, respecting
    circular topology (wrapping at ±π).

    Parameters
    ----------
    x : np.ndarray
        First set of angles in radians
    y : np.ndarray
        Second set of angles in radians (must be same shape as x, or scalar)

    Returns
    -------
    r : np.ndarray
        Angular distances in radians, range [-π, π]

    Examples
    --------
    >>> x = np.array([0.1, np.pi])
    >>> y = np.array([0.0, -np.pi])  # -π and π are same location
    >>> dist = circ_dist(x, y)
    >>> print(dist)  # [0.1, 0.0]

    >>> # Distance wraps around at ±π
    >>> x = np.array([np.pi - 0.1])
    >>> y = np.array([-np.pi + 0.1])
    >>> dist = circ_dist(x, y)
    >>> print(dist)  # Small value, not 2π - 0.2

    Notes
    -----
    Based on CircStat toolbox circ_dist.m by Philipp Berens (2009)
    References: Biostatistical Analysis, J. H. Zar, p. 651
    """
    # Compute angular difference using complex exponentials
    # This automatically wraps to [-π, π]
    r = np.angle(np.exp(1j * x) / np.exp(1j * y))

    return r


def circ_dist2(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """
    All pairwise angular distances (x_i - y_j) around the circle.

    Computes the matrix of all pairwise angular distances between two sets
    of angles, or within one set if y is not provided.

    Parameters
    ----------
    x : np.ndarray
        First set of angles in radians (will be treated as column vector)
    y : np.ndarray, optional
        Second set of angles in radians (will be treated as column vector).
        If None, computes pairwise distances within x. Default is None.

    Returns
    -------
    r : np.ndarray
        Matrix of pairwise angular distances, shape (len(x), len(y))
        Element (i, j) contains the distance from y[j] to x[i]

    Examples
    --------
    >>> x = np.array([0, np.pi/2, np.pi])
    >>> D = circ_dist2(x)  # All pairwise distances within x
    >>> print(D.shape)  # (3, 3)

    >>> y = np.array([0, np.pi])
    >>> D = circ_dist2(x, y)  # All distances from y to x
    >>> print(D.shape)  # (3, 2)

    Notes
    -----
    Based on CircStat toolbox circ_dist2.m by Philipp Berens (2009)
    """
    if y is None:
        y = x

    # Ensure column vectors
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.shape[1] > x.shape[0]:
        x = x.T

    if y.ndim == 1:
        y = y[:, np.newaxis]
    elif y.shape[1] > y.shape[0]:
        y = y.T

    # Compute all pairwise distances using broadcasting
    # Shape: (len(x), len(y))
    r = np.angle(np.exp(1j * x) / np.exp(1j * y.T))

    return r


def circ_rtest(alpha: np.ndarray, w: np.ndarray | None = None) -> float:
    """
    Rayleigh test for non-uniformity of circular data.

    H0: The population is uniformly distributed around the circle.
    HA: The population is not uniformly distributed.

    Parameters
    ----------
    alpha : np.ndarray
        Sample of angles in radians
    w : np.ndarray, optional
        Weights for each angle. If None, uniform weights assumed.

    Returns
    -------
    pval : float
        p-value of Rayleigh test. Small values (< 0.05) indicate
        significant deviation from uniformity.

    Examples
    --------
    >>> # Concentrated distribution
    >>> angles = np.random.normal(0, 0.1, 100)
    >>> p = circ_rtest(angles)
    >>> print(f"p-value: {p:.4f}")  # Should be < 0.05

    >>> # Uniform distribution
    >>> angles = np.random.uniform(-np.pi, np.pi, 100)
    >>> p = circ_rtest(angles)
    >>> print(f"p-value: {p:.4f}")  # Should be > 0.05

    Notes
    -----
    Test statistic: Z = n * r^2, where n is sample size and r is MVL
    Approximation for p-value: p ≈ exp(-Z) * (1 + (2*Z - Z^2)/(4*n))

    References: Topics in Circular Statistics, S.R. Jammalamadaka et al., p. 48
    """
    if w is None:
        w = np.ones_like(alpha)

    # Compute MVL
    r = circ_r(alpha, w=w)

    # Sample size
    n = len(alpha)

    # Rayleigh test statistic
    Z = n * r**2

    # Approximate p-value (good for n > 50, reasonable for n > 20)
    pval = np.exp(-Z) * (
        1 + (2 * Z - Z**2) / (4 * n) - (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2)
    )

    return pval


# Convenience aliases for compatibility
mvl = circ_r  # Mean Vector Length is same as resultant length
angular_mean = circ_mean
angular_std = circ_std
angular_distance = circ_dist


if __name__ == "__main__":
    # Simple tests
    print("Testing circular statistics functions...")

    # Test 1: Concentrated distribution
    angles = np.random.normal(0, 0.1, 100)
    r = circ_r(angles)
    mu = circ_mean(angles)
    s, s0 = circ_std(angles)
    p = circ_rtest(angles)

    print("\nConcentrated distribution (mean=0, std=0.1):")
    print(f"  MVL: {r:.3f} (should be close to 1)")
    print(f"  Mean direction: {mu:.3f} rad (should be close to 0)")
    print(f"  Angular deviation: {s:.3f} rad")
    print(f"  Rayleigh test p-value: {p:.4f} (should be < 0.05)")

    # Test 2: Uniform distribution
    angles = np.random.uniform(-np.pi, np.pi, 100)
    r = circ_r(angles)
    p = circ_rtest(angles)

    print("\nUniform distribution:")
    print(f"  MVL: {r:.3f} (should be close to 0)")
    print(f"  Rayleigh test p-value: {p:.4f} (should be > 0.05)")

    # Test 3: Angular distances
    x = np.array([0.1, np.pi])
    y = np.array([0.0, -np.pi])
    dist = circ_dist(x, y)

    print("\nAngular distances:")
    print(f"  dist([0.1, π], [0, -π]) = {dist}")
    print("  Note: π and -π are the same location, so distance is 0")

    # Test 4: Pairwise distances
    angles = np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    D = circ_dist2(angles)

    print(f"\nPairwise distance matrix shape: {D.shape}")
    print(f"Diagonal should be all zeros: {np.diag(D)}")

    print("\nAll tests completed!")
