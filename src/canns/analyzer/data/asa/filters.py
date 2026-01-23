from __future__ import annotations

import numbers

import numpy as np
from numpy.exceptions import AxisError
from scipy.ndimage import _nd_image, _ni_support
from scipy.ndimage._filters import _invalid_origin


def _gaussian_filter1d(
    input,
    sigma,
    axis=-1,
    order=0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
    *,
    radius=None,
):
    """1-D Gaussian filter.

    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode_reflect)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    radius : None or int, optional
        Radius of the Gaussian kernel. If specified, the size of
        the kernel will be ``2*radius + 1``, and `truncate` is ignored.
        Default is None.

    Returns
    -------
    gaussian_filter1d : ndarray

    Notes
    -----
    The Gaussian kernel will have size ``2*radius + 1`` along each axis. If
    `radius` is None, a default ``radius = round(truncate * sigma)`` will be
    used.

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter1d
    >>> import numpy as np
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
    >>> _gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(101).cumsum()
    >>> y3 = _gaussian_filter1d(x, 3)
    >>> y6 = _gaussian_filter1d(x, 6)
    >>> plt.plot(x, 'k', label='original data')
    >>> plt.plot(y3, '--', label='filtered, sigma=3')
    >>> plt.plot(y6, ':', label='filtered, sigma=6')
    >>> plt.legend()
    >>> plt.grid()
    >>> plt.show()

    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError("Radius must be a nonnegative integer.")
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return _correlate1d(input, weights, axis, output, mode, cval, 0)


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def _correlate1d(input, weights, axis=-1, output=None, mode="reflect", cval=0.0, origin=0):
    """Calculate a 1-D correlation along the given axis.

    The lines of the array along the given axis are correlated with the
    given weights.

    Parameters
    ----------
    %(input)s
    weights : array
        1-D sequence of numbers.
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s

    Returns
    -------
    result : ndarray
        Correlation result. Has the same shape as `input`.

    Examples
    --------
    >>> from scipy.ndimage import correlate1d
    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([ 8, 26,  8, 12,  7, 28, 36,  9])
    """
    input = np.asarray(input)
    weights = np.asarray(weights)
    complex_input = input.dtype.kind == "c"
    complex_weights = weights.dtype.kind == "c"
    if complex_input or complex_weights:
        if complex_weights:
            weights = weights.conj()
            weights = weights.astype(np.complex128, copy=False)
        kwargs = dict(axis=axis, mode=mode, origin=origin)
        output = _ni_support._get_output(output, input, complex_output=True)
        return _complex_via_real_components(_correlate1d, input, weights, output, cval, **kwargs)

    output = _ni_support._get_output(output, input)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError("no filter weights given")
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = _normalize_axis_index(axis, input.ndim)
    if _invalid_origin(origin, len(weights)):
        raise ValueError(
            "Invalid origin; origin must satisfy "
            "-(len(weights) // 2) <= origin <= "
            "(len(weights)-1) // 2"
        )
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate1d(input, weights, axis, output, mode, cval, origin)
    return output


def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    """Complex convolution via a linear combination of real convolutions."""
    complex_input = input.dtype.kind == "c"
    complex_weights = weights.dtype.kind == "c"
    if complex_input and complex_weights:
        # real component of the output
        func(input.real, weights.real, output=output.real, cval=np.real(cval), **kwargs)
        output.real -= func(input.imag, weights.imag, output=None, cval=np.imag(cval), **kwargs)
        # imaginary component of the output
        func(input.real, weights.imag, output=output.imag, cval=np.real(cval), **kwargs)
        output.imag += func(input.imag, weights.real, output=None, cval=np.imag(cval), **kwargs)
    elif complex_input:
        func(input.real, weights, output=output.real, cval=np.real(cval), **kwargs)
        func(input.imag, weights, output=output.imag, cval=np.imag(cval), **kwargs)
    else:
        if np.iscomplexobj(cval):
            raise ValueError("Cannot provide a complex-valued cval when the input is real.")
        func(input, weights.real, output=output.real, cval=cval, **kwargs)
        func(input, weights.imag, output=output.imag, cval=cval, **kwargs)
    return output


def _normalize_axis_index(axis, ndim):
    # Check if `axis` is in the correct range and normalize it
    if axis < -ndim or axis >= ndim:
        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        raise AxisError(msg)

    if axis < 0:
        axis = axis + ndim
    return axis
