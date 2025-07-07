import brainunit as u

__all__ = ["map2pi"]


def map2pi(a):
    """
    Maps an angle 'a' to the interval [-pi, pi] using the modulo operator.

    Args:
        a: The input angle in radians.

    Returns:
        The angle mapped to the interval [-pi, pi].
    """
    # Normalize to [0, 2*pi]
    b = u.math.fmod(a + u.math.pi, 2 * u.math.pi)
    # Map to [-pi, pi]
    c = b - u.math.pi
    return c
