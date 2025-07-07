import brainunit as u

__all__ = ["map2pi"]


def map2pi(a):
    b = u.math.where(a > u.math.pi, a - u.math.pi * 2, a)
    c = u.math.where(b < -u.math.pi, b + u.math.pi * 2, b)
    return c
