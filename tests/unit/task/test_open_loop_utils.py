from __future__ import annotations

import numpy as np
import brainpy.math as bm

from canns.task.open_loop_navigation import map2pi


def test_map2pi_wraps_range():
    angles = bm.array([3.5, -4.0, 0.1])
    wrapped = map2pi(angles)
    wrapped_np = np.asarray(wrapped)
    assert (wrapped_np <= np.pi).all()
    assert (wrapped_np >= -np.pi).all()
