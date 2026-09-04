"""Compare the four CANN2D acceleration modes on a moving stimulus.

The recurrent matvec ``Irec = r @ conn`` can be replaced by a
low-rank SVD factorisation. This example drives a ``CANN2D`` with
``SmoothTracking2D`` and prints the rank chosen for each mode.

For measured speedups (CPU + GPU) see
``benchmarks/canns-accl/``; the ``canns.models.basic.accel``
module docstring has the full decision table.
"""

import brainpy.math as bm
import numpy as np

from canns.models.basic import CANN2D
from canns.task.tracking import SmoothTracking2D

bm.set_dt(0.1)


def main():
    cann = CANN2D(length=32)
    task = SmoothTracking2D(
        cann_instance=cann,
        Iext=([0.0, 0.0], [np.pi / 2, np.pi / 2], [np.pi, np.pi],
               [3 * np.pi / 2, 3 * np.pi / 2]),
        duration=(10.0, 10.0, 10.0),
        time_step=bm.get_dt(),
    )
    task.get_data()
    cann(task.data[0])  # one warmup step so the backend is materialised

    print(f"{'mode':<12}{'k':>6}")
    for mode in ("normal", "fast", "ultra-fast"):
        cann.set_accl_mode(mode)
        print(f"{mode:<12}{cann.accl_k:>6}")

    for budget_mrad in (5.0, 1.0, 0.5):
        cann.set_accl_mode("auto", target_err_mrad=budget_mrad)
        print(f"{'auto':<12}{cann.accl_k:>6}   target={budget_mrad} mrad")


if __name__ == "__main__":
    main()
