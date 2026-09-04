"""CANN2D acceleration modes (``accl_mode``).

The recurrent matvec ``Irec = r @ conn`` can be replaced by a
low-rank SVD factorisation (4 modes) or by an exact doubly-
circulant 2D FFT (the 5th mode). This example drives a ``CANN2D``
with ``SmoothTracking2D`` and prints the rank each mode picks,
then shows the FFT-mode setup on a clean grid.

For measured speedups (CPU + GPU) see
``benchmarks/canns-accl/``; the ``canns.models.basic.accel``
module docstring has the full decision table.
"""

import warnings

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
    cann(task.data[0])  # warmup so the backend is materialised

    # --- Low-rank modes: 4 presets + 'auto' --------------------------------
    print(f"{'mode':<12}{'k':>6}")
    for mode in ("normal", "fast", "ultra-fast"):
        cann.set_accl_mode(mode)
        print(f"{mode:<12}{cann.accl_k:>6}")

    # 'auto' picks the smallest k from the SVD spectrum to satisfy an
    # error budget; a tighter budget needs more components.
    for budget_mrad in (5.0, 1.0, 0.5):
        cann.set_accl_mode("auto", target_err_mrad=budget_mrad)
        print(f"{'auto':<12}{cann.accl_k:>6}   target={budget_mrad} mrad")

    # --- FFT mode: only on a clean endpoint=False grid ---------------------
    print()
    # 1) default grid -> falls back to dense with a warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cann_default = CANN2D(length=16, accl_mode="fft")
    print(f"  default grid: requested=fft, got={cann_default.accl_mode}  (fallback)")
    print(f"  warning: {str(caught[0].message)[:80]}...")

    # 2) clean grid -> 2D FFT path activated, exact to fp32.
    cann_clean = CANN2D(length=16, accl_mode="normal")
    cann_clean.x = bm.linspace(-bm.pi, bm.pi, 16, endpoint=False)
    cann_clean.conn_mat = cann_clean.make_conn()
    cann_clean.set_accl_mode("fft")
    r = bm.random.rand(16 * 16)
    dense = np.asarray(cann_clean.conn_mat @ r)
    fft_out = np.asarray(cann_clean.irec_backend(r))
    rel_err = np.abs(dense - fft_out).max() / np.abs(dense).max()
    print(f"  clean grid:   mode=fft, max|dense-fft|/|dense|={rel_err:.2e}")


if __name__ == "__main__":
    main()
