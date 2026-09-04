"""CANN2D ``accl_mode="fft"``: exact doubly-circulant matvec on a clean torus.

Same setup caveat as :file:`cann1d_fft_mode.py`: the FFT path is
exact only on a uniform ``endpoint=False`` grid. The canns default
``endpoint=True`` grid is not doubly-circulant, so the mode falls
back to dense with a warning.

See :mod:`canns.models.basic.accel` for the underlying decision.
"""

import warnings

import brainpy.math as bm
import numpy as np

from canns.models.basic import CANN2D

bm.set_dt(0.1)


def main():
    # 1) Default grid (endpoint=True) -> falls back to dense.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cann = CANN2D(length=16, accl_mode="fft")
    print(f"  default grid: requested=fft, got={cann.accl_mode}  "
          f"(silent fallback)")
    if caught:
        print(f"  warning: {str(caught[0].message)[:80]}...")

    # 2) Clean grid (endpoint=False) -> FFT path exact.
    cann = CANN2D(length=16, accl_mode="normal")
    cann.x = bm.linspace(-bm.pi, bm.pi, 16, endpoint=False)
    cann.conn_mat = cann.make_conn()
    cann.set_accl_mode("fft")
    print(f"\n  clean grid: mode={cann.accl_mode}  "
          f"backend={type(cann.irec_backend).__name__}")

    # 3) Verify FFT matvec equals dense to fp32 precision.
    r = bm.random.rand(16 * 16)
    dense = np.asarray(cann.conn_mat @ r)
    fft_out = np.asarray(cann.irec_backend(r))
    rel_err = np.abs(dense - fft_out).max() / np.abs(dense).max()
    print(f"  max |dense - fft| / |dense| = {rel_err:.2e}  "
          f"({'exact' if rel_err < 1e-4 else 'NOT exact'})")


if __name__ == "__main__":
    main()
