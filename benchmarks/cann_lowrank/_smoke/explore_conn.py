"""Smoke test: explore conn_mat SVD spectrum for CANN1D and CANN2D.

Goal:
  - Verify CANN1D/CANN2D run in this brainpy env.
  - Extract conn_mat, compute SVD, see how fast the singular values decay.
    (Gaussian distance kernels are smooth functions of position, so SVD
    should decay exponentially — the precondition for low-rank being useful.)
  - Print a few numbers so we know what ranks are sane.
"""
import sys
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/Volumes/data-sch/projects/canns/src")

import numpy as np
import brainpy.math as bm
from canns.models.basic import CANN1D, CANN2D

bm.set_dt(0.1)

print("=" * 60)
print("CANN1D — SVD spectrum")
print("=" * 60)
for num in [64, 128, 256, 512, 1024]:
    m = CANN1D(num=num)
    conn = bm.as_numpy(m.conn_mat)
    s = np.linalg.svd(conn, compute_uv=False)
    # top-k cumulative energy
    total = (s ** 2).sum()
    cum = np.cumsum(s ** 2) / total
    print(
        f"  num={num:>4d} | rank needed for 99% energy: {int(np.searchsorted(cum, 0.99)) + 1:>3d}"
        f" | for 99.9%: {int(np.searchsorted(cum, 0.999)) + 1:>3d}"
        f" | top-1 rel: {s[0] / s.sum():.3f} | top-5 rel: {s[:5].sum() / s.sum():.3f}"
    )

print()
print("=" * 60)
print("CANN2D — SVD spectrum")
print("=" * 60)
for length in [8, 16, 32, 64]:
    m = CANN2D(length=length)
    conn = bm.as_numpy(m.conn_mat)
    s = np.linalg.svd(conn, compute_uv=False)
    total = (s ** 2).sum()
    cum = np.cumsum(s ** 2) / total
    print(
        f"  L={length:>3d} (n={length * length:>6d}) | rank 99%: {int(np.searchsorted(cum, 0.99)) + 1:>3d}"
        f" | 99.9%: {int(np.searchsorted(cum, 0.999)) + 1:>3d}"
        f" | top-1 rel: {s[0] / s.sum():.3f} | top-5 rel: {s[:5].sum() / s.sum():.3f}"
    )

print()
print("=" * 60)
print("One update step — both models")
print("=" * 60)
for num in [64, 256]:
    m = CANN1D(num=num)
    inp = m.get_stimulus_by_pos(0.0)
    m.update(inp)
    r = bm.as_numpy(m.r.value)
    u = bm.as_numpy(m.u.value)
    print(f"  CANN1D num={num}: r.max={r.max():.4f}  r.sum={r.sum():.4f}  u.max={u.max():.4f}")
for L in [16, 32]:
    m = CANN2D(length=L)
    inp = m.get_stimulus_by_pos([0.0, 0.0])
    m.update(inp)
    r = bm.as_numpy(m.r.value)
    u = bm.as_numpy(m.u.value)
    print(f"  CANN2D L={L}: r.max={r.max():.4f}  r.sum={r.sum():.4f}  u.max={u.max():.4f}")
