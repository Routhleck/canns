"""Compare the three acceleration modes on CANN1D.

The recurrent connectivity of CANN1D is a Gaussian distance kernel. Its
singular values decay extremely fast: the top-8 components already
capture 99.4% of the energy, so the matvec

    Irec = conn @ r

can be replaced by a low-rank factorisation

    Irec = U_l @ (V_l.T @ r)

where ``U_l`` and ``V_l`` are ``(n, k)`` matrices. This gives 30-245x
matvec speedup at ``num >= 512`` (CPU) with bump-position error below
5 mrad.

``canns.models.basic.CANN1D`` and ``CANN2D`` expose this through the
``accl_mode`` and ``accl_k`` constructor arguments. Three presets are
defined:

  - ``"normal"``     — full rank (default)
  - ``"fast"``       — low-rank, rank-8 for CANN1D / rank-32 for CANN2D
  - ``"ultra-fast"`` — low-rank, rank-1 for CANN1D / rank-4 for CANN2D

Or pass an explicit ``accl_k`` to override the mode's default.
"""
import time

import brainpy.math as bm
import jax
import numpy as np

from canns.models.basic import CANN1D, CANN2D

bm.set_dt(0.1)


def run(model, T: int) -> float:
    """Run ``T`` update steps under a moving Gaussian stimulus.

    Dispatches on the model's spatial dimensionality: 1D models get a
    sweeping Gaussian along the ring; 2D models get a sweeping Gaussian
    along the diagonal of the toroidal grid.
    """
    x = np.asarray(model.x)
    z_range = float(model.z_range)
    is_2d = hasattr(model, "length")
    if is_2d:
        L = int(model.length)
        xx, yy = np.meshgrid(x, x)
    for t in range(T):
        pos = np.pi * t / max(T - 1, 1)
        if is_2d:
            dx = (xx - pos) % z_range
            dy = (yy - pos) % z_range
            dx = np.where(dx > z_range / 2, dx - z_range, dx)
            dy = np.where(dy > z_range / 2, dy - z_range, dy)
            dist = np.sqrt(dx * dx + dy * dy)
            inp = (model.A * np.exp(-0.25 * (dist / model.a) ** 2)).astype(np.float32)
        else:
            d = (x - pos) % z_range
            d = np.where(d > z_range / 2, d - z_range, d)
            inp = (model.A * np.exp(-0.25 * (d / model.a) ** 2)).astype(np.float32)
        model.update(inp)
    return float(model.r.value.max())


def time_model(build_fn, T: int) -> tuple[float, float, float]:
    """Return (r_max after T steps, median step time, is_accelerated)."""
    model = build_fn()
    # Warmup (JIT compile).
    run(model, T=20)
    # Time
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        run(model, T=T)
        jax.block_until_ready(model.r.value)
        times.append((time.perf_counter() - t0) * 1000.0 / T)
    times.sort()
    return float(model.r.value.max()), times[len(times) // 2], model.is_accelerated


if __name__ == "__main__":
    print("CANN1D num=2048  — accl_mode comparison")
    print("-" * 60)
    T = 200
    for label, build in [
        ("normal         ", lambda: CANN1D(num=2048, accl_mode="normal")),
        ("fast   (k=8)   ", lambda: CANN1D(num=2048, accl_mode="fast")),
        ("fast   (k=4)   ", lambda: CANN1D(num=2048, accl_mode="fast", accl_k=4)),
        ("ultra-fast (k=1)", lambda: CANN1D(num=2048, accl_mode="ultra-fast")),
    ]:
        rmax, ms, accel = time_model(build, T=T)
        print(f"  {label}  r.max={rmax:.4f}  step={ms:7.3f} ms  accel={accel}")

    print()
    print("CANN2D length=64 (n=4096)  — accl_mode comparison")
    print("-" * 60)
    for label, build in [
        ("normal          ", lambda: CANN2D(length=64, accl_mode="normal")),
        ("fast   (k=32)   ", lambda: CANN2D(length=64, accl_mode="fast")),
        ("fast   (k=8)    ", lambda: CANN2D(length=64, accl_mode="fast", accl_k=8)),
        ("ultra-fast (k=4)", lambda: CANN2D(length=64, accl_mode="ultra-fast")),
    ]:
        rmax, ms, accel = time_model(build, T=T)
        print(f"  {label}  r.max={rmax:.4f}  step={ms:7.3f} ms  accel={accel}")

    print()
    print("Runtime mode-switching via set_accl_mode:")
    print("-" * 60)
    model = CANN1D(num=1024)
    print(f"  before: mode={model.accl_mode}, k={model.accl_k}, "
          f"is_accelerated={model.is_accelerated}")
    model.set_accl_mode("fast")
    print(f"  set_accl_mode('fast'): mode={model.accl_mode}, "
          f"k={model.accl_k}, is_accelerated={model.is_accelerated}")
    model.set_accl_mode("fast", k=16)
    print(f"  set_accl_mode('fast', k=16): mode={model.accl_mode}, "
          f"k={model.accl_k}, is_accelerated={model.is_accelerated}")
    model.set_accl_mode("normal")
    print(f"  set_accl_mode('normal'): mode={model.accl_mode}, "
          f"k={model.accl_k}, is_accelerated={model.is_accelerated}")
