"""FFT matvec benchmark for CANN1D and CANN2D.

Goal
----
Compare the per-step and per-rollout cost of the recurrent matvec under
three backends:

  1. ``dense``    : plain ``conn @ r``  (the baseline, O(n²))
  2. ``svd_k{k}`` : rank-k truncated SVD, ``U_l @ (V_l.T @ r)``  (O(nk))
  3. ``fft``      : exact circulant matvec, ``ifft(fft(c) ⊙ fft(r))``  (O(n log n))

The FFT path requires a *clean* circulant, which means the grid must be
``endpoint=False``. The canns default is ``endpoint=True`` (which makes
the matrix non-circulant under the canns wrap convention), so the FFT
numbers below come from a clean-grid model that is not the canns default.
We also report the FFT entry's accuracy against the dense baseline.

For each (model, n, backend) we measure:

  - per-step time: median wall time of a single matvec (ms)
  - scan T=200 time: median per-step wall time inside a ``lax.scan`` (ms)
  - accuracy: max |dense - backend| / max |dense|

This benchmark writes:

  - results/cann_fft_speed.csv         — per-step + scan times
  - results/cann_fft_accuracy.csv      — max-abs error vs dense
  - results/cann_fft_summary.md        — headline numbers
  - figures/*.png                       — per-(model, n) bar plots

Run:

    # CPU
    uv run python benchmarks/cann_fft/cann_fft_bench.py

    # GPU
    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 \\
        uv run python benchmarks/cann_fft/cann_fft_bench.py

    # Quick smoke
    uv run python benchmarks/cann_fft/cann_fft_bench.py --fast
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Backend is set by the user via env; we don't override it here so the
# same script can run on CPU and GPU. The README explains the env vars.

# Add canns source to path so this works without a build step.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))

import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np

from canns.models.basic import CANN1D, CANN2D  # noqa: E402

# ---------------------------------------------------------------------------
# Backend setup helpers
# ---------------------------------------------------------------------------


def make_clean_cann1d(num: int) -> CANN1D:
    """Build a CANN1D on a uniform endpoint=False grid (clean circulant)."""
    m = CANN1D(num=num, accl_mode="normal")
    m.x = bm.linspace(-bm.pi, bm.pi, num, endpoint=False)
    m.conn_mat = m.make_conn()
    return m


def make_clean_cann2d(length: int) -> CANN2D:
    m = CANN2D(length=length, accl_mode="normal")
    m.x = bm.linspace(-bm.pi, bm.pi, length, endpoint=False)
    m.y = bm.linspace(-bm.pi, bm.pi, length, endpoint=False)
    m.conn_mat = m.make_conn()
    return m


def lowrank_factors(conn: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    U, S, Vt = np.linalg.svd(conn, full_matrices=False)
    sqrtS = np.sqrt(S[:k].astype(np.float32))
    U_l = U[:, :k].astype(np.float32) * sqrtS
    V_l = Vt[:k, :].T.astype(np.float32) * sqrtS
    return U_l, V_l


# ---------------------------------------------------------------------------
# Pure step / scan functions
# ---------------------------------------------------------------------------


def make_dense_step(conn: jnp.ndarray):
    @jax.jit
    def step(r):
        return conn @ r

    return step


def make_lowrank_step(U_l: jnp.ndarray, V_l: jnp.ndarray):
    @jax.jit
    def step(r):
        return U_l @ (V_l.T @ r)

    return step


def make_fft1d_step(K_fft: jnp.ndarray):
    @jax.jit
    def step(r):
        return jnp.real(jnp.fft.ifft(K_fft * jnp.fft.fft(r)))

    return step


def make_fft2d_step(K_fft2: jnp.ndarray, L: int):
    @jax.jit
    def step(r):
        r_2d = r.reshape(L, L)
        out = jnp.real(jnp.fft.ifft2(K_fft2 * jnp.fft.fft2(r_2d)))
        return out.ravel()

    return step


def make_scan_dense(conn: jnp.ndarray, T: int):
    def body(carry, _):
        return conn @ carry, None

    @jax.jit
    def run(r0):
        return jax.lax.scan(body, r0, None, length=T)[0]

    return run, T


def make_scan_lowrank(U_l, V_l, T):
    def body(carry, _):
        return U_l @ (V_l.T @ carry), None

    @jax.jit
    def run(r0):
        return jax.lax.scan(body, r0, None, length=T)[0]

    return run, T


def make_scan_fft1d(K_fft, T):
    def body(carry, _):
        return jnp.real(jnp.fft.ifft(K_fft * jnp.fft.fft(carry))), None

    @jax.jit
    def run(r0):
        return jax.lax.scan(body, r0, None, length=T)[0]

    return run, T


def make_scan_fft2d(K_fft2, L, T):
    def body(carry, _):
        r2 = carry.reshape(L, L)
        out = jnp.real(jnp.fft.ifft2(K_fft2 * jnp.fft.fft2(r2)))
        return out.ravel(), None

    @jax.jit
    def run(r0):
        return jax.lax.scan(body, r0, None, length=T)[0]

    return run, T


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def median_ms(fn, args, n_warmup: int, n_iters: int) -> float:
    out = fn(*args)
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    for _ in range(n_warmup):
        out = fn(*args)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = fn(*args)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def per_step_ms_from_scan(scan_fn, r0, T_scan, n_warmup, n_iters) -> float:
    out = scan_fn(r0)
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    for _ in range(n_warmup):
        out = scan_fn(r0)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = scan_fn(r0)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / T_scan)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------


@dataclass
class FFTBenchRow:
    model: str  # "cann1d" / "cann2d"
    n_param: int  # 1D num, or 2D L (we also store n = L² for 2D)
    n_total: int  # 1D num, or 2D L*L
    backend: str  # "dense" / "svd_k1" / "fft"
    per_step_ms: float  # median per-step wall time
    scan_per_step_ms: float  # median per-step inside T=200 scan
    max_abs_err: float  # vs dense (zero for dense)


# ---------------------------------------------------------------------------
# Per-size benchmarks
# ---------------------------------------------------------------------------


def run_1d(
    num: int, ranks: list[int], T_scan: int, n_warmup: int, n_iters: int
) -> list[FFTBenchRow]:
    print(f"  CANN1D num={num}")
    bm.random.seed(0)
    m = make_clean_cann1d(num)
    conn = jnp.asarray(np.asarray(m.conn_mat, dtype=np.float32))
    r0 = jax.random.normal(jax.random.PRNGKey(0), (num,), dtype=jnp.float32)

    # Dense baseline
    step_d = make_dense_step(conn)
    scan_d, _ = make_scan_dense(conn, T_scan)
    t_step_d = median_ms(step_d, (r0,), n_warmup, n_iters)
    t_scan_d = per_step_ms_from_scan(scan_d, r0, T_scan, n_warmup, n_iters)
    out_dense = np.asarray(step_d(r0))

    rows = [FFTBenchRow("cann1d", num, num, "dense", t_step_d, t_scan_d, 0.0)]

    # SVD ranks
    for k in ranks:
        U_l, V_l = lowrank_factors(np.asarray(m.conn_mat, dtype=np.float32), k)
        U_l_j, V_l_j = jnp.asarray(U_l), jnp.asarray(V_l)
        step_l = make_lowrank_step(U_l_j, V_l_j)
        scan_l, _ = make_scan_lowrank(U_l_j, V_l_j, T_scan)
        t_step = median_ms(step_l, (r0,), n_warmup, n_iters)
        t_scan = per_step_ms_from_scan(scan_l, r0, T_scan, n_warmup, n_iters)
        out = np.asarray(step_l(r0))
        err = float(np.max(np.abs(out - out_dense)))
        rows.append(FFTBenchRow("cann1d", num, num, f"svd_k{k}", t_step, t_scan, err))

    # FFT (clean circulant — for endpoint=False grid)
    first_row = np.asarray(m.conn_mat[0, :], dtype=np.float32)
    K_fft = jnp.fft.fft(jnp.asarray(first_row))
    step_f = make_fft1d_step(K_fft)
    scan_f, _ = make_scan_fft1d(K_fft, T_scan)
    t_step = median_ms(step_f, (r0,), n_warmup, n_iters)
    t_scan = per_step_ms_from_scan(scan_f, r0, T_scan, n_warmup, n_iters)
    out = np.asarray(step_f(r0))
    err = float(np.max(np.abs(out - out_dense)))
    rows.append(FFTBenchRow("cann1d", num, num, "fft", t_step, t_scan, err))

    return rows


def run_2d(
    length: int, ranks: list[int], T_scan: int, n_warmup: int, n_iters: int
) -> list[FFTBenchRow]:
    L = length
    n = L * L
    print(f"  CANN2D L={L} (n={n})")
    bm.random.seed(0)
    m = make_clean_cann2d(L)
    conn = jnp.asarray(np.asarray(m.conn_mat, dtype=np.float32))
    r0 = jax.random.normal(jax.random.PRNGKey(0), (n,), dtype=jnp.float32)

    step_d = make_dense_step(conn)
    scan_d, _ = make_scan_dense(conn, T_scan)
    t_step_d = median_ms(step_d, (r0,), n_warmup, n_iters)
    t_scan_d = per_step_ms_from_scan(scan_d, r0, T_scan, n_warmup, n_iters)
    out_dense = np.asarray(step_d(r0))

    rows = [FFTBenchRow("cann2d", L, n, "dense", t_step_d, t_scan_d, 0.0)]

    for k in ranks:
        U_l, V_l = lowrank_factors(np.asarray(m.conn_mat, dtype=np.float32), k)
        U_l_j, V_l_j = jnp.asarray(U_l), jnp.asarray(V_l)
        step_l = make_lowrank_step(U_l_j, V_l_j)
        scan_l, _ = make_scan_lowrank(U_l_j, V_l_j, T_scan)
        t_step = median_ms(step_l, (r0,), n_warmup, n_iters)
        t_scan = per_step_ms_from_scan(scan_l, r0, T_scan, n_warmup, n_iters)
        out = np.asarray(step_l(r0))
        err = float(np.max(np.abs(out - out_dense)))
        rows.append(FFTBenchRow("cann2d", L, n, f"svd_k{k}", t_step, t_scan, err))

    first_row = np.asarray(m.conn_mat[0, :], dtype=np.float32).reshape(L, L)
    K_fft2 = jnp.fft.fft2(jnp.asarray(first_row))
    step_f = make_fft2d_step(K_fft2, L)
    scan_f, _ = make_scan_fft2d(K_fft2, L, T_scan)
    t_step = median_ms(step_f, (r0,), n_warmup, n_iters)
    t_scan = per_step_ms_from_scan(scan_f, r0, T_scan, n_warmup, n_iters)
    out = np.asarray(step_f(r0))
    err = float(np.max(np.abs(out - out_dense)))
    rows.append(FFTBenchRow("cann2d", L, n, "fft", t_step, t_scan, err))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="smaller sweep (smoke test)")
    p.add_argument(
        "--out", default=str(_HERE / "results"), help="output directory for CSVs / figures"
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--T-scan", type=int, default=200, help="scan length for scan-time benchmarks")
    p.add_argument(
        "--cpu-only-1d-max", type=int, default=4096, help="largest 1D num to benchmark on CPU"
    )
    p.add_argument(
        "--cpu-only-2d-max", type=int, default=64, help="largest 2D L to benchmark on CPU"
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_gpu = "cuda" in (os.environ.get("JAX_PLATFORMS", "") or "") or any(
        p.platform == "gpu" for p in jax.devices()
    )
    print(f"# Platform: {jax.devices()}")
    print(f"# is_gpu: {is_gpu}")

    if args.fast:
        sizes_1d = [64, 256, 1024]
        sizes_2d = [8, 16, 32]
        ranks = [1, 4, 16]
    else:
        if is_gpu:
            # Note: n > 4096 makes the CPU SVD (used to factorise the
            # baseline conn) very slow (> 10 min per rank), which
            # dominates the wall time. We cap at 4096 (1D) and 64 (2D;
            # L=64 means n=4096; L=96 is n=9216 which is impractical
            # with the CPU SVD baseline).
            sizes_1d = [64, 128, 256, 512, 1024, 2048, 4096]
            sizes_2d = [4, 8, 16, 32, 48, 64]
        else:
            sizes_1d = [
                n for n in [64, 128, 256, 512, 1024, 2048, 4096] if n <= args.cpu_only_1d_max
            ]
            sizes_2d = [L for L in [4, 8, 16, 32, 48, 64] if L <= args.cpu_only_2d_max]
        ranks = [1, 4, 16, 64]

    all_rows: list[FFTBenchRow] = []
    print("\n# 1D sweep")
    for n in sizes_1d:
        all_rows.extend(run_1d(n, ranks, args.T_scan, args.warmup, args.iters))

    print("\n# 2D sweep")
    for L in sizes_2d:
        all_rows.extend(run_2d(L, ranks, args.T_scan, args.warmup, args.iters))

    # Write CSVs
    speed_csv = out_dir / "cann_fft_speed.csv"
    acc_csv = out_dir / "cann_fft_accuracy.csv"
    with open(speed_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "n_param",
                "n_total",
                "backend",
                "per_step_ms",
                "scan_per_step_ms",
                "speedup_vs_dense_step",
                "speedup_vs_dense_scan",
                "platform",
            ]
        )
        # group by (model, n_param) for speedup calc
        by_key: dict[tuple[str, int], dict[str, FFTBenchRow]] = {}
        for r in all_rows:
            by_key.setdefault((r.model, r.n_param), {})[r.backend] = r
        for (_model, _n_param), mp in by_key.items():
            base = mp.get("dense")
            base_step = base.per_step_ms if base else float("nan")
            base_scan = base.scan_per_step_ms if base else float("nan")
            for r in mp.values():
                step_su = (
                    base_step / r.per_step_ms if (base and r.per_step_ms > 0) else float("nan")
                )
                scan_su = (
                    base_scan / r.scan_per_step_ms
                    if (base and r.scan_per_step_ms > 0)
                    else float("nan")
                )
                w.writerow(
                    [
                        r.model,
                        r.n_param,
                        r.n_total,
                        r.backend,
                        f"{r.per_step_ms:.4f}",
                        f"{r.scan_per_step_ms:.4f}",
                        f"{step_su:.2f}",
                        f"{scan_su:.2f}",
                        "gpu" if is_gpu else "cpu",
                    ]
                )
    with open(acc_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_param", "n_total", "backend", "max_abs_err"])
        for r in all_rows:
            w.writerow([r.model, r.n_param, r.n_total, r.backend, f"{r.max_abs_err:.3e}"])
    print(f"\n# wrote {speed_csv}")
    print(f"# wrote {acc_csv}")


if __name__ == "__main__":
    main()
