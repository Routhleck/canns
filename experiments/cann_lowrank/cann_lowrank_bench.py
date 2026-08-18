"""Low-rank recurrent matrix benchmark for CANN1D and CANN2D.

Goal
----
In the canns package, both CANN1D and CANN2D use a Gaussian distance kernel
as the recurrent connectivity matrix `conn_mat`. At every step the model
performs a (typically large) matvec

    Irec = conn_mat @ r           # CANN1D
    Irec = r.flatten() @ conn_mat # CANN2D

This is the dominant per-step cost at large n. The kernel is smooth in
the feature space, so its SVD decays very fast: only a handful of
singular values are needed to capture 99%+ of the energy (see
_smoke/explore_conn.py).

This benchmark answers two questions for each (model, n):

  1. **Speed.** How much faster is `Irec = U_k @ (V_k.T @ r)` vs
     `Irec = conn @ r` when only k singular components are kept?
  2. **Dynamics preservation.** How well does a low-rank model
     reproduce the full-rank dynamics under a moving stimulus?

The benchmark writes:
  - results/cann_lowrank_speed.csv  — per-step timing in milliseconds
  - results/cann_lowrank_accuracy.csv — per-rank dynamics metrics
  - results/cann_lowrank_summary.md  — writeup of the headline numbers

Run:
    uv run python experiments/cann_lowrank/cann_lowrank_bench.py            # full sweep
    uv run python experiments/cann_lowrank/cann_lowrank_bench.py --fast     # smaller sweep
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Force CPU so the benchmark numbers are reproducible across machines.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Add canns source to path so this works without a build step.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import jax
import jax.numpy as jnp
import brainpy.math as bm

from canns.models.basic import CANN1D, CANN2D  # noqa: E402

# ---------------------------------------------------------------------------
# SVD-based low-rank factorization
# ---------------------------------------------------------------------------

def lowrank_factor(conn: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated SVD: conn ≈ U_k diag(S_k) V_k.T.

    Returns three numpy arrays:
        U : (n, k)
        S : (k,)
        V : (n, k)
    such that U @ diag(S) @ V.T ≈ conn (Frobenius-optimal for rank k).
    """
    U, S, Vt = np.linalg.svd(conn, full_matrices=False)
    U = U[:, :k].copy()
    S = S[:k].copy()
    V = Vt[:k].T.copy()
    return U, S, V


def lowrank_mats(U: np.ndarray, S: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (U_l, V_l) such that U_l @ V_l.T = U diag(S) V.T.

    For each rank-k factor we precompute these once. The forward
    matvec then becomes Irec = U_l @ (V_l.T @ r), i.e. two GEMV calls
    against (n, k) matrices, total 2*n*k FLOPs.
    """
    sqrtS = np.sqrt(S)
    U_l = U * sqrtS  # (n, k)
    V_l = V * sqrtS  # (n, k)
    return U_l, V_l


# ---------------------------------------------------------------------------
# Pure (JIT-friendly) step functions
# ---------------------------------------------------------------------------
# Both CANN1D and CANN2D share the same scalar recurrent dynamics:
#
#     r_t   = u_t**2 / (1 + k * sum(u_t**2))
#     Irec  = conn @ r_t            (or U_k @ V_k.T @ r_t under low-rank)
#     u_t+1 = u_t + dt/tau * (-u_t + Irec + inp)
#
# So we factor the recurrent matvec out, which is the only thing that
# changes between dense and low-rank variants. The divisive norm and
# Euler update are kept identical so the speedup is attributable
# purely to the recurrent matvec.
# ---------------------------------------------------------------------------

def _make_dense_step() -> Callable:
    """Dense step: Irec = conn @ r. Operates on flat (n,) state.

    For CANN1D, u, r, inp are already shape (num,).
    For CANN2D, the caller flattens (L, L) -> (L*L,) and reshapes the
    output back. The recurrent matvec is the same shape either way:
    weight is (n, n), r is (n,), Irec is (n,).
    """
    @jax.jit
    def step_dense(u, r_in, conn, inp, tau, k, dt):
        r1 = jnp.square(u)
        r2 = 1.0 + k * jnp.sum(r1)
        r = r1 / r2
        Irec = conn @ r
        u_new = u + (dt / tau) * (-u + Irec + inp)
        return u_new, r
    return step_dense


def _make_lowrank_step() -> Callable:
    """Low-rank step: Irec = U_l @ (V_l.T @ r). Operates on flat (n,) state."""
    @jax.jit
    def step_lowrank(u, r_in, U_l, V_l, inp, tau, k, dt):
        r1 = jnp.square(u)
        r2 = 1.0 + k * jnp.sum(r1)
        r = r1 / r2
        # Two small GEMVs: V_l.T @ r is (k,), U_l @ that is (n,).
        Irec = U_l @ (V_l.T @ r)
        u_new = u + (dt / tau) * (-u + Irec + inp)
        return u_new, r
    return step_lowrank


# ---------------------------------------------------------------------------
# Stimulus sequences
# ---------------------------------------------------------------------------

def make_moving_stimulus_1d(
    num: int, T: int, x: np.ndarray, a: float, A: float, z_range: float
) -> np.ndarray:
    """Slow sweep across half the ring: pos(t) = pi * t / T_max.

    Returns array of shape (T, num). The CANN bump should track this
    slowly moving Gaussian.
    """
    out = np.empty((T, num), dtype=np.float32)
    for t in range(T):
        pos = math.pi * t / max(T - 1, 1)
        d = (x - pos) % z_range
        d = np.where(d > z_range / 2, d - z_range, d)
        out[t] = A * np.exp(-0.25 * (d / a) ** 2)
    return out


def make_moving_stimulus_2d(
    length: int, T: int, x: np.ndarray, a: float, A: float, z_range: float
) -> np.ndarray:
    """Slow 2D sweep along the diagonal: pos(t) = pi * t/T_max in each dim.

    Returns array of shape (T, length, length).
    """
    L = length
    xx, yy = np.meshgrid(x, x)
    out = np.empty((T, L, L), dtype=np.float32)
    for t in range(T):
        pos = math.pi * t / max(T - 1, 1)
        dx = (xx - pos) % z_range
        dy = (yy - pos) % z_range
        dx = np.where(dx > z_range / 2, dx - z_range, dx)
        dy = np.where(dy > z_range / 2, dy - z_range, dy)
        out[t] = A * np.exp(-0.25 * ((dx ** 2 + dy ** 2) ** 0.5) / a) ** 2
    return out


# ---------------------------------------------------------------------------
# Bump diagnostics
# ---------------------------------------------------------------------------

def bump_position_1d(r: np.ndarray, x: np.ndarray, z_range: float) -> float:
    """Circular-mean of r — robust to noisy bump shapes."""
    weights = np.maximum(r, 0)
    if weights.sum() < 1e-12:
        return float("nan")
    angles = np.exp(1j * x)
    return float(np.angle(np.sum(weights * angles)))


def bump_fwhm_1d(r: np.ndarray, x: np.ndarray) -> float:
    """Approximate FWHM of the bump, measured in arc-length on the ring."""
    rmax = r.max()
    if rmax < 1e-12:
        return float("nan")
    half = rmax / 2
    # count cells above half-max, weighted by dx
    above = r > half
    n_above = int(above.sum())
    dx = float(x[1] - x[0])
    return n_above * dx


def bump_position_2d(r: np.ndarray, x: np.ndarray, z_range: float) -> tuple[float, float]:
    """Circular-mean position in 2D."""
    L = r.shape[0]
    xx, yy = np.meshgrid(x, x)
    weights = np.maximum(r, 0)
    if weights.sum() < 1e-12:
        return float("nan"), float("nan")
    cx = float(np.angle(np.sum(weights * np.exp(1j * xx))))
    cy = float(np.angle(np.sum(weights * np.exp(1j * yy))))
    return cx, cy


# ---------------------------------------------------------------------------
# Per-cell measurement
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    model: str           # "CANN1D" or "CANN2D"
    n: int               # num (1D) or length (2D)
    n_neurons: int       # n (1D) or L*L (2D)
    k: int               # rank ("full" recorded as -1)
    is_lowrank: bool
    # Speed — full update step (includes divisive norm + Euler)
    per_step_ms: float   # median per-step wall time
    speedup_vs_dense: float  # t_dense / t_this (1.0 for dense)
    # Speed — recurrent matvec ONLY (lax.scan of T=200 matvecs)
    matvec_per_step_ms: float
    matvec_speedup: float
    # Accuracy
    max_abs_err_r_max: float
    mean_abs_err_r_max: float
    max_pos_err: float
    mean_pos_err: float
    rmse_r: float
    # Spectral
    relerr_conn: float   # ||conn - U_k V_k.T||_F / ||conn||_F
    captured_energy: float  # sum(S[:k]^2) / sum(S^2)


def measure_step_time(step_fn, args, n_warmup: int, n_iters: int) -> float:
    """Median per-call wall time in milliseconds, with JIT warmup."""
    # Warmup
    out = step_fn(*args)
    if hasattr(out, "block_until_ready"):
        out[0].block_until_ready()
    for _ in range(n_warmup):
        out = step_fn(*args)
        if hasattr(out, "block_until_ready"):
            out[0].block_until_ready()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = step_fn(*args)
        if hasattr(out, "block_until_ready"):
            out[0].block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def make_matvec_scan_dense(conn: jnp.ndarray):
    """JIT'd lax.scan over a body that ONLY does the recurrent matvec.

    This isolates the algorithmic cost of `Irec = conn @ r` from any other
    step work. The body is the same one-liner repeated `T_scan` times,
    which lets XLA fuse everything. We pass `r` through the carry and
    return it, so the work is real, not DCE'd.
    """
    T_scan = 200

    def scan_body(carry_r, _):
        new_r = conn @ carry_r
        return new_r, None

    @jax.jit
    def run(r0):
        final_r, _ = jax.lax.scan(scan_body, r0, None, length=T_scan)
        return final_r

    return run, T_scan


def make_matvec_scan_lowrank(U_l: jnp.ndarray, V_l: jnp.ndarray):
    """JIT'd lax.scan over a body that ONLY does the low-rank matvec."""
    T_scan = 200

    def scan_body(carry_r, _):
        new_r = U_l @ (V_l.T @ carry_r)
        return new_r, None

    @jax.jit
    def run(r0):
        final_r, _ = jax.lax.scan(scan_body, r0, None, length=T_scan)
        return final_r

    return run, T_scan


def measure_matvec_scan_time(scan_fn, r0: jnp.ndarray, n_warmup: int, n_iters: int, T_scan: int) -> float:
    """Time a scan-loop that ONLY does the matvec. Returns per-step ms."""
    # Warmup
    out = scan_fn(r0)
    out.block_until_ready()
    for _ in range(n_warmup):
        out = scan_fn(r0)
        out.block_until_ready()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = scan_fn(r0)
        out.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / T_scan)  # per-step ms
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# CANN1D benchmark cell
# ---------------------------------------------------------------------------

def benchmark_cann1d(num: int, ranks: list[int], T: int, dt: float) -> list[CellResult]:
    bm.set_dt(dt)
    model = CANN1D(num=num)
    conn_np = np.asarray(model.conn_mat)
    x_np = np.asarray(model.x)
    tau = float(model.tau)
    k_global = float(model.k)
    a = float(model.a)
    A = float(model.A)
    z_range = float(model.z_range)
    n_neurons = num

    # Build stimulus (slow sweep)
    stim_np = make_moving_stimulus_1d(num, T, x_np, a, A, z_range)

    # Build JIT'd step functions
    step_dense = _make_dense_step()
    step_low = _make_lowrank_step()

    # Pre-compute low-rank factors (numpy)
    U_full, S_full, V_full = np.linalg.svd(conn_np, full_matrices=False)
    total_energy = float((S_full ** 2).sum())

    # Allocate state — start from a small bump so the first few steps are
    # already on a meaningful trajectory (avoids a "warmup" phase tainting
    # the per-step timing).
    rng = np.random.default_rng(0)
    u0 = np.zeros(num, dtype=np.float32)
    r0 = np.zeros(num, dtype=np.float32)

    # Run dense reference trajectory for accuracy comparison
    u = u0.copy()
    r = r0.copy()
    dense_u = np.empty((T, num), dtype=np.float32)
    dense_r = np.empty((T, num), dtype=np.float32)
    conn_jax = jnp.asarray(conn_np)
    for t in range(T):
        inp = jnp.asarray(stim_np[t])
        u, r = step_dense(u, r, conn_jax, inp, tau, k_global, dt)
        dense_u[t] = np.asarray(u)
        dense_r[t] = np.asarray(r)

    # Reference bump diagnostics
    dense_r_max = dense_r.max(axis=1)
    dense_pos = np.array([bump_position_1d(dense_r[t], x_np, z_range) for t in range(T)])
    dense_fwhm = np.array([bump_fwhm_1d(dense_r[t], x_np) for t in range(T)])

    # Speed-test inputs (use the steady-state-like state from the dense run)
    test_u = jnp.asarray(dense_u[T // 2])
    test_r = jnp.asarray(dense_r[T // 2])
    test_inp = jnp.asarray(stim_np[T // 2])

    results: list[CellResult] = []

    # Time dense (full step)
    t_dense = measure_step_time(
        step_dense,
        (test_u, test_r, conn_jax, test_inp, tau, k_global, dt),
        n_warmup=5,
        n_iters=50,
    )

    # Time dense (matvec-only via lax.scan)
    scan_dense, T_scan = make_matvec_scan_dense(conn_jax)
    # Build a representative r for the scan: bump-shaped
    r_scan = jnp.asarray(np.maximum(dense_r[T // 2], 0.0))
    t_dense_matvec = measure_matvec_scan_time(
        scan_dense, r_scan, n_warmup=3, n_iters=20, T_scan=T_scan
    )

    # Dense result row
    # relerr_conn = 0 (full rank), captured_energy = 1.0
    results.append(
        CellResult(
            model="CANN1D",
            n=num,
            n_neurons=n_neurons,
            k=-1,
            is_lowrank=False,
            per_step_ms=t_dense,
            speedup_vs_dense=1.0,
            matvec_per_step_ms=t_dense_matvec,
            matvec_speedup=1.0,
            max_abs_err_r_max=0.0,
            mean_abs_err_r_max=0.0,
            max_pos_err=0.0,
            mean_pos_err=0.0,
            rmse_r=0.0,
            relerr_conn=0.0,
            captured_energy=1.0,
        )
    )

    # Per-rank lowrank rows
    for k_rank in ranks:
        U, S, V = U_full[:, :k_rank], S_full[:k_rank], V_full[:, :k_rank]
        U_l, V_l = lowrank_mats(U, S, V)
        Ul_jax = jnp.asarray(U_l)
        Vl_jax = jnp.asarray(V_l)

        # Spectral error of the approximation
        approx = (U_l @ V_l.T)
        relerr = float(np.linalg.norm(conn_np - approx) / np.linalg.norm(conn_np))
        captured = float((S ** 2).sum() / total_energy)

        # Speed — full step
        t_low = measure_step_time(
            step_low,
            (test_u, test_r, Ul_jax, Vl_jax, test_inp, tau, k_global, dt),
            n_warmup=5,
            n_iters=50,
        )
        speedup = t_dense / t_low if t_low > 0 else float("inf")

        # Speed — matvec-only via lax.scan
        scan_low, _ = make_matvec_scan_lowrank(Ul_jax, Vl_jax)
        t_low_matvec = measure_matvec_scan_time(
            scan_low, r_scan, n_warmup=3, n_iters=20, T_scan=T_scan
        )
        matvec_speedup = t_dense_matvec / t_low_matvec if t_low_matvec > 0 else float("inf")

        # Accuracy: simulate with low-rank and compare to dense
        u_l = u0.copy()
        r_l = r0.copy()
        low_r = np.empty((T, num), dtype=np.float32)
        for t in range(T):
            inp = jnp.asarray(stim_np[t])
            u_l, r_l = step_low(u_l, r_l, Ul_jax, Vl_jax, inp, tau, k_global, dt)
            low_r[t] = np.asarray(r_l)

        low_r_max = low_r.max(axis=1)
        low_pos = np.array([bump_position_1d(low_r[t], x_np, z_range) for t in range(T)])

        # Wrap pos error on the ring (minimum arc length)
        dpos = np.abs(dense_pos - low_pos)
        dpos = np.minimum(dpos, z_range - dpos)
        max_pos_err = float(np.nanmax(dpos))
        mean_pos_err = float(np.nanmean(dpos))

        rmax_diff = np.abs(dense_r_max - low_r_max)
        max_rmax_err = float(rmax_diff.max())
        mean_rmax_err = float(rmax_diff.mean())
        rmse_r = float(np.sqrt(((dense_r - low_r) ** 2).mean()))

        results.append(
            CellResult(
                model="CANN1D",
                n=num,
                n_neurons=n_neurons,
                k=k_rank,
                is_lowrank=True,
                per_step_ms=t_low,
                speedup_vs_dense=speedup,
                matvec_per_step_ms=t_low_matvec,
                matvec_speedup=matvec_speedup,
                max_abs_err_r_max=max_rmax_err,
                mean_abs_err_r_max=mean_rmax_err,
                max_pos_err=max_pos_err,
                mean_pos_err=mean_pos_err,
                rmse_r=rmse_r,
                relerr_conn=relerr,
                captured_energy=captured,
            )
        )

    return results


# ---------------------------------------------------------------------------
# CANN2D benchmark cell
# ---------------------------------------------------------------------------

def benchmark_cann2d(length: int, ranks: list[int], T: int, dt: float) -> list[CellResult]:
    bm.set_dt(dt)
    model = CANN2D(length=length)
    conn_np = np.asarray(model.conn_mat)
    x_np = np.asarray(model.x)
    tau = float(model.tau)
    k_global = float(model.k)
    a = float(model.a)
    A = float(model.A)
    z_range = float(model.z_range)
    L = length
    n_neurons = L * L

    stim_np = make_moving_stimulus_2d(L, T, x_np, a, A, z_range)

    step_dense = _make_dense_step()
    step_low = _make_lowrank_step()

    U_full, S_full, V_full = np.linalg.svd(conn_np, full_matrices=False)
    total_energy = float((S_full ** 2).sum())

    u0 = np.zeros((L, L), dtype=np.float32)
    r0 = np.zeros((L, L), dtype=np.float32)

    # Dense reference trajectory
    u = u0.copy().reshape(-1)
    r = r0.copy().reshape(-1)
    dense_u = np.empty((T, L, L), dtype=np.float32)
    dense_r = np.empty((T, L, L), dtype=np.float32)
    conn_jax = jnp.asarray(conn_np)
    for t in range(T):
        inp = jnp.asarray(stim_np[t].reshape(-1))
        u, r = step_dense(u, r, conn_jax, inp, tau, k_global, dt)
        dense_u[t] = np.asarray(u).reshape(L, L)
        dense_r[t] = np.asarray(r).reshape(L, L)

    # Reference diagnostics
    dense_r_max = dense_r.reshape(T, -1).max(axis=1)
    dense_pos = np.array([bump_position_2d(dense_r[t], x_np, z_range) for t in range(T)])

    test_u = jnp.asarray(dense_u[T // 2].reshape(-1))
    test_r = jnp.asarray(dense_r[T // 2].reshape(-1))
    test_inp = jnp.asarray(stim_np[T // 2].reshape(-1))

    results: list[CellResult] = []

    # Dense
    t_dense = measure_step_time(
        step_dense,
        (test_u, test_r, conn_jax, test_inp, tau, k_global, dt),
        n_warmup=5,
        n_iters=30,  # fewer iters — 2D is slower
    )

    # Dense matvec-only
    scan_dense, T_scan = make_matvec_scan_dense(conn_jax)
    r_scan = jnp.asarray(np.maximum(dense_r[T // 2].reshape(-1), 0.0))
    t_dense_matvec = measure_matvec_scan_time(
        scan_dense, r_scan, n_warmup=3, n_iters=15, T_scan=T_scan
    )

    results.append(
        CellResult(
            model="CANN2D",
            n=length,
            n_neurons=n_neurons,
            k=-1,
            is_lowrank=False,
            per_step_ms=t_dense,
            speedup_vs_dense=1.0,
            matvec_per_step_ms=t_dense_matvec,
            matvec_speedup=1.0,
            max_abs_err_r_max=0.0,
            mean_abs_err_r_max=0.0,
            max_pos_err=0.0,
            mean_pos_err=0.0,
            rmse_r=0.0,
            relerr_conn=0.0,
            captured_energy=1.0,
        )
    )

    for k_rank in ranks:
        U, S, V = U_full[:, :k_rank], S_full[:k_rank], V_full[:, :k_rank]
        U_l, V_l = lowrank_mats(U, S, V)
        Ul_jax = jnp.asarray(U_l)
        Vl_jax = jnp.asarray(V_l)

        approx = U_l @ V_l.T
        relerr = float(np.linalg.norm(conn_np - approx) / np.linalg.norm(conn_np))
        captured = float((S ** 2).sum() / total_energy)

        t_low = measure_step_time(
            step_low,
            (test_u, test_r, Ul_jax, Vl_jax, test_inp, tau, k_global, dt),
            n_warmup=5,
            n_iters=30,
        )
        speedup = t_dense / t_low if t_low > 0 else float("inf")

        # Matvec-only
        scan_low, _ = make_matvec_scan_lowrank(Ul_jax, Vl_jax)
        t_low_matvec = measure_matvec_scan_time(
            scan_low, r_scan, n_warmup=3, n_iters=15, T_scan=T_scan
        )
        matvec_speedup = t_dense_matvec / t_low_matvec if t_low_matvec > 0 else float("inf")

        # Accuracy (run in flat space)
        u_l = u0.copy().reshape(-1)
        r_l = r0.copy().reshape(-1)
        low_r = np.empty((T, L, L), dtype=np.float32)
        for t in range(T):
            inp = jnp.asarray(stim_np[t].reshape(-1))
            u_l, r_l = step_low(u_l, r_l, Ul_jax, Vl_jax, inp, tau, k_global, dt)
            low_r[t] = np.asarray(r_l).reshape(L, L)

        low_r_max = low_r.reshape(T, -1).max(axis=1)
        low_pos = np.array([bump_position_2d(low_r[t], x_np, z_range) for t in range(T)])

        # 2D circular-mean position error
        dx = np.abs(dense_pos[:, 0] - low_pos[:, 0])
        dy = np.abs(dense_pos[:, 1] - low_pos[:, 1])
        dx = np.minimum(dx, z_range - dx)
        dy = np.minimum(dy, z_range - dy)
        dpos = np.sqrt(dx ** 2 + dy ** 2)
        max_pos_err = float(np.nanmax(dpos))
        mean_pos_err = float(np.nanmean(dpos))

        rmax_diff = np.abs(dense_r_max - low_r_max)
        max_rmax_err = float(rmax_diff.max())
        mean_rmax_err = float(rmax_diff.mean())
        rmse_r = float(np.sqrt(((dense_r - low_r) ** 2).mean()))

        results.append(
            CellResult(
                model="CANN2D",
                n=length,
                n_neurons=n_neurons,
                k=k_rank,
                is_lowrank=True,
                per_step_ms=t_low,
                speedup_vs_dense=speedup,
                matvec_per_step_ms=t_low_matvec,
                matvec_speedup=matvec_speedup,
                max_abs_err_r_max=max_rmax_err,
                mean_abs_err_r_max=mean_rmax_err,
                max_pos_err=max_pos_err,
                mean_pos_err=mean_pos_err,
                rmse_r=rmse_r,
                relerr_conn=relerr,
                captured_energy=captured,
            )
        )

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "model", "n", "n_neurons", "k", "is_lowrank",
    "per_step_ms", "speedup_vs_dense",
    "matvec_per_step_ms", "matvec_speedup",
    "max_abs_err_r_max", "mean_abs_err_r_max",
    "max_pos_err", "mean_pos_err", "rmse_r",
    "relerr_conn", "captured_energy",
]


def write_csv(path: Path, rows: list[CellResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            d = {fld: getattr(r, fld) for fld in CSV_FIELDS}
            # Cast n/k to int for clean CSV
            d["n"] = int(d["n"])
            d["n_neurons"] = int(d["n_neurons"])
            d["k"] = int(d["k"])
            d["is_lowrank"] = int(bool(d["is_lowrank"]))
            w.writerow(d)


# ---------------------------------------------------------------------------
# Bump center trajectory recording (for the "decode over time" figure)
# ---------------------------------------------------------------------------

def _moving_stimulus_1d(x: np.ndarray, a: float, A: float, z_range: float, T: int) -> np.ndarray:
    """Sweep a Gaussian across the ring over ``T`` steps."""
    num = len(x)
    out = np.empty((T, num), dtype=np.float32)
    for t in range(T):
        pos = math.pi * t / max(T - 1, 1)
        d = (x - pos) % z_range
        d = np.where(d > z_range / 2, d - z_range, d)
        out[t] = A * np.exp(-0.25 * (d / a) ** 2)
    return out


def _moving_stimulus_2d(x: np.ndarray, a: float, A: float, z_range: float, T: int) -> np.ndarray:
    L = len(x)
    xx, yy = np.meshgrid(x, x)
    out = np.empty((T, L, L), dtype=np.float32)
    for t in range(T):
        pos = math.pi * t / max(T - 1, 1)
        dx = (xx - pos) % z_range
        dy = (yy - pos) % z_range
        dx = np.where(dx > z_range / 2, dx - z_range, dx)
        dy = np.where(dy > z_range / 2, dy - z_range, dy)
        out[t] = (A * np.exp(-0.25 * ((dx ** 2 + dy ** 2) ** 0.5) / a) ** 2).astype(np.float32)
    return out


def record_bump_trajectories_1d(
    num: int, k_list: list[int], T: int, dt: float = 0.1,
    T_warm: int = 20,
) -> dict:
    """Record the bump center trajectory over T steps for a CANN1D at each
    rank in ``k_list`` (plus the dense baseline).

    A pre-warm phase of ``T_warm`` steps with a *stationary* stimulus at
    ``pos=0`` is run first so the bump is fully formed before the moving
    stimulus starts. This matches the realistic use case where the
    network is initialised with a bump already in place.

    Returns a dict with keys:
        ``dense``     — array of shape ``(T,)``
        ``k{ki}``     — array of shape ``(T,)`` for each rank
        ``k_full``    — same as ``dense``, kept for naming consistency
        ``sv``        — full SVD spectrum of the conn matrix
        ``x``         — feature-space coordinates
    """
    bm.set_dt(dt)
    model = CANN1D(num=num, accl_mode="normal")
    conn_np = np.asarray(model.conn_mat)
    x_np = np.asarray(model.x)
    tau = float(model.tau)
    k_global = float(model.k)
    a, A, z_range = float(model.a), float(model.A), float(model.z_range)

    U_full, S_full, Vt_full = np.linalg.svd(conn_np, full_matrices=False)
    sv = S_full.astype(np.float32)

    step_dense = _make_dense_step()
    step_low = _make_lowrank_step()

    # Moving stimulus (for the recorded trajectory)
    stim_np = _moving_stimulus_1d(x_np, a, A, z_range, T)
    # Stationary stimulus at pos=0 (for warm-up)
    warm_stim = np.broadcast_to(
        (A * np.exp(-0.25 * ((x_np / a) ** 2))).astype(np.float32),
        (T_warm, num),
    ).copy()

    # Pre-build lowrank factors
    factors = {}
    for ki in k_list:
        sqrtS = np.sqrt(S_full[:ki].astype(np.float32))
        factors[ki] = (
            jnp.asarray(U_full[:, :ki].astype(np.float32) * sqrtS),
            jnp.asarray(Vt_full[:ki, :].T.astype(np.float32) * sqrtS),
        )

    def warm_and_record(initial_u, initial_r, step_fn, *extra_args, flat=False):
        u, r = initial_u, initial_r
        # Warm-up
        for t in range(T_warm):
            inp = jnp.asarray(warm_stim[t])
            u, r = step_fn(u, r, *extra_args, inp, tau, k_global, dt)
        # Recorded trajectory
        pos = np.empty(T, dtype=np.float64)
        for t in range(T):
            inp = jnp.asarray(stim_np[t])
            u, r = step_fn(u, r, *extra_args, inp, tau, k_global, dt)
            if flat:
                r_2d = np.asarray(r).reshape(num, num)
            else:
                r_2d = np.asarray(r)
            pos[t] = bump_position_1d(r_2d, x_np, z_range)
        return pos

    # Dense reference
    dense_pos = warm_and_record(
        np.zeros(num, dtype=np.float32),
        np.zeros(num, dtype=np.float32),
        step_dense,
        jnp.asarray(conn_np),
    )

    out: dict = {"dense": dense_pos, "sv": sv, "x": x_np}
    out["k_full"] = dense_pos.copy()

    for ki, (U_l, V_l) in factors.items():
        out[f"k{ki}"] = warm_and_record(
            np.zeros(num, dtype=np.float32),
            np.zeros(num, dtype=np.float32),
            step_low,
            U_l, V_l,
        )

    return out


def record_bump_trajectories_2d(
    length: int, k_list: list[int], T: int, dt: float = 0.1,
    T_warm: int = 20,
) -> dict:
    """Record the bump center trajectory over T steps for a CANN2D at each
    rank in ``k_list`` (plus the dense baseline).

    A pre-warm phase of ``T_warm`` steps with a *stationary* stimulus at
    ``pos=(0, 0)`` is run first so the bump is fully formed before the
    moving stimulus starts.

    Returns a dict with keys:
        ``dense``     — array of shape ``(T, 2)``
        ``k{ki}``     — array of shape ``(T, 2)`` for each rank
        ``k_full``    — same as ``dense``
        ``sv``        — full SVD spectrum of the conn matrix
        ``x``         — feature-space coordinates
    """
    bm.set_dt(dt)
    model = CANN2D(length=length, accl_mode="normal")
    conn_np = np.asarray(model.conn_mat)
    x_np = np.asarray(model.x)
    tau = float(model.tau)
    k_global = float(model.k)
    a, A, z_range = float(model.a), float(model.A), float(model.z_range)
    L = length
    n = L * L

    U_full, S_full, Vt_full = np.linalg.svd(conn_np, full_matrices=False)
    sv = S_full.astype(np.float32)

    step_dense = _make_dense_step()
    step_low = _make_lowrank_step()

    stim_np = _moving_stimulus_2d(x_np, a, A, z_range, T)

    # Stationary warm-up stimulus
    xx, yy = np.meshgrid(x_np, x_np)
    warm_stim_flat = (A * np.exp(-0.25 * ((xx ** 2 + yy ** 2) ** 0.5) / a) ** 2).astype(np.float32).reshape(-1)
    warm_stim = np.broadcast_to(warm_stim_flat, (T_warm, n)).copy()

    factors = {}
    for ki in k_list:
        sqrtS = np.sqrt(S_full[:ki].astype(np.float32))
        factors[ki] = (
            jnp.asarray(U_full[:, :ki].astype(np.float32) * sqrtS),
            jnp.asarray(Vt_full[:ki, :].T.astype(np.float32) * sqrtS),
        )

    def warm_and_record(initial_u, initial_r, step_fn, *extra_args):
        u, r = initial_u, initial_r
        for t in range(T_warm):
            inp = jnp.asarray(warm_stim[t])
            u, r = step_fn(u, r, *extra_args, inp, tau, k_global, dt)
        pos = np.empty((T, 2), dtype=np.float64)
        for t in range(T):
            inp = jnp.asarray(stim_np[t].reshape(-1))
            u, r = step_fn(u, r, *extra_args, inp, tau, k_global, dt)
            cx, cy = bump_position_2d(np.asarray(r).reshape(L, L), x_np, z_range)
            pos[t] = (cx, cy)
        return pos

    dense_pos = warm_and_record(
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        step_dense,
        jnp.asarray(conn_np),
    )

    out: dict = {"dense": dense_pos, "sv": sv, "x": x_np}
    out["k_full"] = dense_pos.copy()

    for ki, (U_l, V_l) in factors.items():
        out[f"k{ki}"] = warm_and_record(
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            step_low,
            U_l, V_l,
        )

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fast", action="store_true",
                   help="smaller sweep (quick smoke test, ~1 min)")
    p.add_argument("--gpu-sweep", action="store_true",
                   help="use the larger n sweep suitable for a GPU (n up to 4096 for 1D, "
                        "L up to 128 for 2D). Implies a tag of 'gpu' if --tag is not given.")
    p.add_argument("--T", type=int, default=200,
                   help="simulation length for accuracy (default 200)")
    p.add_argument("--dt", type=float, default=0.1,
                   help="simulation timestep (default 0.1, matches canns default)")
    p.add_argument("--outdir", type=str, default=None,
                   help="results dir (default: experiments/cann_lowrank/results)")
    p.add_argument("--tag", type=str, default=None,
                   help="filename suffix (e.g. 'cpu', 'gpu'). Default: 'cpu' or 'gpu' "
                        "auto-detected from JAX_PLATFORMS env var, with --gpu-sweep "
                        "forcing 'gpu'.")
    p.add_argument("--no-trajectories", action="store_true",
                   help="skip recording bump center trajectories (faster, less output)")
    args = p.parse_args()

    outdir = Path(args.outdir) if args.outdir else _HERE / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    # Tag: cpu or gpu. auto-detect from JAX_PLATFORMS, allow --tag override.
    if args.tag is not None:
        tag = args.tag
    else:
        platforms = os.environ.get("JAX_PLATFORMS", "").lower()
        tag = "gpu" if "cuda" in platforms or args.gpu_sweep else "cpu"

    if args.fast:
        cann1d_sizes = [64, 128, 256, 512]
        cann2d_sizes = [8, 16, 32]
        ranks_1d = [1, 2, 4, 8, 16]
        ranks_2d = [1, 2, 4, 8, 16, 32, 64]
    elif args.gpu_sweep:
        # Same n as default CPU; the A100 advantage comes through
        # with more iters, not larger sizes. L=128 / n=4096 is
        # already past the size where GPU matvec time dominates
        # dispatch overhead. Going to n=16384 is dominated by the
        # numpy SVD cost (5+ min per cell on CPU) so we cap here.
        cann1d_sizes = [64, 128, 256, 512, 1024, 2048]
        cann2d_sizes = [8, 16, 32, 64]
        ranks_1d = [1, 2, 4, 8, 16, 32]
        ranks_2d = [1, 2, 4, 8, 16, 32, 64, 128]
    else:
        cann1d_sizes = [64, 128, 256, 512, 1024, 2048]
        cann2d_sizes = [8, 16, 32, 64]
        ranks_1d = [1, 2, 4, 8, 16, 32]
        ranks_2d = [1, 2, 4, 8, 16, 32, 64, 128]

    all_rows: list[CellResult] = []

    print("=" * 70)
    print(f"CANN1D — speed + accuracy  [tag={tag}]")
    print("=" * 70)
    for num in cann1d_sizes:
        print(f"  num={num} ...", flush=True)
        rows = benchmark_cann1d(num=num, ranks=ranks_1d, T=args.T, dt=args.dt)
        all_rows.extend(rows)
        for r in rows:
            if r.k in (-1, 8):
                klabel = 'full' if r.k == -1 else str(r.k)
                print(
                    f"    k={klabel:>4s} | "
                    f"per_step={r.per_step_ms:7.3f} ms | "
                    f"speedup={r.speedup_vs_dense:5.2f}x | "
                    f"matvec={r.matvec_per_step_ms:7.4f} ms | "
                    f"mv_spu={r.matvec_speedup:5.2f}x | "
                    f"r_max_err={r.max_abs_err_r_max:.4f} | "
                    f"pos_err={r.max_pos_err:.4f} | "
                    f"energy={r.captured_energy:.4f}"
                )

    print()
    print("=" * 70)
    print(f"CANN2D — speed + accuracy  [tag={tag}]")
    print("=" * 70)
    for length in cann2d_sizes:
        print(f"  length={length} (n={length*length}) ...", flush=True)
        rows = benchmark_cann2d(length=length, ranks=ranks_2d, T=args.T, dt=args.dt)
        all_rows.extend(rows)
        for r in rows:
            if r.k in (-1, 8, 32):
                klabel = 'full' if r.k == -1 else str(r.k)
                print(
                    f"    k={klabel:>4s} | "
                    f"per_step={r.per_step_ms:7.3f} ms | "
                    f"speedup={r.speedup_vs_dense:5.2f}x | "
                    f"matvec={r.matvec_per_step_ms:7.4f} ms | "
                    f"mv_spu={r.matvec_speedup:5.2f}x | "
                    f"r_max_err={r.max_abs_err_r_max:.4f} | "
                    f"pos_err={r.max_pos_err:.4f} | "
                    f"energy={r.captured_energy:.4f}"
                )

    # Output CSV files
    speed_csv = outdir / f"cann_lowrank_speed_{tag}.csv"
    acc_csv = outdir / f"cann_lowrank_accuracy_{tag}.csv"
    full_csv = outdir / f"cann_lowrank_all_{tag}.csv"
    write_csv(speed_csv, all_rows)
    write_csv(acc_csv, all_rows)
    write_csv(full_csv, all_rows)

    # Bump center trajectory recording (for the "decode over time" figure)
    if not args.no_trajectories:
        print()
        print("=" * 70)
        print("Bump center trajectory recording")
        print("=" * 70)
        traj_npz = outdir / f"bump_trajectories_{tag}.npz"
        traj_data: dict = {}
        # Pick representative sizes for the figures
        traj_1d_n = 256
        traj_2d_L = 16
        traj_1d_ks = [1, 2, 4, 8, 16, 32]
        traj_2d_ks = [1, 4, 8, 16, 32, 64]
        print(f"  CANN1D num={traj_1d_n} ...", flush=True)
        traj_1d = record_bump_trajectories_1d(traj_1d_n, traj_1d_ks, T=args.T, dt=args.dt)
        traj_data["traj_1d_n"] = traj_1d_n
        traj_data["traj_1d_ks"] = np.array(traj_1d_ks)
        traj_data["sv_1d"] = traj_1d["sv"]
        traj_data["x_1d"] = traj_1d["x"]
        for k_name, arr in traj_1d.items():
            if k_name in ("sv", "x"):
                continue
            traj_data[f"traj_1d_{k_name}"] = arr
        print(f"  CANN2D L={traj_2d_L} ...", flush=True)
        traj_2d = record_bump_trajectories_2d(traj_2d_L, traj_2d_ks, T=args.T, dt=args.dt)
        traj_data["traj_2d_L"] = traj_2d_L
        traj_data["traj_2d_ks"] = np.array(traj_2d_ks)
        traj_data["sv_2d"] = traj_2d["sv"]
        traj_data["x_2d"] = traj_2d["x"]
        for k_name, arr in traj_2d.items():
            if k_name in ("sv", "x"):
                continue
            traj_data[f"traj_2d_{k_name}"] = arr
        np.savez(traj_npz, **traj_data)
        print(f"  Wrote {traj_npz}")

    print()
    print(f"  Wrote {speed_csv}")
    print(f"  Wrote {acc_csv}")
    print(f"  Wrote {full_csv}")
    print()
    print("Done. Run the analysis companion to format the writeup:")
    print(f"  python experiments/cann_lowrank/cann_lowrank_report.py --tag {tag}")


if __name__ == "__main__":
    main()
