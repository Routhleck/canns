"""Shuffle null-model comparison: canns-lib FFI vs mp.Pool legacy path.

Covers the matrix:
  T    (time-points / rows)  in {60, 300, 1000}
  N    (neurons / cols)      in {20, 40, 80, 160}
  n_shuffles                 in {10, 50, 200, 1000}

For each (T, N, n_shuffles), runs:
  - FFI:    canns.analyzer.data.asa._run_shuffle_analysis with
            use_ffi_shuffle=True, measures wall time.
  - legacy: same call with use_ffi_shuffle=False (mp.Pool).
Records:
  - wall time (ms) for both
  - speedup = legacy_ms / ffi_ms
  - dim-1 max-lifetime mean (FFI vs legacy) — both distributions are reported,
    NOT directly compared, because the FFI uses raw spike-train Euclidean
    and the legacy path uses the full (sampled + PCA + UMAP-denoised + nbs)
    point cloud. The note is in the writeup.

Writes benchmarks/results/shuffle_results_<ts>.csv by default.
"""
import argparse
import csv
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# The legacy shuffle path inside canns uses multiprocessing.Pool to fan out
# the K shuffles. The default Linux start method is 'fork', which copies
# the parent's full thread state into each worker — including JAX's BLAS
# threadpool. That combination deadlocks on the first pickling barrier.
# 'forkserver' forks from a clean helper process, so workers start
# without the parent's auxiliary threads. This matches what the canns
# maintainers recommend on Linux+JAX systems.
import multiprocessing as mp
try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass  # start method already set (e.g. when run under a wrapper)

# benchmarks/ is a sibling of src/, so go up one level to find src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Record the canns-lib version actually exercised by this benchmark so the
# results are reproducible and obviously tied to a specific commit.
try:
    import canns_lib
    print(f"canns_lib version: {canns_lib.__version__}", flush=True)
except Exception as _exc:
    print(f"canns_lib not importable: {_exc}", flush=True)

try:
    from canns_lib import _ripser_core as _core
    has_shuffle = hasattr(_core, "shuffle_null_model")
    print(f"FFI shuffle_null_model present: {has_shuffle}", flush=True)
except Exception as _exc:
    has_shuffle = False
    print(f"FFI not importable: {_exc}", flush=True)

from canns.analyzer.data.asa.tda import _run_shuffle_analysis  # noqa: E402


@dataclass
class Row:
    T: int
    N: int
    n_shuffles: int
    seed: int
    ffi_ms: float
    legacy_ms: float
    speedup: float
    ffi_h1_mean: float
    legacy_h1_mean: float
    ffi_h2_mean: float
    legacy_h2_mean: float
    ffi_h1_count: int
    legacy_h1_count: int


def _hmean(out, dim):
    v = out.get(dim, [])
    return float(np.mean(v)) if v else 0.0


def _hcount(out, dim):
    return int(len(out.get(dim, [])))


def timeit(fn, repeats):
    out = None
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best * 1000.0, out


def run(T, N, n_shuffles, seed, ffi_args, legacy_args, repeats):
    spikes = np.random.RandomState(seed).poisson(0.3, size=(T, N)).astype(np.float32)
    ffi_ms, ffi_out = timeit(lambda: _run_shuffle_analysis(spikes, num_shuffles=n_shuffles, **ffi_args), repeats)
    legacy_ms, legacy_out = timeit(lambda: _run_shuffle_analysis(spikes, num_shuffles=n_shuffles, **legacy_args), repeats)
    return Row(
        T=T, N=N, n_shuffles=n_shuffles, seed=seed,
        ffi_ms=ffi_ms, legacy_ms=legacy_ms,
        speedup=legacy_ms / ffi_ms if ffi_ms > 0 else float("nan"),
        ffi_h1_mean=_hmean(ffi_out, 1),
        legacy_h1_mean=_hmean(legacy_out, 1),
        ffi_h2_mean=_hmean(ffi_out, 2),
        legacy_h2_mean=_hmean(legacy_out, 2),
        ffi_h1_count=_hcount(ffi_out, 1),
        legacy_h1_count=_hcount(legacy_out, 1),
    )


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    grid_args = {
        (T, N, K): dict(
            ffi_args=dict(maxdim=args.maxdim, coeff=args.coeff, use_ffi_shuffle=True, progress_bar=False),
            legacy_args=dict(maxdim=args.maxdim, coeff=args.coeff, use_ffi_shuffle=False, progress_bar=False),
        )
        for T in args.Ts
        for N in args.Ns
        for K in args.shuffles
    }

    for (T, N, K), kwa in grid_args.items():
        try:
            r = run(T, N, K, args.seed, kwa["ffi_args"], kwa["legacy_args"], repeats=args.repeats)
        except Exception as exc:
            print(f"FAIL T={T} N={N} K={K}: {exc}")
            continue
        print(f"  T={T:5d} N={N:4d} K={K:5d}  FFI={r.ffi_ms:8.1f}ms  legacy={r.legacy_ms:9.1f}ms  speedup={r.speedup:7.1f}x  "
              f"(h1 counts: ffi={r.ffi_h1_count}, legacy={r.legacy_h1_count})", flush=True)
        rows.append(r)

    if not rows:
        return None

    csv_path = out_dir / f"shuffle_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"\nsaved {csv_path}")

    # Aggregate summary
    print("\n=== speedup median by (T,N,K) ===", flush=True)
    print(f"{'T':>5}  {'N':>4}  {'K':>5}  {'FFI ms':>9}  {'Legacy ms':>10}  {'speedup':>9}")
    for r in rows:
        print(f"{r.T:5d}  {r.N:4d}  {r.n_shuffles:5d}  {r.ffi_ms:9.1f}  {r.legacy_ms:10.1f}  {r.speedup:8.1f}x")

    # Speedup grouped by K and by (T*N)
    import statistics
    by_K = {}
    for r in rows:
        by_K.setdefault(r.n_shuffles, []).append(r.speedup)
    print("\n=== speedup median by n_shuffles ===", flush=True)
    for K in sorted(by_K):
        m = statistics.median(by_K[K])
        print(f"  K={K:5d}  median speedup {m:7.1f}x  (n={len(by_K[K])})")
    return csv_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--T", dest="Ts", type=int, nargs="+", default=[60, 300, 1000])
    p.add_argument("--N", dest="Ns", type=int, nargs="+", default=[20, 40, 80, 160])
    p.add_argument("--shuffles", type=int, nargs="+", default=[10, 50, 200, 1000])
    p.add_argument("--maxdim", type=int, default=2)
    p.add_argument("--coeff", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str,
                   default=str(Path(__file__).resolve().parent / "results"),
                   help="Directory to write the CSV into. Defaults to benchmarks/results/.")
    main(p.parse_args())
