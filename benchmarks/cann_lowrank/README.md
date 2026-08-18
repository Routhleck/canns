# Low-rank recurrent matvec for CANN1D / CANN2D

This directory contains a self-contained, paper-style benchmark of
the **low-rank approximation of the recurrent matvec** in the
standard `CANN1D` and `CANN2D` models from `canns.models.basic`,
plus the correctness audit of the benchmark and the API
documentation for the `accl_mode` / `accl_k` feature that is
exposed on the model classes.

## What it tests

In `CANN1D` / `CANN2D`, the recurrent connectivity `conn_mat` is a
Gaussian distance kernel. At every step the model performs

```
Irec = conn @ r                 # CANN1D
Irec = r.flatten() @ conn_mat   # CANN2D
```

For a ring / grid with `n` neurons this is an `O(n²)` matvec and
becomes the dominant cost at large `n`.

The benchmark replaces the dense matvec with a truncated-SVD
factorisation `conn ≈ U_l @ V_l.T` (where `U_l`, `V_l` are
`(n, k)`, `k ≪ n`). The forward matvec becomes two small GEMVs,
`Irec = U_l @ (V_l.T @ r)`, costing `2·n·k` FLOPs.

## What it measures

For each `(model, n, k)` cell:

- **matvec per-step** (μs) — median of a `lax.scan` body that does
  *only* the recurrent matvec, 200 steps per call. Isolates the
  algorithmic cost of the low-rank substitution.
- **full step per-step** (μs) — median of the entire CANN update
  function (divisive norm + matvec + Euler).
- **bump-position error** (mrad) — max circular-distance error in
  bump center between low-rank and dense, over a 200-step moving
  Gaussian stimulus.
- **`r_max` error** — max absolute error in `max(r)` over the same
  trajectory.
- **captured energy** — `Σ S[:k]² / Σ S²`, where `S` is the SVD of
  `conn_mat`.
- **bump center trajectory** (1D and 2D) — recorded for one
  representative cell per model, decoded via circular mean, for
  every k value plus the dense reference.
- **long-trajectory drift** (optional, `--long-trajectory`) — a
  T = 2000 slow-sweep protocol that measures the *boundedness* of
  the low-rank error over a long horizon (one full ring per trial).
  Reported as `|pos_lowrank(t) - pos_dense(t)|` on a log scale for
  each k. See §3.7 in the report.

## Paper-style writeup

The benchmark report is at
[`results/cann_lowrank_summary.md`](results/cann_lowrank_summary.md)
and follows a small-paper structure:

- **Abstract** — one-paragraph motivation, methods, headline numbers.
- **§1 Introduction** — background, motivation, the cost question.
- **§2 Methods** — CANN dynamics, low-rank SVD factorisation, bump
  decoding, stimulus protocol, metrics, hardware.
- **§3 Results** — 11 figures total:
  1. SVD spectrum of the Gaussian kernel (1D + 2D, one figure)
  2. 1D bump center trajectory over time (all k values overlaid)
  3. 2D bump center trajectory in feature space (all k values overlaid)
  4. CPU matvec speedup vs n — 2 figures (1D + 2D)
  5. GPU matvec speedup vs n — 2 figures (1D + 2D), only if a GPU
     sweep was run
  6. Speed-accuracy Pareto frontier — 2 figures (1D + 2D)
  7. 1D long-trajectory drift (T = 2000 slow sweep), only with
     `--long-trajectory`
  8. 2D long-trajectory drift (T = 2000 slow sweep), only with
     `--long-trajectory`
- **§4 Discussion** — when low-rank helps, when it doesn't, recommended
  strategy.
- **§5 Limitations** — what was *not* measured.
- **§6 Conclusion** — recap and API pointer.
- **References** — Wu et al. 2008 (the original CANN paper), the
  `canns` package.
- **Appendix A** — reproduction commands.
- **Appendix B** — raw data files.

The figures are in
[`results/figures/`](results/figures/) as both PNG (for web) and
PDF (for paper inclusion).

## Files

| File | Purpose |
| --- | --- |
| `cann_lowrank_bench.py` | Main benchmark. Runs the full sweep, records bump trajectories, writes CSVs and npz. |
| `cann_lowrank_report.py` | Reads the CSVs + npz, generates 11 figures, writes the paper-style markdown. |
| `REVIEW.md` | Correctness audit of the benchmark (SVD math, symmetric conn_mat argument, stimulus, bump-position, error metrics, timing methodology, edge cases). |
| `_smoke/explore_conn.py` | Initial spectrum exploration that informed the rank sweep. |
| `results/cann_lowrank_all_{cpu,gpu}.csv` | Raw per-cell numbers (one row per `(model, n, k)`). |
| `results/bump_trajectories_{cpu,gpu}.npz` | Bump center trajectories for one representative cell per model, at all k values plus the dense reference. |
| `results/bump_drift_{cpu,gpu}.npz` | Long-trajectory drift (T = 2000 slow sweep), one representative cell per model. |
| `results/figures/fig_*.{png,pdf}` | The figures embedded in the report. |
| `results/cann_lowrank_summary.md` | The paper-style writeup. |

## Reproducing

From the repo root, with the `canns` source on `PYTHONPATH` and
`brainpy.math` + `jax` installed:

```bash
# CPU sweep (Apple M3 Pro, single core, ~15 min wall):
python benchmarks/cann_lowrank/cann_lowrank_bench.py --T 200 --tag cpu

# Optional: also run the long-trajectory drift test (T=2000):
python benchmarks/cann_lowrank/cann_lowrank_bench.py --T 200 --long-trajectory --tag cpu

# GPU sweep (NVIDIA A100, GPU 1, ~13 min wall — includes n=6144/8192
# which need ~100s of numpy SVD on CPU, plus all the 2D sizes):
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  python benchmarks/cann_lowrank/cann_lowrank_bench.py --gpu-sweep --T 200 --tag gpu

# Format the paper-style report (figures + markdown only):
python benchmarks/cann_lowrank/cann_lowrank_report.py --tag cpu

# Also write a styled HTML version (NeurIPS-like):
python benchmarks/cann_lowrank/cann_lowrank_report.py --tag cpu --html

# Also write a PDF version (NeurIPS-like, requires weasyprint):
python benchmarks/cann_lowrank/cann_lowrank_report.py --tag cpu --pdf
```

The report is reproducible as long as at least the CPU sweep has
been run. The GPU sweep adds the two GPU-speedup figures; if it
hasn't been run, the report falls back to the CPU data.

## Branch

This work lives on the `canns-lowrank-bench` branch.
