# Low-rank recurrent matvec for CANN1D / CANN2D

This directory contains a self-contained benchmark of the
**low-rank approximation of the recurrent matvec** in the standard
`CANN1D` and `CANN2D` models from `canns.models.basic`.

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

## Files

| File | Purpose |
| --- | --- |
| `cann_lowrank_bench.py` | Main benchmark. Runs the full sweep and writes CSVs. |
| `cann_lowrank_report.py` | Reads the CSVs and writes the markdown writeup. |
| `_smoke/explore_conn.py` | Initial spectrum exploration that informed the rank sweep. |
| `results/cann_lowrank_all.csv` | Raw per-cell numbers (one row per `(model, n, k)`). |
| `results/cann_lowrank_summary.md` | Human-readable writeup with tables and findings. |

## Reproducing

From the repo root, with the canns-accel venv active (it has
`brainpy`, `jax`, and an editable `canns` install):

```bash
python experiments/cann_lowrank/cann_lowrank_bench.py --T 200
python experiments/cann_lowrank/cann_lowrank_report.py
```

The full sweep (CANN1D `num ∈ {64..2048}`, CANN2D `length ∈ {8..64}`,
ranks `k ∈ {1, 2, 4, 8, 16, 32}` and `+64, +128` for 2D) takes
~10 minutes on a CPU-only machine. Pass `--fast` for a smaller
sweep (1D up to 512, 2D up to L=32).

## Branch

This work lives on the `canns-lowrank-bench` branch.
