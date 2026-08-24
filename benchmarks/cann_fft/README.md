# FFT matvec benchmark (`accl_mode='fft'`)

Compares three recurrent-matvec backends for `CANN1D` and `CANN2D`:

| backend | formula | complexity | error |
|---|---|---|---|
| `dense` | `conn @ r` | O(n²) | 0 (baseline) |
| `svd_k{k}` | `U_l @ (V_l.T @ r)` | O(nk) | 10⁻² – 10² mrad depending on k |
| `fft` | `real(ifft(fft(c) ⊙ fft(r)))` | O(n log n) | **0** (exact) on a clean circulant |

The `fft` mode exploits the circulant structure of the Gaussian
distance kernel on a uniform ring (1D) or torus (2D). The canns
default `endpoint=True` grid is **not** circulant (the wrap convention
breaks the cyclic shift symmetry), so `fft` on the default grid
silently falls back to `dense` with a `UserWarning`. To get the FFT
speedup, override the grid:

```python
model = CANN1D(num=4096, accl_mode="normal")
model.x = bm.linspace(-bm.pi, bm.pi, 4096, endpoint=False)
model.conn_mat = model.make_conn()       # rebuild K for the new grid
model.set_accl_mode("fft")               # or pass it to the constructor
# for CANN2D, also override model.y and rebuild
```

## Run

```bash
# CPU
JAX_PLATFORMS=cpu python benchmarks/cann_fft/cann_fft_bench.py

# GPU
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 \
    python benchmarks/cann_fft/cann_fft_bench.py

# Report
python benchmarks/cann_fft/cann_fft_report.py --tag cpu
python benchmarks/cann_fft/cann_fft_report.py --tag gpu
python benchmarks/cann_fft/combine_reports.py
```

## Headline numbers (CPU)

| model | n | backend | step_ms | scan_ms | speedup_step | speedup_scan |
|---|---|---|---|---|---|---|
| CANN1D | 4096 | dense | 0.80 | 0.80 | 1.00× | 1.00× |
| CANN1D | 4096 | **fft** | **0.032** | **0.021** | **25×** | **39×** |
| CANN1D | 4096 | svd_k1 | 0.005 | 0.001 | 168× | 965× |
| CANN2D | 4096 (L=64) | dense | 0.83 | 0.80 | 1.00× | 1.00× |
| CANN2D | 4096 (L=64) | **fft** | **0.027** | **0.016** | **31×** | **50×** |
| CANN2D | 4096 (L=64) | svd_k1 | 0.003 | 0.001 | 247× | 944× |

SVD k=1 is much faster but has 30-50 mrad position error. FFT is the
right choice when you need exact dynamics (parameter sweeps,
bifurcation analysis, regression tests).

## Headline numbers (A100 GPU)

| model | n | backend | step_ms | scan_ms | speedup_step | speedup_scan |
|---|---|---|---|---|---|---|
| CANN1D | 4096 | dense | 0.23 | 0.053 | 1.00× | 1.00× |
| CANN1D | 4096 | fft | 0.21 | 0.027 | 1.10× | **1.96×** |
| CANN1D | 4096 | svd_k1 | 0.094 | 0.010 | 2.40× | 5.03× |

On A100 the dense matvec is already well-optimized (cuBLAS sgemv) and
the per-step win for FFT is only ~10%. FFT wins more decisively on
the scan path (1.6-2.0×) where XLA fuses the FFT and avoids per-step
launch overhead.

## Endpoint gotcha

The canns default grid is `bm.linspace(-π, π, n, endpoint=True)`, so
`x[0] = -π` and `x[-1] = +π`. The wrap convention then makes
`K[0, 0] = K[0, n-1] = f(0) = max`, but `K[0, 1] ≠ K[1, 0]` after the
wrap (the differences wrap inconsistently near the boundary). The
matrix is symmetric (Gaussian is even) but not circulant. The FFT
path detects this and falls back to dense.

To enable FFT, use a uniform periodic grid (no endpoint):
`bm.linspace(-π, π, n, endpoint=False)`. Then `K[0, 0] = f(0) = max`
and `K[0, n-1] = f(2π/n)`, so the wrap consistently gives the same
result for `K[i, j] = c[(j - i) mod n]` (right-circulant).

## Layout

- `cann_fft_bench.py` — the benchmark (run on CPU or GPU)
- `cann_fft_report.py` — per-platform markdown report + figures
- `combine_reports.py` — combined CPU vs GPU report
- `results/cann_fft_speed_{cpu,gpu}.csv` — per-step + scan times
- `results/cann_fft_accuracy_{cpu,gpu}.csv` — max-abs error vs dense
- `results/cann_fft_summary.md` — combined report
- `figures/*.png` — speed / scan / accuracy / Pareto / CPU vs GPU plots
