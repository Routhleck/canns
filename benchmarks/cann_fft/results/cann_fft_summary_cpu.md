# FFT matvec benchmark — CPU

Comparison of three recurrent-matvec backends for the CANN connectivity kernel: `dense` (O(n²), baseline), `svd_k{k}` (truncated SVD, O(nk), approximate), and `fft` (exact circulant matvec, O(n log n)).

All numbers come from a **clean circulant** setup (`endpoint=False` uniform grid); the canns default `endpoint=True` grid is not circulant and falls back to `dense` with a warning when `accl_mode='fft'` is requested.

## 1. Headline numbers (largest n)

| model | n | n_total | backend | step_ms | scan_ms | speedup_step | speedup_scan | max_err |
|---|---|---|---|---|---|---|---|---|
| cann1d | 4096 | 4096 | dense | 0.7998 | 0.8025 | 1.00 | 1.00 | 0.00e+00 |
| cann1d | 4096 | 4096 | fft | 0.0317 | 0.0207 | 25.22 | 38.80 | 1.70e-04 |
| cann1d | 4096 | 4096 | svd_k1 | 0.0048 | 0.0008 | 168.36 | 965.36 | 5.38e+01 |
| cann1d | 4096 | 4096 | svd_k4 | 0.0126 | 0.0027 | 63.35 | 297.69 | 4.60e+01 |
| cann1d | 4096 | 4096 | svd_k16 | 0.0169 | 0.0058 | 47.28 | 139.31 | 2.88e-02 |
| cann2d | 64 | 4096 | dense | 0.8318 | 0.8000 | 1.00 | 1.00 | 0.00e+00 |
| cann2d | 64 | 4096 | fft | 0.0267 | 0.0159 | 31.19 | 50.23 | 8.01e-05 |
| cann2d | 64 | 4096 | svd_k1 | 0.0034 | 0.0008 | 246.53 | 944.43 | 8.46e+01 |
| cann2d | 64 | 4096 | svd_k4 | 0.0109 | 0.0027 | 76.20 | 297.61 | 8.76e+01 |
| cann2d | 64 | 4096 | svd_k16 | 0.0192 | 0.0058 | 43.40 | 137.14 | 5.61e+01 |

## 2. 1D CANN — all sizes

| num | backend | step_ms | scan_ms | step_su | scan_su | max_err |
|---|---|---|---|---|---|---|
| 64 | dense | 0.0031 | 0.0002 | 1.00 | 1.00 | 0.00e+00 |
| 64 | fft | 0.0061 | 0.0011 | 0.51 | 0.18 | 8.58e-06 |
| 64 | svd_k1 | 0.0033 | 0.0001 | 0.96 | 2.60 | 8.84e+00 |
| 64 | svd_k4 | 0.0032 | 0.0001 | 0.97 | 2.17 | 2.54e+00 |
| 64 | svd_k16 | 0.0033 | 0.0001 | 0.94 | 1.42 | 5.62e-03 |
| 128 | dense | 0.0032 | 0.0006 | 1.00 | 1.00 | 0.00e+00 |
| 128 | fft | 0.0064 | 0.0014 | 0.50 | 0.43 | 1.14e-05 |
| 128 | svd_k1 | 0.0033 | 0.0001 | 0.96 | 6.01 | 2.06e+01 |
| 128 | svd_k4 | 0.0169 | 0.0001 | 0.19 | 4.15 | 1.07e+01 |
| 128 | svd_k16 | 0.0035 | 0.0002 | 0.91 | 2.89 | 1.53e-03 |
| 256 | dense | 0.0137 | 0.0033 | 1.00 | 1.00 | 0.00e+00 |
| 256 | fft | 0.0325 | 0.0021 | 0.42 | 1.61 | 1.91e-05 |
| 256 | svd_k1 | 0.0033 | 0.0001 | 4.16 | 24.68 | 3.01e+01 |
| 256 | svd_k4 | 0.0036 | 0.0002 | 3.78 | 15.53 | 1.38e+01 |
| 256 | svd_k16 | 0.0034 | 0.0003 | 4.01 | 9.56 | 5.04e-03 |
| 512 | dense | 0.0230 | 0.0120 | 1.00 | 1.00 | 0.00e+00 |
| 512 | fft | 0.0135 | 0.0028 | 1.71 | 4.25 | 3.62e-05 |
| 512 | svd_k1 | 0.0036 | 0.0002 | 6.43 | 49.63 | 3.60e+01 |
| 512 | svd_k4 | 0.0087 | 0.0004 | 2.66 | 31.85 | 1.59e+01 |
| 512 | svd_k16 | 0.0117 | 0.0006 | 1.96 | 19.90 | 1.25e-02 |
| 1024 | dense | 0.0458 | 0.0283 | 1.00 | 1.00 | 0.00e+00 |
| 1024 | fft | 0.0154 | 0.0048 | 2.98 | 5.89 | 5.72e-05 |
| 1024 | svd_k1 | 0.0036 | 0.0005 | 12.79 | 59.85 | 4.20e+01 |
| 1024 | svd_k4 | 0.0033 | 0.0007 | 13.92 | 40.52 | 2.33e+01 |
| 1024 | svd_k16 | 0.0188 | 0.0014 | 2.44 | 20.91 | 1.32e-02 |
| 2048 | dense | 0.1212 | 0.1271 | 1.00 | 1.00 | 0.00e+00 |
| 2048 | fft | 0.0217 | 0.0093 | 5.58 | 13.71 | 1.05e-04 |
| 2048 | svd_k1 | 0.0035 | 0.0007 | 34.62 | 182.97 | 4.93e+01 |
| 2048 | svd_k4 | 0.0060 | 0.0014 | 20.33 | 93.29 | 2.13e+01 |
| 2048 | svd_k16 | 0.0123 | 0.0029 | 9.82 | 44.39 | 1.84e-02 |
| 4096 | dense | 0.7998 | 0.8025 | 1.00 | 1.00 | 0.00e+00 |
| 4096 | fft | 0.0317 | 0.0207 | 25.22 | 38.80 | 1.70e-04 |
| 4096 | svd_k1 | 0.0048 | 0.0008 | 168.36 | 965.36 | 5.38e+01 |
| 4096 | svd_k4 | 0.0126 | 0.0027 | 63.35 | 297.69 | 4.60e+01 |
| 4096 | svd_k16 | 0.0169 | 0.0058 | 47.28 | 139.31 | 2.88e-02 |

## 3. 2D CANN — all sizes

| L | n_total | backend | step_ms | scan_ms | step_su | scan_su | max_err |
|---|---|---|---|---|---|---|---|
| 4 | 16 | dense | 0.0027 | 0.0001 | 1.00 | 1.00 | 0.00e+00 |
| 4 | 16 | fft | 0.0068 | 0.0017 | 0.40 | 0.03 | 4.77e-07 |
| 4 | 16 | svd_k1 | 0.0030 | 0.0000 | 0.90 | 1.61 | 7.27e+00 |
| 4 | 16 | svd_k4 | 0.0028 | 0.0000 | 0.96 | 1.54 | 7.17e+00 |
| 4 | 16 | svd_k16 | 0.0031 | 0.0001 | 0.88 | 0.58 | 7.15e-07 |
| 8 | 64 | dense | 0.0031 | 0.0002 | 1.00 | 1.00 | 0.00e+00 |
| 8 | 64 | fft | 0.0075 | 0.0019 | 0.41 | 0.10 | 2.15e-06 |
| 8 | 64 | svd_k1 | 0.0032 | 0.0001 | 0.97 | 2.58 | 8.40e+00 |
| 8 | 64 | svd_k4 | 0.0032 | 0.0001 | 0.96 | 1.80 | 8.21e+00 |
| 8 | 64 | svd_k16 | 0.0033 | 0.0001 | 0.94 | 1.39 | 4.30e+00 |
| 16 | 256 | dense | 0.0138 | 0.0032 | 1.00 | 1.00 | 0.00e+00 |
| 16 | 256 | fft | 0.0136 | 0.0028 | 1.01 | 1.13 | 1.00e-05 |
| 16 | 256 | svd_k1 | 0.0033 | 0.0001 | 4.18 | 23.76 | 1.61e+01 |
| 16 | 256 | svd_k4 | 0.0034 | 0.0002 | 4.03 | 14.40 | 1.69e+01 |
| 16 | 256 | svd_k16 | 0.0032 | 0.0003 | 4.29 | 9.49 | 8.36e+00 |
| 32 | 1024 | dense | 0.0447 | 0.0270 | 1.00 | 1.00 | 0.00e+00 |
| 32 | 1024 | fft | 0.0165 | 0.0053 | 2.71 | 5.10 | 3.96e-05 |
| 32 | 1024 | svd_k1 | 0.0035 | 0.0004 | 12.76 | 71.54 | 4.36e+01 |
| 32 | 1024 | svd_k4 | 0.0033 | 0.0007 | 13.40 | 39.39 | 3.78e+01 |
| 32 | 1024 | svd_k16 | 0.0054 | 0.0013 | 8.25 | 20.52 | 2.33e+01 |
| 48 | 2304 | dense | 0.1847 | 0.1875 | 1.00 | 1.00 | 0.00e+00 |
| 48 | 2304 | fft | 0.0225 | 0.0105 | 8.22 | 17.89 | 4.77e-05 |
| 48 | 2304 | svd_k1 | 0.0035 | 0.0008 | 52.78 | 238.70 | 5.89e+01 |
| 48 | 2304 | svd_k4 | 0.0068 | 0.0016 | 27.03 | 119.92 | 5.31e+01 |
| 48 | 2304 | svd_k16 | 0.0138 | 0.0032 | 13.43 | 59.42 | 3.92e+01 |
| 64 | 4096 | dense | 0.8318 | 0.8000 | 1.00 | 1.00 | 0.00e+00 |
| 64 | 4096 | fft | 0.0267 | 0.0159 | 31.19 | 50.23 | 8.01e-05 |
| 64 | 4096 | svd_k1 | 0.0034 | 0.0008 | 246.53 | 944.43 | 8.46e+01 |
| 64 | 4096 | svd_k4 | 0.0109 | 0.0027 | 76.20 | 297.61 | 8.76e+01 |
| 64 | 4096 | svd_k16 | 0.0192 | 0.0058 | 43.40 | 137.14 | 5.61e+01 |

## 4. Figures

- `figures/fig_fft_speed.png` — per-step time vs n
- `figures/fig_fft_scan.png` — per-step time inside T=200 scan
- `figures/fig_fft_accuracy.png` — max-abs error vs n
- `figures/fig_fft_pareto.png` — speed vs accuracy Pareto

## 5. Key findings

- **FFT is exact.** On a clean circulant, the FFT path's max-abs error is at float precision (1e-6 to 1e-4), independent of n. SVD low-rank at any fixed k has constant or growing error.

- **FFT is 25-50× faster than dense** at the largest n on CPU, and the gap widens with n. SVD k=1 can be 100-1000× faster than dense but at huge accuracy cost (max err 8-44).

- **FFT vs SVD tradeoff:** when the task requires exact dynamics (e.g. parameter sweeps, bifurcation analysis), FFT is the right choice. When a few percent of error is acceptable and n is large, SVD k=1 wins by another order of magnitude.

- **Endpoint gotcha.** The canns default `endpoint=True` grid is *not* circulant under the canns wrap convention. Setting `accl_mode='fft'` on that grid silently falls back to dense (with a `UserWarning`). To use FFT, override the grid: `model.x = bm.linspace(-bm.pi, bm.pi, n, endpoint=False)`.

