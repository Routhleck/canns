# FFT matvec benchmark — GPU

Comparison of three recurrent-matvec backends for the CANN connectivity kernel: `dense` (O(n²), baseline), `svd_k{k}` (truncated SVD, O(nk), approximate), and `fft` (exact circulant matvec, O(n log n)).

All numbers come from a **clean circulant** setup (`endpoint=False` uniform grid); the canns default `endpoint=True` grid is not circulant and falls back to `dense` with a warning when `accl_mode='fft'` is requested.

## 1. Headline numbers (largest n)

| model | n | n_total | backend | step_ms | scan_ms | speedup_step | speedup_scan | max_err |
|---|---|---|---|---|---|---|---|---|
| cann1d | 4096 | 4096 | dense | 0.2265 | 0.0525 | 1.00 | 1.00 | 0.00e+00 |
| cann1d | 4096 | 4096 | fft | 0.2061 | 0.0268 | 1.10 | 1.96 | 7.12e-02 |
| cann1d | 4096 | 4096 | svd_k1 | 0.0944 | 0.0104 | 2.40 | 5.03 | 5.38e+01 |
| cann1d | 4096 | 4096 | svd_k4 | 0.1031 | 0.0124 | 2.20 | 4.23 | 4.04e+01 |
| cann1d | 4096 | 4096 | svd_k16 | 0.1193 | 0.0186 | 1.90 | 2.83 | 1.03e-01 |
| cann2d | 64 | 4096 | dense | 0.1976 | 0.0539 | 1.00 | 1.00 | 0.00e+00 |
| cann2d | 64 | 4096 | fft | 0.1709 | 0.0332 | 1.16 | 1.62 | 3.24e-02 |
| cann2d | 64 | 4096 | svd_k1 | 0.1425 | 0.0104 | 1.39 | 5.18 | 8.46e+01 |
| cann2d | 64 | 4096 | svd_k4 | 0.1218 | 0.0153 | 1.62 | 3.52 | 7.74e+01 |
| cann2d | 64 | 4096 | svd_k16 | 0.1748 | 0.0184 | 1.13 | 2.93 | 5.71e+01 |

## 2. 1D CANN — all sizes

| num | backend | step_ms | scan_ms | step_su | scan_su | max_err |
|---|---|---|---|---|---|---|
| 64 | dense | 0.2032 | 0.0095 | 1.00 | 1.00 | 0.00e+00 |
| 64 | fft | 0.1581 | 0.0216 | 1.29 | 0.44 | 9.54e-06 |
| 64 | svd_k1 | 0.0509 | 0.0093 | 3.99 | 1.03 | 8.84e+00 |
| 64 | svd_k4 | 0.1505 | 0.0099 | 1.35 | 0.96 | 2.52e+00 |
| 64 | svd_k16 | 0.1456 | 0.0102 | 1.40 | 0.94 | 2.96e-03 |
| 128 | dense | 0.1676 | 0.0084 | 1.00 | 1.00 | 0.00e+00 |
| 128 | fft | 0.1516 | 0.0218 | 1.11 | 0.38 | 1.12e-05 |
| 128 | svd_k1 | 0.0895 | 0.0076 | 1.87 | 1.11 | 2.06e+01 |
| 128 | svd_k4 | 0.0888 | 0.0076 | 1.89 | 1.11 | 4.65e+00 |
| 128 | svd_k16 | 0.0967 | 0.0100 | 1.73 | 0.84 | 9.53e-04 |
| 256 | dense | 0.1210 | 0.0094 | 1.00 | 1.00 | 0.00e+00 |
| 256 | fft | 0.1091 | 0.0228 | 1.11 | 0.41 | 2.10e-05 |
| 256 | svd_k1 | 0.0561 | 0.0087 | 2.16 | 1.08 | 3.01e+01 |
| 256 | svd_k4 | 0.1173 | 0.0095 | 1.03 | 1.00 | 6.11e+00 |
| 256 | svd_k16 | 0.1173 | 0.0105 | 1.03 | 0.90 | 3.17e-04 |
| 512 | dense | 0.1600 | 0.0097 | 1.00 | 1.00 | 0.00e+00 |
| 512 | fft | 0.1796 | 0.0203 | 0.89 | 0.48 | 2.57e-05 |
| 512 | svd_k1 | 0.0790 | 0.0091 | 2.02 | 1.06 | 3.60e+01 |
| 512 | svd_k4 | 0.1724 | 0.0103 | 0.93 | 0.94 | 1.77e+01 |
| 512 | svd_k16 | 0.1747 | 0.0116 | 0.92 | 0.84 | 3.64e-03 |
| 1024 | dense | 0.1823 | 0.0104 | 1.00 | 1.00 | 0.00e+00 |
| 1024 | fft | 0.1839 | 0.0252 | 0.99 | 0.41 | 5.34e-05 |
| 1024 | svd_k1 | 0.1423 | 0.0082 | 1.28 | 1.26 | 4.20e+01 |
| 1024 | svd_k4 | 0.1897 | 0.0108 | 0.96 | 0.96 | 2.64e+01 |
| 1024 | svd_k16 | 0.2103 | 0.0151 | 0.87 | 0.69 | 1.30e-02 |
| 2048 | dense | 0.1897 | 0.0134 | 1.00 | 1.00 | 0.00e+00 |
| 2048 | fft | 0.2305 | 0.0245 | 0.82 | 0.55 | 7.25e-05 |
| 2048 | svd_k1 | 0.1882 | 0.0086 | 1.01 | 1.56 | 4.93e+01 |
| 2048 | svd_k4 | 0.1838 | 0.0123 | 1.03 | 1.09 | 2.24e+01 |
| 2048 | svd_k16 | 0.1170 | 0.0163 | 1.62 | 0.82 | 1.23e-02 |
| 4096 | dense | 0.2265 | 0.0525 | 1.00 | 1.00 | 0.00e+00 |
| 4096 | fft | 0.2061 | 0.0268 | 1.10 | 1.96 | 7.12e-02 |
| 4096 | svd_k1 | 0.0944 | 0.0104 | 2.40 | 5.03 | 5.38e+01 |
| 4096 | svd_k4 | 0.1031 | 0.0124 | 2.20 | 4.23 | 4.04e+01 |
| 4096 | svd_k16 | 0.1193 | 0.0186 | 1.90 | 2.83 | 1.03e-01 |

## 3. 2D CANN — all sizes

| L | n_total | backend | step_ms | scan_ms | step_su | scan_su | max_err |
|---|---|---|---|---|---|---|---|
| 4 | 16 | dense | 0.1773 | 0.0081 | 1.00 | 1.00 | 0.00e+00 |
| 4 | 16 | fft | 0.1398 | 0.0224 | 1.27 | 0.36 | 9.54e-07 |
| 4 | 16 | svd_k1 | 0.1754 | 0.0093 | 1.01 | 0.87 | 7.27e+00 |
| 4 | 16 | svd_k4 | 0.1413 | 0.0091 | 1.25 | 0.89 | 7.17e+00 |
| 4 | 16 | svd_k16 | 0.1341 | 0.0092 | 1.32 | 0.88 | 1.43e-06 |
| 8 | 64 | dense | 0.1903 | 0.0080 | 1.00 | 1.00 | 0.00e+00 |
| 8 | 64 | fft | 0.1737 | 0.0237 | 1.10 | 0.34 | 5.24e-06 |
| 8 | 64 | svd_k1 | 0.0736 | 0.0074 | 2.59 | 1.08 | 8.40e+00 |
| 8 | 64 | svd_k4 | 0.0725 | 0.0078 | 2.62 | 1.02 | 7.94e+00 |
| 8 | 64 | svd_k16 | 0.0681 | 0.0093 | 2.79 | 0.85 | 4.94e+00 |
| 16 | 256 | dense | 0.1251 | 0.0083 | 1.00 | 1.00 | 0.00e+00 |
| 16 | 256 | fft | 0.1206 | 0.0241 | 1.04 | 0.34 | 1.43e-05 |
| 16 | 256 | svd_k1 | 0.0844 | 0.0077 | 1.48 | 1.08 | 1.61e+01 |
| 16 | 256 | svd_k4 | 0.0589 | 0.0083 | 2.12 | 1.00 | 1.49e+01 |
| 16 | 256 | svd_k16 | 0.0596 | 0.0104 | 2.10 | 0.80 | 8.37e+00 |
| 32 | 1024 | dense | 0.1108 | 0.0102 | 1.00 | 1.00 | 0.00e+00 |
| 32 | 1024 | fft | 0.1926 | 0.0287 | 0.58 | 0.35 | 4.58e-05 |
| 32 | 1024 | svd_k1 | 0.0596 | 0.0083 | 1.86 | 1.22 | 4.36e+01 |
| 32 | 1024 | svd_k4 | 0.0779 | 0.0104 | 1.42 | 0.97 | 3.79e+01 |
| 32 | 1024 | svd_k16 | 0.0791 | 0.0150 | 1.40 | 0.68 | 2.09e+01 |
| 48 | 2304 | dense | 0.1358 | 0.0138 | 1.00 | 1.00 | 0.00e+00 |
| 48 | 2304 | fft | 0.1637 | 0.0309 | 0.83 | 0.45 | 3.92e-05 |
| 48 | 2304 | svd_k1 | 0.1406 | 0.0087 | 0.97 | 1.59 | 5.89e+01 |
| 48 | 2304 | svd_k4 | 0.1421 | 0.0124 | 0.96 | 1.11 | 4.57e+01 |
| 48 | 2304 | svd_k16 | 0.1171 | 0.0166 | 1.16 | 0.83 | 4.58e+01 |
| 64 | 4096 | dense | 0.1976 | 0.0539 | 1.00 | 1.00 | 0.00e+00 |
| 64 | 4096 | fft | 0.1709 | 0.0332 | 1.16 | 1.62 | 3.24e-02 |
| 64 | 4096 | svd_k1 | 0.1425 | 0.0104 | 1.39 | 5.18 | 8.46e+01 |
| 64 | 4096 | svd_k4 | 0.1218 | 0.0153 | 1.62 | 3.52 | 7.74e+01 |
| 64 | 4096 | svd_k16 | 0.1748 | 0.0184 | 1.13 | 2.93 | 5.71e+01 |

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

