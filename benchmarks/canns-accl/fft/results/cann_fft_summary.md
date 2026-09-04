# FFT matvec benchmark — CPU vs GPU

Headline numbers (largest n per platform) for the recurrent matvec under three backends: `dense` (cuBLAS sgemv / numpy matmul), `svd_k{k}` (rank-k truncated SVD approximation), and `fft` (exact circulant matvec via cuFFT / numpy FFT).

## CPU headline

| model | n | backend | step_ms | scan_ms | speedup_step | speedup_scan |
|---|---|---|---|---|---|---|
| cann1d | 4096 | dense | 0.7998 | 0.8025 | 1.00 | 1.00 |
| cann1d | 4096 | fft | 0.0317 | 0.0207 | 25.22 | 38.80 |
| cann1d | 4096 | svd_k1 | 0.0048 | 0.0008 | 168.36 | 965.36 |
| cann1d | 4096 | svd_k4 | 0.0126 | 0.0027 | 63.35 | 297.69 |
| cann1d | 4096 | svd_k16 | 0.0169 | 0.0058 | 47.28 | 139.31 |
| cann2d | 64 | dense | 0.8318 | 0.8000 | 1.00 | 1.00 |
| cann2d | 64 | fft | 0.0267 | 0.0159 | 31.19 | 50.23 |
| cann2d | 64 | svd_k1 | 0.0034 | 0.0008 | 246.53 | 944.43 |
| cann2d | 64 | svd_k4 | 0.0109 | 0.0027 | 76.20 | 297.61 |
| cann2d | 64 | svd_k16 | 0.0192 | 0.0058 | 43.40 | 137.14 |

## GPU headline

| model | n | backend | step_ms | scan_ms | speedup_step | speedup_scan |
|---|---|---|---|---|---|---|
| cann1d | 4096 | dense | 0.2265 | 0.0525 | 1.00 | 1.00 |
| cann1d | 4096 | fft | 0.2061 | 0.0268 | 1.10 | 1.96 |
| cann1d | 4096 | svd_k1 | 0.0944 | 0.0104 | 2.40 | 5.03 |
| cann1d | 4096 | svd_k4 | 0.1031 | 0.0124 | 2.20 | 4.23 |
| cann1d | 4096 | svd_k16 | 0.1193 | 0.0186 | 1.90 | 2.83 |
| cann2d | 64 | dense | 0.1976 | 0.0539 | 1.00 | 1.00 |
| cann2d | 64 | fft | 0.1709 | 0.0332 | 1.16 | 1.62 |
| cann2d | 64 | svd_k1 | 0.1425 | 0.0104 | 1.39 | 5.18 |
| cann2d | 64 | svd_k4 | 0.1218 | 0.0153 | 1.62 | 3.52 |
| cann2d | 64 | svd_k16 | 0.1748 | 0.0184 | 1.13 | 2.93 |

## Take-aways

- **CPU: FFT is 25-50× faster than dense** and is *exact*. SVD k=1 is 100-1000× faster but has 30-50 mrad position error. FFT is the right choice for the high-fidelity regime on CPU.

- **GPU: FFT is barely faster than dense on per-step** (1.1-1.2× at n=4096) because cuBLAS sgemv is already very well-optimized for the matmul shape. FFT only wins meaningfully on the *scan* (rollout) path (1.6-2.0× at n=4096), where XLA can fuse FFT and avoid per-step launch overhead.

- **Accuracy on GPU** is ~1e-2 (not 1e-5) because cuBLAS uses TF32 (10-bit mantissa) by default on Ampere. Disable TF32 (`torch.backends.cuda.matmul.allow_tf32 = False` or the JAX equivalent) if you need full FP32 precision in the dense baseline.

- **Endpoint gotcha.** The canns default `endpoint=True` grid is not circulant; `accl_mode='fft'` on that grid silently falls back to dense with a `UserWarning`. To use FFT, set `model.x = bm.linspace(-bm.pi, bm.pi, n, endpoint=False)` and rebuild `model.conn_mat = model.make_conn()`.


## Figures

- `figures/fig_fft_speed_cpu.png` — per-step time vs n (CPU)
- `figures/fig_fft_speed_gpu.png` — per-step time vs n (GPU)
- `figures/fig_fft_accuracy_cpu.png` — max-abs error vs n (CPU)
- `figures/fig_fft_pareto_cpu.png` — speed vs accuracy (CPU)
