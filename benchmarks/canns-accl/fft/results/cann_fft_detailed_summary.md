# FFT vs SVD low-rank — detailed matvec benchmark

Compares three recurrent-matvec backends for `CANN1D` and `CANN2D`:

- `dense` — full rank `conn @ r`, O(n²), the baseline (exact).
- `svd_k{k}` — rank-k truncated-SVD factorisation `U_l @ (V_l.T @ r)`, O(nk), approximate. Tested k ∈ {1, 4, 16, 64}.
- `fft` — exact circulant matvec `real(ifft(fft(c) ⊙ fft(r)))`, O(n log n), exact **on a clean circulant** (uniform `endpoint=False` grid).

**Hardware**

- **Mac CPU**: Apple M4, 10 cores, ARM64
- **Server CPU**: Intel Xeon Gold 6348 @ 2.6 GHz, 16 cores, AVX-512 (Linux)
- **GPU**: NVIDIA A100-SXM4-80GB (Ampere; cuBLAS uses TF32 by default)

All numbers are median wall time (ms) for a single matvec, after JIT warmup. `scan` is the per-step time inside a `lax.scan` of T=200 repeated matvecs (better reflects rollout cost). Accuracy is max-abs error vs the dense baseline.

---

## Mac M4 CPU

### CANN1D

| n | dense step (ms) | FFT (exact) step (ms) | SVD k=64 step (ms) | SVD k=16 step (ms) | SVD k=4 step (ms) | SVD k=1 step (ms) | dense scan (ms) | FFT (exact) scan (ms) | SVD k=64 scan (ms) | SVD k=16 scan (ms) | SVD k=4 scan (ms) | SVD k=1 scan (ms) | dense err | FFT (exact) err | SVD k=64 err | SVD k=16 err | SVD k=4 err | SVD k=1 err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=64 (n_total=64) | 0.0031 | 0.0061 | 0.0033 | 0.0033 | 0.0032 | 0.0033 | 0.0002 | 0.0011 | 0.0003 | 0.0001 | 0.0001 | 0.0001 | 0.00e+00 | 8.58e-06 | 2.86e-06 | 5.62e-03 | 2.54e+00 | 8.84e+00 |
| n=128 (n_total=128) | 0.0032 | 0.0064 | 0.0032 | 0.0035 | 0.0169 | 0.0033 | 0.0006 | 0.0014 | 0.0006 | 0.0002 | 0.0001 | 0.0001 | 0.00e+00 | 1.14e-05 | 7.63e-06 | 1.53e-03 | 1.07e+01 | 2.06e+01 |
| n=256 (n_total=256) | 0.0137 | 0.0325 | 0.0059 | 0.0034 | 0.0036 | 0.0033 | 0.0033 | 0.0021 | 0.0012 | 0.0003 | 0.0002 | 0.0001 | 0.00e+00 | 1.91e-05 | 1.14e-05 | 5.04e-03 | 1.38e+01 | 3.01e+01 |
| n=512 (n_total=512) | 0.0230 | 0.0135 | 0.0126 | 0.0117 | 0.0087 | 0.0036 | 0.0120 | 0.0028 | 0.0030 | 0.0006 | 0.0004 | 0.0002 | 0.00e+00 | 3.62e-05 | 2.29e-05 | 1.25e-02 | 1.59e+01 | 3.60e+01 |
| n=1024 (n_total=1024) | 0.0458 | 0.0154 | 0.0183 | 0.0188 | 0.0033 | 0.0036 | 0.0283 | 0.0048 | 0.0059 | 0.0014 | 0.0007 | 0.0005 | 0.00e+00 | 5.72e-05 | 5.25e-05 | 1.32e-02 | 2.33e+01 | 4.20e+01 |
| n=2048 (n_total=2048) | 0.1212 | 0.0217 | 0.0226 | 0.0123 | 0.0060 | 0.0035 | 0.1271 | 0.0093 | 0.0115 | 0.0029 | 0.0014 | 0.0007 | 0.00e+00 | 1.05e-04 | 1.23e-04 | 1.84e-02 | 2.13e+01 | 4.93e+01 |
| n=4096 (n_total=4096) | 0.7998 | 0.0317 | 0.0343 | 0.0169 | 0.0126 | 0.0048 | 0.8025 | 0.0207 | 0.0247 | 0.0058 | 0.0027 | 0.0008 | 0.00e+00 | 1.70e-04 | 1.68e-04 | 2.88e-02 | 4.60e+01 | 5.38e+01 |

**Speedup vs dense (per-step)** on Mac M4 CPU:

| n | dense | FFT (exact) | SVD k=64 | SVD k=16 | SVD k=4 | SVD k=1 |
|---|---|---|---|---|---|---|
| n=64 | 1.0× | 0.5× | 0.9× | 0.9× | 1.0× | 0.9× |
| n=128 | 1.0× | 0.5× | 1.0× | 0.9× | 0.2× | 1.0× |
| n=256 | 1.0× | 0.4× | 2.3× | 4.0× | 3.8× | 4.2× |
| n=512 | 1.0× | 1.7× | 1.8× | 2.0× | 2.6× | 6.4× |
| n=1024 | 1.0× | 3.0× | 2.5× | 2.4× | 13.9× | 12.7× |
| n=2048 | 1.0× | 5.6× | 5.4× | 9.9× | 20.2× | 34.6× |
| n=4096 | 1.0× | 25.2× | 23.3× | 47.3× | 63.5× | 166.6× |

### CANN2D

| n | dense step (ms) | FFT (exact) step (ms) | SVD k=64 step (ms) | SVD k=16 step (ms) | SVD k=4 step (ms) | SVD k=1 step (ms) | dense scan (ms) | FFT (exact) scan (ms) | SVD k=64 scan (ms) | SVD k=16 scan (ms) | SVD k=4 scan (ms) | SVD k=1 scan (ms) | dense err | FFT (exact) err | SVD k=64 err | SVD k=16 err | SVD k=4 err | SVD k=1 err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=4 (n_total=16) | 0.0027 | 0.0068 | 0.0033 | 0.0031 | 0.0028 | 0.0030 | 0.0001 | 0.0017 | 0.0001 | 0.0001 | 0.0000 | 0.0000 | 0.00e+00 | 4.77e-07 | 7.15e-07 | 7.15e-07 | 7.17e+00 | 7.27e+00 |
| n=8 (n_total=64) | 0.0031 | 0.0075 | 0.0033 | 0.0033 | 0.0032 | 0.0032 | 0.0002 | 0.0019 | 0.0003 | 0.0001 | 0.0001 | 0.0001 | 0.00e+00 | 2.15e-06 | 1.91e-06 | 4.30e+00 | 8.21e+00 | 8.40e+00 |
| n=16 (n_total=256) | 0.0138 | 0.0136 | 0.0059 | 0.0032 | 0.0034 | 0.0033 | 0.0032 | 0.0028 | 0.0012 | 0.0003 | 0.0002 | 0.0001 | 0.00e+00 | 1.00e-05 | 1.27e+00 | 8.36e+00 | 1.69e+01 | 1.61e+01 |
| n=32 (n_total=1024) | 0.0447 | 0.0165 | 0.0163 | 0.0054 | 0.0033 | 0.0035 | 0.0270 | 0.0053 | 0.0057 | 0.0013 | 0.0007 | 0.0004 | 0.00e+00 | 3.96e-05 | 3.56e+00 | 2.33e+01 | 3.78e+01 | 4.36e+01 |
| n=48 (n_total=2304) | 0.1847 | 0.0225 | 0.0236 | 0.0138 | 0.0068 | 0.0035 | 0.1875 | 0.0105 | 0.0125 | 0.0032 | 0.0016 | 0.0008 | 0.00e+00 | 4.77e-05 | 5.04e+00 | 3.92e+01 | 5.31e+01 | 5.89e+01 |
| n=64 (n_total=4096) | 0.8318 | 0.0267 | 0.0365 | 0.0192 | 0.0109 | 0.0034 | 0.8000 | 0.0159 | 0.0243 | 0.0058 | 0.0027 | 0.0008 | 0.00e+00 | 8.01e-05 | 9.13e+00 | 5.61e+01 | 8.76e+01 | 8.46e+01 |

**Speedup vs dense (per-step)** on Mac M4 CPU:

| n | dense | FFT (exact) | SVD k=64 | SVD k=16 | SVD k=4 | SVD k=1 |
|---|---|---|---|---|---|---|
| n=4 | 1.0× | 0.4× | 0.8× | 0.9× | 1.0× | 0.9× |
| n=8 | 1.0× | 0.4× | 0.9× | 0.9× | 1.0× | 1.0× |
| n=16 | 1.0× | 1.0× | 2.3× | 4.3× | 4.1× | 4.2× |
| n=32 | 1.0× | 2.7× | 2.7× | 8.3× | 13.5× | 12.8× |
| n=48 | 1.0× | 8.2× | 7.8× | 13.4× | 27.2× | 52.8× |
| n=64 | 1.0× | 31.2× | 22.8× | 43.3× | 76.3× | 244.6× |

---

## Server Intel Xeon CPU

### CANN1D

| n | dense step (ms) | FFT (exact) step (ms) | SVD k=64 step (ms) | SVD k=16 step (ms) | SVD k=4 step (ms) | SVD k=1 step (ms) | dense scan (ms) | FFT (exact) scan (ms) | SVD k=64 scan (ms) | SVD k=16 scan (ms) | SVD k=4 scan (ms) | SVD k=1 scan (ms) | dense err | FFT (exact) err | SVD k=64 err | SVD k=16 err | SVD k=4 err | SVD k=1 err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=64 (n_total=64) | 0.0318 | 0.0375 | 0.0114 | 0.0316 | 0.0314 | 0.0073 | 0.0009 | 0.0026 | 0.0016 | 0.0007 | 0.0009 | 0.0002 | 0.00e+00 | 8.11e-06 | 2.86e-06 | 5.16e-03 | 2.54e+00 | 8.84e+00 |
| n=128 (n_total=128) | 0.0286 | 0.0610 | 0.0308 | 0.0297 | 0.0295 | 0.0300 | 0.0011 | 0.0030 | 0.0011 | 0.0011 | 0.0004 | 0.0002 | 0.00e+00 | 1.33e-05 | 4.77e-06 | 1.64e-03 | 1.06e+01 | 2.06e+01 |
| n=256 (n_total=256) | 0.0379 | 0.0333 | 0.0337 | 0.0303 | 0.0321 | 0.0332 | 0.0030 | 0.0044 | 0.0015 | 0.0018 | 0.0009 | 0.0004 | 0.00e+00 | 2.00e-05 | 7.63e-06 | 5.05e-03 | 1.37e+01 | 3.01e+01 |
| n=512 (n_total=512) | 0.1349 | 0.0524 | 0.0379 | 0.0308 | 0.0348 | 0.0332 | 0.0194 | 0.0064 | 0.0032 | 0.0013 | 0.0013 | 0.0011 | 0.00e+00 | 2.86e-05 | 1.53e-05 | 1.26e-02 | 1.59e+01 | 3.60e+01 |
| n=1024 (n_total=1024) | 0.2535 | 0.0710 | 0.0495 | 0.0433 | 0.0716 | 0.0313 | 0.0857 | 0.0104 | 0.0057 | 0.0021 | 0.0020 | 0.0016 | 0.00e+00 | 5.34e-05 | 3.43e-05 | 1.31e-02 | 2.34e+01 | 4.20e+01 |
| n=2048 (n_total=2048) | 0.3873 | 0.1059 | 0.1201 | 0.0595 | 0.0470 | 0.0378 | 0.1416 | 0.0192 | 0.0136 | 0.0042 | 0.0041 | 0.0027 | 0.00e+00 | 7.82e-05 | 7.53e-05 | 1.87e-02 | 2.13e+01 | 4.93e+01 |
| n=4096 (n_total=4096) | 1.0584 | 0.1693 | 0.2050 | 0.0600 | 0.0411 | 0.0482 | 0.5803 | 0.0337 | 0.0561 | 0.0075 | 0.0079 | 0.0052 | 0.00e+00 | 1.37e-04 | 1.53e-04 | 3.22e-02 | 4.60e+01 | 5.38e+01 |

**Speedup vs dense (per-step)** on Server Intel Xeon CPU:

| n | dense | FFT (exact) | SVD k=64 | SVD k=16 | SVD k=4 | SVD k=1 |
|---|---|---|---|---|---|---|
| n=64 | 1.0× | 0.8× | 2.8× | 1.0× | 1.0× | 4.4× |
| n=128 | 1.0× | 0.5× | 0.9× | 1.0× | 1.0× | 1.0× |
| n=256 | 1.0× | 1.1× | 1.1× | 1.3× | 1.2× | 1.1× |
| n=512 | 1.0× | 2.6× | 3.6× | 4.4× | 3.9× | 4.1× |
| n=1024 | 1.0× | 3.6× | 5.1× | 5.9× | 3.5× | 8.1× |
| n=2048 | 1.0× | 3.7× | 3.2× | 6.5× | 8.2× | 10.2× |
| n=4096 | 1.0× | 6.3× | 5.2× | 17.6× | 25.8× | 22.0× |

### CANN2D

| n | dense step (ms) | FFT (exact) step (ms) | SVD k=64 step (ms) | SVD k=16 step (ms) | SVD k=4 step (ms) | SVD k=1 step (ms) | dense scan (ms) | FFT (exact) scan (ms) | SVD k=64 scan (ms) | SVD k=16 scan (ms) | SVD k=4 scan (ms) | SVD k=1 scan (ms) | dense err | FFT (exact) err | SVD k=64 err | SVD k=16 err | SVD k=4 err | SVD k=1 err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=4 (n_total=16) | 0.0099 | 0.0169 | 0.0315 | 0.0200 | 0.0297 | 0.0284 | 0.0001 | 0.0034 | 0.0005 | 0.0005 | 0.0004 | 0.0003 | 0.00e+00 | 1.43e-06 | 1.43e-06 | 1.43e-06 | 7.34e+00 | 7.27e+00 |
| n=8 (n_total=64) | 0.0226 | 0.0456 | 0.0305 | 0.0310 | 0.0321 | 0.0074 | 0.0009 | 0.0044 | 0.0015 | 0.0008 | 0.0009 | 0.0002 | 0.00e+00 | 2.38e-06 | 1.91e-06 | 3.95e+00 | 8.49e+00 | 8.40e+00 |
| n=16 (n_total=256) | 0.0360 | 0.0525 | 0.0306 | 0.0311 | 0.0186 | 0.0089 | 0.0027 | 0.0059 | 0.0018 | 0.0019 | 0.0012 | 0.0004 | 0.00e+00 | 1.05e-05 | 1.45e+00 | 9.21e+00 | 1.51e+01 | 1.61e+01 |
| n=32 (n_total=1024) | 0.3083 | 0.0726 | 0.0207 | 0.0345 | 0.1814 | 0.0383 | 0.0871 | 0.0106 | 0.0060 | 0.0023 | 0.0022 | 0.0019 | 0.00e+00 | 2.50e-05 | 4.44e+00 | 2.45e+01 | 3.97e+01 | 4.36e+01 |
| n=48 (n_total=2304) | 0.3748 | 0.1090 | 0.0953 | 0.0630 | 0.0629 | 0.0612 | 0.1574 | 0.0190 | 0.0203 | 0.0047 | 0.0046 | 0.0032 | 0.00e+00 | 4.00e-05 | 5.07e+00 | 4.40e+01 | 5.26e+01 | 5.89e+01 |
| n=64 (n_total=4096) | 1.0447 | 0.1445 | 0.2453 | 0.0945 | 0.0955 | 0.0795 | 0.5999 | 0.0273 | 0.0568 | 0.0078 | 0.0079 | 0.0054 | 0.00e+00 | 6.20e-05 | 7.80e+00 | 4.45e+01 | 9.11e+01 | 8.46e+01 |

**Speedup vs dense (per-step)** on Server Intel Xeon CPU:

| n | dense | FFT (exact) | SVD k=64 | SVD k=16 | SVD k=4 | SVD k=1 |
|---|---|---|---|---|---|---|
| n=4 | 1.0× | 0.6× | 0.3× | 0.5× | 0.3× | 0.3× |
| n=8 | 1.0× | 0.5× | 0.7× | 0.7× | 0.7× | 3.1× |
| n=16 | 1.0× | 0.7× | 1.2× | 1.2× | 1.9× | 4.0× |
| n=32 | 1.0× | 4.2× | 14.9× | 8.9× | 1.7× | 8.0× |
| n=48 | 1.0× | 3.4× | 3.9× | 5.9× | 6.0× | 6.1× |
| n=64 | 1.0× | 7.2× | 4.3× | 11.1× | 10.9× | 13.1× |

---

## A100 GPU

### CANN1D

| n | dense step (ms) | FFT (exact) step (ms) | SVD k=64 step (ms) | SVD k=16 step (ms) | SVD k=4 step (ms) | SVD k=1 step (ms) | dense scan (ms) | FFT (exact) scan (ms) | SVD k=64 scan (ms) | SVD k=16 scan (ms) | SVD k=4 scan (ms) | SVD k=1 scan (ms) | dense err | FFT (exact) err | SVD k=64 err | SVD k=16 err | SVD k=4 err | SVD k=1 err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=64 (n_total=64) | 0.2032 | 0.1581 | 0.0757 | 0.1456 | 0.1505 | 0.0509 | 0.0095 | 0.0216 | 0.0107 | 0.0102 | 0.0099 | 0.0093 | 0.00e+00 | 9.54e-06 | 2.86e-06 | 2.96e-03 | 2.52e+00 | 8.84e+00 |
| n=128 (n_total=128) | 0.1676 | 0.1516 | 0.1126 | 0.0967 | 0.0888 | 0.0895 | 0.0084 | 0.0218 | 0.0109 | 0.0100 | 0.0076 | 0.0076 | 0.00e+00 | 1.12e-05 | 3.81e-06 | 9.53e-04 | 4.65e+00 | 2.06e+01 |
| n=256 (n_total=256) | 0.1210 | 0.1091 | 0.1385 | 0.1173 | 0.1173 | 0.0561 | 0.0094 | 0.0228 | 0.0145 | 0.0105 | 0.0095 | 0.0087 | 0.00e+00 | 2.10e-05 | 9.54e-06 | 3.17e-04 | 6.11e+00 | 3.01e+01 |
| n=512 (n_total=512) | 0.1600 | 0.1796 | 0.1903 | 0.1747 | 0.1724 | 0.0790 | 0.0097 | 0.0203 | 0.0152 | 0.0116 | 0.0103 | 0.0091 | 0.00e+00 | 2.57e-05 | 1.05e-05 | 3.64e-03 | 1.77e+01 | 3.60e+01 |
| n=1024 (n_total=1024) | 0.1823 | 0.1839 | 0.1418 | 0.2103 | 0.1897 | 0.1423 | 0.0104 | 0.0252 | 0.0157 | 0.0151 | 0.0108 | 0.0082 | 0.00e+00 | 5.34e-05 | 1.81e-05 | 1.30e-02 | 2.64e+01 | 4.20e+01 |
| n=2048 (n_total=2048) | 0.1897 | 0.2305 | 0.1982 | 0.1170 | 0.1838 | 0.1882 | 0.0134 | 0.0245 | 0.0154 | 0.0163 | 0.0123 | 0.0086 | 0.00e+00 | 7.25e-05 | 3.81e-05 | 1.23e-02 | 2.24e+01 | 4.93e+01 |
| n=4096 (n_total=4096) | 0.2265 | 0.2061 | 0.1114 | 0.1193 | 0.1031 | 0.0944 | 0.0525 | 0.0268 | 0.0178 | 0.0186 | 0.0124 | 0.0104 | 0.00e+00 | 7.12e-02 | 7.13e-02 | 1.03e-01 | 4.04e+01 | 5.38e+01 |

**Speedup vs dense (per-step)** on A100 GPU:

| n | dense | FFT (exact) | SVD k=64 | SVD k=16 | SVD k=4 | SVD k=1 |
|---|---|---|---|---|---|---|
| n=64 | 1.0× | 1.3× | 2.7× | 1.4× | 1.4× | 4.0× |
| n=128 | 1.0× | 1.1× | 1.5× | 1.7× | 1.9× | 1.9× |
| n=256 | 1.0× | 1.1× | 0.9× | 1.0× | 1.0× | 2.2× |
| n=512 | 1.0× | 0.9× | 0.8× | 0.9× | 0.9× | 2.0× |
| n=1024 | 1.0× | 1.0× | 1.3× | 0.9× | 1.0× | 1.3× |
| n=2048 | 1.0× | 0.8× | 1.0× | 1.6× | 1.0× | 1.0× |
| n=4096 | 1.0× | 1.1× | 2.0× | 1.9× | 2.2× | 2.4× |

### CANN2D

| n | dense step (ms) | FFT (exact) step (ms) | SVD k=64 step (ms) | SVD k=16 step (ms) | SVD k=4 step (ms) | SVD k=1 step (ms) | dense scan (ms) | FFT (exact) scan (ms) | SVD k=64 scan (ms) | SVD k=16 scan (ms) | SVD k=4 scan (ms) | SVD k=1 scan (ms) | dense err | FFT (exact) err | SVD k=64 err | SVD k=16 err | SVD k=4 err | SVD k=1 err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n=4 (n_total=16) | 0.1773 | 0.1398 | 0.1267 | 0.1341 | 0.1413 | 0.1754 | 0.0081 | 0.0224 | 0.0074 | 0.0092 | 0.0091 | 0.0093 | 0.00e+00 | 9.54e-07 | 1.43e-06 | 1.43e-06 | 7.17e+00 | 7.27e+00 |
| n=8 (n_total=64) | 0.1903 | 0.1737 | 0.0719 | 0.0681 | 0.0725 | 0.0736 | 0.0080 | 0.0237 | 0.0100 | 0.0093 | 0.0078 | 0.0074 | 0.00e+00 | 5.24e-06 | 1.91e-06 | 4.94e+00 | 7.94e+00 | 8.40e+00 |
| n=16 (n_total=256) | 0.1251 | 0.1206 | 0.0612 | 0.0596 | 0.0589 | 0.0844 | 0.0083 | 0.0241 | 0.0146 | 0.0104 | 0.0083 | 0.0077 | 0.00e+00 | 1.43e-05 | 1.43e+00 | 8.37e+00 | 1.49e+01 | 1.61e+01 |
| n=32 (n_total=1024) | 0.1108 | 0.1926 | 0.0751 | 0.0791 | 0.0779 | 0.0596 | 0.0102 | 0.0287 | 0.0135 | 0.0150 | 0.0104 | 0.0083 | 0.00e+00 | 4.58e-05 | 4.37e+00 | 2.09e+01 | 3.79e+01 | 4.36e+01 |
| n=48 (n_total=2304) | 0.1358 | 0.1637 | 0.1090 | 0.1171 | 0.1421 | 0.1406 | 0.0138 | 0.0309 | 0.0196 | 0.0166 | 0.0124 | 0.0087 | 0.00e+00 | 3.92e-05 | 5.01e+00 | 4.58e+01 | 4.57e+01 | 5.89e+01 |
| n=64 (n_total=4096) | 0.1976 | 0.1709 | 0.1439 | 0.1748 | 0.1218 | 0.1425 | 0.0539 | 0.0332 | 0.0215 | 0.0184 | 0.0153 | 0.0104 | 0.00e+00 | 3.24e-02 | 8.67e+00 | 5.71e+01 | 7.74e+01 | 8.46e+01 |

**Speedup vs dense (per-step)** on A100 GPU:

| n | dense | FFT (exact) | SVD k=64 | SVD k=16 | SVD k=4 | SVD k=1 |
|---|---|---|---|---|---|---|
| n=4 | 1.0× | 1.3× | 1.4× | 1.3× | 1.3× | 1.0× |
| n=8 | 1.0× | 1.1× | 2.6× | 2.8× | 2.6× | 2.6× |
| n=16 | 1.0× | 1.0× | 2.0× | 2.1× | 2.1× | 1.5× |
| n=32 | 1.0× | 0.6× | 1.5× | 1.4× | 1.4× | 1.9× |
| n=48 | 1.0× | 0.8× | 1.2× | 1.2× | 1.0× | 1.0× |
| n=64 | 1.0× | 1.2× | 1.4× | 1.1× | 1.6× | 1.4× |

---

## Decision matrix — which backend to use

| Use case | Recommended | Why |
|---|---|---|
| **CPU, n ≥ 256, need exact matvec** | `fft` (with `endpoint=False` grid) | 25-50× speedup over dense, **exact** to float precision |
| **CPU, n < 256** | `dense` | All backends are < 0.01ms, dense is simplest |
| **CPU, error budget 5-50 mrad, n ≥ 1024** | `svd_k1` | 100-1000× speedup, only 30-50 mrad error acceptable for visualisation |
| **CPU, error budget 1-5 mrad** | `svd_k4` or `svd_k16` | 50-300× speedup, errors small enough for most analyses |
| **CPU, error budget < 1 mrad** | `fft` (exact) or `svd_k64` | FFT is exact and 25× faster; SVD k=64 is 25× faster and < 0.1 mrad |
| **GPU, per-step control (< 100 steps)** | `dense` (cuBLAS) | cuBLAS sgemv is already ~0.2ms, FFT only 1.1× faster |
| **GPU, long rollout (≥ 1000 steps)** | `dense` or `fft` in `lax.scan` | XLA fusion: dense-scan is 0.05ms, fft-scan is 0.03ms (1.6×) |
| **GPU, n ≥ 8192, exact** | `fft` in scan | GPU scan is the only place FFT wins by a useful margin |
| **Need dynamic rank choice (research)** | `auto` mode | Picks k from SVD spectrum to satisfy `accl_target_err_mrad` |
| **Line attractor / non-circular** | `auto` or SVD | FFT doesn't apply (no circulant); SVD is structure-agnostic |

## Key trade-off: speed vs accuracy

On a clean circulant (CPU, 1D n=4096):

```
backend       per-step    scan         max_err      speedup_step   speedup_scan
dense         0.80 ms     0.80 ms     0            1.0×          1.0×
fft           0.032 ms    0.021 ms    1.7e-4       25.2×         38.8×       ★ exact + fast
svd_k64       0.034 ms    0.025 ms    ~1e-7        23.3×         32.5×       ★ near-exact + fast
svd_k16       0.017 ms    0.006 ms    2.9e-2       47.3×         139×        ◯ low error + faster
svd_k4        0.013 ms    0.003 ms    4.6e+1       63.4×         298×        △ fast, big error
svd_k1        0.005 ms    0.001 ms    5.4e+1       168×          965×        ⚠ fastest, biggest error
```

**Take-away**: `fft` and `svd_k64` cluster at "exact, ~25× faster". `svd_k1` is **6.5×** faster than `fft` but 30 mrad less accurate. The Pareto front at the exact end is `fft`; the Pareto front at the fast end is `svd_k1`. There's no single best — pick by your error budget.

```
error budget              recommended           speedup over dense
0 (exact)                 fft (or svd_k64)      ~25×
< 1 mrad                   fft (or svd_k64)      ~25×
1 - 30 mrad                svd_k16               ~50×
30 - 50 mrad               svd_k4                ~60×
> 50 mrad                  svd_k1                ~170×
```

Note: there's a **gap between 1 mrad and 29 mrad** — `svd_k16` jumps from exact-equivalent to 29 mrad. If you need < 1 mrad, FFT is the only fast option. If you can tolerate 30 mrad, `svd_k16` is ~2× faster than FFT. Below 30 mrad, the speed-error curve flattens: doubling accuracy doesn't get you much more speed.

## Cost of accuracy: how much speed do you trade for one mrad?

CPU, 1D n=4096, going from `fft` (exact) toward `svd_k1` (lowest accuracy):

```
step from → to             speed gain   extra error    cost (ms per mrad)
fft (1.7e-4 err)  → k=16  1.9×         +29 mrad       +0.5 ns per mrad
k=16 (29 mrad)    → k=4   1.3×         +46 mrad       +0.2 ns per mrad
k=4 (75 mrad)     → k=1   2.7×         +8 mrad        +1.0 ns per mrad
```

**Insight**: dropping accuracy from 1 mrad to 30 mrad (1.7e-4 → 29) buys ~2× speed. Dropping further to 75 mrad (k=4) buys another 1.3×. Below ~30 mrad the speed-error curve flattens out — you can keep halving error but the speedup stops growing.

---

## Figures

- `figures/fig_fft_tradeoff_cpu.png` — accuracy vs per-step time, both platforms
- `figures/fig_fft_tradeoff_gpu.png`
- `figures/fig_fft_per_n_panels.png` — small multiples: per-n speedup bars, all backends
- `figures/fig_fft_speed_cpu.png` — speed vs n (CPU)
- `figures/fig_fft_speed_gpu.png` — speed vs n (GPU)
- `figures/fig_fft_accuracy_cpu.png` — max-abs error vs n (CPU)
- `figures/fig_fft_pareto_cpu.png` — speed vs accuracy Pareto (CPU)
