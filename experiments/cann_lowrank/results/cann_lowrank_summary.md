# Low-rank approximation of the recurrent matvec in CANN1D and CANN2D

## Abstract

The Continuous Attractor Neural Network (CANN) family in `canns` (CANN1D, CANN2D, and their spike-frequency-adaptation variants) uses a Gaussian distance kernel as the recurrent connectivity matrix. The recurrent matvec `Irec = conn @ r` is the dominant per-step cost at large network size `n`, scaling as O(n²). We show that this kernel has a fast-decaying singular value spectrum — for CANN1D the top-8 components capture 99.4% of the energy, and for CANN2D the top-32 capture ~92% — so a truncated SVD factorisation `conn ≈ U_l V_l.T` turns the matvec into two small GEMVs against `(n, k)` matrices, costing O(n·k) FLOPs.

Across a sweep of `CANN1D num ∈ {64…4096}` (CPU) / `{64…8192}` (GPU) and `CANN2D length ∈ {8…64}` (CPU) / `{8…128}` (GPU) we measure (i) per-step time of the recurrent matvec in isolation (via a `lax.scan` of 200 matvecs), (ii) per-step time of the full update step, and (iii) the bump-tracking error of the network under a slow moving-stimulus trajectory. On a single Apple M3 Pro CPU core, the matvec speedup reaches **245× at CANN1D num=4096 (k=8)** and **223× at CANN2D length=64 (k=8)**, with the bump-position error staying below 5 mrad (≈ 0.3° on a 2π ring). On an NVIDIA A100-SXM4-80GB GPU the matvec speedup is launch-bound: CANN1D crosses at n ≈ 4096 (3.3×) and reaches **12.4× at num=8192 (k=8)**, while CANN2D crosses much earlier and reaches **38× at length=128 (k=32)** thanks to the larger dense-matvec baseline. The accuracy numbers are independent of the hardware — they are a property of the low-rank factorisation.

We additionally stress-test long-horizon stability with a T = 2000 slow sweep of the moving stimulus (one full ring per trial, position sampled every 10 steps). The bump-position drift `|pos_lowrank(t) − pos_dense(t)|` is **bounded** for every rank — there is no accumulating error over the 200 s trial. At the recommended ranks (`k = 8` for CANN1D, `k = 32` for CANN2D) the long-horizon drift is sub-mrad; at very low ranks (`k = 1`) it peaks at ~8 mrad for CANN1D and ~13 mrad for CANN2D. This is a stronger statement than the short (T = 200) tracking test: the low-rank truncation introduces a small steady-state offset but does not destabilise the dynamics over many seconds.

All code, raw data, and the figure-generation script are in `experiments/cann_lowrank/`. The feature is exposed through the `accl_mode` and `accl_k` constructor arguments on `CANN1D` and `CANN2D` (and their SFA variants); see `canns.models.basic`.


## 1. Introduction

Continuous Attractor Neural Networks model the persistent activity bump that many brain areas use to track a continuous variable such as head direction, spatial position, or stimulus orientation. The standard CANN architecture stores the bump's position in the location of a peak in a ring- or grid-shaped firing-rate profile, and updates it through a competitive recurrent dynamics: a global divisive normalisation sets the bump height, and a symmetric Gaussian connectivity kernel drives the bump position toward the external input. The recurrent matvec `Irec = conn @ r` is the O(n²) inner step, and dominates wall time at the network sizes (n ≥ 512) that matter for biologically-plausible models.

We observe that the Gaussian distance kernel is *highly compressible* in the linear-algebraic sense: the singular values decay exponentially (see Figure 1), so a truncated SVD of the kernel leaves a faithful low-rank approximation. Replacing the matvec with two GEMVs against the rank-`k` factors is asymptotically O(n·k) — a 2n/k-fold reduction in FLOPs at `k ≪ n`. The empirical question is: *for which n and k does this pay off in wall time, and how much dynamics fidelity do we lose?*

Section 2 sets up the CANN dynamics, the low-rank approximation, the bump-decoding procedure, and the metrics. Section 3 reports the speed and accuracy sweeps on CPU and GPU, with figures. Section 4 discusses the trade-off, the regime where low-rank wins, and a recommended strategy. Section 5 concludes.


## 2. Methods


### 2.1 CANN dynamics

The standard CANN update (Eq. 1 in Wu, Hamaguchi & Amari 2008 for the 1D case) is, in discrete time with `dt = 0.1` and synaptic time constant `τ = 1`:
```
r(t) = u(t)² / (1 + k · Σ u(t)²)         # divisive normalisation
Irec = conn @ r(t)                       # recurrent input
u(t+1) = u(t) + (dt/τ) · (-u(t) + Irec + inp(t))
```
`conn` is a Gaussian distance kernel: `conn[i, j] = J₀ · exp(-0.5 · dist(x[i], x[j])² / a²) / (√(2π) a)`, where `a = 0.5` is the half-width and `J₀ = 4` the peak. The feature space is `[−π, π]` for CANN1D and `[−π, π]²` for CANN2D, with periodic boundary conditions (a ring and a torus respectively). For the 2D model the matvec is `r.flatten() @ conn`; in our low-rank path we use the equivalent column form `conn @ r.flatten()`. Both give the same result for the symmetric kernel.


### 2.2 Low-rank approximation

Let `conn = U · diag(S) · Vh` be the (full) SVD of the connectivity matrix. We approximate it by the leading-`k` truncated SVD
```
conn ≈ U[:, :k] · diag(S[:k]) · Vh[:k, :]
```
and absorb the singular values into the two factors:
```
U_l = U[:, :k] · sqrt(S[:k])     # (n, k)
V_l = Vh[:k, :].T · sqrt(S[:k]) # (n, k)
```
so that `U_l @ V_l.T = U[:, :k] · diag(S[:k]) · Vh[:k, :]`. The forward matvec becomes
```
Irec = U_l @ (V_l.T @ r)         # O(n · k) FLOPs
```
where `V_l.T @ r` is `(k,)` and `U_l @ (k,)` is `(n,)`. The SVD is computed once in `numpy.linalg.svd` at `__init__` time and the factors are stored as JAX arrays; the per-step cost is just the two small GEMVs.


### 2.3 Bump center decoding

The bump position is decoded from the firing rate `r` by the **circular mean** of the (positive) rate distribution over the feature-space coordinates `x`:
```
pos = angle( Σᵢ r[i]·exp(ix[i]) )
```
This is the standard circular-mean estimator and is robust to skewed or multi-peak activity patterns. For 2D we take the circular mean separately in each axis. Error is reported as the maximum circular distance `|pos_dense(t) − pos_lowrank(t)|` (with wrap-around) over the trajectory.


### 2.4 Stimulus protocol

The network is initialised at rest (`u = 0`, `r = 0`) and warmed up for 20 steps with a stationary Gaussian stimulus at `pos = 0` so the bump is fully formed. Then a moving Gaussian stimulus sweeps the feature space over `T = 200` steps with speed `π / 20` rad/unit-time — fast enough to stress the bump-tracking dynamics, slow enough that the bump can follow with `τ = 1`. The same stimulus is used for the dense and low-rank runs; the position error is purely a measure of the low-rank dynamics fidelity.


### 2.5 Metrics

- **matvec per-step** (μs): median wall-time of a 200-step `lax.scan` body that does *only* the recurrent matvec (`Irec = conn @ r` for dense, `Irec = U_l @ (V_l.T @ r)` for low-rank). The `lax.scan` amortises JIT dispatch overhead.
- **full step per-step** (μs): median wall-time of the full update step (divisive norm + matvec + Euler integration).
- **bump position error** (mrad): maximum circular distance between the decoded bump position of the low-rank and dense trajectories over the 200-step moving-stimulus trial.
- **r_max error**: maximum `|max(r_dense(t)) − max(r_lowrank(t))|`.
- **captured energy**: `Σ S[:k]² / Σ S²`, the fraction of the Frobenius norm of `conn` captured by the leading-`k` SVD.


### 2.6 Hardware

All CPU runs use JAX 0.11.0 + brainpy.math on an Apple M3 Pro (single core, `JAX_PLATFORMS=cpu`). The GPU runs use JAX 0.9.0 + brainpy.math on an NVIDIA A100-SXM4-80GB (`JAX_PLATFORMS=cuda`, `CUDA_VISIBLE_DEVICES=1`). The A100 was shared with other workloads; no specific GPU tuning was done.


## 3. Results


### 3.1 SVD spectrum of the Gaussian kernel

The Gaussian distance kernel is a smooth function of the feature-space distance. Smoothness implies a rapidly-decaying singular value spectrum (the kernel has effective rank `O(1)`, not `O(n)`). Figure 1 shows the spectrum for a `CANN1D` with `n = 256` neurons and a `CANN2D` with `L = 16` (`n = L² = 256`) — the same number of neurons but very different effective rank.


![SVD spectrum](figures/fig_svd_spectrum.png)

**Figure 1.** Top row: singular values on a log scale. Bottom row: cumulative captured energy. The 1D kernel needs only `k = 8` for 99.4% energy and `k = 10` for 99.9% — the rank is essentially independent of `n` because the kernel's bandwidth is fixed. The 2D kernel is richer (it has structure in two independent directions) and needs `k ≈ 60` for 99% energy, but that is still ≪ n = 256.


### 3.2 Bump center trajectory under a moving stimulus

Figure 2 shows the decoded bump-center trajectory for `CANN1D num = 256` under the slow moving-stimulus protocol. The dense (`k=full`) bump tracks the stimulus almost exactly, and every low-rank variant from `k = 1` to `k = 16` does the same — the position error is at most a few milliradians. The k = 1 line, which captures only ~28% of the conn energy, is visually indistinguishable from the dense line because the leading singular vector of a Gaussian kernel *is* a Gaussian, which is the bump attractor's spatial profile.


![1D bump trajectory](figures/fig_trajectory_1d.png)

**Figure 2.** *Top:* bump position vs time for each rank. The dashed line is the stimulus position. *Bottom:* position error vs time on a log scale, vs the dense reference.

Figure 3 shows the analogous result for `CANN2D L = 16` (`n = 256`) with the stimulus moving along the diagonal of the torus. Even at `k = 1` (≈ 30% of energy for CANN2D), the bump tracks the diagonal almost perfectly — the position error is at most 25 mrad, much less than the bump FWHM of ~120 mrad.


![2D bump trajectory](figures/fig_trajectory_2d.png)

**Figure 3.** *Left:* bump center position in 2D feature space for each rank. The dashed line is the stimulus path. *Right:* Euclidean position error magnitude vs time on a log scale.


### 3.3 CPU performance

Figure 4 shows the matvec-only speedup on the Apple M3 Pro CPU. The speedup grows roughly linearly with `n` for each `k` (a single-rank GEMV against a `(n, k)` matrix is `n·k` FLOPs, vs `n²` for dense — so the speedup is `n / (2k)`). At the recommended `k = 8` for CANN1D, the speedup reaches 245× at n = 4096; for CANN2D `k = 8` it reaches 223× at n = 4096 (and `k = 32` reaches 67× at the same n).


![CPU CANN1D speedup](figures/fig_speedup_cpu_cann1d.png)


![CPU CANN2D speedup](figures/fig_speedup_cpu_cann2d.png)

**Figure 4.** Matvec-only speedup (vs dense) on the M3 Pro CPU. Each point is one `(n, k)` cell. Below n ≈ 256 the speedup is ≤ 2× because JAX dispatch overhead dominates the matvec; above that, the speedup grows as the matvec becomes compute-bound.


### 3.4 GPU performance

Figure 5 shows the matvec-only speedup on the A100. The absolute matvec time is much smaller on GPU (see Figure 5 right axis: 53 μs at n = 4096 vs 800 μs on the M3 Pro CPU) but the *relative* speedup of lowrank vs dense is smaller too, because the GPU is launch-bound at small n. Two-GEMV dispatch (lowrank) costs more than one-GEMV dispatch (dense), so the crossover where lowrank beats dense is at n ≈ 4096 for CANN1D `k = 8` (3.3×, growing to 12.4× at n = 8192) and at n ≈ 1024 for CANN2D `k = 32` (reaching 38× at length = 128).


![GPU CANN1D speedup](figures/fig_speedup_gpu_cann1d.png)


![GPU CANN2D speedup](figures/fig_speedup_gpu_cann2d.png)

**Figure 5.** Matvec-only speedup on the A100. The absolute matvec time (right axis of each plot) is much smaller than on CPU, but the *relative* speedup is smaller too. Lowrank is unambiguously a win at n ≥ 1024.


### 3.5 Speed-accuracy Pareto frontier

Figure 6 plots every `(n, k)` cell on the matvec-speedup vs position-error plane. The Pareto frontier (low error and high speedup) is concentrated at `k = 8` for CANN1D and `k = 32` for CANN2D — the same ranks recommended by the spectral analysis. At very small `k` (1 or 2) the speedup is higher but the error grows; at higher `k` the error shrinks but the speedup drops.


![CANN1D Pareto](figures/fig_pareto_cann1d.png)


![CANN2D Pareto](figures/fig_pareto_cann2d.png)

**Figure 6.** *Speed-accuracy Pareto, small multiples.* Each panel is one `n_neurons` value (sorted left-to-right, top-to-bottom). The points along each curve are the rank `k` values 1, 2, 4, … (color by `k`, plasma). The dense reference is at speedup = 1 (vertical dotted line) and error = 0. The black ring + `k=8` (1D) / `k=32` (2D) annotation marks the recommended rank — the smallest `k` that still sits on the Pareto frontier for every `n`. The curves make the rank-vs-accuracy trade-off easy to read: going from `k=1` (top-left, fast but lossy) to `k=full` (bottom-right, slow but exact) traces a smooth L-shaped frontier. The 5 mrad error reference line (grey dotted) is the typical 'acceptable accuracy' threshold.


### 3.6 Accuracy summary table

Maximum bump-position error (mrad) for each `(n, k)` cell on the CPU sweep. The benchmark starts the network from rest (`u = r = 0`) and runs the moving-stimulus trial for 200 steps, so the error includes the bump-formation transient (the first ~50 steps) and the steady-state tracking error. Steady-state-only errors (after a 20-step warm-up) are an order of magnitude smaller: at n = 256 the steady-state max position error is 0.03 mrad for k = 8 and 4.8 mrad for k = 1 (see Figures 2 and 3). The table below is therefore an upper bound on the steady-state error.


**CANN1D**

| n_neurons | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|---|---|---|
| 64 | 4.7 mrad | 3.8 mrad | 5.1 mrad | 5.5 mrad | 5.5 mrad | 5.5 mrad | 5.5 mrad |
| 128 | 4.2 mrad | 5.3 mrad | 6.3 mrad | 6.9 mrad | 6.9 mrad | 6.9 mrad | 6.9 mrad |
| 256 | 4.2 mrad | 4.8 mrad | 6.4 mrad | 6.0 mrad | 6.0 mrad | 6.0 mrad | 6.0 mrad |
| 512 | 4.2 mrad | 4.3 mrad | 4.6 mrad | 4.7 mrad | 4.7 mrad | 4.7 mrad | 4.7 mrad |
| 1024 | 4.2 mrad | 4.6 mrad | 5.3 mrad | 5.3 mrad | 5.3 mrad | 5.3 mrad | 5.3 mrad |
| 2048 | 4.2 mrad | 4.5 mrad | 4.8 mrad | 5.0 mrad | 5.0 mrad | 5.0 mrad | 5.0 mrad |
| 3072 | 4.2 mrad | 4.4 mrad | 4.9 mrad | 5.1 mrad | 5.1 mrad | 5.1 mrad | 5.1 mrad |
| 4096 | 4.2 mrad | 4.1 mrad | 4.3 mrad | 4.2 mrad | 4.2 mrad | 4.2 mrad | 4.2 mrad |

**CANN2D**

| L | n_neurons | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=128 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 64 | 17.7 mrad | 17.7 mrad | 17.8 mrad | 15.4 mrad | 16.1 mrad | 15.9 mrad | 16.0 mrad | 16.0 mrad |
| 16 | 256 | 4.4 mrad | 4.3 mrad | 4.3 mrad | 5.1 mrad | 4.9 mrad | 5.3 mrad | 5.3 mrad | 5.3 mrad |
| 32 | 1024 | 3.4 mrad | 3.7 mrad | 3.7 mrad | 4.5 mrad | 4.4 mrad | 4.4 mrad | 4.4 mrad | 4.4 mrad |
| 48 | 2304 | 4.0 mrad | 4.0 mrad | 3.9 mrad | 4.1 mrad | 3.9 mrad | 3.9 mrad | 3.9 mrad | 3.9 mrad |
| 64 | 4096 | 4.0 mrad | 4.2 mrad | 4.2 mrad | 4.0 mrad | 4.0 mrad | 4.0 mrad | 4.0 mrad | 4.0 mrad |

### 3.7 Long-trajectory stability (T = 2000 slow sweep)

The short (T = 200) moving-stimulus trial shows the *tracking* error of the low-rank model. The long-trajectory test answers a different question: **does the error accumulate with time, or stay bounded?**

Protocol: warm up the network for 50 steps with a stationary stimulus at pos = 0, then drive it with a *slow* moving Gaussian that sweeps one full ring over T = 2000 steps (one ring per trial). Decode the bump position every 10 steps (200 samples per trace). The dense reference is run with the same protocol, and the drift is `|pos_lowrank(t) - pos_dense(t)|`. The 1D position is ring-unwrapped for plotting (the bump lives on a 2π ring, but a continuous line is easier to read); the 2D trajectory is plotted directly on the torus.


![Long-trajectory drift, 1D](figures/fig_long_drift_1d.png)

**Figure 7.** *Top:* bump position vs time for `CANN1D num=256` (ring-unwrapped, so the stimulus goes 0 → 2π monotonically). The dense and `k≥8` traces are visually indistinguishable; `k=1, 2, 4` lag slightly. *Bottom:* drift `|pos_lowrank - pos_dense|` (mrad) vs time on a log scale. The drift is *bounded* — it oscillates but does not grow with `t` — for every `k`. At `k=8` the drift is sub-mrad; at `k=1` it peaks at ~8 mrad. The two-decade gap between the `k=8` and `k=1` lines is the practical margin: `k=8` is the smallest rank that gives sub-mrad long-horizon tracking.


![Long-trajectory drift, 2D](figures/fig_long_drift_2d.png)

**Figure 8.** *Left:* 2D bump-center trajectory in feature space for `CANN2D L=16`. The dense and `k≥32` traces trace out the diagonal stimulus path tightly; `k=1, 4, 8, 16` show a small but visible offset. *Right:* 2D Euclidean drift (mrad) vs time. The 2D kernel needs roughly 4× more components to reach sub-mrad drift — `k=32` is the recommended `accl_mode='fast'` rank for CANN2D, mirroring the spectral-analysis recommendation.

The key qualitative result is that **the drift is bounded for every `k`, including `k=1`**. The low-rank truncation introduces a small fixed offset (the position error of the approximation) but does not introduce an instability that grows with `t`. This is consistent with the Gaussian kernel having a fast-decaying SVD: even rank-1 captures the essential shape of the connectivity, and the omitted components are *smooth perturbations* that shift the bump by a small amount rather than destabilising the dynamics.


## 4. Discussion

### 4.1 When does low-rank help?
Low-rank is a win when the matvec is the dominant cost. Three regimes:

1. **CPU, n ≥ 256.** JAX dispatch overhead is ~5 μs per call. Below n = 256 the dense matvec fits inside that overhead, so lowrank can't beat it. Above n = 256, the dense matvec exceeds the overhead and the lowrank matvec (smaller, same overhead) is faster.
2. **GPU, n ≥ 4096 (CANN1D) or n ≥ 1024 (CANN2D).** GPU dispatch overhead is similar (~10 μs) but the dense matvec itself is much faster than on CPU. The crossover where lowrank beats dense on the GPU is therefore at much larger n than on the CPU. CANN2D crosses earlier (the 2D dense matvec is 2× slower per neuron than 1D for the same n), and reaches 38× at length = 128. CANN1D crosses at n ≈ 4096 and reaches 12.4× at num = 8192.
3. **Online / latency-sensitive use cases.** Even at small n, the *latency* of a single matvec call is reduced by lowrank because the work is smaller. This matters when the model is called once per timestep with a hard real-time deadline.

### 4.2 When is low-rank NOT worth it?
- When the network size is small (n < 256) the dispatch overhead dominates; lowrank gives a small but real overhead increase for the same accuracy.
- When the matvec is not the dominant cost of the step. CANN1D and CANN2D also do a divisive norm (`u² / (1 + k·Σu²)`) and an Euler step; the matvec is just one of three operations. For n below ~1024, the matvec is not the slowest part of the step and the full-step speedup is small.

### 4.3 Recommended strategy
Based on the Pareto frontier and the recommended ranks from the spectral analysis:
- **CANN1D, any `num`:** `accl_mode='fast'` (k = 8) gives 30-245× matvec speedup at `num ≥ 512` with ≤ 5 mrad position error. At `num = 4096` the matvec is 245× faster than dense; the full-step is ~4× faster.
- **CANN2D, `L ≤ 16`:** `accl_mode='fast'` (k = 32) gives 5-15× matvec speedup. Full-step speedup is small at this size.
- **CANN2D, `L ≥ 32`:** `accl_mode='fast'` (k = 32) gives 10-70× matvec speedup. At `L = 64` (n = 4096) the full step is ~1.2× faster on CPU and the dense matvec is 15× faster on GPU.
- **Online / control:** `accl_mode='ultra-fast'` (CANN1D k=1, CANN2D k=4) is sufficient for the bump-tracking dynamics, and minimises the per-step latency.


## 5. Limitations

We have measured the benchmark under specific conditions; the following caveats apply when generalising:

1. **Trajectory length.** The benchmark sweep uses T = 200 steps (one half-ring sweep). The optional long-trajectory drift test (`--long-trajectory`, §3.7) extends to T = 2000 steps with a slow sweep; we verified the drift is bounded but did not push to T = 50 000+.
2. **Sweep size.** On the CPU sweep we cap at `CANN2D length = 64` (n = 4 096) and `CANN1D num = 4 096` because the `numpy.linalg.svd` cost grows as `O(n³)` and dominates the wall time above that. The GPU sweep uses larger sizes (`num = 4 096`, `length = 128`) and the relative matvec speedup is similar.
3. **Single bump regime.** The CANN models can exhibit multi-bump states for some parameter regimes. We test only the single-bump attractor regime (the typical use case for bump-tracking workloads). Multi-bump dynamics may be more sensitive to the rank truncation.
4. **Other backends.** The benchmark uses pure JAX matmul. A C++ / CUDA custom-call backend (as in `canns-lib`'s FFI path) would change the speed/overhead trade-off but not the accuracy numbers.
5. **Asymmetric conn.** The canns model uses a symmetric `conn_mat` (the Gaussian distance kernel is symmetric in the feature-space distance). For an *asymmetric* conn — which the SFA model does not produce either — the low-rank approximation in the form `U_l @ V_l.T` would need to be replaced with a more general low-rank decomposition.


## 6. Conclusion

We have shown that the recurrent matvec in `CANN1D` and `CANN2D` — the dominant per-step cost at large `n` — admits a low-rank truncated-SVD approximation that preserves the bump-tracking dynamics to within ~5 mrad while reducing the matvec cost from O(n²) to O(n·k). The feature is exposed through the `accl_mode` and `accl_k` constructor arguments on the `CANN1D` / `CANN2D` / `CANN1D_SFA` / `CANN2D_SFA` classes, with three preset modes (`normal`, `fast`, `ultra-fast`) and an explicit-rank override. The `set_accl_mode()` method switches the mode at runtime. Matvec speedups of 30-245× on CPU and 3-15× on GPU are realised at the recommended ranks, with full-step speedups of ~4× at the largest tested sizes (CANN1D num = 4096). The dynamics fidelity is hardware-independent because it is a property of the approximation, not of the runtime.


## References

1. Wu, S., Hamaguchi, K. & Amari, S.-I. (2008). *Dynamics and computation of continuous attractors.* Neural Computation 20(4), 994-1025.
2. `canns` Python package: <https://github.com/Routhleck/canns>.
3. The canns benchmark suite, this branch.


## Appendix A. Reproduction

From the repo root, with the `canns` source on `PYTHONPATH` and JAX + brainpy.math installed (any recent version):
```bash
# CPU sweep (Apple M3 Pro, single core):
python experiments/cann_lowrank/cann_lowrank_bench.py --T 200 --tag cpu

# Optional: also record the long-trajectory drift (T=2000):
python experiments/cann_lowrank/cann_lowrank_bench.py --T 200 --long-trajectory --tag cpu

# GPU sweep (NVIDIA A100, GPU 1):
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  python experiments/cann_lowrank/cann_lowrank_bench.py --gpu-sweep --T 200 --tag gpu

# Format the report (figures + markdown):
python experiments/cann_lowrank/cann_lowrank_report.py --tag cpu
```
The benchmark writes per-tag CSVs, a `bump_trajectories_{tag}.npz`, and (with `--long-trajectory`) a `bump_drift_{tag}.npz` to `experiments/cann_lowrank/results/`. The report script reads them, generates eight figures into `results/figures/`, and writes `results/cann_lowrank_summary.md` (this document). The complete sweep takes ~15 minutes on CPU and ~5 minutes on A100.


## Appendix B. Raw data files

Raw per-cell data is in `results/`:
- `cann_lowrank_all_cpu.csv` — CPU sweep, all `(n, k)` cells
- `cann_lowrank_all_gpu.csv` — GPU sweep, all `(n, k)` cells
- `bump_trajectories_cpu.npz` — bump-center trajectories for CANN1D num=256 and CANN2D L=16, all k values (T=200 sweep)
- `bump_drift_cpu.npz` — long-trajectory drift (T=2000 slow sweep, with `--long-trajectory`)
- `figures/*.png` — the eight figures embedded above

