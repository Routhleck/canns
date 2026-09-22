# Complete ASA shuffle benchmark

This benchmark compares the original Python `multiprocessing.Pool` shuffle
against the complete ASA pipeline in CANNs PR #103 with canns-lib PR #5.
It measures a batch of independent neuron circular shifts, activity selection,
standardization, PCA, density sampling, final graph construction and PH.
The input is an **already embedded** activity matrix `(timepoints, neurons)`;
spike embedding, real-data PH and plotting are outside the timed shuffle batch.

The old Python path already calls Rust PH. The candidate retains Python ASA
orchestration, uses a bounded thread scheduler and Rust sampling/PH kernels.
This is **not** a comparison of pure Python PH with pure Rust PH.

## Measured results

The [accepted grid_1 report](results/RESULTS.md) includes all 36 jobs, raw
per-job measurements and a time/memory plot. Peak PSS falls by 65.8–68.3%;
single-worker batches are faster, while two-worker H1 is slower. The report
separates threshold controls and lists the limitations and next profiling work.

## Fixed versions

| Component | Revision |
|---|---|
| Original Python ASA, v1.1.0 | `508a71b2cf9fa08990efb083b8961b45376e35f2` |
| Pre-PR canns-lib | `67e7b7ecbc8e2543acc2340bda57f42a5bdcba8a` |
| Candidate CANNs numerical code | `4340e8f32b70d490d323e3a1fc8d1a3426164c89` |
| Candidate canns-lib numerical code | `b9429682b024fefe5ae3f89cf69a654cea81205b` |

The v1.1.0 `_run_shuffle_analysis_multiprocessing`, worker, roll, persistence,
PCA, sampling and graph functions are AST-identical to the legacy functions
retained in CANNs `4cfa142ce5f641c18f3dbfcc31f3f96bcd40c03e`.
The old native neuron-distance shortcut is not used. Its historical
100–3000× claims concern a different null model and do not apply here.
The baseline combines the original Python algorithm with the pre-PR library
under the same Python numerical dependencies as the candidate; it is not a
recreation of every package version shipped with v1.1.0.

## Reproduce

Use Linux with Python 3.11+, the CANNs dependencies, psutil and a thread-safe
Numba OpenMP backend. Install/build the two canns-lib revisions independently
in release mode. Put the candidate CANNs and canns-lib wheels into one isolated
site directory, and the baseline canns-lib wheel into a different site directory.
Do not let one native extension overwrite the other. The harness records the
actual loaded files and their SHA256 hashes for each subprocess.
The exact Rust lockfiles used here are supplied in `locks/`. Copy the matching
lockfile to each canns-lib checkout before building with `maturin build --release
--locked`. All 62 shared package/version entries have matching checksums;
the candidate adds `thiserror`, `thiserror-impl` and an additional `syn` version.
Build directories and installed site directories must be separate.

Obtain the original two ASA files without editing them:

```bash
mkdir legacy_asa
git -C /path/to/canns show 508a71b2cf9fa08990efb083b8961b45376e35f2:src/canns/analyzer/data/asa/tda.py > legacy_asa/tda.py
git -C /path/to/canns show 508a71b2cf9fa08990efb083b8961b45376e35f2:src/canns/analyzer/data/asa/config.py > legacy_asa/config.py
```

First run the synthetic smoke plan (no dataset download required):

```bash
python benchmarks/asa_shuffle/benchmark.py \
  --legacy-asa legacy_asa --legacy-lib /path/to/baseline/site \
  --candidate-site /path/to/candidate/site \
  --plan benchmarks/asa_shuffle/plans/smoke.json --output /tmp/asa-smoke
```

For grid_1, obtain `grid_1.npz` from the
[CANNs dataset repository](https://huggingface.co/datasets/canns-team/data-analysis-datasets).
With the candidate CANNs package active, prepare the activity once:

```bash
python benchmarks/asa_shuffle/prepare_activity.py /path/to/grid_1.npz /tmp/activity.npy
```

The preparation checks both the raw archive and the resulting activity against
the measured input hashes. It uses 10 ms bins, Gaussian smoothing with
`sigma=5000` at `res=100000`, and speed filtering at 2.5 cm/s. The output has
126,729 timepoints and 172 neurons; its NPY SHA256 is
`da1e351789d176ee739531f551be3dba0d0b43364efdc27178a6b55fcbc132e5`.
Raw archive SHA256:
`76bf469e0eb0c23f9c4bf2c296452470a92a9360ae904d8193c27eea6cf736ec`.

Use that same saved activity matrix for both suites:

```bash
python benchmarks/asa_shuffle/benchmark.py \
  --legacy-asa legacy_asa --legacy-lib /path/to/baseline/site \
  --candidate-site /path/to/candidate/site --activity /path/to/activity.npy \
  --plan benchmarks/asa_shuffle/plans/main.json --output /tmp/asa-main
python benchmarks/asa_shuffle/benchmark.py \
  --legacy-asa legacy_asa --legacy-lib /path/to/baseline/site \
  --candidate-site /path/to/candidate/site --activity /path/to/activity.npy \
  --plan benchmarks/asa_shuffle/plans/controls.json --output /tmp/asa-controls
python benchmarks/asa_shuffle/summarize.py /tmp/asa-main /tmp/asa-controls \
  --output /tmp/asa-report
```

Output directories must not exist. The harness retains all failures, refuses
to replace failed draws and stops on any incomplete/unequal result. Use a new
directory for an explicitly documented fix/retry with the same inputs and seed.
The default per-job timeout is 1,800 seconds, and the supervisor stops only its
own process group if summed process-tree RSS exceeds 64 GiB. These limits are
configurable with `--timeout` and `--max-tree-rss-gib`; a limit breach is a failed
measurement, never an empty or zero-valued shuffle result.

## What is controlled

- All scientific parameters are explicit in the plan. Both variants rerun
  the entire analysis after each shift. Standardization and cocycles are enabled.
- The full grid_1 activity matrix is retained. H1 uses 1,200 representative
  points; the H2 timing case uses 400 points and must not be described as a
  1,200-point H2 benchmark. Both use 15,000 activity candidates and PCA dimension 6.
- Each configuration has four fixed shifts and three performance repetitions.
  Repetitions reuse the same shifts; they are not additional independent null draws.
- For legacy round `i`, the benchmark seeds the original worker with
  `1729 + i`. Candidate shifts replay those exact `RandomState` draws; using
  the same integer seed with `default_rng` would not reproduce them.
- Both variants use one BLAS, Numba and Rayon thread per analysis and identical
  CPU affinity. Worker counts are 1 and 2. Legacy uses Linux `fork` before
  any Numba parallel warm-up; the candidate uses its actual bounded scheduler.
- Each timed batch runs in a new subprocess. Imports and input loading finish
  before the timer. **JIT compilation, process/thread startup, IPC and all rounds
  are included.** This reports cold batch latency, not steady-state PH latency.
- An external supervisor samples the whole process tree. RSS is sampled about
  every 20 ms; PSS about every 200 ms. Sampling can miss brief peaks. Summed RSS
  double-counts shared fork pages; PSS is the primary memory comparison.

## Separate the engineering effects

| Variant | ASA scheduler / sampling | PH threshold |
|---|---|---|
| `legacy-default` | Original Pool / Python sampling | Original infinity |
| `pr-legacy` | PR bounded scheduler / Rust sampling | Infinity |
| `legacy-finite` | Original Pool / Python sampling | Inject maximum finite float32 graph edge |
| `pr-finite` | PR bounded scheduler / Rust sampling | Explicit `max_finite_float32` option |

`legacy-finite` is an instrumented control, not an unmodified historical default.
Its only scientific-call adaptation is passing `thresh` at the PH boundary.
The main `legacy-default → pr-finite` ratio combines several changes; it cannot
be attributed solely to Rust. Controls compare both implementations at the
same threshold and compare thresholds within the PR implementation.

## Correctness before performance

The original worker, original Pool dispatch and original persistence function
are called directly. Benchmark wrappers assign reproducible seeds, record
intermediate hashes and intercept the PH return value. They do not replace
the numerical pipeline. The same recording overhead is present in both variants.

Each round saves **every** applicable H0/H1/H2 bar, including short and essential
bars, plus all returned cocycles. Acceptance reopens every NPZ and compares
shape, dtype and exact bytes, together with offsets, PCA output, sampled indices
and the final graph hash. A maximum-lifetime match alone is insufficient.

The old collector omits empty finite diagrams. The harness distinguishes a
verified successful empty diagram from a swallowed exception using its saved
complete PH payload. It checks the returned maxima against those payloads;
new zero entries for genuine empty diagrams are not fabricated failed draws.
For maxima only, comparison allows float32 subtraction rounding versus the
new float64 subtraction; full diagrams and cocycles must still match exactly.

```bash
python -m pytest benchmarks/asa_shuffle/test_benchmark.py benchmarks/asa_shuffle/test_summary.py -q
```

`summarize.py` rechecks full-array equality across all suites before writing
paired medians/ranges, a machine-readable JSON report and per-run CSV. Stage
times sum work across workers and must not be mistaken for parallel wall time.
Do not pool unlike point budgets or interpret three repetitions as a general
performance guarantee.

Regenerate the figure from an accepted summary:

```bash
python benchmarks/asa_shuffle/plot_results.py /tmp/asa-report/summary.json /tmp/asa-report/comparison
```
