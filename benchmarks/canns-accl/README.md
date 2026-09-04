# CANN acceleration benchmarks (`accl_mode`)

This directory holds the two complementary benchmark suites for the
`accl_mode` feature on `CANN1D` / `CANN2D` (and their SFA variants):

- **`lowrank/`** — the SVD truncated-rank sweep. Compares
  `accl_mode="normal" | "fast" | "ultra-fast" | "auto"` over a
  wide range of `num` (1D) and `length` (2D) on CPU and GPU.
  Includes the `REVIEW.md` correctness audit and the per-platform
  CSV / markdown / HTML / PDF writeup.
- **`fft/`** — the circulant-FFT sweep. Compares `accl_mode="fft"`
  against the dense baseline and the best low-rank mode, on a
  clean ring/torus (`endpoint=False`) and on the canns default
  `endpoint=True` grid. Includes `TEACHING_FFT.md` (442-line
  pedagogical walkthrough of the FFT principle).

The two suites share the same underlying model API
(`CANN1D(accl_mode=...)`, `set_accl_mode(...)`) and the same
default-rank table (`ACCL_DEFAULT_K`), so the headline numbers can
be compared side-by-side. See `canns.models.basic.accel` for the
strategy-pattern implementation that both suites exercise.

## Running

```bash
# lowrank suite (CPU)
python benchmarks/canns-accl/lowrank/bench.py --T 200 --tag cpu
python benchmarks/canns-accl/lowrank/bench.py --T 200 --long-trajectory --tag cpu
python benchmarks/canns-accl/lowrank/bench.py --gpu-sweep --T 200 --tag gpu
python benchmarks/canns-accl/lowrank/report.py --tag cpu
python benchmarks/canns-accl/lowrank/report.py --tag cpu --html
python benchmarks/canns-accl/lowrank/report.py --tag cpu --pdf

# fft suite
python benchmarks/canns-accl/fft/bench.py                  # 3 platforms, ~10 min
python benchmarks/canns-accl/fft/bench.py --fast           # 1 platform, ~2 min
python benchmarks/canns-accl/fft/report.py --tag maccpu    # writeup for one platform
python benchmarks/canns-accl/fft/combine_reports.py        # combined 3-platform summary
python benchmarks/canns-accl/fft/detailed_report.py        # per-n per-platform table
python benchmarks/canns-accl/fft/triple_report.py          # 3-platform wall-time chart
```

Or run everything end-to-end:

```bash
bash benchmarks/canns-accl/run_all.sh
```

## Output layout

Each suite writes to its own `results/` subdir:

```
benchmarks/canns-accl/
├── lowrank/results/
│   ├── cann_lowrank_all_cpu.csv           # raw per-(n, k) measurements
│   ├── cann_lowrank_all_gpu.csv
│   ├── cann_lowrank_speed_cpu.csv
│   ├── cann_lowrank_speed_gpu.csv
│   ├── cann_lowrank_accuracy_cpu.csv
│   ├── cann_lowrank_accuracy_gpu.csv
│   ├── cann_lowrank_summary.md            # headline writeup
│   ├── cann_lowrank_summary.html          # NeurIPS-styled
│   ├── cann_lowrank_summary.pdf
│   └── figures/
│       ├── fig_pareto.png
│       ├── fig_long_drift_1d.png          # only if --long-trajectory
│       └── fig_long_drift_2d.png
└── fft/results/
    ├── cann_fft_speed_<plat>.csv          # one per platform (maccpu, servercpu, gpu)
    ├── cann_fft_accuracy_<plat>.csv
    ├── cann_fft_summary.md
    ├── cann_fft_summary_cpu.md
    ├── cann_fft_summary_gpu.md
    ├── cann_fft_detailed_summary.md
    ├── cann_fft_triple_summary.md
    └── figures/
        ├── fig_fft_speed_per_n.png
        └── fig_fft_error_per_n.png
```

Output filenames keep the `cann_lowrank_*` / `cann_fft_*` prefix so
that old reports / changelog entries still link correctly; the
scripts themselves drop the prefix because the parent
`lowrank/` or `fft/` directory already disambiguates.

## Adding a new mode benchmark

If you implement a new `accl_mode` (e.g. `block-circulant`) and want
to benchmark it:

1. Write the benchmark driver in
   `benchmarks/canns-accl/<new-suite>/bench.py` (or
   `block_circulant/bench.py` for a new suite).
2. Write a report in `report.py` if you want a per-suite writeup.
3. Append the new suite to `benchmarks/canns-accl/run_all.sh`.
4. Update this README's "Output layout" section.
5. Add the new mode to `ACCL_MODES` in
   `src/canns/models/basic/accel/__init__.py`.

## See also

- `src/canns/models/basic/accel/` — the strategy-pattern backend
  implementation that these benchmarks exercise.
- `docs/.../09_acceleration_modes.ipynb` — a tutorial that walks
  through all 5 modes with code.
- `CHANGELOG.md` `[Unreleased]` — what changed and when.
