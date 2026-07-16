# `benchmarks/` — performance receipts for the canns Python package

This directory is the canonical home for **reproducible performance numbers**
that get quoted in canns release notes. Each canns release that touches a
performance-sensitive path should add a new result file here.

> **Scope:** this is for the **Python `canns` package** only — Ripser /
> `canns_lib` engine benchmarks live in the
> [canns-lib repo](https://github.com/Routhleck/canns-lib/tree/master/benchmarks/ripser).

## Layout

```
benchmarks/
├── README.md                  # this file
├── shuffle_null_model.py      # current benchmark: FFI vs mp.Pool
├── run_all.sh                 # one-shot re-runner (add new benchmarks here)
└── results/
    ├── v1.2.0-shuffle.csv     # raw data, version-keyed, immutable
    ├── v1.2.0-shuffle.md      # human-readable writeup, immutable
    └── ...                    # future: v1.3.0-shuffle.md, etc.
```

## Immutability rule for `results/`

**Files under `results/` are immutable once committed.**

When a new release needs a fresh benchmark run, do not edit the existing
writeup or CSV — create a new file with the new version tag
(`v1.3.0-shuffle.md`, `v1.3.0-shuffle.csv`, …). This is what makes the
release notes link survive being re-checked six months later.

If you spot a factual error in a committed result, do **not** rewrite it.
Either add a new result that supersedes it and reference both in the release
notes, or annotate it with a "RETRACTED" notice at the top.

## Running an existing benchmark

`shuffle_null_model.py` compares `canns_lib._ripser_core.shuffle_null_model`
(FFI) against the `multiprocessing.Pool` legacy path:

```bash
# from the repo root, with the canns dev venv active:
uv run python benchmarks/shuffle_null_model.py
```

Default matrix is `T ∈ {60, 300, 1000} × N ∈ {20, 40, 80, 160} ×
n_shuffles ∈ {10, 50, 200, 1000}` = 48 cells, but you can shrink it
for a quick smoke test:

```bash
uv run python benchmarks/shuffle_null_model.py \
    --T 60 --N 20 --shuffles 10 50 \
    --maxdim 1 --coeff 2 --repeats 1
```

Output: a timestamped CSV in `benchmarks/results/`, plus an aggregate
median summary printed to stdout.

## Adding a new result (release checklist)

When a release introduces (or re-tunes) a performance-sensitive path:

1. Add a new benchmark script under `benchmarks/` (or extend an existing
   one) covering the new path.
2. Wire it into `benchmarks/run_all.sh` so a single command reproduces
   every committed result.
3. Run the benchmark on the target hardware. Capture:
   - Hardware (CPU model + count, RAM, OS + version)
   - Software (Python version, `canns` version, `canns-lib` version, key
     library versions like `numpy` / `ripser`)
   - The exact command (copy-pasteable)
4. Commit the raw CSV into `results/<version>-<short-name>.csv`.
5. Write a writeup `results/<version>-<short-name>.md` with:
   - TL;DR (the headline number(s))
   - Hardware & software table
   - Reproducibility (exact command)
   - Methodology (what was measured, how)
   - Per-cell detail (table or link to CSV)
   - Notes & caveats (what this number does **not** cover)
6. Update the canns release notes to link to the new writeup.

## What "official" means here

A number is "official" once it lives in a committed `results/<version>-*.md`
file. Numbers in PR descriptions, Slack threads, or unreleased branch
writeups are **not** quotable in release notes. The release-notes
writeup link is the only thing users will follow, so the link must resolve
to a file that survives in `master` long-term.

## What this directory is **not**

- **Not a CI benchmark runner.** Ripser + shuffle null-model together take
  ~36 minutes wall time on the slowest legacy cells; running on every PR
  is not viable. Re-runs are a release-time activity, not a CI activity.
- **Not a profiling tool.** If you need to drill into why a cell is slow,
  use a profiler (`py-spy`, `cProfile`, the `rayon`/`perf` tooling on the
  Rust side) — not this harness.
- **Not a regression alarm.** We don't have a baseline-vs-current
  comparison; the only regression check is a human reading the new
  writeup against the previous one.
