#!/usr/bin/env bash
# Re-run every committed benchmark in this directory. New benchmark scripts
# should be added here.
#
# Usage:  bash benchmarks/run_all.sh
# Output: each script writes its own timestamped CSV into results/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "==> shuffle_null_model.py"
uv run python benchmarks/shuffle_null_model.py "$@"

# `canns_lowrank/` is a sub-benchmark for the accl_mode / accl_k low-rank
# matvec feature on CANN1D / CANN2D. It produces a paper-style writeup
# (cann_lowrank_summary.{md,html,pdf}). Two stages: sweep the cells, then
# render the report. The CPU sweep is ~15 min on Apple M3 Pro; the GPU
# sweep needs an A100 with `--gpu-sweep`. Both are optional — the report
# gracefully falls back to whichever tag exists.
echo "==> canns_lowrank — CPU sweep (~15 min on Apple M3 Pro)"
uv run python benchmarks/canns_lowrank/cann_lowrank_bench.py --T 200 --tag cpu "$@"
echo "==> canns_lowrank — render report (markdown + HTML + PDF)"
uv run python benchmarks/canns_lowrank/cann_lowrank_report.py --tag cpu --html --pdf "$@"

# Add new benchmarks below as they are introduced.
