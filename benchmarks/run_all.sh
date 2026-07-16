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

# Add new benchmarks below as they are introduced.
