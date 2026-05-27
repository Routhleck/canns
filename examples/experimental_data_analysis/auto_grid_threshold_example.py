#!/usr/bin/env python3
"""
Minimal example for automatic grid-score threshold selection.

This example is meant for conversational/CLI workflows:
1. load one ASA dataset,
2. sweep top-k grid-score subsets,
3. report the best threshold candidate,
4. optionally refine using shuffle-based TDA validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from canns.analyzer.data.asa.path import load_npz_any
from canns.analyzer.workflows import analyze_auto_grid_threshold
from canns.utils.example_outputs import get_example_output_dir

DEFAULT_OUT_DIR = get_example_output_dir(__file__)


def _jsonify(result: dict) -> dict:
    out = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, dict):
            out[key] = _jsonify(value)
        elif isinstance(value, list):
            out[key] = [_jsonify(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep grid-score thresholds for ASA data.")
    parser.add_argument("--data", required=True, help="ASA .npz with spike/x/y/t.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--use-precomputed-spikes", action="store_true", help="Treat spike as dense (T,N).")
    parser.add_argument("--min-topk", type=int, default=50, help="Minimum top-k subset size.")
    parser.add_argument("--max-topk", type=int, default=None, help="Maximum top-k subset size.")
    parser.add_argument("--topk-step", type=int, default=25, help="Sweep step for top-k.")
    parser.add_argument("--topology-mode", choices=["h1", "h1h2"], default="h1")
    parser.add_argument("--shuffle-refine", action="store_true", help="Run shuffle refinement on shortlist.")
    parser.add_argument("--shuffle-num-shuffles", type=int, default=50, help="Number of shuffle refinements.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asa = load_npz_any(args.data)
    result = analyze_auto_grid_threshold(
        asa,
        use_precomputed_spikes=args.use_precomputed_spikes,
        min_topk=args.min_topk,
        max_topk=args.max_topk,
        topk_step=args.topk_step,
        topology_mode=args.topology_mode,
        use_shuffle_refinement=args.shuffle_refine,
        shuffle_num_shuffles=args.shuffle_num_shuffles,
    )

    out_json = out_dir / "auto_grid_threshold_summary.json"
    out_json.write_text(json.dumps(_jsonify(result), indent=2), encoding="utf-8")

    best = result.get("best_result")
    if best is None:
        print("No valid threshold candidate found.")
    else:
        print(
            f"Best top-k={best['topk']} threshold={best['grid_threshold']:.4f} "
            f"pearson_r={best.get('correlation', {}).get('pearson_r', float('nan')):.4f}"
        )
    print(f"Saved summary: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
