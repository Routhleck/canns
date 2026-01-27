#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal grid module classification demo using the built-in grid dataset.

Steps:
1) Load grid dataset
2) Bin spikes into time windows (with positions aligned)
3) Build autocorrelograms
4) Identify grid modules with Leiden clustering
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from canns.analyzer import data
from canns.analyzer.data.cell_classification import (
    GridnessAnalyzer,
    compute_2d_autocorrelation,
    compute_rate_map_from_binned,
    identify_grid_modules_and_stats,
)
from canns.data import load_grid_data, load_left_right_npz


def main() -> None:
    # grid_data = load_grid_data()
    grid_data = load_left_right_npz(
        session_id="28304_1",
        filename="28304_1_ASA_mec_full_cm.npz",
    )

    cfg = data.SpikeEmbeddingConfig(smooth=True, speed_filter=False, min_speed=2.5)
    spikes, xx, yy, _tt = data.embed_spike_trains(grid_data, config=cfg)
    if xx is None or yy is None:
        raise SystemExit("Position data missing in grid dataset.")

    max_neurons = spikes.shape[1]
    autocorrs = []
    for nid in tqdm(range(max_neurons)):
        rate_map, _, _, _ = compute_rate_map_from_binned(xx, yy, spikes[:, nid], bins=35)
        autocorr = compute_2d_autocorrelation(rate_map)
        autocorrs.append(autocorr.astype(np.float32, copy=False))

    autocorrs = np.stack(autocorrs, axis=0)

    analyzer = GridnessAnalyzer()
    out = identify_grid_modules_and_stats(
        autocorrs,
        gridness_analyzer=analyzer,
        k=30,
        resolution=1.0,
        score_thr=0.3,
        consistency_thr=0.5,
        min_cells=10,
        merge_corr_thr=0.7,
        metric="manhattan",
    )

    print(
        f"[OK] n_units={out['n_units']}  n_grid_cells={out['n_grid_cells']}  "
        f"n_modules={out['n_modules']}"
    )
    if out["n_modules"]:
        sizes = [m["size"] for m in out["modules"]]
        print(f"Module sizes: {sizes}")
    np.savez_compressed("grid_modules_demo_output.npz", **out)


if __name__ == "__main__":
    main()
