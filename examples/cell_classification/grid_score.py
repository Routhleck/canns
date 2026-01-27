#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal grid score example using the built-in grid dataset.

Steps:
1) Load grid dataset
2) Bin spikes into time windows (with positions aligned)
3) Score gridness on a subset of neurons
4) Plot rate map, autocorrelogram, and score histogram
"""

from __future__ import annotations

import numpy as np

from canns.analyzer import data
from canns.analyzer.data.cell_classification import (
    GridnessAnalyzer,
    compute_2d_autocorrelation,
    compute_rate_map_from_binned,
    plot_autocorrelogram,
    plot_grid_score_histogram,
    plot_rate_map,
)
from canns.data.loaders import load_grid_data


def main() -> None:
    grid_data = load_grid_data()
    if grid_data is None:
        raise SystemExit("Grid dataset not found.")

    cfg = data.SpikeEmbeddingConfig(smooth=False, speed_filter=True, min_speed=0.0)
    spikes, xx, yy, _tt = data.embed_spike_trains(grid_data, config=cfg)
    if xx is None or yy is None:
        raise SystemExit("Position data missing in grid dataset.")

    # pick the best-scoring neuron among the first N for a clean demo
    max_neurons = spikes.shape[1]
    best_score = -np.inf
    best_id = 0
    best_result = None
    scores: list[float] = []

    analyzer = GridnessAnalyzer()
    for nid in range(max_neurons):
        rm, _, _, _ = compute_rate_map_from_binned(xx, yy, spikes[:, nid], bins=35)
        ac = compute_2d_autocorrelation(rm)
        result = analyzer.compute_gridness_score(ac)
        if np.isfinite(result.score):
            scores.append(float(result.score))
            if result.score > best_score:
                best_score = float(result.score)
                best_id = nid
                best_result = result

    if best_result is None:
        raise SystemExit("Failed to compute any grid scores.")

    print(f"Best grid score (neuron {best_id}): {best_result.score:.3f}")
    print(f"Spacing: {best_result.spacing}")
    print(f"Orientation: {best_result.orientation}")

    rm, _, _, _ = compute_rate_map_from_binned(xx, yy, spikes[:, best_id], bins=35)
    ac = compute_2d_autocorrelation(rm)

    plot_rate_map(rm, show=True)
    plot_autocorrelogram(ac, gridness_score=best_result.score, show=True)

    if scores:
        plot_grid_score_histogram(np.asarray(scores), bins=30, show=True)


if __name__ == "__main__":
    main()
