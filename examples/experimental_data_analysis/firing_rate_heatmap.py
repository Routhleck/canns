#!/usr/bin/env python3
"""
Simple firing-rate heatmap example using the built-in grid dataset.
"""

from pathlib import Path

from canns.analyzer import data
from canns.analyzer.visualization import PlotConfigs
from canns.data.loaders import load_grid_data

grid_data = load_grid_data()

spike_cfg = data.SpikeEmbeddingConfig(smooth=False, speed_filter=True, min_speed=2.5)
spikes, *_ = data.embed_spike_trains(grid_data, config=spike_cfg)

mat = data.compute_fr_heatmap_matrix(spikes, transpose=True, normalize=None)

out_dir = Path("Results/examples/fr")
out_dir.mkdir(parents=True, exist_ok=True)

config = PlotConfigs.fr_heatmap(
    show=True,
    save_path=out_dir / "fr.png",
)

data.save_fr_heatmap_png(
    mat,
    config=config,
    dpi=200,
)
