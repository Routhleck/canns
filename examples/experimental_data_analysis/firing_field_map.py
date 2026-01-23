#!/usr/bin/env python3
"""
Simple firing-rate map example using the built-in grid dataset.
"""

from pathlib import Path

from canns.analyzer import data
from canns.analyzer.visualization import PlotConfigs
from canns.data.loaders import load_grid_data

grid_data = load_grid_data()

spike_cfg = data.SpikeEmbeddingConfig(smooth=True, speed_filter=False, min_speed=2.5)
spikes, *_ = data.embed_spike_trains(grid_data, config=spike_cfg)

x = grid_data["x"]
y = grid_data["y"]
neuron_id = 130

frm_res = data.compute_frm(
    spikes,
    x,
    y,
    neuron_id,
    bins=50,
    min_occupancy=1,
    smoothing=True,
    sigma=1.0,
    nan_for_empty=True,
)

out_dir = Path("Results/examples/frm")
out_dir.mkdir(parents=True, exist_ok=True)

config = PlotConfigs.frm(
    show=True,
    save_path=f"{out_dir}/frm.png",
)

data.plot_frm(
    frm_res.frm,
    config=config,
    dpi=200,
)
