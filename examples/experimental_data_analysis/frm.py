#!/usr/bin/env python3
"""
Simple firing-rate map example using the built-in grid dataset.
"""

from pathlib import Path

from canns.analyzer import data
from canns.analyzer.visualization import PlotConfigs
from canns.data.loaders import load_grid_data

asa = load_grid_data()

spike_cfg = data.SpikeEmbeddingConfig(smooth=False, speed_filter=True, min_speed=2.5)
spikes, *_ = data.embed_spike_trains(asa, config=spike_cfg)

x = asa["x"]
y = asa["y"]
neuron_id = 0

frm_res = data.compute_frm(
    spikes,
    x,
    y,
    neuron_id,
    bins=50,
    min_occupancy=1,
    smoothing=False,
    sigma=1.0,
    nan_for_empty=True,
)

out_dir = Path("Results/examples/frm")
out_dir.mkdir(parents=True, exist_ok=True)

config = PlotConfigs.frm(show=True)

data.save_frm_png(
    frm_res.frm,
    str(out_dir / "frm_neuron0000.png"),
    config=config,
    dpi=200,
)
