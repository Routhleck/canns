#!/usr/bin/env python3
"""
Simple TDA example using the built-in grid dataset.
"""

from canns.analyzer import data
from canns.data.loaders import load_grid_data

grid_data = load_grid_data()

spike_cfg = data.SpikeEmbeddingConfig(smooth=True, speed_filter=False, min_speed=2.5)
spikes, *_ = data.embed_spike_trains(grid_data, config=spike_cfg)

tda_cfg = data.TDAConfig(maxdim=1, do_shuffle=False, show=True, progress_bar=True)

data.tda_vis(embed_data=spikes, config=tda_cfg)
