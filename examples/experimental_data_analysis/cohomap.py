#!/usr/bin/env python3
"""
Simple CohoMap example using the built-in grid dataset.
"""

from canns.analyzer import data
from canns.analyzer.visualization import PlotConfigs
from canns.data.loaders import load_grid_data

grid_data = load_grid_data()

spike_cfg = data.SpikeEmbeddingConfig(smooth=True, speed_filter=False, min_speed=2.5)
spikes, *_ = data.embed_spike_trains(grid_data, config=spike_cfg)

grid_data_embedded = dict(grid_data)
grid_data_embedded["spike"] = spikes

tda_cfg = data.TDAConfig(maxdim=1, do_shuffle=False, show=True, progress_bar=True)
result = data.tda_vis(embed_data=spikes, config=tda_cfg)

decoding = data.decode_circular_coordinates2(
    persistence_result=result,
    spike_data=grid_data_embedded,
    num_circ=2,
)

config = PlotConfigs.cohomap(show=True)

data.plot_cohomap1(
    decoding_result=decoding,
    position_data={"x": grid_data["x"], "y": grid_data["y"]},
    config=config,
)
