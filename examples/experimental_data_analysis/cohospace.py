#!/usr/bin/env python3
"""
Simple CohoSpace example using the built-in grid dataset.
"""

from canns.analyzer import data
from canns.analyzer.visualization import PlotConfigs
from canns.data.loaders import load_grid_data

asa = load_grid_data()

spike_cfg = data.SpikeEmbeddingConfig(smooth=True, speed_filter=True, min_speed=2.5)
spikes, *_ = data.embed_spike_trains(asa, config=spike_cfg)

asa_embedded = dict(asa)
asa_embedded["spike"] = spikes

tda_cfg = data.TDAConfig(maxdim=1, do_shuffle=False, show=True, progress_bar=True)
result = data.tda_vis(embed_data=spikes, config=tda_cfg)

decoding = data.decode_circular_coordinates2(
    persistence_result=result,
    spike_data=asa_embedded,
    num_circ=2,
)

coords = decoding.get("coords")
coordsbox = decoding.get("coordsbox")
times_box = decoding.get("times_box")
if coords is None:
    raise KeyError("decoding is missing 'coords'.")

# config = PlotConfigs.cohospace_trajectory(show=True)
#
# data.plot_cohospace_trajectory(
#     coords=coords[:, :2],
#     times=None,
#     subsample=2,
#     config=config,
# )

config = PlotConfigs.cohospace_neuron(show=True)

data.plot_cohospace_neuron(
    coords=coordsbox[:, :2],
    activity=spikes,
    times=times_box,
    neuron_id=31,
    mode='spike',
    top_percent=1,
    config=config,
)
