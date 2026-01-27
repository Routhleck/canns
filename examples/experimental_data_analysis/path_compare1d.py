#!/usr/bin/env python3
"""
Simple 1D path-compare example using the built-in grid dataset.
"""

import numpy as np

from canns.analyzer import data
from canns.analyzer.visualization import PlotConfigs
from canns.data.loaders import load_grid_data


asa = load_grid_data()

spike_cfg = data.SpikeEmbeddingConfig(smooth=True, speed_filter=False, min_speed=2.5)
spikes, *_ = data.embed_spike_trains(asa, config=spike_cfg)

asa_embedded = dict(asa)
asa_embedded["spike"] = spikes

tda_cfg = data.TDAConfig(maxdim=1, do_shuffle=False, show=True, progress_bar=True)
result = data.tda_vis(embed_data=spikes, config=tda_cfg)

decoding = data.decode_circular_coordinates_multi(
    persistence_result=result,
    spike_data=asa_embedded,
    num_circ=1,
)

coords = np.asarray(decoding.get("coords"))
if coords.ndim == 2 and coords.shape[1] == 1:
    coords = coords[:, 0]
if coords.ndim != 1:
    raise ValueError(f"decoding['coords'] must be 1D, got {coords.shape}")

# align decoded coords to full trajectory
_t = np.asarray(asa["t"]).ravel()
_x = np.asarray(asa["x"]).ravel()
_y = np.asarray(asa["y"]).ravel()

_t_use, x_use, y_use, coords_use, _ = data.align_coords_to_position_1d(
    t_full=_t,
    x_full=_x,
    y_full=_y,
    coords1=coords,
    use_box=True,
    times_box=decoding.get("times_box", None),
    interp_to_full=True,
)

coords_use = data.apply_angle_scale(coords_use, "rad")

plot_config = PlotConfigs.path_compare_1d(show=True)

data.plot_path_compare_1d(x_use, y_use, coords_use, config=plot_config)
