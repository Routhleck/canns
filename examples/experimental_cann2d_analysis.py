#!/usr/bin/env python3
"""
CANN 2D Analysis

This example demonstrates how to use the bump_fits and create_1d_bump_animation functions
from the experimental data analyzer to analyze 1D CANN bumps.
"""

import numpy as np
from canns.analyzer.experimental_data import bump_fits, create_1d_bump_animation, load_roi_data, embed_spike_trains, \
    plot_projection, tda_vis, decode_circular_coordinates, plot_3d_bump_on_torus

from canns.analyzer.experimental_data._datasets_utils import load_grid_data

data = load_grid_data()

embed_spike, *_ = embed_spike_trains(data)

import umap

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    metric='euclidean',
    random_state=42
)

reduce_func = reducer.fit_transform

plot_projection(reduce_func=reduce_func, embed_data=embed_spike, show=True)
persistence_result = tda_vis(
    embed_data=embed_spike, maxdim=1, do_shuffle=False, show=True
)

# results = tda_vis(
#     embed_data=embed_spike, maxdim=1, do_shuffle=True, num_shuffles=10, show=True
# )

decode = decode_circular_coordinates(
    persistence_result=persistence_result,
    spike_data=data,
)

plot_3d_bump_on_torus(
    decoding_result=decode,
    spike_data=data,
    show=True,
    save_path='experimental_cann2d_analysis_torus.gif',
    n_frames=20
)
