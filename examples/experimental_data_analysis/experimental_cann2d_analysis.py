#!/usr/bin/env python3
"""
CANN 2D Analysis

This example demonstrates how to use the bump_fits and create_1d_bump_animation functions
from the experimental data analyzer to analyze 1D CANN bumps.
"""

from canns.analyzer.data.cann2d_metrics import (
    embed_spike_trains,
    plot_projection, tda_vis, decode_circular_coordinates, plot_3d_bump_on_torus,
    CANN2DPlotConfig, SpikeEmbeddingConfig, TDAConfig
)

from canns.data.loaders import load_grid_data

data = load_grid_data()

# Using config-based approach for spike embedding
spike_config = SpikeEmbeddingConfig(
    smooth=True,
    speed_filter=True,
    min_speed=2.5
)

embed_spike, *_ = embed_spike_trains(data, config=spike_config)

import umap

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    metric='euclidean',
    random_state=42
)

reduce_func = reducer.fit_transform

# Using config-based approach for 3D projection
proj_config = CANN2DPlotConfig.for_projection_3d(
    title="3D Spike Data Projection",
    xlabel="UMAP Component 1",
    ylabel="UMAP Component 2",
    zlabel="UMAP Component 3",
    show=True,
    dpi=150
)

plot_projection(
    reduce_func=reduce_func,
    embed_data=embed_spike,
    config=proj_config
)

# Old-style approach still works:
# plot_projection(reduce_func=reduce_func, embed_data=embed_spike, show=False)
# Using config-based approach for TDA
tda_config = TDAConfig(
    maxdim=1,
    do_shuffle=False,
    # num_shuffles=10,
    show=True,
    dim=6,
    n_points=1200,
    progress_bar=True,
)

persistence_result = tda_vis(
    embed_data=embed_spike,
    config=tda_config
)

# Old-style approach still works:
# persistence_result = tda_vis(
#     embed_data=embed_spike, maxdim=1, do_shuffle=False, show=False
# )

# results = tda_vis(
#     embed_data=embed_spike, maxdim=1, do_shuffle=True, num_shuffles=10, show=True
# )

decode = decode_circular_coordinates(
    persistence_result=persistence_result,
    spike_data=data,
)

# Using config-based approach for torus animation
torus_config = CANN2DPlotConfig.for_torus_animation(
    show=True,
    save_path='../experimental_cann2d_analysis_torus.mp4',
    n_frames=20,
    fps=5,
    title="3D Bump Movement on Torus",
    window_size=300,
    show_progress_bar=True
)

plot_3d_bump_on_torus(
    decoding_result=decode,
    spike_data=data,
    config=torus_config
)

# Old-style approach still works:
# plot_3d_bump_on_torus(
#     decoding_result=decode,
#     spike_data=data,
#     show=False,
#     save_path='experimental_cann2d_analysis_torus_old.mp4',
#     n_frames=20
# )
