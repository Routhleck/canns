#!/usr/bin/env python3
"""
Simple path-compare example using the built-in grid dataset.
"""

import numpy as np

from canns.analyzer import data
from canns.analyzer.visualization import PlotConfig, finalize_figure
from canns.data.loaders import load_grid_data


def _plot_static(
    x: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    config: PlotConfig,
) -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    coords = np.asarray(coords)

    fig, axes = plt.subplots(1, 2, figsize=config.figsize)

    ax0 = axes[0]
    ax0.set_title("Physical path (x,y)")
    ax0.set_aspect("equal", "box")
    ax0.axis("off")
    ax0.plot(x, y, lw=0.9, alpha=0.8)

    ax1 = axes[1]
    ax1.set_title("Decoded coho path")
    ax1.set_aspect("equal", "box")
    ax1.axis("off")

    if coords.shape[1] >= 2:
        theta2 = coords[:, :2] % (2 * np.pi)
        xy = data.skew_transform(theta2)
        data.draw_base_parallelogram(ax1)
        trail = data.snake_wrap_trail_in_parallelogram(
            xy, np.array([2 * np.pi, 0.0]), np.array([np.pi, np.sqrt(3) * np.pi])
        )
        ax1.plot(trail[:, 0], trail[:, 1], lw=0.9, alpha=0.9)
    else:
        th = coords[:, 0] % (2 * np.pi)
        ax1.plot(np.cos(th), np.sin(th), lw=0.9, alpha=0.9)
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)

    fig.tight_layout()
    finalize_figure(fig, config)


asa = load_grid_data()

spike_cfg = data.SpikeEmbeddingConfig(smooth=False, speed_filter=True, min_speed=2.5)
spikes, *_ = data.embed_spike_trains(asa, config=spike_cfg)

asa_embedded = dict(asa)
asa_embedded["spike"] = spikes

tda_cfg = data.TDAConfig(maxdim=1, do_shuffle=False, show=False, progress_bar=True)
result = data.tda_vis(embed_data=spikes, config=tda_cfg)

decoding = data.decode_circular_coordinates2(
    persistence_result=result,
    spike_data=asa_embedded,
    num_circ=2,
)

coords = np.asarray(decoding.get("coords"))
if coords.ndim != 2:
    raise ValueError(f"decoding['coords'] must be 2D, got {coords.shape}")

# align decoded coords to full trajectory
_t = np.asarray(asa["t"]).ravel()
_x = np.asarray(asa["x"]).ravel()
_y = np.asarray(asa["y"]).ravel()

_t_use, x_use, y_use, coords_use, _ = data.align_coords_to_position(
    t_full=_t,
    x_full=_x,
    y_full=_y,
    coords2=coords,
    use_box=True,
    times_box=decoding.get("times_box", None),
    interp_to_full=True,
)

coords_use = data.apply_angle_scale(coords_use, "rad")

plot_config = PlotConfig.for_static_plot(
    title="Path Compare",
    figsize=(12, 5),
    show=True,
)

_plot_static(x_use, y_use, coords_use, plot_config)
