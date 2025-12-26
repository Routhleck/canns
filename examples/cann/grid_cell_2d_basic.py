"""
Basic Grid Cell 2D Network Example

Demonstrates GridCell2D model with hexagonal firing patterns.
"""

import brainpy.math as bm
import numpy as np
from pathlib import Path
from tqdm import tqdm

from canns.models.basic import GridCell2D
from canns.task.open_loop_navigation import OpenLoopNavigationTask
from canns.analyzer.metrics.spatial_metrics import compute_firing_field, gaussian_smooth_heatmaps
from canns.analyzer.visualization import PlotConfig, plot_firing_field_heatmap

# Setup
dt = 1.0
bm.set_dt(dt)
output_dir = Path("outputs/grid_cell_2d_basic")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate navigation trajectory
task = OpenLoopNavigationTask(
    duration=40000.0,
    width=3.0,
    height=3.0,
    start_pos=[1.0, 1.0],
    speed_mean=0.6,
    speed_std=0.1,
    dt=dt / 10.0,
)
task.get_data()
position = task.data.position

# Initialize network
gc_model = GridCell2D(
    length=40,
    mapping_ratio=6.0,  # Reduced grid spacing for visible hexagonal pattern
    tau=10.0,
    k=0.5,
    a=0.8,
    A=15.0,
    J0=10.0,
    noise_strength=0.2,
    g=100.0,
)


# Run simulation
def run_step(_i, pos):
    gc_model(pos)
    return gc_model.r.value, gc_model.center_position.value


activity, decoded_positions = bm.for_loop(
    run_step,
    (bm.arange(len(position)), position),
    progress_bar=True
)

# Compute and smooth firing fields
firing_fields = compute_firing_field(
    np.asarray(activity),
    np.asarray(position),
    width=3.0,
    height=3.0,
    M=80,
    K=80
)
firing_fields_smooth = gaussian_smooth_heatmaps(firing_fields, sigma=1.0)

# Save top 8 cells
max_rates = np.max(firing_fields_smooth, axis=(1, 2))
top_indices = np.argsort(max_rates)[-8:][::-1]

for cell_idx in tqdm(top_indices, desc='Saving firing fields'):
    config = PlotConfig(
        figsize=(5, 5),
        save_path=str(output_dir / f'grid_cell_{cell_idx}_firing_field.png'),
        show=False
    )
    plot_firing_field_heatmap(firing_fields_smooth[cell_idx], config=config)

print(f"Saved {len(top_indices)} firing field heatmaps to {output_dir}/")
