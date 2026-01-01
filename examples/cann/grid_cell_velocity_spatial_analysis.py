"""
Grid Cell Velocity Model - Spatial Analysis

Demonstrates spatial analysis of GridCell2DVelocity model using systematic
spatial sampling for high-quality rate maps and grid scores (typically >0.6).

Based on Burak & Fiete (2009) grid cell model with velocity-based path integration.
"""

import brainpy.math as bm
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from canns.models.basic import GridCell2DVelocity
from canns.analyzer.metrics.spatial_metrics import (
    gaussian_smooth_heatmaps,
    compute_spatial_autocorrelation,
    compute_grid_score,
    find_grid_spacing,
)
from canns.analyzer.visualization import (
    plot_firing_field_heatmap,
    plot_autocorrelation,
    plot_grid_score,
    PlotConfigs,
)
from canns.analyzer.metrics.systematic_ratemap import compute_systematic_ratemap

# Setup
dt_sim = 5e-4
bm.set_dt(dt_sim)

output_dir = Path("outputs/burak_spatial_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

num_cells_to_analyze = 5
spatial_resolution = 100

# Initialize model
model = GridCell2DVelocity(
    length=50,
    tau=0.01,
    alpha=0.1,
    W_l=2.0,
    lambda_net=17.0,
)

print(f"Network: {model.num} neurons")

# Healing
print("Running healing process...")
model.heal_network(num_healing_steps=10000, dt_healing=1e-4)

# Compute rate maps via systematic sampling
print("Computing rate maps via systematic sampling...")

box_width = 2.2
box_height = 2.2

ratemaps = compute_systematic_ratemap(
    model,
    box_width=box_width,
    box_height=box_height,
    resolution=spatial_resolution,
    speed=0.3,
    num_batches=10,
    verbose=True,
)

# Transpose to (num_neurons, height, width) format
firing_fields = np.transpose(ratemaps, (2, 0, 1))
firing_fields_smooth = gaussian_smooth_heatmaps(firing_fields, sigma=2.0)

# Select top cells for analysis
max_rates = np.max(firing_fields_smooth, axis=(1, 2))
top_cell_indices = np.argsort(max_rates)[-num_cells_to_analyze:][::-1]

# Analyze each cell
for cell_idx in top_cell_indices:
    rate_map = firing_fields_smooth[cell_idx]

    autocorr = compute_spatial_autocorrelation(rate_map)
    grid_score, rotated_corrs = compute_grid_score(autocorr)

    bin_size = box_width / spatial_resolution
    spacing_bins, spacing_real = find_grid_spacing(autocorr, bin_size=bin_size)

    print(f"Cell {cell_idx}: Grid Score={grid_score:.3f}, Spacing={spacing_real:.3f}m")

    # Visualizations
    plot_firing_field_heatmap(
        rate_map,
        config=PlotConfigs.firing_field_heatmap(
            title=f"Cell {cell_idx} Firing Field (Grid Score={grid_score:.3f})",
            save_path=str(output_dir / f"cell{cell_idx}_ratemap.png"),
            show=False,
        ),
    )

    plot_autocorrelation(
        autocorr,
        config=PlotConfigs.grid_autocorrelation(
            title=f"Cell {cell_idx} Spatial Autocorrelation",
            save_path=str(output_dir / f"cell{cell_idx}_autocorr.png"),
            show=False,
        ),
    )

    plot_grid_score(
        rotated_corrs,
        grid_score,
        config=PlotConfigs.grid_score_plot(
            save_path=str(output_dir / f"cell{cell_idx}_gridscore.png"), show=False
        ),
    )

# Summary figure with all rate maps
fig, axes = plt.subplots(1, num_cells_to_analyze, figsize=(4 * num_cells_to_analyze, 4))

if num_cells_to_analyze == 1:
    axes = [axes]

for i, cell_idx in enumerate(top_cell_indices):
    rate_map = firing_fields_smooth[cell_idx]
    autocorr = compute_spatial_autocorrelation(rate_map)
    grid_score, _ = compute_grid_score(autocorr)

    im = axes[i].imshow(rate_map, cmap="jet", origin="lower", aspect="auto")
    axes[i].set_title(f"Cell {cell_idx}\nGrid Score: {grid_score:.3f}", fontsize=10)
    axes[i].set_xlabel("X bin")
    axes[i].set_ylabel("Y bin")
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()
summary_path = output_dir / "summary_all_cells.png"
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()

# Summary
grid_scores = [
    compute_grid_score(compute_spatial_autocorrelation(firing_fields_smooth[c]))[0]
    for c in top_cell_indices
]
avg_grid_score = np.mean(grid_scores)

print(f"\nMean grid score: {avg_grid_score:.3f}")
print(f"Outputs saved to: {output_dir}/")
