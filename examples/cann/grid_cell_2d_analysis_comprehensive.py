"""
Grid Cell 2D Network - Comprehensive Analysis Example

Demonstrates complete grid cell analysis workflow including:
- Firing field heatmaps (spatial firing patterns)
- Spatial autocorrelation (reveals hexagonal periodicity)
- Grid score calculation (quantifies hexagonal symmetry)
- Grid spacing measurement (measures field spacing)
- Tracking animation (visualizes real-time behavior)
"""

import brainpy.math as bm
import numpy as np
from pathlib import Path

from canns.models.basic import GridCell2D
from canns.task.open_loop_navigation import OpenLoopNavigationTask
from canns.analyzer.metrics.spatial_metrics import (
    compute_firing_field,
    gaussian_smooth_heatmaps,
    compute_spatial_autocorrelation,
    compute_grid_score,
    find_grid_spacing,
)
from canns.analyzer.visualization import (
    plot_firing_field_heatmap,
    plot_autocorrelation,
    plot_grid_score,
    plot_grid_spacing_analysis,
    create_grid_cell_tracking_animation,
    PlotConfig,
    PlotConfigs,
)

# Setup
dt = 1.0
bm.set_dt(dt)
env_size = 3.0
spatial_bins = 80
num_cells_to_analyze = 3
output_dir = Path("outputs/grid_cell_2d_analysis_comprehensive")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate trajectory
task = OpenLoopNavigationTask(
    duration=4000.0,
    width=env_size,
    height=env_size,
    start_pos=[0.5, 0.5],
    speed_mean=0.4,
    speed_std=0.1,
    dt=dt / 10.0,
)
task.get_data()
position = task.data.position

# Initialize and run network
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


def run_step(_i, pos):
    gc_model(pos)
    return gc_model.r.value, gc_model.center_position.value


activity, decoded_positions = bm.for_loop(
    run_step,
    (bm.arange(len(position)), position),
    progress_bar=True
)

print(f"Simulation complete. Position error: {bm.mean(bm.linalg.norm(position - decoded_positions, axis=1)):.4f}m")

# Compute firing fields
activity_np = np.asarray(activity)
position_np = np.asarray(position)

firing_fields = compute_firing_field(
    activity_np, position_np, width=env_size, height=env_size, M=spatial_bins, K=spatial_bins
)
firing_fields_smooth = gaussian_smooth_heatmaps(firing_fields, sigma=1.0)

# Select top cells for analysis
max_rates = np.max(firing_fields_smooth, axis=(1, 2))
top_cell_indices = np.argsort(max_rates)[-num_cells_to_analyze:][::-1]

# Analyze each top cell
print(f"\nAnalyzing top {num_cells_to_analyze} cells:")
print("=" * 60)

for idx, cell_idx in enumerate(top_cell_indices):
    rate_map = firing_fields_smooth[cell_idx]

    # Compute grid cell metrics
    autocorr = compute_spatial_autocorrelation(rate_map)
    grid_score, rotated_corrs = compute_grid_score(autocorr)
    spacing_bins, spacing_real = find_grid_spacing(autocorr, bin_size=env_size / spatial_bins)

    # Print results
    print(f"\nCell {cell_idx} (Rank #{idx+1}):")
    print(f"  Grid Score: {grid_score:.3f} {'✓' if grid_score > 0.3 else ''}")
    print(f"  Spacing: {spacing_bins:.1f} bins = {spacing_real:.3f}m (theory: {gc_model.Lambda:.3f}m)")
    print(f"  Correlations: 60°={rotated_corrs[60]:.2f}, 120°={rotated_corrs[120]:.2f}")

    # Generate visualizations
    plot_firing_field_heatmap(
        rate_map,
        config=PlotConfig(
            title=f"Cell {cell_idx} Firing Field",
            figsize=(5, 5),
            save_path=str(output_dir / f"cell_{cell_idx}_firing_field.png"),
            show=False,
        ),
    )

    plot_autocorrelation(
        autocorr,
        config=PlotConfigs.grid_autocorrelation(
            title=f"Cell {cell_idx} Autocorrelation",
            save_path=str(output_dir / f"cell_{cell_idx}_autocorr.png"),
            show=False,
        ),
    )

    plot_grid_score(
        rotated_corrs,
        grid_score,
        config=PlotConfigs.grid_score_plot(
            title=f"Cell {cell_idx} Grid Score",
            save_path=str(output_dir / f"cell_{cell_idx}_gridscore.png"),
            show=False,
        ),
    )

    plot_grid_spacing_analysis(
        autocorr,
        spacing_bins,
        bin_size=env_size / spatial_bins,
        config=PlotConfigs.grid_spacing_plot(
            title=f"Cell {cell_idx} Spacing",
            save_path=str(output_dir / f"cell_{cell_idx}_spacing.png"),
            show=False,
        ),
    )

# Create tracking animation for best cell
best_cell_idx = top_cell_indices[0]
activity_single = activity_np[:, best_cell_idx]
rate_map_best = firing_fields_smooth[best_cell_idx]

print(f"\nCreating animation for Cell {best_cell_idx}...")
create_grid_cell_tracking_animation(
    position_np,
    activity_single,
    rate_map_best,
    config=PlotConfigs.grid_cell_tracking_animation(
        time_steps_per_second=int(500 / dt),
        fps=20,
        title=f"Cell {best_cell_idx} Tracking",
        save_path=str(output_dir / f"cell_{best_cell_idx}_tracking.mp4"),
        show=False,
    ),
    env_size=env_size,
    dt=dt,
)

# Summary
all_grid_scores = []
all_spacings = []
for cell_idx in top_cell_indices:
    autocorr = compute_spatial_autocorrelation(firing_fields_smooth[cell_idx])
    grid_score, _ = compute_grid_score(autocorr)
    spacing_bins, spacing_real = find_grid_spacing(autocorr, bin_size=env_size / spatial_bins)
    all_grid_scores.append(grid_score)
    all_spacings.append(spacing_real)

print("\n" + "=" * 60)
print("Summary:")
print(f"  Grid scores: mean={np.mean(all_grid_scores):.3f}, range=[{np.min(all_grid_scores):.3f}, {np.max(all_grid_scores):.3f}]")
print(f"  Cells above threshold (>0.3): {np.sum(np.array(all_grid_scores) > 0.3)}/{len(all_grid_scores)}")
print(f"  Spacing: mean={np.mean(all_spacings):.3f}m, std={np.std(all_spacings):.3f}m (theory={gc_model.Lambda:.3f}m)")
print(f"\nAll outputs saved to {output_dir}/")
print("Generated files:")
for cell_idx in top_cell_indices:
    print(f"  • cell_{cell_idx}_{{firing_field,autocorr,gridscore,spacing}}.png")
print(f"  • cell_{best_cell_idx}_tracking.mp4")
