import os
import time

import brainpy.math as bm
import numpy as np

from canns.models.basic import HierarchicalNetwork
from canns.task.open_loop_navigation import OpenLoopNavigationTask

PATH = os.path.dirname(os.path.abspath(__file__))

bm.set_dt(dt=0.05)
task_sn = OpenLoopNavigationTask(
    width=5,
    height=5,
    speed_mean=0.04,
    speed_std=0.016,
    duration=50000.0,
    dt=0.05,
    start_pos=(2.5, 2.5),
    progress_bar=True,
)

trajectory_file_path = os.path.join(PATH, 'trajectory_test.npz')
trajectory_graph_file_path = os.path.join(PATH, 'trajectory_graph.png')

# if os.path.exists(trajectory_file_path):
#     print(f"Loading trajectory from {trajectory_file_path}")
#     task_sn.load_data(trajectory_file_path)
# else:
print(f"Generating new trajectory and saving to {trajectory_file_path}")
task_sn.get_data()
task_sn.show_data(show=False, save_path=trajectory_graph_file_path)
# task_sn.save_data(trajectory_file_path)

hierarchical_net = HierarchicalNetwork(num_module=5, num_place=30)


def initialize(t, input_stre):
    hierarchical_net(
        velocity=bm.zeros(2, ),
        loc=task_sn.data.position[0],
        loc_input_stre=input_stre,
    )


init_time = 500
indices = np.arange(init_time)
input_stre = np.zeros(init_time)
input_stre[:400] = 100.
bm.for_loop(
    initialize,
    (
        bm.asarray(indices), bm.asarray(input_stre)
    ),
    progress_bar=100
)


def run_step(t, vel, loc):
    hierarchical_net(velocity=vel, loc=loc, loc_input_stre=0.)
    band_x_r = hierarchical_net.band_x_fr.value
    band_y_r = hierarchical_net.band_y_fr.value
    grid_r = hierarchical_net.grid_fr.value
    place_r = hierarchical_net.place_fr.value
    return band_x_r, band_y_r, grid_r, place_r


total_time = task_sn.data.velocity.shape[0]
indices = np.arange(total_time)

band_x_r, band_y_r, grid_r, place_r = bm.for_loop(
    run_step,
    (
        bm.asarray(indices),
        bm.asarray(task_sn.data.velocity),
        bm.asarray(task_sn.data.position)
    ),
    progress_bar=10000
)

# activity_file_path = os.path.join(PATH, 'band_grid_place_activity.npz')
#
# np.savez(
#     activity_file_path,
#     band_x_r=band_x_r,
#     band_y_r=band_y_r,
#     grid_r=grid_r,
#     place_r=place_r,
# )

#### Visualization
from tqdm import tqdm

from canns.analyzer.metrics.spatial_metrics import compute_firing_field, gaussian_smooth_heatmaps
from canns.analyzer.visualization import PlotConfig, plot_firing_field_heatmap

np.random.seed(10)

# trajectory = np.load(trajectory_file_path)
loc = task_sn.data.position

# load the neuron activity
# data = np.load(activity_file_path)
# band_x_r = data['band_x_r']
# band_y_r = data['band_y_r']
# grid_r = data['grid_r']
# place_r = data['place_r']

loc = np.array(loc)
width = 5
height = 5
M = int(width * 10)
K = int(height * 10)

T = grid_r.shape[0]

print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - Reshaping data...")

# Reshape directly without unnecessary np.array() conversion
grid_r = grid_r.reshape(T, -1)
band_x_r = band_x_r.reshape(T, -1)
band_y_r = band_y_r.reshape(T, -1)
place_r = place_r.reshape(T, -1)

print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - Computing heatmaps...")

heatmaps_grid = compute_firing_field(np.array(grid_r), loc, width, height, M, K)
heatmaps_band_x = compute_firing_field(np.array(band_x_r), loc, width, height, M, K)
heatmaps_band_y = compute_firing_field(np.array(band_y_r), loc, width, height, M, K)
heatmaps_place = compute_firing_field(np.array(place_r), loc, width, height, M, K)

# heatmap_file_path = os.path.join(PATH, 'band_grid_place_heatmap.npz')
# np.savez(
#     heatmap_file_path,
#     heatmaps_grid=heatmaps_grid,
#     heatmaps_band_x=heatmaps_band_x,
#     heatmaps_band_y=heatmaps_band_y,
#     heatmaps_place=heatmaps_place,
# )

print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - Plotting heatmaps...")

heatmaps_grid = gaussian_smooth_heatmaps(heatmaps_grid)
heatmaps_band_x = gaussian_smooth_heatmaps(heatmaps_band_x)
heatmaps_band_y = gaussian_smooth_heatmaps(heatmaps_band_y)
heatmaps_place = gaussian_smooth_heatmaps(heatmaps_place)

print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - Reshaping heatmaps...")

heatmaps_band_x = heatmaps_band_x.reshape(5, -1, M, K)
heatmaps_band_y = heatmaps_band_y.reshape(5, -1, M, K)
heatmaps_grid = heatmaps_grid.reshape(5, -1, M, K)

output_dir = os.path.join(PATH, 'heatmap_figures')
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Configuration for selective saving of heatmaps
# ============================================================================
# SAVE_MODULES: List of module indices to save for band/grid cells
#   - [0]: Save only module 0
#   - [0, 1, 2]: Save modules 0, 1, and 2
#   - None: Save all modules (default behavior)
SAVE_MODULES = None

# SAVE_CELLS: List of cell indices to save within each module
#   - [0, 1, 2]: Save only cells 0, 1, and 2 from each selected module
#   - None: Save all cells in each selected module (default behavior)
SAVE_CELLS = [0, 1, 2, 3, 4]

# SAVE_PLACE_CELLS: List of place cell indices to save
#   - [0, 5, 10]: Save only place cells 0, 5, and 10
#   - None: Save all place cells (default behavior)
SAVE_PLACE_CELLS = [0, 5, 10, 15, 20]

# Save band cell heatmaps
for heatmaps, prefix in [(heatmaps_band_x, 'heatmap_band_x'),
                         (heatmaps_band_y, 'heatmap_band_y'),
                         (heatmaps_grid, 'heatmap_grid')]:
    num_modules, num_cells = heatmaps.shape[:2]

    # Determine which modules and cells to save
    modules_to_save = SAVE_MODULES if SAVE_MODULES is not None else range(num_modules)
    cells_to_save = SAVE_CELLS if SAVE_CELLS is not None else range(num_cells)

    total = len(modules_to_save) * len(cells_to_save)
    with tqdm(total=total, desc=f'Saving {prefix} heatmaps') as pbar:
        for module_idx in modules_to_save:
            for cell_idx in cells_to_save:
                filename = f'{prefix}_module_{module_idx}_cell_{cell_idx}.png'
                save_path = os.path.join(output_dir, filename)
                config = PlotConfig(figsize=(5, 5), save_path=save_path, show=False)
                plot_firing_field_heatmap(heatmaps[module_idx, cell_idx], config=config)
                pbar.update(1)

# Save place cell heatmaps
place_cells_to_save = SAVE_PLACE_CELLS if SAVE_PLACE_CELLS is not None else range(heatmaps_place.shape[0])
for cell_idx in tqdm(place_cells_to_save, desc='Saving place cell heatmaps'):
    filename = f'heatmap_place_cell_{cell_idx}.png'
    save_path = os.path.join(output_dir, filename)
    config = PlotConfig(figsize=(5, 5), save_path=save_path, show=False)
    plot_firing_field_heatmap(heatmaps_place[cell_idx], config=config)
