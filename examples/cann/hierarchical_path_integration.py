import time

import brainstate
import brainunit as u
import jax
import numpy as np
import os

from canns.models.basic import HierarchicalNetwork
from canns.task.open_loop_navigation import OpenLoopNavigationTask

PATH = os.path.dirname(os.path.abspath(__file__))


brainstate.environ.set(dt=0.05)
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

if os.path.exists(trajectory_file_path):
    print(f"Loading trajectory from {trajectory_file_path}")
    task_sn.load_data(trajectory_file_path)
else:
    print(f"Generating new trajectory and saving to {trajectory_file_path}")
    task_sn.get_data()
    task_sn.show_data(show=False, save_path=trajectory_graph_file_path)
    task_sn.save_data(trajectory_file_path)

hierarchical_net = HierarchicalNetwork(num_module=5, num_place=30)
hierarchical_net.init_state()

def initialize(t, input_stre):
    hierarchical_net(
        velocity=u.math.zeros(2, ),
        loc=task_sn.data.position[0],
        loc_input_stre=input_stre,
    )

init_time = 500
indices = np.arange(init_time)
input_stre = np.zeros(init_time)
input_stre[:400]=100.
brainstate.compile.for_loop(
    initialize,
    u.math.asarray(indices),
    u.math.asarray(input_stre),
    pbar=brainstate.compile.ProgressBar(10),
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


band_x_r, band_y_r, grid_r, place_r = brainstate.compile.for_loop(
    run_step,
    u.math.asarray(indices),
    u.math.asarray(task_sn.data.velocity),
    u.math.asarray(task_sn.data.position),
    pbar=brainstate.compile.ProgressBar(10),
)


activity_file_path = os.path.join(PATH, 'band_grid_place_activity.npz')

np.savez(
    activity_file_path,
    band_x_r=band_x_r,
    band_y_r=band_y_r,
    grid_r=grid_r,
    place_r=place_r,
)

#### Visualization
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


np.random.seed(10)


def gauss_filter(heatmaps, sigma: float = 1.0):
    """Apply Gaussian smoothing to each heatmap without mixing channels."""
    filtered = gaussian_filter(heatmaps, sigma=(0, sigma, sigma))
    return np.where(heatmaps == 0, 0, filtered)


@njit(parallel=True)
def compute_firing_field(A, positions, width, height, M, K):
    T, N = A.shape  # Number of time steps and neurons
    # Initialize the heatmaps and bin counters
    heatmaps = np.zeros((N, M, K))
    bin_counts = np.zeros((M, K))

    # Determine bin sizes
    bin_width = width / M
    bin_height = height / K
    # Assign positions to bins
    x_bins = np.clip(((positions[:, 0]) // bin_width).astype(np.int32), 0, M - 1)
    y_bins = np.clip(((positions[:, 1]) // bin_height).astype(np.int32), 0, K - 1)

    # Accumulate activity in each bin
    for t in prange(T):
        x_bin = x_bins[t]
        y_bin = y_bins[t]
        heatmaps[:, x_bin, y_bin] += A[t, :]
        bin_counts[x_bin, y_bin] += 1

    # Compute average firing rate per bin (avoid division by zero)
    for n in range(N):
        heatmaps[n] = np.where(bin_counts > 0, heatmaps[n] / bin_counts, 0)

    return heatmaps


trajectory = np.load(trajectory_file_path)
loc = trajectory['position']

# load the neuron activity
data = np.load(activity_file_path)
band_x_r = data['band_x_r']
band_y_r = data['band_y_r']
grid_r = data['grid_r']
place_r = data['place_r']

loc = np.array(loc)
width = 5
height = 5
M = int(width * 10)
K = int(height * 10)

T = grid_r.shape[0]

grid_r = np.array(grid_r).reshape(T, -1)
band_x_r = np.array(band_x_r).reshape(T, -1)
band_y_r = np.array(band_y_r).reshape(T, -1)
place_r = np.array(place_r).reshape(T, -1)

heatmaps_grid = compute_firing_field(grid_r, loc, width, height, M, K)
heatmaps_band_x = compute_firing_field(band_x_r, loc, width, height, M, K)
heatmaps_band_y = compute_firing_field(band_y_r, loc, width, height, M, K)
heatmaps_place = compute_firing_field(place_r, loc, width, height, M, K)

heatmap_file_path = os.path.join(PATH, 'band_grid_place_heatmap.npz')
np.savez(
    heatmap_file_path,
    heatmaps_grid=heatmaps_grid,
    heatmaps_band_x=heatmaps_band_x,
    heatmaps_band_y=heatmaps_band_y,
    heatmaps_place=heatmaps_place,
)

heatmaps_grid = gauss_filter(heatmaps_grid)
heatmaps_band_x = gauss_filter(heatmaps_band_x)
heatmaps_band_y = gauss_filter(heatmaps_band_y)
heatmaps_place = gauss_filter(heatmaps_place)

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
SAVE_MODULES = [0]

# SAVE_CELLS: List of cell indices to save within each module
#   - [0, 1, 2]: Save only cells 0, 1, and 2 from each selected module
#   - None: Save all cells in each selected module (default behavior)
SAVE_CELLS = None

# SAVE_PLACE_CELLS: List of place cell indices to save
#   - [0, 5, 10]: Save only place cells 0, 5, and 10
#   - None: Save all place cells (default behavior)
SAVE_PLACE_CELLS = None


def save_heatmap(image: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image.T, cmap='jet', interpolation='nearest', origin='lower')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def save_module_heatmaps(heatmaps: np.ndarray, prefix: str):
    num_modules, num_cells = heatmaps.shape[:2]

    # Determine which modules and cells to save
    modules_to_save = SAVE_MODULES if SAVE_MODULES is not None else range(num_modules)
    cells_to_save = SAVE_CELLS if SAVE_CELLS is not None else range(num_cells)

    total = len(modules_to_save) * len(cells_to_save)
    with tqdm(total=total, desc=f'Saving {prefix} heatmaps') as pbar:
        for module_idx in modules_to_save:
            for cell_idx in cells_to_save:
                filename = f'{prefix}_module_{module_idx}_cell_{cell_idx}.png'
                save_heatmap(heatmaps[module_idx, cell_idx], os.path.join(output_dir, filename))
                pbar.update(1)


save_module_heatmaps(heatmaps_band_x, 'heatmap_band_x')
save_module_heatmaps(heatmaps_band_y, 'heatmap_band_y')
save_module_heatmaps(heatmaps_grid, 'heatmap_grid')

# Save place cell heatmaps
place_cells_to_save = SAVE_PLACE_CELLS if SAVE_PLACE_CELLS is not None else range(heatmaps_place.shape[0])
for cell_idx in tqdm(place_cells_to_save, desc='Saving place cell heatmaps'):
    filename = f'heatmap_place_cell_{cell_idx}.png'
    save_heatmap(heatmaps_place[cell_idx], os.path.join(output_dir, filename))
