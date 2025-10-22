import time

import brainstate
import brainunit as u
import jax
import numpy as np
import os

from canns.models.basic import HierarchicalNetwork
from canns.task.open_loop_navigation import OpenLoopNavigationTask

PATH = os.path.dirname(os.path.abspath(__file__))


brainstate.environ.set(dt=0.1)
task_sn = OpenLoopNavigationTask(
    width=5,
    height=5,
    speed_mean=0.16,
    speed_std=0.016,
    duration=10000.0,
    dt=0.1,
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

hierarchical_net = HierarchicalNetwork(num_module=5, num_place=10)
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

trajectory = np.load('trajectory_test.npz')

# 提取各个矩阵
loc = trajectory['position']

# load the neuron activity
data = np.load('band_grid_place_activity.npz')
band_x_r = data['band_x_r']
band_y_r = data['band_y_r']
grid_r = data['grid_r']
place_r = data['place_r']

from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from numba import prange
from scipy.ndimage import gaussian_filter


np.random.seed(10)


def gauss_filter(heatmaps):
    # Gaussian 平滑参数
    sigma = 1.0  # 平滑的标准差
    N = heatmaps.shape[-1]
    for k in range(N):
        map_k = heatmaps[:, :, k]
        filtered_map = gaussian_filter(map_k, sigma=sigma)
        filtered_map = np.where(map_k == 0, 0, filtered_map)
        heatmaps[:, :, k] = filtered_map

    return heatmaps


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


grid_r = np.array(grid_r)
band_x_r = np.array(band_x_r)
band_y_r = np.array(band_y_r)
place_r = np.array(place_r)
loc = np.array(loc)
width = 5
height = 5
M = int(width * 10)
K = int(height * 10)
T = grid_r.shape[0]
grid_r = grid_r.reshape(T, -1)
band_x_r = band_x_r.reshape(T, -1)
band_y_r = band_y_r.reshape(T, -1)
heatmaps_grid = compute_firing_field(grid_r, loc, width, height, M, K)
heatmaps_band_x = compute_firing_field(band_x_r, loc, width, height, M, K)
heatmaps_band_y = compute_firing_field(band_y_r, loc, width, height, M, K)
heatmaps_place = compute_firing_field(place_r, loc, width, height, M, K)
# save the heatmap
np.savez('band_grid_place_heatmap.npz', heatmaps_grid=heatmaps_grid, heatmaps_band_x=heatmaps_band_x,
         heatmaps_band_y=heatmaps_band_y, heatmaps_place=heatmaps_place)


heatmaps_grid = gauss_filter(heatmaps_grid)
heatmaps_band_x = gauss_filter(heatmaps_band_x)
heatmaps_band_y = gauss_filter(heatmaps_band_y)
heatmaps_place = gauss_filter(heatmaps_place)

heatmaps_band_x = heatmaps_band_x.reshape(5, -1, M, K)
heatmaps_band_y = heatmaps_band_y.reshape(5, -1, M, K)
heatmaps_grid = heatmaps_grid.reshape(5, -1, M, K)

# plot the heatmap
probe_index = np.random.choice(heatmaps_band_x.shape[1], 1)[0]
for module_index in range(5):
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmaps_band_x[module_index, probe_index].T, cmap='jet', interpolation='nearest', origin='lower')
    # plt.title('Band module 1')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    figname = 'heatmap_band_1_module_{}.png'.format(module_index)
    plt.savefig(figname)

    plt.figure(figsize=(5, 5))
    plt.imshow(heatmaps_band_y[module_index, probe_index].T, cmap='jet', interpolation='nearest', origin='lower')
    # plt.title('Band module 2')
    # 去除 x 轴和 y 轴的刻度
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    figname = 'heatmap_band_2_module_{}.png'.format(module_index)
    plt.savefig(figname)

    plt.figure(figsize=(5, 5))
    plt.imshow(heatmaps_grid[module_index, probe_index].T, cmap='jet', interpolation='nearest', origin='lower')
    # plt.title('Grid cell')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    figname = 'heatmap_grid_module_{}.png'.format(module_index)
    plt.savefig(figname)

data = np.load('band_grid_place_heatmap.npz')
heatmaps_place = data['heatmaps_place']
# plot the data
# plot three cells, random shuffle 3 cells for 800
probe_index = np.random.choice(heatmaps_place.shape[0], 3, replace=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, probe_index in enumerate(probe_index):
    ax[i].imshow(heatmaps_place[probe_index].T, cmap='jet', interpolation='nearest', origin='lower')
    # ax[i].set_title(f'Place cell {i+1}')
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.tight_layout()
plt.savefig('heatmap_place.png')