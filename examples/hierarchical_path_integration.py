import brainstate
import brainunit as u
import jax
import numpy as np
import os

from canns.models.basic import HierarchicalNetwork
from canns.task.path_integration import PathIntegrationTask

PATH = os.path.dirname(os.path.abspath(__file__))


brainstate.environ.set(dt=0.1)
task_pi = PathIntegrationTask(
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
    data = np.load(trajectory_file_path)
    trajectory = task_pi.get_empty_trajectory()
    trajectory.position = data['position']
    trajectory.velocity = data['velocity']
    trajectory.speed = data['speed']
    trajectory.hd_angle = data['hd_angle']
    trajectory.rot_vel = data['rot_vel']
else:
    print(f"Generating new trajectory and saving to {trajectory_file_path}")
    trajectory = task_pi.generate_trajectory()
    task_pi.show_trajectory(show=False, save_path=trajectory_graph_file_path)
    np.savez(
        trajectory_file_path,
        position=trajectory.position,
        velocity=trajectory.velocity,
        speed=trajectory.speed,
        hd_angle=trajectory.hd_angle,
        rot_vel=trajectory.rot_vel,
    )

hierarchical_net = HierarchicalNetwork(num_module=5, num_place=30)
hierarchical_net.init_state()

def initialize(t, input_stre):
    hierarchical_net(
        velocity=u.math.zeros(2, ),
        loc=trajectory.position[0],
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

total_time = trajectory.velocity.shape[0]
indices = np.arange(total_time)

# with jax.disable_jit():
band_x_r, band_y_r, grid_r, place_r = brainstate.compile.for_loop(
    run_step,
    u.math.asarray(indices),
    u.math.asarray(trajectory.velocity),
    u.math.asarray(trajectory.position),
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