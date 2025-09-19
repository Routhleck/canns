import brainstate
import brainunit as u
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy.diffgeom.rn import theta

from canns.models.basic.theta_sweep_model import DirectionCellNetwork, GridCellNetwork
from canns.task.spatial_navigation import SpatialNavigationTask

np.random.seed(10)
Env_size = 1.5
simulate_time = 2
dt = 0.001
brainstate.environ.set(dt=1.0)

snt = SpatialNavigationTask(
    duration=simulate_time,
    initial_head_direction=11/12*u.math.pi,
    width=Env_size,
    height=Env_size,
    start_pos=[Env_size*15/16, Env_size*1/16],
    speed_mean=2.,
    speed_std=0.,
    dt=dt,
    speed_coherence_time=10,
    rotational_velocity_std=40*np.pi/180,
)

snt.get_data()
snt_data = snt.data

time_steps = snt.run_steps
position = snt_data.position
direction = snt_data.hd_angle
velocity = snt_data.rot_vel
moving_speed = snt_data.speed
direction_unwrapped = np.unwrap(direction)
ang_velocity = np.diff(direction_unwrapped) / dt
ang_velocity = np.insert(ang_velocity, 0, 0)

# plot
fig, axs = plt.subplots(1,3,figsize=(12,3),dpi=100, width_ratios=[1,2,2])

ax = axs[0]
ax.plot(position[:,0], position[:,1], lw=1, color='black')
ax.set_xlim(0, Env_size)
ax.set_ylim(0, Env_size)
#equal axis
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([0, Env_size])
ax.set_yticks([0, Env_size])

ax = axs[1]
sns.despine(ax=axs[1])
ax.plot(time_steps, moving_speed, lw=1, color='#009FB9')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (m/s)')

ax = axs[2]
sns.despine(ax=axs[2])
# find the jump points where the difference between two adjacent points is greater than pi
jumps = np.where(np.abs(np.diff(direction)) > np.pi)[0]
# set the jump points to NaN for plotting
direction_plot = direction.copy()
direction_plot[jumps + 1] = np.nan
ax.plot(time_steps, direction_plot, lw=1, color='#009FB9')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Direction (rad)')

plt.tight_layout()
plt.show()

dc_net = DirectionCellNetwork(
    num=100,
    adaptation_strength=15,
    noise_strength=0.
)

mapping_ratio=5
gc_net = GridCellNetwork(
    num_dc = dc_net.num,
    num_gc_x=100,
    adaptation_strength=8,
    mapping_ratio=mapping_ratio,
    noise_strength=0.,
)
dc_net.init_state()
gc_net.init_state()

linear_speed_gains = moving_speed/np.max(moving_speed)
ang_speed_gains = ang_velocity/np.max(np.abs(ang_velocity))


def simulate_dcgcnet(
    dcnet: DirectionCellNetwork, gcnet: GridCellNetwork,
    positions, directions,
    linear_speed_gains, ang_speed_gains,
    theta_strength_hd=0, theta_strength_gc=0,
    theta_cycle_len=100
):
    """Simulate direction-conjg grid cell-grid cell network over a trajectory of directions and velocity gains."""

    def step(i, position, direction, linear_gain, ang_gain):
        # theta oscillation phase
        t = i * brainstate.environ.get_dt()
        theta_phase = u.math.mod(t, theta_cycle_len) / theta_cycle_len
        theta_phase = theta_phase * 2 * np.pi - np.pi

        # get theta modulation for both direction cell and grid cell
        # theta modulation scales with angular speed in direction cell network, and scales with linear speed in grid cell network

        theta_modulation_hd = 1 + theta_strength_hd * (0.5 + ang_gain) * u.math.cos(theta_phase)
        theta_modulation_gc = 1 + theta_strength_gc * (0.5 + linear_gain) * u.math.cos(theta_phase)

        # update direction cell network
        dcnet(direction, theta_modulation_hd)
        # get direction cell network activity
        dc_net_activity = dcnet.r.value
        # get internal direction
        internal_direction = dcnet.center.value

        # update grid cell network
        gcnet(position, dc_net_activity, theta_modulation_gc)

        # get results
        internal_position = gcnet.center_position.value
        gc_net_activity = gcnet.r.value
        gC_bump = gcnet.gc_bump.value

        return (
            internal_position,
            internal_direction,
            gc_net_activity,
            gC_bump,
            dc_net_activity,
            theta_phase,
            theta_modulation_hd,
            theta_modulation_gc,
        )

    return brainstate.compile.for_loop(
        step,
        u.math.arange(len(positions)), positions, directions, linear_speed_gains, ang_speed_gains,
        pbar=True
    )

internal_position, internal_direction, gc_netactivity, \
    gc_bump, dc_netactivity, theta_phase, theta_modulation_hd,\
    theta_modulation_gc = simulate_dcgcnet(dc_net, gc_net,
                                           positions=position,
                                           directions=direction,
                                           linear_speed_gains=linear_speed_gains,
                                           ang_speed_gains=ang_speed_gains,
                                           theta_strength_hd=1.0, theta_strength_gc=0.5,
                                           theta_cycle_len=100
                                           )


def plot_population_activity(ax, time_steps, theta_phase, net_activity, direction, add_lines=True, atol=1e-2, **kwargs):
    """Plot HD network population activity + direction trace."""
    cmap = kwargs.pop("cmap", "jet")  # 取出 cmap，如果没传就用 jet
    im = ax.imshow(
        net_activity.T * 100,
        aspect="auto",
        extent=[time_steps[0], time_steps[-1], -np.pi, np.pi],
        origin="lower",
        cmap=cmap,
        **kwargs
    )

    # find the jump points where the difference between two adjacent points is greater than pi
    jumps = np.where(np.abs(np.diff(direction)) > np.pi)[0]
    # set the jump points to NaN for plotting
    direction_plot = direction.copy()
    direction_plot[jumps + 1] = np.nan
    ax.plot(time_steps, direction_plot, color="white", lw=3)

    if add_lines:
        zero_phase_index = np.where(np.isclose(theta_phase, 0, atol=atol))[0]
        for i in zero_phase_index:
            ax.axvline(x=time_steps[i], color="grey", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_yticks([-np.pi, np.pi])
    ax.set_yticklabels([0, 360])
    ax.set_ylabel("Direction(°)")

    sns.despine(ax=ax)
    return ax

max_gc_activity = np.max(gc_netactivity, axis=1)
max_dc_activity = np.max(dc_netactivity, axis=1)
fig, axs = plt.subplots(1, 2, figsize=(8, 2), dpi=300, width_ratios=[2,1])
s_size = 2

ax = axs[0]
ax = plot_population_activity(ax, time_steps, theta_phase, dc_netactivity, direction,
                                       add_lines=True, atol=5e-2, cmap="jet")
ax.set_title('Internal direction')

ax = axs[1]
sc = ax.scatter(
                internal_position[:, 0],
                internal_position[:, 1],
                c=max_gc_activity[:],
                cmap="cool",
                s=s_size,
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
#add real position of the animal
ax.plot(position[:, 0], position[:, 1], color="black")

ax.set_title('Internal position')
#equal axis
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, Env_size)
ax.set_ylim(0, Env_size)
ax.set_xticks([0, Env_size])
ax.set_yticks([0, Env_size])

plt.tight_layout()
plt.show()

value_grid_twisted = np.dot(gc_net.coor_transform_inv, gc_net.value_grid.T).T
grid_cell_activity = gc_netactivity.reshape(-1, gc_net.num_gc_1side, gc_net.num_gc_1side)
frame_idx = 900

fig, axs = plt.subplots(1,2,figsize=(6,3),dpi=150, width_ratios=[1,2])

ax = axs[0]

value_grid_twisted_2_realspace = value_grid_twisted/mapping_ratio
ax.scatter(value_grid_twisted_2_realspace[:, 0], value_grid_twisted_2_realspace[:, 1], c=grid_cell_activity[frame_idx].flatten(), cmap="jet", s=5)
#equal axis
ax.set_aspect('equal', adjustable='box')
ax.set_title("GC bump atcivity on the\ntwisted torus manifold")
sns.despine(ax=axs[0])

ax = axs[1]

points = value_grid_twisted / mapping_ratio

colors =  np.array(grid_cell_activity[frame_idx])
colors[:3, :] = np.nan
colors[-3:, :] = np.nan
colors[:, :3] = np.nan
colors[:, -3:] = np.nan
colors = colors.flatten()


nx = (np.sqrt(gc_net.candidate_centers.shape)[0]).astype(int)
ny = (np.sqrt(gc_net.candidate_centers.shape)[0]).astype(int)

candidate_centers = gc_net.candidate_centers.reshape(nx, ny, 2)

for i in range(nx//2-1, nx):
    for j in range(ny//2-1, ny):
        ax.scatter(points[:, 0] + candidate_centers[i, j, 0],
                   points[:, 1] + candidate_centers[i, j, 1],
                   c=colors, cmap="jet", s=5)

ax.set_xlim(0, Env_size)
ax.set_ylim(0, Env_size)
ax.set_xticks([0, Env_size])
ax.set_yticks([0, Env_size])
ax.set_aspect('equal', adjustable='box')
ax.set_title("GC bump atcivity\nin the environment")

plt.tight_layout()
plt.show()

import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 50  # unit is MB to set the limit of embedding animation

grid_cell_activity = gc_netactivity.reshape(-1, gc_net.num_gc_1side, gc_net.num_gc_1side)

n_step = 10

direction4ani = direction[::n_step]
frames = len(direction4ani)

direction_bins = np.linspace(-np.pi, np.pi, 100, endpoint=False)

fig, ax_ani = plt.subplots(1, 4, figsize=(12, 3), dpi=100, width_ratios=[1, 0.8, 1, 1])

ax_traj = ax_ani[0]
position4ani = position[::n_step, :]
ax_traj.plot(position4ani[:, 0], position4ani[:, 1], color="#F18D00", lw=1)
red_dot_on_realspace, = ax_traj.plot([], [], 'ro', markersize=5)
ax_traj.set_xlim(0, Env_size)
ax_traj.set_ylim(0, Env_size)
ax_traj.set_aspect('equal', adjustable='box')
ax_traj.set_xticks([0, Env_size])
ax_traj.set_yticks([0, Env_size])

sns.despine(ax=ax_ani[1], left=True, bottom=True, right=True, top=True)
time_text = ax_ani[1].text(1.2, 1.3, '', transform=ax_ani[0].transAxes,
                           ha='center', va='center', fontsize=12)
ax_ani[1].set_xticks([])
ax_ani[1].set_yticks([])

ax_dc = plt.subplot(1, 4, 2, projection='polar')
dc_activity_4_ani = dc_netactivity[::n_step, :]
ax_dc.set_ylim(0., np.max(dc_activity_4_ani[:, :]) * 1.2)
ax_dc.set_yticks([])
ax_dc.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax_dc.set_xticklabels(['0', r'90$^\circ$', r'180$^\circ$', r'270$^\circ$'])
ax_dc.set_title("Direction sweep")

hd_line, = ax_dc.plot([], [], 'k-')
id_line, = ax_dc.plot([], [], color='#009FB9')

ax_gc = ax_ani[2]
pos2phase = np.array(gc_net.position2phase(position.T))
pos2phase4ani = pos2phase[:, ::n_step]
pos2phase4ani_twisted = np.dot(gc_net.coor_transform_inv, pos2phase4ani)

x = pos2phase4ani_twisted[0, :]
y = pos2phase4ani_twisted[1, :]
jumps_x = np.where(np.abs(np.diff(x)) > np.pi)[0]
jumps_y = np.where(np.abs(np.diff(y)) > np.pi)[0]
jumps = np.unique(np.concatenate([jumps_x, jumps_y]))
x_plot = x.copy()
y_plot = y.copy()
x_plot[jumps + 1] = np.nan
y_plot[jumps + 1] = np.nan
ax_gc.plot(x_plot, y_plot, color="#F18D00")

gc_bump4ani = grid_cell_activity[::n_step, :, :]
vmin1 = 0
vmax1 = np.max(gc_bump4ani)
heatmap = ax_gc.scatter([], [], c=[], s=1, cmap="jet", vmin=vmin1, vmax=vmax1)

value_grid_twisted = np.dot(gc_net.coor_transform_inv, gc_net.value_grid.T).T
red_dot_on_manifold, = ax_gc.plot([], [], 'ro', markersize=5)

ax_gc.set_title("GC sweep on the manifold")
x_min, x_max = value_grid_twisted[:, 0].min(), value_grid_twisted[:, 0].max()
y_min, y_max = value_grid_twisted[:, 1].min(), value_grid_twisted[:, 1].max()
ax_gc.set_xlim(x_min, x_max)
ax_gc.set_ylim(y_min, y_max)
ax_gc.axis('off')
ax_gc.set_aspect("equal")

ax_realgc = ax_ani[3]
heatmap_realgc = ax_realgc.scatter([], [], c=[], s=1, cmap="jet", vmin=vmin1, vmax=vmax1)
ax_realgc.plot(position4ani[:, 0], position4ani[:, 1], color="#F18D00", lw=1)
red_dot_on_realgc, = ax_realgc.plot([], [], 'ro', markersize=5)
ax_realgc.set_xlim(0, Env_size)
ax_realgc.set_ylim(0, Env_size)
ax_realgc.set_aspect('equal', adjustable='box')
ax_realgc.set_title("GC sweep in real space")
ax_realgc.set_xticks([])
ax_realgc.set_yticks([])

def update(frame):
    # trajectory
    red_dot_on_realspace.set_data([position4ani[frame, 0]], [position4ani[frame, 1]])
    ax_traj.set_title(f"Animal trajectory (time: {frame*n_step/1000:.1f} s)")

    # direction cell polar
    hd_line.set_data([direction4ani[frame], direction4ani[frame]],
                     [0, np.max(dc_activity_4_ani[:, :])])
    id_line.set_data(direction_bins, dc_activity_4_ani[frame])

    # manifold
    heatmap.set_offsets(value_grid_twisted)
    heatmap.set_array(gc_bump4ani[frame].flatten())
    red_dot_on_manifold.set_data([pos2phase4ani_twisted[0, frame]],
                                 [pos2phase4ani_twisted[1, frame]])

    # real space
    ax_realgc.cla()
    ax_realgc.set_xlim(0, Env_size)
    ax_realgc.set_ylim(0, Env_size)
    ax_realgc.set_aspect('equal', adjustable='box')
    ax_realgc.set_title("GC sweep in real space")
    ax_realgc.set_xticks([])
    ax_realgc.set_yticks([])

    points = value_grid_twisted / mapping_ratio
    colors = np.array(gc_bump4ani[frame])  # convert from JAX to numpy
    # colors = gc_bump4ani[frame].copy()  # shape (n,n)

    # mask the outer ring of each tile with NaN
    colors[:3, :] = np.nan
    colors[-3:, :] = np.nan
    colors[:, :3] = np.nan
    colors[:, -3:] = np.nan
    colors = colors.flatten()

    nx = int(np.sqrt(gc_net.candidate_centers.shape[0]))
    ny = int(np.sqrt(gc_net.candidate_centers.shape[0]))
    candidate_centers = gc_net.candidate_centers.reshape(nx, ny, 2)

    for i in range(nx//2 - 1, nx):
        for j in range(ny//2 - 1, ny):
            ax_realgc.scatter(points[:, 0] + candidate_centers[i, j, 0],
                              points[:, 1] + candidate_centers[i, j, 1],
                              c=colors, cmap="jet", s=5,
                              vmin=vmin1, vmax=vmax1)

    # trajectory overlay
    ax_realgc.plot(position4ani[:, 0], position4ani[:, 1], color="#F18D00", lw=2)
    ax_realgc.plot(position4ani[frame, 0], position4ani[frame, 1], 'ro', markersize=5)

    return hd_line, id_line, heatmap, red_dot_on_manifold

plt.tight_layout()
plt.close(fig)

ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
ani.save("gridcell_sweeps.gif", writer="pillow", fps=10)
