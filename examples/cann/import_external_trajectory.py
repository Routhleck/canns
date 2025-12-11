"""
Example: Import External Trajectory Data

This example demonstrates how to import external position coordinates
into OpenLoopNavigationTask instead of using the built-in random motion model.
"""

import numpy as np

from canns.task.open_loop_navigation import OpenLoopNavigationTask

# Environment parameters
Env_size = 2.2
dt = 0.01

# Create external trajectory data (smooth circular path)
simulate_time = 10.0
n_steps = int(simulate_time / dt)
times = np.linspace(0, simulate_time, n_steps)

# Random walk with drift parameters
np.random.seed(42)  # For reproducible results
center = [Env_size / 2, Env_size / 2]
speed_mean = 0.15
speed_std = 0.05
direction_change_rate = 0.1  # How often direction changes

# Generate random trajectory with realistic animal movement
positions = np.zeros((n_steps, 2))
positions[0] = center  # Start at center

# Initialize movement parameters
current_direction = np.random.uniform(0, 2 * np.pi)
current_speed = np.random.normal(speed_mean, speed_std)

for i in range(1, n_steps):
    # Randomly change direction and speed
    if np.random.random() < direction_change_rate:
        current_direction += np.random.normal(0, 0.3)  # Small direction changes
    if np.random.random() < 0.05:  # Occasionally change speed
        current_speed = np.abs(np.random.normal(speed_mean, speed_std))

    # Calculate step
    dt_step = times[i] - times[i - 1]
    dx = current_speed * np.cos(current_direction) * dt_step
    dy = current_speed * np.sin(current_direction) * dt_step

    # Add some noise
    dx += np.random.normal(0, 0.002)
    dy += np.random.normal(0, 0.002)

    # Update position
    new_pos = positions[i - 1] + [dx, dy]

    # Keep within bounds (simple wall bounce)
    if new_pos[0] < 0.1 or new_pos[0] > Env_size - 0.1:
        current_direction = np.pi - current_direction
        new_pos[0] = np.clip(new_pos[0], 0.1, Env_size - 0.1)
    if new_pos[1] < 0.1 or new_pos[1] > Env_size - 0.1:
        current_direction = -current_direction
        new_pos[1] = np.clip(new_pos[1], 0.1, Env_size - 0.1)

    positions[i] = new_pos

# Head direction based on movement direction (will be calculated automatically if not provided)
# For this example, we'll let import_data calculate it from velocity
head_directions = None

# Create spatial navigation task
snt = OpenLoopNavigationTask(
    duration=simulate_time,
    width=Env_size,
    height=Env_size,
    dt=dt,
    progress_bar=True
)

# Import external trajectory data
print("Importing external trajectory data...")
snt.import_data(
    position_data=positions,
    times=times,
    head_direction=head_directions
)

# Calculate theta sweep data (if needed)
snt.calculate_theta_sweep_data()
snt_data = snt.data

# Extract trajectory data
time_steps = snt.total_steps
position = snt_data.position
direction = snt_data.hd_angle
velocity = snt_data.velocity
speed = snt_data.speed
linear_speed_gains = snt_data.linear_speed_gains
ang_speed_gains = snt_data.ang_speed_gains

# Show trajectory analysis with smoothing
print("Displaying trajectory analysis...")
snt.show_trajectory_analysis(save_path="import_external_trajectory.png", show=False, smooth_window=50)

# Also plot our calculated data directly for comparison
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

# Plot 1: Our imported trajectory
axs[0].plot(snt_data.position[:, 0], snt_data.position[:, 1], 'b-', linewidth=2, label='Our data')
axs[0].set_title('Our Imported Trajectory')
axs[0].set_xlabel('X Position')
axs[0].set_ylabel('Y Position')
axs[0].set_aspect('equal')

# Plot 2: Our calculated speed
time_our = times
axs[1].plot(time_our, snt_data.speed, 'b-', linewidth=2, label='Our speed')
axs[1].set_title('Our Calculated Speed')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Speed (m/s)')

# Plot 3: Our head direction
axs[2].plot(time_our, snt_data.hd_angle, 'b-', linewidth=2, label='Our head direction')
axs[2].set_title('Our Head Direction')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Direction (rad)')

plt.tight_layout()
plt.savefig('our_data_comparison.png')
plt.close()

print(f"Imported {time_steps} time steps")
print(
    f"Position range: X=[{position[:, 0].min():.3f}, {position[:, 0].max():.3f}], Y=[{position[:, 1].min():.3f}, {position[:, 1].max():.3f}]")
print(f"Speed range: [{speed.min():.3f}, {speed.max():.3f}] units/s")

# Debug: Check some calculation details
print(f"\nDebug info:")
print(f"Expected speed range: {speed_mean - speed_std:.3f} to {speed_mean + speed_std:.3f} units/s")
print(f"Time step (dt): {dt}")
print(f"Calculated speed mean: {snt_data.speed.mean():.6f}")
print(f"Speed std deviation: {snt_data.speed.std():.6f}")
print(f"Speed range: [{snt_data.speed.min():.6f}, {snt_data.speed.max():.6f}]")

print(f"\nNote: This random trajectory shows realistic speed and direction variations")
print(f"that would be typical in experimental animal movement data.")
