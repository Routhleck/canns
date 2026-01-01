"""
Grid Cell Velocity Model - Path Integration Verification

Verifies path integration accuracy of GridCell2DVelocity model following
Burak & Fiete (2009). Uses robust blob tracking to achieve R² > 0.99.
"""

import brainpy.math as bm
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from canns.models.basic import GridCell2DVelocity
from canns.task.open_loop_navigation import OpenLoopNavigationTask

# Setup
dt = 5e-4
dt_task = 0.01
bm.set_dt(dt)

box_size = 2.2
duration = 10.0

output_dir = Path("outputs/velocity_path_integration")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Grid Cell Velocity Model - Path Integration Verification")
print("=" * 70)

# Generate trajectory
task = OpenLoopNavigationTask(
    duration=duration,
    width=box_size,
    height=box_size,
    start_pos=[box_size/2, box_size/2],
    speed_mean=0.5,
    speed_std=0.05,
    dt=dt_task,
)
task.get_data()

true_positions = np.asarray(task.data.position)
velocities = task.data.velocity

print(f"\nTrajectory: {len(true_positions)} steps, {duration}s")

# Initialize model
model = GridCell2DVelocity(
    length=40,
    tau=0.01,
    alpha=0.1,
    W_l=3.0,
    lambda_net=17.0,
)

print(f"Network: {model.num} neurons ({model.length}×{model.length})")

# Healing
print("\nHealing network...")
model.heal_network(num_healing_steps=5000, dt_healing=1e-4)

# Run simulation
print("Running path integration...")


def run_step(vel):
    model(vel)
    return model.r.value


activities = bm.for_loop(run_step, (velocities,), progress_bar=True)
activities = np.asarray(activities)

# Track blob centers
print("Tracking blob centers...")
blob_centers = GridCell2DVelocity.track_blob_centers(activities, model.length)

# Compute displacements
blob_displacement = np.diff(blob_centers, axis=0)

# Jump correction
threshold = 3.0
displacement_norm = np.linalg.norm(blob_displacement, axis=1)
jump_indices = np.where(displacement_norm > threshold)[0]

if len(jump_indices) > 0:
    print(f"Correcting {len(jump_indices)} jumps (periodic boundaries)")
    for idx in jump_indices:
        if idx > 0 and idx < len(blob_displacement) - 1:
            blob_displacement[idx] = (blob_displacement[idx - 1] + blob_displacement[idx + 1]) / 2

# Integrate to get estimated trajectory
estimated_pos_neuron = np.cumsum(blob_displacement, axis=0)

# Align with true positions
true_pos_rel = true_positions - true_positions[0]
true_pos_aligned = true_pos_rel[:len(estimated_pos_neuron)]

# Linear regression for scaling
X = estimated_pos_neuron.flatten().reshape(-1, 1)
y = true_pos_aligned.flatten()

reg = LinearRegression(fit_intercept=False)
reg.fit(X, y)
prop_factor = reg.coef_[0]
r2_score = reg.score(X, y)

# Rescale and compute errors
estimated_pos_physical = prop_factor * estimated_pos_neuron + true_positions[0]
position_errors = np.linalg.norm(
    estimated_pos_physical - true_positions[:len(estimated_pos_physical)], axis=1
)

# Results
print(f"\nPath Integration Quality:")
print(f"  R² score: {r2_score:.6f}")
print(f"  Proportional factor: {prop_factor:.6f}")
print(f"\nPosition Errors:")
print(f"  Mean: {np.mean(position_errors):.4f}m")
print(f"  Max: {np.max(position_errors):.4f}m")
print(f"  Final: {position_errors[-1]:.4f}m")

if r2_score > 0.99:
    print(f"\n✅ EXCELLENT: R² > 0.99 (high-quality path integration)")
elif r2_score > 0.95:
    print(f"\n✓ GOOD: R² > 0.95")
else:
    print(f"\n⚠️  LOW R²: Path integration quality needs improvement")

# Visualizations
print("\nCreating visualizations...")

# Trajectory comparison
fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(
    true_positions[:len(estimated_pos_physical), 0],
    true_positions[:len(estimated_pos_physical), 1],
    "b-", alpha=0.3, linewidth=0.5, label="True"
)
ax.plot(
    estimated_pos_physical[:, 0],
    estimated_pos_physical[:, 1],
    "r-", alpha=0.5, linewidth=0.5, label="Estimated"
)

ax.scatter(true_positions[0, 0], true_positions[0, 1],
           c="green", s=100, marker="o", label="Start", zorder=5)
ax.scatter(true_positions[len(estimated_pos_physical)-1, 0],
           true_positions[len(estimated_pos_physical)-1, 1],
           c="black", s=100, marker="x", label="End (True)", zorder=5)
ax.scatter(estimated_pos_physical[-1, 0], estimated_pos_physical[-1, 1],
           c="red", s=100, marker="x", label="End (Est.)", zorder=5)

ax.set_xlabel("X Position (m)", fontsize=12)
ax.set_ylabel("Y Position (m)", fontsize=12)
ax.set_title(
    f"Path Integration: R²={r2_score:.4f}, Mean Error={np.mean(position_errors):.4f}m",
    fontsize=14
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

plt.savefig(output_dir / "trajectory_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Error over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

time_steps = np.arange(len(position_errors)) * dt_task

ax1.plot(time_steps, position_errors, "r-", linewidth=0.5, alpha=0.7)
ax1.axhline(np.mean(position_errors), color="blue", linestyle="--",
            label=f"Mean: {np.mean(position_errors):.4f}m")
ax1.axhline(np.mean(position_errors) + np.std(position_errors),
            color="orange", linestyle=":", alpha=0.5, label="±1 Std")
ax1.axhline(np.mean(position_errors) - np.std(position_errors),
            color="orange", linestyle=":", alpha=0.5)
ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Position Error (m)", fontsize=12)
ax1.set_title("Position Error Over Time", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

cumulative_mean = np.cumsum(position_errors) / np.arange(1, len(position_errors) + 1)
ax2.plot(time_steps, cumulative_mean, "b-", linewidth=1)
ax2.set_xlabel("Time (s)", fontsize=12)
ax2.set_ylabel("Cumulative Mean Error (m)", fontsize=12)
ax2.set_title("Cumulative Mean Error", fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "error_over_time.png", dpi=150, bbox_inches="tight")
plt.close()

# Error spatial distribution
fig, ax = plt.subplots(figsize=(10, 10))

nbins = 30
H, xedges, yedges = np.histogram2d(
    true_positions[:len(estimated_pos_physical), 0],
    true_positions[:len(estimated_pos_physical), 1],
    bins=nbins,
    range=[[0, box_size], [0, box_size]],
    weights=position_errors,
)

counts, _, _ = np.histogram2d(
    true_positions[:len(estimated_pos_physical), 0],
    true_positions[:len(estimated_pos_physical), 1],
    bins=nbins,
    range=[[0, box_size], [0, box_size]],
)

mean_error_map = np.divide(H, counts, where=counts > 0, out=np.zeros_like(H))

im = ax.imshow(
    mean_error_map.T,
    origin="lower",
    extent=[0, box_size, 0, box_size],
    cmap="hot",
    aspect="auto",
)
plt.colorbar(im, ax=ax, label="Mean Position Error (m)")
ax.set_xlabel("X Position (m)", fontsize=12)
ax.set_ylabel("Y Position (m)", fontsize=12)
ax.set_title("Spatial Distribution of Position Errors", fontsize=14)

plt.savefig(output_dir / "error_spatial_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nOutputs saved to: {output_dir}")
print("=" * 70)
