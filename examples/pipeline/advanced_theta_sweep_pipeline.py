"""Advanced Pipeline Example: Complete Parameter Customization

This example demonstrates how to use all parameters of ThetaSweepPipeline
for advanced users who want full control over the neural network models.
"""

import numpy as np

from canns.pipeline import ThetaSweepPipeline


def _catmull_rom_chain(control_points: np.ndarray, samples: int) -> np.ndarray:
    """Generate a smooth Catmull–Rom spline through the given control points."""

    if control_points.shape[0] < 4:
        raise ValueError("Catmull–Rom spline requires at least 4 control points.")

    extended = np.vstack([control_points[0], control_points, control_points[-1]])
    segment_count = control_points.shape[0] - 1
    sample_bins = np.linspace(0, samples, num=segment_count + 1, dtype=int)

    curve: list[np.ndarray] = []
    for seg_idx in range(segment_count):
        seg_samples = sample_bins[seg_idx + 1] - sample_bins[seg_idx]
        if seg_samples <= 0:
            continue
        p0, p1, p2, p3 = extended[seg_idx: seg_idx + 4]
        t_values = np.linspace(0.0, 1.0, seg_samples, endpoint=seg_idx == segment_count - 1)
        for t in t_values:
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            curve.append(point)

    return np.asarray(curve)


def main():
    # Create deterministic trajectory with perturbations forming a smooth multi-node L-curve
    n_steps = 800
    dt = 0.002
    env_size = 1.8
    times = np.linspace(0.0, (n_steps - 1) * dt, n_steps)

    start = np.array([0.2, 0.2])
    corner = np.array([1.4, 0.2])
    end = np.array([1.4, 1.4])

    # Control points loosely following an L-trajectory, perturbed for a more organic path
    base_nodes = np.array(
        [
            start,
            start + np.array([0.5, 0.05]),
            corner + np.array([-0.3, 0.12]),
            corner + np.array([-0.05, 0.45]),
            corner + np.array([0.05, 0.85]),
            end,
        ]
    )
    rng = np.random.default_rng(2024)
    perturb = rng.normal(scale=0.06, size=base_nodes.shape)
    perturb[0] = 0.0
    perturb[-1] = 0.0
    control_points = np.clip(base_nodes + perturb, 0.1, env_size - 0.1)

    positions = _catmull_rom_chain(control_points, n_steps)

    # Apply gentle deterministic wobble along the curve for additional texture
    param = np.linspace(0.0, 1.0, len(positions))
    wobble = 0.02 * np.column_stack(
        (
            np.sin(4 * np.pi * param),
            np.sin(2.5 * np.pi * param + np.pi / 4),
        )
    )
    wobble[[0, -1]] = 0.0
    positions = np.clip(positions + wobble, 0.05, env_size - 0.05)

    print("🔬 Advanced Theta Sweep Pipeline Example")
    print("=========================================")
    print(f"📊 Trajectory: {len(positions)} steps, duration: {times[-1]:.2f}s")
    print(f"🎯 Pattern: Perturbed multi-node L-curve (deterministic, seeded noise)")

    # Configure all pipeline parameters for maximum customization
    pipeline = ThetaSweepPipeline(
        # === Required Parameters ===
        trajectory_data=positions,
        times=times,

        # === Environment Configuration ===
        env_size=env_size,  # Larger environment to accommodate trajectory
        dt=dt,  # Match trajectory sampling rate

        # === Direction Cell Network Parameters ===
        direction_cell_params={
            "num": 100,
            "adaptation_strength": 15,
            "noise_strength": 0.0,
        },

        # === Grid Cell Network Parameters ===
        grid_cell_params={
            "num_gc_x": 100,
            "adaptation_strength": 8,
            "noise_strength": 0.0,
        },

        # === Theta Rhythm Parameters ===
        theta_params={
            "theta_strength_hd": 1.0,  # Strong theta modulation in head direction
            "theta_strength_gc": 0.5,  # Strong theta modulation in grid cells
            "theta_cycle_len": 100.0,  # Longer theta cycle (slower rhythm)
        },

        # === Spatial Navigation Task Parameters ===
        spatial_nav_params={
            "width": env_size,  # Match env_size
            "height": env_size,  # Square environment
            "dt": dt,  # Consistent time step
            "progress_bar": True,  # Show import progress
        },
    )

    print("\n🧠 Network Configuration:")
    print(f"  • Direction cells: {pipeline.direction_cell_params['num']}")
    print(f"  • Grid cells: {pipeline.grid_cell_params['num_gc_x']}×{pipeline.grid_cell_params['num_gc_x']}")
    print(f"  • Theta cycle length: {pipeline.theta_params['theta_cycle_len']} steps")
    print(f"  • Grid mapping ratio: {pipeline.grid_cell_params['mapping_ratio']}")

    # Run with custom output configuration
    results = pipeline.run(
        output_dir="advanced_theta_sweep_results",
        save_animation=True,
        save_plots=True,
        show_plots=False,  # Set to True for interactive display
        animation_fps=15,  # Higher frame rate for smoother animation
        animation_dpi=200,  # High quality animation
        verbose=True,
    )

    print(f"\n📊 Analysis Results:")
    print(f"  • Animation: {results['animation_path']}")
    print(f"  • Trajectory analysis: {results['trajectory_analysis']}")
    print(f"  • Population activity: {results['population_activity']}")

    # Access simulation data for custom analysis
    sim_data = results["data"]

    print(f"\n🔍 Simulation Data Available:")
    for key, value in sim_data.items():
        if isinstance(value, np.ndarray):
            print(f"  • {key}: {value.shape} ({value.dtype})")

    # Example custom analysis
    gc_activity = sim_data["gc_activity"]
    dc_activity = sim_data["dc_activity"]

    # Find peak activity moments
    gc_peak_frame = np.argmax(np.max(gc_activity, axis=1))
    dc_peak_frame = np.argmax(np.max(dc_activity, axis=1))

    print(f"\n📈 Peak Activity Analysis:")
    print(f"  • Grid cell peak at frame {gc_peak_frame} (t={gc_peak_frame * dt:.3f}s)")
    print(f"    Position: [{positions[gc_peak_frame, 0]:.3f}, {positions[gc_peak_frame, 1]:.3f}]")
    print(f"  • Direction cell peak at frame {dc_peak_frame} (t={dc_peak_frame * dt:.3f}s)")
    print(f"    Head direction: {sim_data['direction'][dc_peak_frame]:.3f} rad")

    # Analyze theta modulation strength
    theta_phase = sim_data["theta_phase"]
    theta_range = theta_phase.max() - theta_phase.min()

    print(f"\n🌊 Theta Rhythm Analysis:")
    print(f"  • Phase range: {theta_range:.3f} rad")
    print(f"  • Estimated cycles: {theta_range / (2 * np.pi):.1f}")

    print(f"\n✅ Advanced pipeline analysis complete!")
    print(f"📁 All results saved to: advanced_theta_sweep_results/")

    return results


if __name__ == "__main__":
    main()
