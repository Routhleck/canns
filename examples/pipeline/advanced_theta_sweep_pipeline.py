"""Advanced Pipeline Example: Complete Parameter Customization

This example demonstrates how to use all parameters of ThetaSweepPipeline
for advanced users who want full control over the neural network models.
"""

import numpy as np

from canns.pipeline import ThetaSweepPipeline


def main():
    # Create deterministic two-segment trajectory (L-shaped walk without noise)
    n_steps = 800
    dt = 0.002
    times = np.linspace(0.0, (n_steps - 1) * dt, n_steps)

    start = np.array([0.2, 0.2])
    corner = np.array([1.4, 0.2])
    end = np.array([1.4, 1.4])

    first_leg_steps = n_steps // 2
    second_leg_steps = n_steps - first_leg_steps

    first_leg_t = np.linspace(0.0, 1.0, first_leg_steps, endpoint=False)
    second_leg_t = np.linspace(0.0, 1.0, second_leg_steps, endpoint=True)

    first_leg = start + np.outer(first_leg_t, corner - start)
    second_leg = corner + np.outer(second_leg_t, end - corner)

    positions = np.vstack([first_leg, second_leg])
    
    print("🔬 Advanced Theta Sweep Pipeline Example")
    print("=========================================")
    print(f"📊 Trajectory: {len(positions)} steps, duration: {times[-1]:.2f}s")
    print(f"🎯 Pattern: Two straight segments (deterministic, no noise)")
    
    # Configure all pipeline parameters for maximum customization
    pipeline = ThetaSweepPipeline(
        # === Required Parameters ===
        trajectory_data=positions,
        times=times,
        
        # === Environment Configuration ===
        env_size=1.8,  # Larger environment to accommodate trajectory
        dt=dt,         # Match trajectory sampling rate
        
        # === Direction Cell Network Parameters ===
        direction_cell_params={
            "num": 200,                    # Higher resolution direction representation
            "adaptation_strength": 25,     # Stronger adaptation for sharper tuning
            "noise_strength": 0.08,        # Add biological noise
        },
        
        # === Grid Cell Network Parameters ===
        grid_cell_params={
            "num_gc_x": 150,              # High-resolution grid (150x150 = 22,500 cells)
            "adaptation_strength": 12,     # Moderate adaptation for grid patterns
            "mapping_ratio": 4,            # Smaller grid scale (higher frequency)
            "noise_strength": 0.05,        # Moderate noise in grid cells
        },
        
        # === Theta Rhythm Parameters ===
        theta_params={
            "theta_strength_hd": 1.8,      # Strong theta modulation in head direction
            "theta_strength_gc": 1.2,      # Strong theta modulation in grid cells
            "theta_cycle_len": 120.0,      # Longer theta cycle (slower rhythm)
        },
        
        # === Spatial Navigation Task Parameters ===
        spatial_nav_params={
            "width": 1.8,                  # Match env_size
            "height": 1.8,                 # Square environment
            "dt": dt,                      # Consistent time step
            "progress_bar": True,          # Show import progress
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
        show_plots=False,              # Set to True for interactive display
        animation_fps=15,              # Higher frame rate for smoother animation
        animation_dpi=200,             # High quality animation
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
    print(f"  • Grid cell peak at frame {gc_peak_frame} (t={gc_peak_frame*dt:.3f}s)")
    print(f"    Position: [{positions[gc_peak_frame, 0]:.3f}, {positions[gc_peak_frame, 1]:.3f}]")
    print(f"  • Direction cell peak at frame {dc_peak_frame} (t={dc_peak_frame*dt:.3f}s)")
    print(f"    Head direction: {sim_data['direction'][dc_peak_frame]:.3f} rad")
    
    # Analyze theta modulation strength
    theta_phase = sim_data["theta_phase"]
    theta_range = theta_phase.max() - theta_phase.min()
    
    print(f"\n🌊 Theta Rhythm Analysis:")
    print(f"  • Phase range: {theta_range:.3f} rad")
    print(f"  • Estimated cycles: {theta_range / (2*np.pi):.1f}")
    
    print(f"\n✅ Advanced pipeline analysis complete!")
    print(f"📁 All results saved to: advanced_theta_sweep_results/")
    
    return results


if __name__ == "__main__":
    main()
