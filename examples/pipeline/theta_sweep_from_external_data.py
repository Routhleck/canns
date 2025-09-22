"""Pipeline demo using external trajectory data for theta sweep analysis.

This example demonstrates how to analyze experimental trajectory data using
the ThetaSweepPipeline without needing to understand CANN implementation details.
"""

import numpy as np

from canns.pipeline import ThetaSweepPipeline


def main() -> None:
    # Create example external trajectory (circular motion)
    np.random.seed(42)
    n_steps = 1000
    dt = 0.002
    times = np.linspace(0, n_steps * dt, n_steps)
    
    # Generate circular trajectory with some noise
    radius = 0.4
    center = [0.75, 0.75]
    theta = 2 * np.pi * times / times[-1]  # One full circle
    
    positions = np.column_stack([
        center[0] + radius * np.cos(theta) + np.random.normal(0, 0.01, n_steps),
        center[1] + radius * np.sin(theta) + np.random.normal(0, 0.01, n_steps),
    ])
    
    print("Running theta sweep analysis on external trajectory...")
    print(f"Trajectory: {len(positions)} steps, duration: {times[-1]:.2f}s")
    
    # Run complete analysis using the pipeline
    pipeline = ThetaSweepPipeline(
        trajectory_data=positions,
        times=times,
        env_size=1.5,
    )
    results = pipeline.run(output_dir="theta_sweep_results")
    
    print(f"\nAnalysis complete!")
    print(f"Animation saved to: {results['animation_path']}")
    print(f"Plots saved to: theta_sweep_results/")


if __name__ == "__main__":
    main()