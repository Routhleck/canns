#!/usr/bin/env python3
"""
CANN 1D Bump Analysis Example

This example demonstrates how to use the bump_fits and create_1d_bump_animation functions
from the experimental data analyzer to analyze 1D CANN bumps.
"""

import numpy as np
from canns.analyzer.experimental_data import bump_fits, create_1d_bump_animation


def main():
    """Demonstrate bump analysis and animation creation."""
    # Generate sample data for demonstration
    # In practice, you would load your experimental data
    np.random.seed(42)
    n_steps = 1000
    n_roi = 16
    
    # Simulate some bump-like activity data
    data = np.random.rand(n_steps, n_roi) * 0.1
    
    # Add some bump-like patterns
    for i in range(n_steps):
        center = (i * 0.01) % n_roi
        for j in range(n_roi):
            distance = min(abs(j - center), n_roi - abs(j - center))
            data[i, j] += 0.5 * np.exp(-distance**2 / 2.0)
    
    print("Running bump fitting analysis...")
    
    # Run bump fitting analysis
    bumps, fits, nbump, centrbump = bump_fits(
        data,
        n_steps=n_steps,
        n_roi=n_roi,
        n_bump_max=2,
        sigma_diff=0.5,
        ampli_min=2.0,
        random_seed=42
    )
    
    print(f"Analysis complete!")
    print(f"Found {len(fits)} time steps with bump data")
    print(f"Average number of bumps: {np.mean(nbump):.2f}")
    
    # Create animation of the bump evolution
    print("Creating bump animation...")
    
    create_1d_bump_animation(
        fits,
        show=False,
        save_path="examples/bump_analysis_demo.gif",
        max_height_value=0.8,
        nframes=min(100, len(fits)),
        fps=10,
        title="1D CANN Bump Analysis Demo"
    )
    
    print("Animation saved as 'examples/bump_analysis_demo.gif'")
    
    return bumps, fits, nbump, centrbump


if __name__ == "__main__":
    results = main()