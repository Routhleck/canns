#!/usr/bin/env python3
"""
CANN 1D Bump Analysis Example

This example demonstrates how to use the bump_fits and create_1d_bump_animation functions
from the experimental data analyzer to analyze 1D CANN bumps.
"""

import numpy as np
from canns.analyzer.experimental_data import bump_fits, create_1d_bump_animation, load_roi_data


def main():
    """Demonstrate bump analysis and animation creation."""
    # Generate sample data for demonstration
    # In practice, you would load your experimental data
    data = load_roi_data()
    
    # Run bump fitting analysis
    bumps, fits, nbump, centrbump = bump_fits(
        data,
        n_steps=5000,
        n_roi=16,
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
        nframes=100,
        fps=10,
        title="1D CANN Bump Analysis Demo"
    )
    
    print("Animation saved as 'examples/bump_analysis_demo.gif'")
    
    return bumps, fits, nbump, centrbump


if __name__ == "__main__":
    results = main()