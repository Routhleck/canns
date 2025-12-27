#!/usr/bin/env python3
"""
CANN 1D Bump Analysis Example

This example demonstrates how to use the bump_fits and create_1d_bump_animation functions
from the experimental data analyzer to analyze 1D CANN bumps.
"""

import numpy as np

from canns.analyzer.data.cann1d_metrics import (
    bump_fits, create_1d_bump_animation, CANN1DPlotConfig
)
from canns.data.loaders import load_roi_data

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

# Create animation of the bump evolution using new config approach
print("Creating bump animation...")

# Using new config-based approach
config = CANN1DPlotConfig.for_bump_animation(
    show=False,
    save_path="bump_analysis_demo.mp4",
    nframes=100,
    fps=10,
    title="1D CANN Bump Analysis Demo",
    max_height_value=0.6,
    show_progress_bar=True
)

create_1d_bump_animation(
    fits_data=fits,
    config=config
)

print("Animation saved as 'bump_analysis_demo.mp4'")

# For comparison, the old-style approach still works:
# create_1d_bump_animation(
#     fits,
#     show=False,
#     save_path="bump_analysis_demo_old.mp4",
#     nframes=100,
#     fps=10,
#     title="1D CANN Bump Analysis Demo (Old Style)"
# )
# print("Old-style animation also saved as 'bump_analysis_demo_old.mp4'")
