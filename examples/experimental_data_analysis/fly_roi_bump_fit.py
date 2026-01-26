#!/usr/bin/env python3
"""
Fly ROI bump fitting demo.

Shows how to fit 1D bumps on ROI time series and render an animation.
"""

import numpy as np

from canns.analyzer.data.asa import (
    CANN1DPlotConfig,
    create_1d_bump_animation,
    roi_bump_fits,
)
from canns.data.loaders import load_roi_data

# Load sample ROI data (replace with your experimental data in practice)
data = load_roi_data()

# Run bump fitting
bumps, fits, nbump, centrbump = roi_bump_fits(
    data,
    n_steps=5000,
    n_roi=16,
    random_seed=42
)

print(f"Analysis complete!")
print(f"Found {len(fits)} time steps with bump data")
print(f"Average number of bumps: {np.mean(nbump):.2f}")

# Create bump animation using config-based setup
print("Creating bump animation...")

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
