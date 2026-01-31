"""
Cell Classification Package

Python implementation of grid cell and head direction cell classification algorithms.

Based on the MATLAB code from:
Vollan, Gardner, Moser & Moser (Nature, 2025)
"Left-right-alternating sweeps in entorhinal-hippocampal maps of space"
"""

__version__ = "0.1.0"

from .core import (  # noqa: F401
    BTNAnalyzer,
    BTNConfig,
    BTNResult,
    GridnessAnalyzer,
    GridnessResult,
    HDCellResult,
    HeadDirectionAnalyzer,
    compute_2d_autocorrelation,
    compute_field_statistics,
    compute_grid_spacing,
    compute_rate_map,
    compute_rate_map_from_binned,
    compute_spatial_information,
    identify_grid_modules_and_stats,
)
from .io import MATFileLoader, TuningCurve, Unit  # noqa: F401
from .utils import (  # noqa: F401
    autocorrelation_2d,
    cart2pol,
    circ_dist,
    circ_dist2,
    circ_mean,
    circ_r,
    circ_rtest,
    circ_std,
    fit_ellipse,
    label_connected_components,
    normalized_xcorr2,
    pearson_correlation,
    pol2cart,
    polyarea,
    regionprops,
    rotate_image,
    squared_distance,
    wrap_to_pi,
)
from .visualization import (  # noqa: F401
    plot_autocorrelogram,
    plot_btn_autocorr_summary,
    plot_btn_distance_matrix,
    plot_grid_score_histogram,
    plot_gridness_analysis,
    plot_hd_analysis,
    plot_polar_tuning,
    plot_rate_map,
    plot_temporal_autocorr,
)

__all__ = [
    "GridnessAnalyzer",
    "GridnessResult",
    "BTNAnalyzer",
    "BTNConfig",
    "BTNResult",
    "HeadDirectionAnalyzer",
    "HDCellResult",
    "compute_2d_autocorrelation",
    "compute_rate_map",
    "compute_rate_map_from_binned",
    "compute_spatial_information",
    "compute_field_statistics",
    "compute_grid_spacing",
    "identify_grid_modules_and_stats",
    "MATFileLoader",
    "TuningCurve",
    "Unit",
    "circ_r",
    "circ_mean",
    "circ_std",
    "circ_dist",
    "circ_dist2",
    "circ_rtest",
    "pearson_correlation",
    "normalized_xcorr2",
    "autocorrelation_2d",
    "fit_ellipse",
    "squared_distance",
    "polyarea",
    "wrap_to_pi",
    "cart2pol",
    "pol2cart",
    "rotate_image",
    "label_connected_components",
    "regionprops",
    "plot_autocorrelogram",
    "plot_gridness_analysis",
    "plot_rate_map",
    "plot_grid_score_histogram",
    "plot_polar_tuning",
    "plot_temporal_autocorr",
    "plot_hd_analysis",
    "plot_btn_autocorr_summary",
    "plot_btn_distance_matrix",
]
