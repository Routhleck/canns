from __future__ import annotations

# Coho-space analysis + visualization
from .cohospace import (
    compute_cohoscore_1d,
    compute_cohoscore_2d,
    plot_cohospace_neuron_1d,
    plot_cohospace_neuron_2d,
    plot_cohospace_population_1d,
    plot_cohospace_population_2d,
    plot_cohospace_trajectory_1d,
    plot_cohospace_trajectory_2d,
)
from .config import (
    CANN2DError,
    CANN2DPlotConfig,
    Constants,
    DataLoadError,
    ProcessingError,
    SpikeEmbeddingConfig,
    TDAConfig,
)
from .decode import decode_circular_coordinates, decode_circular_coordinates_multi
from .embedding import embed_spike_trains
from .fly_roi import (
    BumpFitsConfig,
    CANN1DPlotConfig,
    create_1d_bump_animation,
    roi_bump_fits,
)
from .fr import (
    FRMResult,
    compute_fr_heatmap_matrix,
    compute_frm,
    plot_frm,
    save_fr_heatmap_png,
)

# Path utilities
from .path import (
    align_coords_to_position_1d,
    align_coords_to_position_2d,
    apply_angle_scale,
)

# Higher-level plotting helpers
from .plotting import (
    plot_2d_bump_on_manifold,
    plot_3d_bump_on_torus,
    plot_cohomap,
    plot_cohomap_multi,
    plot_path_compare_1d,
    plot_path_compare_2d,
    plot_projection,
)

# TDA entry point
from .tda import tda_vis

__all__ = [
    "SpikeEmbeddingConfig",
    "TDAConfig",
    "CANN2DPlotConfig",
    "Constants",
    "CANN2DError",
    "DataLoadError",
    "ProcessingError",
    "embed_spike_trains",
    "tda_vis",
    "decode_circular_coordinates",
    "decode_circular_coordinates_multi",
    "plot_projection",
    "plot_path_compare_1d",
    "plot_path_compare_2d",
    "plot_cohomap",
    "plot_cohomap_multi",
    "plot_3d_bump_on_torus",
    "plot_2d_bump_on_manifold",
    "BumpFitsConfig",
    "CANN1DPlotConfig",
    "create_1d_bump_animation",
    "roi_bump_fits",
    "compute_fr_heatmap_matrix",
    "save_fr_heatmap_png",
    "FRMResult",
    "compute_frm",
    "plot_frm",
    "plot_cohospace_trajectory_1d",
    "plot_cohospace_trajectory_2d",
    "plot_cohospace_neuron_1d",
    "plot_cohospace_neuron_2d",
    "plot_cohospace_population_1d",
    "plot_cohospace_population_2d",
    "compute_cohoscore_1d",
    "compute_cohoscore_2d",
    "align_coords_to_position_1d",
    "align_coords_to_position_2d",
    "apply_angle_scale",
]
