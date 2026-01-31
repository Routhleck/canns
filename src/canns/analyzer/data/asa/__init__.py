from __future__ import annotations

from .cohomap import (
    cohomap,
    fit_cohomap_stripes,
    plot_cohomap,
)
from .cohomap_vectors import (
    cohomap_vectors,
    plot_cohomap_stripes,
    plot_cohomap_vectors,
)
from .cohospace import (
    cohospace,
    plot_cohospace,
    plot_cohospace_skewed,
)
from .cohospace_phase_centers import (
    cohospace_phase_centers,
    plot_cohospace_phase_centers,
)

# Coho-space (scatter) analysis + visualization
from .cohospace_scatter import (
    compute_cohoscore_scatter_1d,
    compute_cohoscore_scatter_2d,
    plot_cohospace_scatter_neuron_1d,
    plot_cohospace_scatter_neuron_2d,
    plot_cohospace_scatter_neuron_skewed,
    plot_cohospace_scatter_population_1d,
    plot_cohospace_scatter_population_2d,
    plot_cohospace_scatter_population_skewed,
    plot_cohospace_scatter_trajectory_1d,
    plot_cohospace_scatter_trajectory_2d,
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
    plot_cohomap_scatter,
    plot_cohomap_scatter_multi,
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
    "plot_cohomap_scatter",
    "plot_cohomap_scatter_multi",
    "plot_3d_bump_on_torus",
    "plot_2d_bump_on_manifold",
    "cohomap",
    "fit_cohomap_stripes",
    "plot_cohomap",
    "cohospace",
    "plot_cohospace",
    "plot_cohospace_skewed",
    "cohomap_vectors",
    "plot_cohomap_stripes",
    "plot_cohomap_vectors",
    "cohospace_phase_centers",
    "plot_cohospace_phase_centers",
    "BumpFitsConfig",
    "CANN1DPlotConfig",
    "create_1d_bump_animation",
    "roi_bump_fits",
    "compute_fr_heatmap_matrix",
    "save_fr_heatmap_png",
    "FRMResult",
    "compute_frm",
    "plot_frm",
    "plot_cohospace_scatter_trajectory_1d",
    "plot_cohospace_scatter_trajectory_2d",
    "plot_cohospace_scatter_neuron_1d",
    "plot_cohospace_scatter_neuron_2d",
    "plot_cohospace_scatter_population_1d",
    "plot_cohospace_scatter_population_2d",
    "plot_cohospace_scatter_neuron_skewed",
    "plot_cohospace_scatter_population_skewed",
    "compute_cohoscore_scatter_1d",
    "compute_cohoscore_scatter_2d",
    "align_coords_to_position_1d",
    "align_coords_to_position_2d",
    "apply_angle_scale",
]
