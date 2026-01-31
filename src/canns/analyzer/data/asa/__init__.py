from __future__ import annotations

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
from .cohomap_scatter import plot_cohomap_scatter, plot_cohomap_scatter_multi
from .config import (
    CANN2DError,
    CANN2DPlotConfig,
    Constants,
    DataLoadError,
    ProcessingError,
    SpikeEmbeddingConfig,
    TDAConfig,
)
from .cohomap import (
    cohomap,
    cohomap_upgrade,
    ecohomap,
    fit_cohomap_stripes,
    fit_cohomap_stripes_upgrade,
    plot_cohomap,
    plot_cohomap_upgrade,
    plot_ecohomap,
)
from .cohospace import (
    cohospace,
    cohospace_upgrade,
    ecohospace,
    plot_cohospace,
    plot_cohospace_skewed,
    plot_cohospace_upgrade,
    plot_cohospace_upgrade_skewed,
    plot_ecohospace,
    plot_ecohospace_skewed,
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
    plot_path_compare_1d,
    plot_path_compare_2d,
    plot_projection,
)
from .cohomap_vectors import (
    cohomap_vectors,
    plot_cohomap_stripes,
    plot_cohomap_vectors,
)
from .cohospace_phase_centers import (
    cohospace_phase_centers,
    plot_cohospace_phase_centers,
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
    "cohomap_upgrade",
    "fit_cohomap_stripes",
    "fit_cohomap_stripes_upgrade",
    "plot_cohomap",
    "plot_cohomap_upgrade",
    "cohospace",
    "cohospace_upgrade",
    "plot_cohospace",
    "plot_cohospace_skewed",
    "plot_cohospace_upgrade",
    "plot_cohospace_upgrade_skewed",
    "ecohomap",
    "ecohospace",
    "plot_ecohomap",
    "plot_ecohospace",
    "plot_ecohospace_skewed",
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
