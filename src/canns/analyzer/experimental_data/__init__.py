from ..visualize import PlotConfig
from ._datasets_utils import load_grid_data, load_roi_data, validate_grid_data, validate_roi_data
from .cann1d import CANN1DPlotConfig, bump_fits, create_1d_bump_animation
from .cann2d import (
    CANN2DPlotConfig,
    SpikeEmbeddingConfig,
    TDAConfig,
    decode_circular_coordinates,
    embed_spike_trains,
    plot_3d_bump_on_torus,
    plot_projection,
    tda_vis,
)

__all__ = [
    # CANN1D functions
    "bump_fits",
    "create_1d_bump_animation",
    # CANN2D functions
    "embed_spike_trains",
    "tda_vis",
    "plot_projection",
    "decode_circular_coordinates",
    "plot_3d_bump_on_torus",
    # Configuration classes
    "SpikeEmbeddingConfig",
    "TDAConfig",
    "PlotConfig",
    "CANN1DPlotConfig",
    "CANN2DPlotConfig",
    # Data utilities
    "load_roi_data",
    "load_grid_data",
    "validate_roi_data",
    "validate_grid_data",
]
