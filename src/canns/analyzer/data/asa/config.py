from __future__ import annotations

from dataclasses import dataclass

from ...visualization import PlotConfig


@dataclass
class SpikeEmbeddingConfig:
    """Configuration for spike train embedding.

    Attributes
    ----------
    res : int
        Time scaling factor that converts seconds to integer bins.
    dt : int
        Bin width in the same scaled units as ``res``.
    sigma : int
        Gaussian smoothing width (scaled units).
    smooth : bool
        Whether to apply temporal smoothing to spike counts.
    speed_filter : bool
        Whether to filter by animal speed (requires x/y/t in the input).
    min_speed : float
        Minimum speed threshold for ``speed_filter`` (cm/s by convention).

    Examples
    --------
    >>> from canns.analyzer.data import SpikeEmbeddingConfig
    >>> cfg = SpikeEmbeddingConfig(smooth=False, speed_filter=False)
    >>> cfg.min_speed
    2.5
    """

    res: int = 100000
    dt: int = 1000
    sigma: int = 5000
    smooth: bool = True
    speed_filter: bool = True
    min_speed: float = 2.5


@dataclass
class TDAConfig:
    """Configuration for Topological Data Analysis (TDA).

    Attributes
    ----------
    dim : int
        Target PCA dimension before TDA.
    num_times : int
        Downsampling stride in time.
    active_times : int
        Number of most active time points to keep.
    k : int
        Number of neighbors used in denoising.
    n_points : int
        Number of points sampled for persistent homology.
    metric : str
        Distance metric for point cloud (e.g., "cosine").
    nbs : int
        Number of neighbors for distance matrix construction.
    maxdim : int
        Maximum homology dimension for persistence.
    coeff : int
        Field coefficient for persistent homology.
    show : bool
        Whether to show barcode plots.
    do_shuffle : bool
        Whether to run shuffle analysis.
    num_shuffles : int
        Number of shuffles for null distribution.
    progress_bar : bool
        Whether to show progress bars.
    standardize : bool
        Whether to standardize data before PCA (z-score).

    Examples
    --------
    >>> from canns.analyzer.data import TDAConfig
    >>> cfg = TDAConfig(maxdim=1, do_shuffle=False, show=False)
    >>> cfg.maxdim
    1
    """

    dim: int = 6
    num_times: int = 5
    active_times: int = 15000
    k: int = 1000
    n_points: int = 1200
    metric: str = "cosine"
    nbs: int = 800
    maxdim: int = 1
    coeff: int = 47
    show: bool = True
    do_shuffle: bool = False
    num_shuffles: int = 1000
    progress_bar: bool = True
    standardize: bool = True


@dataclass
class CANN2DPlotConfig(PlotConfig):
    """Specialized PlotConfig for CANN2D visualizations.

    Extends :class:`canns.analyzer.visualization.PlotConfig` with fields that
    control 3D projection and torus animation parameters.

    Examples
    --------
    >>> from canns.analyzer.data import CANN2DPlotConfig
    >>> cfg = CANN2DPlotConfig.for_projection_3d(title="Projection")
    >>> cfg.zlabel
    'Component 3'
    """

    # 3D projection specific parameters
    zlabel: str = "Component 3"
    dpi: int = 300

    # Torus animation specific parameters
    numangsint: int = 51
    r1: float = 1.5  # Major radius
    r2: float = 1.0  # Minor radius
    window_size: int = 300
    frame_step: int = 5
    n_frames: int = 20

    @classmethod
    def for_projection_3d(cls, **kwargs) -> CANN2DPlotConfig:
        """Create configuration for 3D projection plots.

        Examples
        --------
        >>> cfg = CANN2DPlotConfig.for_projection_3d(figsize=(6, 5))
        >>> cfg.figsize
        (6, 5)
        """
        defaults = {
            "title": "3D Data Projection",
            "xlabel": "Component 1",
            "ylabel": "Component 2",
            "zlabel": "Component 3",
            "figsize": (10, 8),
            "dpi": 300,
        }
        defaults.update(kwargs)
        return cls.for_static_plot(**defaults)

    @classmethod
    def for_torus_animation(cls, **kwargs) -> CANN2DPlotConfig:
        """Create configuration for 3D torus bump animations.

        Examples
        --------
        >>> cfg = CANN2DPlotConfig.for_torus_animation(fps=10, n_frames=50)
        >>> cfg.fps, cfg.n_frames
        (10, 50)
        """
        defaults = {
            "title": "3D Bump on Torus",
            "figsize": (8, 8),
            "fps": 5,
            "repeat": True,
            "show_progress_bar": True,
            "numangsint": 51,
            "r1": 1.5,
            "r2": 1.0,
            "window_size": 300,
            "frame_step": 5,
            "n_frames": 20,
        }
        defaults.update(kwargs)
        time_steps = kwargs.get("time_steps_per_second", 1000)
        config = cls.for_animation(time_steps, **defaults)
        # Add torus-specific attributes
        config.numangsint = defaults["numangsint"]
        config.r1 = defaults["r1"]
        config.r2 = defaults["r2"]
        config.window_size = defaults["window_size"]
        config.frame_step = defaults["frame_step"]
        config.n_frames = defaults["n_frames"]
        return config


# ==================== Constants ====================


class Constants:
    """Constants used throughout CANN2D analysis.

    Examples
    --------
    >>> from canns.analyzer.data import Constants
    >>> Constants.DEFAULT_DPI
    300
    """

    DEFAULT_FIGSIZE = (10, 8)
    DEFAULT_DPI = 300
    GAUSSIAN_SIGMA_FACTOR = 100
    SPEED_CONVERSION_FACTOR = 100
    TIME_CONVERSION_FACTOR = 0.01
    MULTIPROCESSING_CORES = 4


# ==================== Custom Exceptions ====================


class CANN2DError(Exception):
    """Base exception for CANN2D analysis errors.

    Examples
    --------
    >>> try:  # doctest: +SKIP
    ...     raise CANN2DError("boom")
    ... except CANN2DError:
    ...     pass
    """

    pass


class DataLoadError(CANN2DError):
    """Raised when data loading fails.

    Examples
    --------
    >>> try:  # doctest: +SKIP
    ...     raise DataLoadError("missing data")
    ... except DataLoadError:
    ...     pass
    """

    pass


class ProcessingError(CANN2DError):
    """Raised when data processing fails.

    Examples
    --------
    >>> try:  # doctest: +SKIP
    ...     raise ProcessingError("processing failed")
    ... except ProcessingError:
    ...     pass
    """

    pass
