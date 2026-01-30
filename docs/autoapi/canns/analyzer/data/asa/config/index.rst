canns.analyzer.data.asa.config
==============================

.. py:module:: canns.analyzer.data.asa.config


Exceptions
----------

.. autoapisummary::

   canns.analyzer.data.asa.config.CANN2DError
   canns.analyzer.data.asa.config.DataLoadError
   canns.analyzer.data.asa.config.ProcessingError


Classes
-------

.. autoapisummary::

   canns.analyzer.data.asa.config.CANN2DPlotConfig
   canns.analyzer.data.asa.config.Constants
   canns.analyzer.data.asa.config.SpikeEmbeddingConfig
   canns.analyzer.data.asa.config.TDAConfig


Module Contents
---------------

.. py:exception:: CANN2DError

   Bases: :py:obj:`Exception`


   Base exception for CANN2D analysis errors.

   .. rubric:: Examples

   >>> try:  # doctest: +SKIP
   ...     raise CANN2DError("boom")
   ... except CANN2DError:
   ...     pass

   Initialize self.  See help(type(self)) for accurate signature.


.. py:exception:: DataLoadError

   Bases: :py:obj:`CANN2DError`


   Raised when data loading fails.

   .. rubric:: Examples

   >>> try:  # doctest: +SKIP
   ...     raise DataLoadError("missing data")
   ... except DataLoadError:
   ...     pass

   Initialize self.  See help(type(self)) for accurate signature.


.. py:exception:: ProcessingError

   Bases: :py:obj:`CANN2DError`


   Raised when data processing fails.

   .. rubric:: Examples

   >>> try:  # doctest: +SKIP
   ...     raise ProcessingError("processing failed")
   ... except ProcessingError:
   ...     pass

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: CANN2DPlotConfig

   Bases: :py:obj:`canns.analyzer.visualization.PlotConfig`


   Specialized PlotConfig for CANN2D visualizations.

   Extends :class:`canns.analyzer.visualization.PlotConfig` with fields that
   control 3D projection and torus animation parameters.

   .. rubric:: Examples

   >>> from canns.analyzer.data import CANN2DPlotConfig
   >>> cfg = CANN2DPlotConfig.for_projection_3d(title="Projection")
   >>> cfg.zlabel
   'Component 3'


   .. py:method:: for_projection_3d(**kwargs)
      :classmethod:


      Create configuration for 3D projection plots.

      .. rubric:: Examples

      >>> cfg = CANN2DPlotConfig.for_projection_3d(figsize=(6, 5))
      >>> cfg.figsize
      (6, 5)



   .. py:method:: for_torus_animation(**kwargs)
      :classmethod:


      Create configuration for 3D torus bump animations.

      .. rubric:: Examples

      >>> cfg = CANN2DPlotConfig.for_torus_animation(fps=10, n_frames=50)
      >>> cfg.fps, cfg.n_frames
      (10, 50)



   .. py:attribute:: dpi
      :type:  int
      :value: 300



   .. py:attribute:: frame_step
      :type:  int
      :value: 5



   .. py:attribute:: n_frames
      :type:  int
      :value: 20



   .. py:attribute:: numangsint
      :type:  int
      :value: 51



   .. py:attribute:: r1
      :type:  float
      :value: 1.5



   .. py:attribute:: r2
      :type:  float
      :value: 1.0



   .. py:attribute:: window_size
      :type:  int
      :value: 300



   .. py:attribute:: zlabel
      :type:  str
      :value: 'Component 3'



.. py:class:: Constants

   Constants used throughout CANN2D analysis.

   .. rubric:: Examples

   >>> from canns.analyzer.data import Constants
   >>> Constants.DEFAULT_DPI
   300


   .. py:attribute:: DEFAULT_DPI
      :value: 300



   .. py:attribute:: DEFAULT_FIGSIZE
      :value: (10, 8)



   .. py:attribute:: GAUSSIAN_SIGMA_FACTOR
      :value: 100



   .. py:attribute:: MULTIPROCESSING_CORES
      :value: 4



   .. py:attribute:: SPEED_CONVERSION_FACTOR
      :value: 100



   .. py:attribute:: TIME_CONVERSION_FACTOR
      :value: 0.01



.. py:class:: SpikeEmbeddingConfig

   Configuration for spike train embedding.

   .. attribute:: res

      Time scaling factor that converts seconds to integer bins.

      :type: int

   .. attribute:: dt

      Bin width in the same scaled units as ``res``.

      :type: int

   .. attribute:: sigma

      Gaussian smoothing width (scaled units).

      :type: int

   .. attribute:: smooth

      Whether to apply temporal smoothing to spike counts.

      :type: bool

   .. attribute:: speed_filter

      Whether to filter by animal speed (requires x/y/t in the input).

      :type: bool

   .. attribute:: min_speed

      Minimum speed threshold for ``speed_filter`` (cm/s by convention).

      :type: float

   .. rubric:: Examples

   >>> from canns.analyzer.data import SpikeEmbeddingConfig
   >>> cfg = SpikeEmbeddingConfig(smooth=False, speed_filter=False)
   >>> cfg.min_speed
   2.5


   .. py:attribute:: dt
      :type:  int
      :value: 1000



   .. py:attribute:: min_speed
      :type:  float
      :value: 2.5



   .. py:attribute:: res
      :type:  int
      :value: 100000



   .. py:attribute:: sigma
      :type:  int
      :value: 5000



   .. py:attribute:: smooth
      :type:  bool
      :value: True



   .. py:attribute:: speed_filter
      :type:  bool
      :value: True



.. py:class:: TDAConfig

   Configuration for Topological Data Analysis (TDA).

   .. attribute:: dim

      Target PCA dimension before TDA.

      :type: int

   .. attribute:: num_times

      Downsampling stride in time.

      :type: int

   .. attribute:: active_times

      Number of most active time points to keep.

      :type: int

   .. attribute:: k

      Number of neighbors used in denoising.

      :type: int

   .. attribute:: n_points

      Number of points sampled for persistent homology.

      :type: int

   .. attribute:: metric

      Distance metric for point cloud (e.g., "cosine").

      :type: str

   .. attribute:: nbs

      Number of neighbors for distance matrix construction.

      :type: int

   .. attribute:: maxdim

      Maximum homology dimension for persistence.

      :type: int

   .. attribute:: coeff

      Field coefficient for persistent homology.

      :type: int

   .. attribute:: show

      Whether to show barcode plots.

      :type: bool

   .. attribute:: do_shuffle

      Whether to run shuffle analysis.

      :type: bool

   .. attribute:: num_shuffles

      Number of shuffles for null distribution.

      :type: int

   .. attribute:: progress_bar

      Whether to show progress bars.

      :type: bool

   .. attribute:: standardize

      Whether to standardize data before PCA (z-score).

      :type: bool

   .. rubric:: Examples

   >>> from canns.analyzer.data import TDAConfig
   >>> cfg = TDAConfig(maxdim=1, do_shuffle=False, show=False)
   >>> cfg.maxdim
   1


   .. py:attribute:: active_times
      :type:  int
      :value: 15000



   .. py:attribute:: coeff
      :type:  int
      :value: 47



   .. py:attribute:: dim
      :type:  int
      :value: 6



   .. py:attribute:: do_shuffle
      :type:  bool
      :value: False



   .. py:attribute:: k
      :type:  int
      :value: 1000



   .. py:attribute:: maxdim
      :type:  int
      :value: 1



   .. py:attribute:: metric
      :type:  str
      :value: 'cosine'



   .. py:attribute:: n_points
      :type:  int
      :value: 1200



   .. py:attribute:: nbs
      :type:  int
      :value: 800



   .. py:attribute:: num_shuffles
      :type:  int
      :value: 1000



   .. py:attribute:: num_times
      :type:  int
      :value: 5



   .. py:attribute:: progress_bar
      :type:  bool
      :value: True



   .. py:attribute:: show
      :type:  bool
      :value: True



   .. py:attribute:: standardize
      :type:  bool
      :value: True



