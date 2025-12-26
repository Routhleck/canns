src.canns.analyzer.data.cann1d
==============================

.. py:module:: src.canns.analyzer.data.cann1d


Attributes
----------

.. autoapisummary::

   src.canns.analyzer.data.cann1d.HAS_NUMBA
   src.canns.analyzer.data.cann1d.data


Exceptions
----------

.. autoapisummary::

   src.canns.analyzer.data.cann1d.AnimationError
   src.canns.analyzer.data.cann1d.CANN1DError
   src.canns.analyzer.data.cann1d.FittingError


Classes
-------

.. autoapisummary::

   src.canns.analyzer.data.cann1d.AnimationConfig
   src.canns.analyzer.data.cann1d.BumpFitsConfig
   src.canns.analyzer.data.cann1d.CANN1DPlotConfig
   src.canns.analyzer.data.cann1d.Constants
   src.canns.analyzer.data.cann1d.SiteBump


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.cann1d.bump_fits
   src.canns.analyzer.data.cann1d.create_1d_bump_animation


Module Contents
---------------

.. py:exception:: AnimationError

   Bases: :py:obj:`CANN1DError`


   Raised when animation creation fails.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:exception:: CANN1DError

   Bases: :py:obj:`Exception`


   Base exception for CANN1D analysis errors.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:exception:: FittingError

   Bases: :py:obj:`CANN1DError`


   Raised when bump fitting fails.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:class:: AnimationConfig

   Configuration for 1D CANN bump animation.


   .. py:attribute:: bump_selection
      :type:  str
      :value: 'strongest'



   .. py:attribute:: fps
      :type:  int
      :value: 5



   .. py:attribute:: max_height_value
      :type:  float
      :value: 0.5



   .. py:attribute:: max_width_range
      :type:  int
      :value: 40



   .. py:attribute:: nframes
      :type:  int | None
      :value: None



   .. py:attribute:: npoints
      :type:  int
      :value: 300



   .. py:attribute:: repeat
      :type:  bool
      :value: False



   .. py:attribute:: show
      :type:  bool
      :value: False



   .. py:attribute:: show_progress_bar
      :type:  bool
      :value: True



   .. py:attribute:: title
      :type:  str
      :value: '1D CANN Bump Animation'



.. py:class:: BumpFitsConfig

   Configuration for CANN1D bump fitting.


   .. py:attribute:: ampli_min
      :type:  float
      :value: 2.0



   .. py:attribute:: beta
      :type:  float
      :value: 5.0



   .. py:attribute:: jc
      :type:  float
      :value: 1.8



   .. py:attribute:: kappa_mean
      :type:  float
      :value: 2.5



   .. py:attribute:: n_bump_max
      :type:  int
      :value: 4



   .. py:attribute:: n_roi
      :type:  int
      :value: 16



   .. py:attribute:: n_steps
      :type:  int
      :value: 20000



   .. py:attribute:: penbump
      :type:  float
      :value: 0.4



   .. py:attribute:: random_seed
      :type:  int | None
      :value: None



   .. py:attribute:: sig2
      :type:  float
      :value: 1.0



   .. py:attribute:: sigma_diff
      :type:  float
      :value: 0.5



.. py:class:: CANN1DPlotConfig

   Bases: :py:obj:`src.canns.analyzer.visualization.PlotConfig`


   Specialized PlotConfig for CANN1D visualizations.


   .. py:method:: for_bump_animation(**kwargs)
      :classmethod:


      Create configuration for 1D CANN bump animation.



   .. py:attribute:: bump_selection
      :type:  str
      :value: 'strongest'



   .. py:attribute:: max_height_value
      :type:  float
      :value: 0.5



   .. py:attribute:: max_width_range
      :type:  int
      :value: 40



   .. py:attribute:: nframes
      :type:  int | None
      :value: None



   .. py:attribute:: npoints
      :type:  int
      :value: 300



.. py:class:: Constants

   Constants used throughout CANN1D analysis.


   .. py:attribute:: BASE_RADIUS
      :value: 1.0



   .. py:attribute:: DEFAULT_DPI
      :value: 100



   .. py:attribute:: DEFAULT_FIGSIZE
      :value: (4, 4)



   .. py:attribute:: MAX_KERNEL_SIZE
      :value: 60



   .. py:attribute:: NUMBA_THRESHOLD
      :value: 64



.. py:class:: SiteBump

   .. py:method:: clone()


   .. py:attribute:: ampli
      :value: []



   .. py:attribute:: kappa
      :value: []



   .. py:attribute:: logl
      :value: 0.0



   .. py:attribute:: nbump
      :value: 0



   .. py:attribute:: pos
      :value: []



.. py:function:: bump_fits(data, config = None, save_path=None, **kwargs)

   Fit CANN1D bumps to data using MCMC optimization.

   :param data: numpy.ndarray
                Input data for bump fitting
   :param config: BumpFitsConfig, optional
                  Configuration object with all fitting parameters
   :param save_path: str, optional
                     Path to save the results
   :param \*\*kwargs: backward compatibility parameters

   :returns:

             list
                 List of fitted bump objects
             fits_array : numpy.ndarray
                 Array of fitted bump parameters
             nbump_array : numpy.ndarray
                 Array of bump counts and reconstructed signals
             centrbump_array : numpy.ndarray
                 Array of centered bump data
   :rtype: bumps


.. py:function:: create_1d_bump_animation(fits_data, config = None, save_path=None, **kwargs)

   Create 1D CANN bump animation using vectorized operations.

   :param fits_data: numpy.ndarray
                     Shape (n_fits, 4) array with columns [time, position, amplitude, kappa]
   :param config: AnimationConfig, optional
                  Configuration object with all animation parameters
   :param save_path: str, optional
                     Output path for the generated GIF
   :param \*\*kwargs: backward compatibility parameters

   :returns:

             matplotlib.animation.FuncAnimation
                 The animation object


.. py:data:: HAS_NUMBA
   :value: True


.. py:data:: data
   :value: None


