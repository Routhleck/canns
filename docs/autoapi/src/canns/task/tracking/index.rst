src.canns.task.tracking
=======================

.. py:module:: src.canns.task.tracking


Classes
-------

.. autoapisummary::

   src.canns.task.tracking.PopulationCoding1D
   src.canns.task.tracking.PopulationCoding2D
   src.canns.task.tracking.SmoothTracking1D
   src.canns.task.tracking.SmoothTracking2D
   src.canns.task.tracking.TemplateMatching1D
   src.canns.task.tracking.TemplateMatching2D


Module Contents
---------------

.. py:class:: PopulationCoding1D(cann_instance, before_duration, after_duration, Iext, duration, time_step = 0.1)

   Bases: :py:obj:`PopulationCoding`


   Population coding task for 1D continuous attractor networks.

   A stimulus is presented for a specific duration, preceded and followed by
   periods of no stimulation.

   Workflow:
       Setup -> Create a 1D CANN and the task.
       Execute -> Call ``get_data()``.
       Result -> Use ``task.data`` as the input sequence.

   .. rubric:: Examples

   >>> import brainpy.math as bm
   >>> from canns.models.basic import CANN1D
   >>> from canns.task.tracking import PopulationCoding1D
   >>>
   >>> bm.set_dt(0.1)
   >>> model = CANN1D(num=64)
   >>> task = PopulationCoding1D(
   ...     cann_instance=model,
   ...     before_duration=1.0,
   ...     after_duration=1.0,
   ...     Iext=0.0,
   ...     duration=2.0,
   ...     time_step=bm.get_dt(),
   ... )
   >>> task.get_data()
   >>> task.data.shape[0] == task.total_steps
   True

   Initializes the Population Coding task.

   :param cann_instance: An instance of the 1D CANN model.
   :type cann_instance: BaseCANN1D
   :param before_duration: Duration of the initial period with no stimulus.
   :type before_duration: float | Quantity
   :param after_duration: Duration of the final period with no stimulus.
   :type after_duration: float | Quantity
   :param Iext: The position of the external input during the stimulation period.
   :type Iext: float | Quantity
   :param duration: The duration of the stimulation period.
   :type duration: float | Quantity
   :param time_step: The simulation time step. Defaults to 0.1.
   :type time_step: float | Quantity, optional


   .. py:attribute:: after_duration


   .. py:attribute:: before_duration


.. py:class:: PopulationCoding2D(cann_instance, before_duration, after_duration, Iext, duration, time_step = 0.1)

   Bases: :py:obj:`PopulationCoding`


   Population coding task for 2D continuous attractor networks.

   A 2D stimulus is presented for a duration with pre- and post-silence.

   Workflow:
       Setup -> Create a 2D CANN and the task.
       Execute -> Call ``get_data()``.
       Result -> Use ``task.data`` as the input sequence.

   .. rubric:: Examples

   >>> import brainpy.math as bm
   >>> from canns.models.basic import CANN2D
   >>> from canns.task.tracking import PopulationCoding2D
   >>>
   >>> bm.set_dt(0.1)
   >>> model = CANN2D(length=8)
   >>> task = PopulationCoding2D(
   ...     cann_instance=model,
   ...     before_duration=1.0,
   ...     after_duration=1.0,
   ...     Iext=(0.0, 0.0),
   ...     duration=1.0,
   ...     time_step=bm.get_dt(),
   ... )
   >>> task.get_data()
   >>> task.data.shape[1:] == model.shape
   True

   Initializes the Population Coding task.

   :param cann_instance: An instance of the 2D CANN model.
   :type cann_instance: BaseCANN2D
   :param before_duration: Duration of the initial period with no stimulus.
   :type before_duration: float | Quantity
   :param after_duration: Duration of the final period with no stimulus.
   :type after_duration: float | Quantity
   :param Iext: The position of the external input during the stimulation period.
   :type Iext: float | Quantity
   :param duration: The duration of the stimulation period.
   :type duration: float | Quantity
   :param time_step: The simulation time step. Defaults to 0.1.
   :type time_step: float | Quantity, optional


   .. py:attribute:: after_duration


   .. py:attribute:: before_duration


.. py:class:: SmoothTracking1D(cann_instance, Iext, duration, time_step = 0.1)

   Bases: :py:obj:`SmoothTracking`


   Smooth tracking task for 1D continuous attractor networks.

   The external input moves smoothly between key positions.

   Workflow:
       Setup -> Create a 1D CANN and the task.
       Execute -> Call ``get_data()``.
       Result -> ``task.data`` contains the smoothly varying stimulus.

   .. rubric:: Examples

   >>> import brainpy.math as bm
   >>> from canns.models.basic import CANN1D
   >>> from canns.task.tracking import SmoothTracking1D
   >>>
   >>> bm.set_dt(0.1)
   >>> model = CANN1D(num=64)
   >>> task = SmoothTracking1D(
   ...     cann_instance=model,
   ...     Iext=(0.0, 1.0, 0.5),
   ...     duration=(0.5, 0.5),
   ...     time_step=bm.get_dt(),
   ... )
   >>> task.get_data()
   >>> task.data.shape[0] == task.total_steps
   True

   Initializes the Smooth Tracking task.

   :param cann_instance: An instance of the 1D CANN model.
   :type cann_instance: BaseCANN1D
   :param Iext: A sequence of keypoint positions for the input.
   :type Iext: Sequence[float | Quantity]
   :param duration: The duration of each segment of smooth movement.
   :type duration: Sequence[float | Quantity]
   :param time_step: The simulation time step. Defaults to 0.1.
   :type time_step: float | Quantity, optional


.. py:class:: SmoothTracking2D(cann_instance, Iext, duration, time_step = 0.1)

   Bases: :py:obj:`SmoothTracking`


   Smooth tracking task for 2D continuous attractor networks.

   The external 2D input moves smoothly between key positions.

   Workflow:
       Setup -> Create a 2D CANN and the task.
       Execute -> Call ``get_data()``.
       Result -> ``task.data`` contains smoothly varying 2D inputs.

   .. rubric:: Examples

   >>> import brainpy.math as bm
   >>> from canns.models.basic import CANN2D
   >>> from canns.task.tracking import SmoothTracking2D
   >>>
   >>> bm.set_dt(0.1)
   >>> model = CANN2D(length=8)
   >>> task = SmoothTracking2D(
   ...     cann_instance=model,
   ...     Iext=((0.0, 0.0), (1.0, 1.0), (0.5, 0.5)),
   ...     duration=(0.5, 0.5),
   ...     time_step=bm.get_dt(),
   ... )
   >>> task.get_data()
   >>> task.data.shape[1:] == model.shape
   True

   Initializes the Smooth Tracking task.

   :param cann_instance: An instance of the 2D CANN model.
   :type cann_instance: BaseCANN2D
   :param Iext: A sequence of 2D keypoint positions for the input.
   :type Iext: Sequence[tuple[float, float] | Quantity]
   :param duration: The duration of each segment of smooth movement.
   :type duration: Sequence[float | Quantity]
   :param time_step: The simulation time step. Defaults to 0.1.
   :type time_step: float | Quantity, optional


.. py:class:: TemplateMatching1D(cann_instance, Iext, duration, time_step = 0.1)

   Bases: :py:obj:`TemplateMatching`


   Template matching task for 1D continuous attractor networks.

   A fixed stimulus template is presented with noise at each step, testing
   the network's denoising dynamics.

   Workflow:
       Setup -> Create a 1D CANN and the task.
       Execute -> Call ``get_data()``.
       Result -> Use ``task.data`` as the noisy input sequence.

   .. rubric:: Examples

   >>> import brainpy.math as bm
   >>> from canns.models.basic import CANN1D
   >>> from canns.task.tracking import TemplateMatching1D
   >>>
   >>> bm.set_dt(0.1)
   >>> model = CANN1D(num=64)
   >>> task = TemplateMatching1D(
   ...     cann_instance=model,
   ...     Iext=0.0,
   ...     duration=1.0,
   ...     time_step=bm.get_dt(),
   ... )
   >>> task.get_data()
   >>> task.data.shape[1] == model.shape[0]
   True

   Initializes the Template Matching task.

   :param cann_instance: An instance of the 1D CANN model.
   :type cann_instance: BaseCANN1D
   :param Iext: The position of the external input.
   :type Iext: float | Quantity
   :param duration: The duration for which the noisy stimulus is presented.
   :type duration: float | Quantity
   :param time_step: The simulation time step. Defaults to 0.1.
   :type time_step: float | Quantity, optional


.. py:class:: TemplateMatching2D(cann_instance, Iext, duration, time_step = 0.1)

   Bases: :py:obj:`TemplateMatching`


   Template matching task for 2D continuous attractor networks.

   A 2D template is presented with noise at each step.

   Workflow:
       Setup -> Create a 2D CANN and the task.
       Execute -> Call ``get_data()``.
       Result -> ``task.data`` contains noisy 2D inputs.

   .. rubric:: Examples

   >>> import brainpy.math as bm
   >>> from canns.models.basic import CANN2D
   >>> from canns.task.tracking import TemplateMatching2D
   >>>
   >>> bm.set_dt(0.1)
   >>> model = CANN2D(length=8)
   >>> task = TemplateMatching2D(
   ...     cann_instance=model,
   ...     Iext=(0.0, 0.0),
   ...     duration=1.0,
   ...     time_step=bm.get_dt(),
   ... )
   >>> task.get_data()
   >>> task.data.shape[1:] == model.shape
   True

   Initializes the Template Matching task.

   :param cann_instance: An instance of the 2D CANN model.
   :type cann_instance: BaseCANN2D
   :param Iext: The 2D position of the external input.
   :type Iext: tuple[float, float] | Quantity
   :param duration: The duration for which the noisy stimulus is presented.
   :type duration: float | Quantity
   :param time_step: The simulation time step. Defaults to 0.1.
   :type time_step: float | Quantity, optional


