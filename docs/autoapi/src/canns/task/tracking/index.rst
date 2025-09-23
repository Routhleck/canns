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
   In this task, a stimulus is presented for a specific duration, preceded and followed by
   periods of no stimulation, to test the network's ability to form and maintain a memory bump.

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
   In this task, a stimulus is presented for a specific duration, preceded and followed by
   periods of no stimulation, to test the network's ability to form and maintain a memory bump.

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
   This task provides an external input that moves smoothly over time, testing the network's
   ability to track a continuously changing stimulus.

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
   This task provides an external input that moves smoothly over time, testing the network's
   ability to track a continuously changing stimulus.

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
   This task presents a stimulus with added noise to test the network's ability to
   denoise the input and settle on the correct underlying pattern (template).

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
   This task presents a stimulus with added noise to test the network's ability to
   denoise the input and settle on the correct underlying pattern (template).

   Initializes the Template Matching task.

   :param cann_instance: An instance of the 2D CANN model.
   :type cann_instance: BaseCANN2D
   :param Iext: The 2D position of the external input.
   :type Iext: tuple[float, float] | Quantity
   :param duration: The duration for which the noisy stimulus is presented.
   :type duration: float | Quantity
   :param time_step: The simulation time step. Defaults to 0.1.
   :type time_step: float | Quantity, optional


