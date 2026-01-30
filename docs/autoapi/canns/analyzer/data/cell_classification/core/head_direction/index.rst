canns.analyzer.data.cell_classification.core.head_direction
===========================================================

.. py:module:: canns.analyzer.data.cell_classification.core.head_direction

.. autoapi-nested-parse::

   Head Direction Cell Classification

   Implementation of head direction cell identification based on Mean Vector Length (MVL).

   Based on MATLAB code from the sweeps analysis pipeline.



Attributes
----------

.. autoapisummary::

   canns.analyzer.data.cell_classification.core.head_direction.time_stamps


Classes
-------

.. autoapisummary::

   canns.analyzer.data.cell_classification.core.head_direction.HDCellResult
   canns.analyzer.data.cell_classification.core.head_direction.HeadDirectionAnalyzer


Module Contents
---------------

.. py:class:: HDCellResult

   Results from head direction cell classification.

   .. attribute:: is_hd

      Whether the cell is classified as a head direction cell

      :type: bool

   .. attribute:: mvl_hd

      Mean Vector Length for head direction tuning

      :type: float

   .. attribute:: preferred_direction

      Preferred head direction in radians

      :type: float

   .. attribute:: mvl_theta

      Mean Vector Length for theta phase tuning (if provided)

      :type: float or None

   .. attribute:: tuning_curve

      Tuple of (bin_centers, firing_rates)

      :type: tuple

   .. attribute:: rayleigh_p

      P-value from Rayleigh test for non-uniformity

      :type: float


   .. py:attribute:: is_hd
      :type:  bool


   .. py:attribute:: mvl_hd
      :type:  float


   .. py:attribute:: mvl_theta
      :type:  float | None


   .. py:attribute:: preferred_direction
      :type:  float


   .. py:attribute:: rayleigh_p
      :type:  float


   .. py:attribute:: tuning_curve
      :type:  tuple[numpy.ndarray, numpy.ndarray]


.. py:class:: HeadDirectionAnalyzer(mvl_hd_threshold = 0.4, mvl_theta_threshold = 0.3, strict_mode = True, n_bins = 60)

   Analyzer for classifying head direction cells based on directional tuning.

   Head direction cells fire when the animal's head points in a specific direction.
   Classification is based on the strength of directional tuning measured by
   Mean Vector Length (MVL).

   :param mvl_hd_threshold: MVL threshold for head direction. Default is 0.4 (strict).
                            Use 0.2 for looser threshold.
   :type mvl_hd_threshold: float, optional
   :param mvl_theta_threshold: MVL threshold for theta phase modulation. Default is 0.3.
   :type mvl_theta_threshold: float, optional
   :param strict_mode: If True, requires both HD and theta criteria. Default is True.
   :type strict_mode: bool, optional
   :param n_bins: Number of directional bins for tuning curve. Default is 60 (6° bins).
   :type n_bins: int, optional

   .. rubric:: Examples

   >>> analyzer = HeadDirectionAnalyzer(mvl_hd_threshold=0.4, strict_mode=True)
   >>> result = analyzer.classify_hd_cell(spike_times, head_directions, time_stamps)
   >>> print(f"Is HD cell: {result.is_hd}")
   >>> print(f"MVL: {result.mvl_hd:.3f}")
   >>> print(f"Preferred direction: {np.rad2deg(result.preferred_direction):.1f}°")

   .. rubric:: Notes

   Based on MATLAB classification from fig2.m and plotSwsExample.m:
   - Strict: MVL_hd > 0.4 AND MVL_theta > 0.3
   - Loose: MVL_hd > 0.2 AND MVL_theta > 0.3

   .. rubric:: References

   Classification thresholds follow standard conventions in head direction
   cell literature and the CircStat toolbox.


   .. py:method:: classify_hd_cell(spike_times, head_directions, time_stamps, theta_phases = None)

      Classify a cell as head direction cell based on MVL thresholds.

      :param spike_times: Spike times in seconds
      :type spike_times: np.ndarray
      :param head_directions: Head direction at each time point (radians)
      :type head_directions: np.ndarray
      :param time_stamps: Time stamps corresponding to head_directions (seconds)
      :type time_stamps: np.ndarray
      :param theta_phases: Theta phase at each time point (radians). If None, theta
                           criterion is not checked.
      :type theta_phases: np.ndarray, optional

      :returns: **result** -- Classification result with MVL, preferred direction, and tuning curve
      :rtype: HDCellResult

      .. rubric:: Examples

      >>> # Simulate a head direction cell
      >>> time_stamps = np.linspace(0, 100, 10000)
      >>> head_directions = np.linspace(0, 20*np.pi, 10000) % (2*np.pi) - np.pi
      >>> preferred_dir = 0.5
      >>> spike_times = time_stamps[np.abs(head_directions - preferred_dir) < 0.3]
      >>> result = analyzer.classify_hd_cell(spike_times, head_directions, time_stamps)



   .. py:method:: compute_mvl(angles, weights = None)

      Compute Mean Vector Length (MVL).

      The MVL is a measure of circular variance, ranging from 0 (uniform
      distribution) to 1 (concentrated distribution).

      :param angles: Angles in radians
      :type angles: np.ndarray
      :param weights: Weights for each angle (e.g., firing rates). If None, uniform weights.
      :type weights: np.ndarray, optional

      :returns: **mvl** -- Mean vector length
      :rtype: float

      .. rubric:: Examples

      >>> # Concentrated distribution
      >>> angles = np.random.normal(0, 0.1, 100)
      >>> mvl = analyzer.compute_mvl(angles)
      >>> print(f"MVL: {mvl:.3f}")  # Should be close to 1

      >>> # Uniform distribution
      >>> angles = np.random.uniform(-np.pi, np.pi, 100)
      >>> mvl = analyzer.compute_mvl(angles)
      >>> print(f"MVL: {mvl:.3f}")  # Should be close to 0

      .. rubric:: Notes

      Uses the circ_r function from circular statistics utilities.



   .. py:method:: compute_tuning_curve(spike_times, head_directions, time_stamps, n_bins = None)

      Compute directional tuning curve.

      :param spike_times: Spike times in seconds
      :type spike_times: np.ndarray
      :param head_directions: Head direction at each time point (radians)
      :type head_directions: np.ndarray
      :param time_stamps: Time stamps corresponding to head_directions (seconds)
      :type time_stamps: np.ndarray
      :param n_bins: Number of bins. If None, uses self.n_bins.
      :type n_bins: int, optional

      :returns: * **bin_centers** (*np.ndarray*) -- Center of each directional bin (radians)
                * **firing_rates** (*np.ndarray*) -- Firing rate in each bin (Hz)
                * **occupancy** (*np.ndarray*) -- Time spent in each bin (seconds)

      .. rubric:: Examples

      >>> bins, rates, occ = analyzer.compute_tuning_curve(
      ...     spike_times, head_directions, time_stamps
      ... )
      >>> # Plot polar tuning curve
      >>> import matplotlib.pyplot as plt
      >>> ax = plt.subplot(111, projection='polar')
      >>> ax.plot(bins, rates)



   .. py:attribute:: mvl_hd_threshold
      :value: 0.4



   .. py:attribute:: mvl_theta_threshold
      :value: 0.3



   .. py:attribute:: n_bins
      :value: 60



   .. py:attribute:: strict_mode
      :value: True



.. py:data:: time_stamps

