"""
Head Direction Cell Classification

Implementation of head direction cell identification based on Mean Vector Length (MVL).

Based on MATLAB code from the sweeps analysis pipeline.
"""

from dataclasses import dataclass

import numpy as np

from ..utils.circular_stats import circ_mean, circ_r, circ_rtest


@dataclass
class HDCellResult:
    """
    Results from head direction cell classification.

    Attributes
    ----------
    is_hd : bool
        Whether the cell is classified as a head direction cell
    mvl_hd : float
        Mean Vector Length for head direction tuning
    preferred_direction : float
        Preferred head direction in radians
    mvl_theta : float or None
        Mean Vector Length for theta phase tuning (if provided)
    tuning_curve : tuple
        Tuple of (bin_centers, firing_rates)
    rayleigh_p : float
        P-value from Rayleigh test for non-uniformity
    """

    is_hd: bool
    mvl_hd: float
    preferred_direction: float
    mvl_theta: float | None
    tuning_curve: tuple[np.ndarray, np.ndarray]
    rayleigh_p: float


class HeadDirectionAnalyzer:
    """
    Analyzer for classifying head direction cells based on directional tuning.

    Head direction cells fire when the animal's head points in a specific direction.
    Classification is based on the strength of directional tuning measured by
    Mean Vector Length (MVL).

    Parameters
    ----------
    mvl_hd_threshold : float, optional
        MVL threshold for head direction. Default is 0.4 (strict).
        Use 0.2 for looser threshold.
    mvl_theta_threshold : float, optional
        MVL threshold for theta phase modulation. Default is 0.3.
    strict_mode : bool, optional
        If True, requires both HD and theta criteria. Default is True.
    n_bins : int, optional
        Number of directional bins for tuning curve. Default is 60 (6° bins).

    Examples
    --------
    >>> analyzer = HeadDirectionAnalyzer(mvl_hd_threshold=0.4, strict_mode=True)
    >>> result = analyzer.classify_hd_cell(spike_times, head_directions, time_stamps)
    >>> print(f"Is HD cell: {result.is_hd}")
    >>> print(f"MVL: {result.mvl_hd:.3f}")
    >>> print(f"Preferred direction: {np.rad2deg(result.preferred_direction):.1f}°")

    Notes
    -----
    Based on MATLAB classification from fig2.m and plotSwsExample.m:
    - Strict: MVL_hd > 0.4 AND MVL_theta > 0.3
    - Loose: MVL_hd > 0.2 AND MVL_theta > 0.3

    References
    ----------
    Classification thresholds follow standard conventions in head direction
    cell literature and the CircStat toolbox.
    """

    def __init__(
        self,
        mvl_hd_threshold: float = 0.4,
        mvl_theta_threshold: float = 0.3,
        strict_mode: bool = True,
        n_bins: int = 60,
    ):
        self.mvl_hd_threshold = mvl_hd_threshold
        self.mvl_theta_threshold = mvl_theta_threshold
        self.strict_mode = strict_mode
        self.n_bins = n_bins

    def classify_hd_cell(
        self,
        spike_times: np.ndarray,
        head_directions: np.ndarray,
        time_stamps: np.ndarray,
        theta_phases: np.ndarray | None = None,
    ) -> HDCellResult:
        """
        Classify a cell as head direction cell based on MVL thresholds.

        Parameters
        ----------
        spike_times : np.ndarray
            Spike times in seconds
        head_directions : np.ndarray
            Head direction at each time point (radians)
        time_stamps : np.ndarray
            Time stamps corresponding to head_directions (seconds)
        theta_phases : np.ndarray, optional
            Theta phase at each time point (radians). If None, theta
            criterion is not checked.

        Returns
        -------
        result : HDCellResult
            Classification result with MVL, preferred direction, and tuning curve

        Examples
        --------
        >>> # Simulate a head direction cell
        >>> time_stamps = np.linspace(0, 100, 10000)
        >>> head_directions = np.linspace(0, 20*np.pi, 10000) % (2*np.pi) - np.pi
        >>> preferred_dir = 0.5
        >>> spike_times = time_stamps[np.abs(head_directions - preferred_dir) < 0.3]
        >>> result = analyzer.classify_hd_cell(spike_times, head_directions, time_stamps)
        """
        # Compute directional tuning curve
        bin_centers, firing_rates, occupancy = self.compute_tuning_curve(
            spike_times, head_directions, time_stamps
        )

        # Compute MVL for head direction
        mvl_hd = self.compute_mvl(bin_centers, weights=firing_rates)

        # Compute preferred direction
        preferred_direction = circ_mean(bin_centers, w=firing_rates)

        # Rayleigh test for non-uniformity
        rayleigh_p = circ_rtest(bin_centers, w=firing_rates)

        # Compute MVL for theta phase if provided
        mvl_theta = None
        if theta_phases is not None:
            # Get theta phases at spike times
            spike_theta = np.interp(spike_times, time_stamps, theta_phases)
            mvl_theta = self.compute_mvl(spike_theta)

        # Classification logic
        is_hd = mvl_hd > self.mvl_hd_threshold

        if self.strict_mode and theta_phases is not None:
            # Strict mode: require both HD and theta criteria
            is_hd = is_hd and (mvl_theta > self.mvl_theta_threshold)
        elif self.strict_mode and theta_phases is None:
            # If strict mode but no theta data, just use HD threshold
            pass

        # Create result
        result = HDCellResult(
            is_hd=is_hd,
            mvl_hd=mvl_hd,
            preferred_direction=preferred_direction,
            mvl_theta=mvl_theta,
            tuning_curve=(bin_centers, firing_rates),
            rayleigh_p=rayleigh_p,
        )

        return result

    def compute_tuning_curve(
        self,
        spike_times: np.ndarray,
        head_directions: np.ndarray,
        time_stamps: np.ndarray,
        n_bins: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute directional tuning curve.

        Parameters
        ----------
        spike_times : np.ndarray
            Spike times in seconds
        head_directions : np.ndarray
            Head direction at each time point (radians)
        time_stamps : np.ndarray
            Time stamps corresponding to head_directions (seconds)
        n_bins : int, optional
            Number of bins. If None, uses self.n_bins.

        Returns
        -------
        bin_centers : np.ndarray
            Center of each directional bin (radians)
        firing_rates : np.ndarray
            Firing rate in each bin (Hz)
        occupancy : np.ndarray
            Time spent in each bin (seconds)

        Examples
        --------
        >>> bins, rates, occ = analyzer.compute_tuning_curve(
        ...     spike_times, head_directions, time_stamps
        ... )
        >>> # Plot polar tuning curve
        >>> import matplotlib.pyplot as plt
        >>> ax = plt.subplot(111, projection='polar')
        >>> ax.plot(bins, rates)
        """
        if n_bins is None:
            n_bins = self.n_bins

        # Define bin edges
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute occupancy (time spent in each bin)
        dt = np.median(np.diff(time_stamps))  # Sampling interval
        hd_binned = np.digitize(head_directions, bin_edges) - 1  # 0-indexed
        hd_binned = np.clip(hd_binned, 0, n_bins - 1)  # Handle edge cases

        occupancy = np.bincount(hd_binned, minlength=n_bins) * dt

        # Count spikes in each bin
        # Interpolate HD at spike times
        spike_hd = np.interp(spike_times, time_stamps, head_directions)
        spike_bins = np.digitize(spike_hd, bin_edges) - 1
        spike_bins = np.clip(spike_bins, 0, n_bins - 1)

        spike_counts = np.bincount(spike_bins, minlength=n_bins)

        # Compute firing rates
        firing_rates = np.zeros(n_bins)
        valid = occupancy > 0
        firing_rates[valid] = spike_counts[valid] / occupancy[valid]

        return bin_centers, firing_rates, occupancy

    def compute_mvl(self, angles: np.ndarray, weights: np.ndarray | None = None) -> float:
        """
        Compute Mean Vector Length (MVL).

        The MVL is a measure of circular variance, ranging from 0 (uniform
        distribution) to 1 (concentrated distribution).

        Parameters
        ----------
        angles : np.ndarray
            Angles in radians
        weights : np.ndarray, optional
            Weights for each angle (e.g., firing rates). If None, uniform weights.

        Returns
        -------
        mvl : float
            Mean vector length

        Examples
        --------
        >>> # Concentrated distribution
        >>> angles = np.random.normal(0, 0.1, 100)
        >>> mvl = analyzer.compute_mvl(angles)
        >>> print(f"MVL: {mvl:.3f}")  # Should be close to 1

        >>> # Uniform distribution
        >>> angles = np.random.uniform(-np.pi, np.pi, 100)
        >>> mvl = analyzer.compute_mvl(angles)
        >>> print(f"MVL: {mvl:.3f}")  # Should be close to 0

        Notes
        -----
        Uses the circ_r function from circular statistics utilities.
        """
        return circ_r(angles, w=weights)


if __name__ == "__main__":
    print("Testing HeadDirectionAnalyzer...")

    # Simulate a head direction cell
    print("\nSimulating head direction cell...")
    time_stamps = np.linspace(0, 100, 10000)  # 100 seconds, 100 Hz sampling
    dt = time_stamps[1] - time_stamps[0]

    # Animal rotates and explores
    angular_velocity = 0.5  # rad/s average
    head_directions = np.cumsum(np.random.randn(len(time_stamps)) * angular_velocity * dt)
    head_directions = np.arctan2(
        np.sin(head_directions), np.cos(head_directions)
    )  # Wrap to [-π, π]

    # Cell fires preferentially at 0.5 radians (~28°)
    preferred_dir = 0.5
    tuning_width = 0.5  # radians (~28° width)

    # Generate spikes based on von Mises tuning
    from scipy.stats import vonmises

    firing_prob = vonmises.pdf(head_directions - preferred_dir, kappa=1 / tuning_width**2)
    firing_prob = firing_prob / firing_prob.max() * 0.1  # Max 10% per bin

    # Poisson spike generation
    spike_mask = np.random.rand(len(time_stamps)) < firing_prob
    spike_times = time_stamps[spike_mask]

    print(f"Generated {len(spike_times)} spikes")
    print(f"Mean firing rate: {len(spike_times) / time_stamps[-1]:.2f} Hz")

    # Classify
    print("\nClassifying cell...")
    analyzer = HeadDirectionAnalyzer(mvl_hd_threshold=0.4, strict_mode=False)
    result = analyzer.classify_hd_cell(spike_times, head_directions, time_stamps)

    print("\nResults:")
    print(f"  Is HD cell: {result.is_hd}")
    print(f"  MVL: {result.mvl_hd:.3f}")
    print(f"  Preferred direction: {np.rad2deg(result.preferred_direction):.1f}°")
    print(f"  True preferred direction: {np.rad2deg(preferred_dir):.1f}°")
    print(f"  Rayleigh test p-value: {result.rayleigh_p:.6f}")

    # Check tuning curve
    bin_centers, firing_rates = result.tuning_curve
    max_rate_idx = np.argmax(firing_rates)
    print(f"  Peak firing rate: {firing_rates[max_rate_idx]:.2f} Hz")
    print(f"  Peak at direction: {np.rad2deg(bin_centers[max_rate_idx]):.1f}°")

    # Test with non-directional cell
    print("\n\nSimulating non-directional cell...")
    # Random spikes (Poisson process, no directional tuning)
    mean_rate = 5  # Hz
    n_spikes = int(mean_rate * time_stamps[-1])
    spike_times_random = np.sort(np.random.uniform(0, time_stamps[-1], n_spikes))

    result_random = analyzer.classify_hd_cell(spike_times_random, head_directions, time_stamps)

    print("Results for non-directional cell:")
    print(f"  Is HD cell: {result_random.is_hd}")
    print(f"  MVL: {result_random.mvl_hd:.3f}")
    print(f"  Rayleigh test p-value: {result_random.rayleigh_p:.3f}")

    print("\nHeadDirectionAnalyzer test completed!")
