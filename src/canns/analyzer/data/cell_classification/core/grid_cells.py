"""
Grid Cell Classification

Implementation of gridness score algorithm for identifying and characterizing grid cells.

Based on the MATLAB gridnessScore.m implementation.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from ..utils.circular_stats import circ_dist2
from ..utils.correlation import autocorrelation_2d, pearson_correlation
from ..utils.geometry import fit_ellipse, polyarea, squared_distance, wrap_to_pi
from ..utils.image_processing import (
    dilate_image,
    find_contours_at_level,
    find_regional_maxima,
    gaussian_filter_2d,
    label_connected_components,
    regionprops,
    rotate_image,
)


@dataclass
class GridnessResult:
    """
    Results from gridness score computation.

    Attributes
    ----------
    score : float
        Gridness score (range -2 to 2, typical grid cells: 0.3-1.3)
    spacing : np.ndarray
        Array of 3 grid field spacings (distances from center)
    orientation : np.ndarray
        Array of 3 grid field orientations (angles in degrees)
    ellipse : np.ndarray
        Fitted ellipse parameters [cx, cy, rx, ry, theta]
    ellipse_theta_deg : float
        Ellipse orientation in degrees [0, 180]
    center_radius : float
        Radius of the central autocorrelation field
    optimal_radius : float
        Radius at which gridness score is maximized
    peak_locations : np.ndarray
        Coordinates of detected grid peaks (N x 2 array)
    """

    score: float
    spacing: np.ndarray
    orientation: np.ndarray
    ellipse: np.ndarray
    ellipse_theta_deg: float
    center_radius: float
    optimal_radius: float
    peak_locations: np.ndarray | None = None


class GridnessAnalyzer:
    """
    Analyzer for computing gridness scores from spatial autocorrelograms.

    This implements the rotation-correlation method for quantifying hexagonal
    grid patterns in neural firing rate maps.

    Parameters
    ----------
    threshold : float, optional
        Normalized threshold for contour detection (0-1). Default is 0.2.
    min_orientation : float, optional
        Minimum angular difference between fields (degrees). Default is 15.
    min_center_radius : int, optional
        Minimum center field radius in pixels. Default is 2.
    num_gridness_radii : int, optional
        Number of adjacent radii to average for gridness score. Default is 3.

    Examples
    --------
    >>> analyzer = GridnessAnalyzer()
    >>> # Assume we have a 2D rate map
    >>> autocorr = compute_2d_autocorrelation(rate_map)
    >>> result = analyzer.compute_gridness_score(autocorr)
    >>> print(f"Gridness score: {result.score:.3f}")
    >>> print(f"Grid spacing: {result.spacing}")

    Notes
    -----
    Based on gridnessScore.m from the MATLAB codebase.

    References
    ----------
    The gridness score algorithm computes correlations between the autocorrelogram
    and rotated versions at 30°, 60°, 90°, 120°, and 150°. The score is:
    min(r_60°, r_120°) - max(r_30°, r_90°, r_150°)

    This exploits the 60° rotational symmetry of hexagonal grids.
    """

    def __init__(
        self,
        threshold: float = 0.2,
        min_orientation: float = 15.0,
        min_center_radius: int = 2,
        num_gridness_radii: int = 3,
    ):
        self.threshold = threshold
        self.min_orientation = min_orientation
        self.min_center_radius = min_center_radius
        self.num_gridness_radii = num_gridness_radii

    def compute_gridness_score(self, autocorr: np.ndarray) -> GridnessResult:
        """
        Compute gridness score from a 2D autocorrelogram.

        Parameters
        ----------
        autocorr : np.ndarray
            2D autocorrelogram of a firing rate map

        Returns
        -------
        result : GridnessResult
            Complete gridness analysis results

        Raises
        ------
        ValueError
            If autocorr is not 2D or if center field cannot be detected
        """
        if autocorr.ndim != 1 and (autocorr.shape[0] == 1 or autocorr.shape[1] == 1):
            # Degenerate case: 1D array
            return self._create_nan_result()

        # Normalize autocorrelogram to [0, 1]
        autocorr = autocorr / np.max(autocorr)

        # Find central field radius using contour detection
        center_radius = self._find_center_radius(autocorr, self.threshold)

        if center_radius < self.min_center_radius:
            return self._create_nan_result()

        # Get autocorr dimensions
        half_height, half_width = np.array(autocorr.shape) // 2 + 1
        autocorr_rad = min(half_height, half_width)

        if center_radius >= autocorr_rad:
            return self._create_nan_result()

        # Compute rotation correlations and gridness score
        gridness_scores, rad_steps = self._compute_rotation_correlations(
            autocorr, center_radius, autocorr_rad, half_height, half_width
        )

        # Find optimal radius with maximum gridness
        score, optimal_radius = self._find_optimal_gridness(gridness_scores, rad_steps)

        # Extract grid statistics (spacing, orientation, ellipse fit)
        grid_stats = self._extract_grid_statistics(
            autocorr, optimal_radius, half_height, half_width
        )

        # Create result object
        result = GridnessResult(
            score=score,
            spacing=grid_stats["spacing"],
            orientation=grid_stats["orientation"],
            ellipse=grid_stats["ellipse"],
            ellipse_theta_deg=grid_stats["ellipse_theta_deg"],
            center_radius=center_radius,
            optimal_radius=optimal_radius,
            peak_locations=grid_stats.get("peak_locations"),
        )

        return result

    def _find_center_radius(self, autocorr: np.ndarray, threshold: float) -> float:
        """
        Find the radius of the central autocorrelation field.

        Uses contour detection at the specified threshold level.

        Parameters
        ----------
        autocorr : np.ndarray
            Normalized autocorrelogram
        threshold : float
            Contour detection threshold

        Returns
        -------
        radius : float
            Radius of central field in pixels
        """
        half_height, half_width = np.array(autocorr.shape) // 2 + 1

        # Find contours at threshold level
        contours = find_contours_at_level(autocorr, threshold)

        if len(contours) == 0:
            return -1

        # Find contour closest to center
        # Note: find_contours returns (row, col) = (y, x)
        center_point = np.array([half_height - 1, half_width - 1])  # -1 for 0-indexing

        min_dist = np.inf
        center_contour = None

        for contour in contours:
            # Compute mean position of contour
            mean_pos = np.mean(contour, axis=0)
            dist = np.linalg.norm(mean_pos - center_point)

            if dist < min_dist:
                min_dist = dist
                center_contour = contour

        if center_contour is None:
            return -1

        # Compute area of central field contour
        # Note: contour is (row, col) = (y, x), need to swap for polyarea
        area = polyarea(center_contour[:, 1], center_contour[:, 0])

        # Radius from area: r = sqrt(area / pi)
        radius = np.floor(np.sqrt(area / np.pi))

        return float(radius)

    def _compute_rotation_correlations(
        self,
        autocorr: np.ndarray,
        center_radius: float,
        autocorr_rad: float,
        half_height: int,
        half_width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute correlations between autocorr and rotated versions.

        Rotates at 30°, 60°, 90°, 120°, 150° and computes Pearson correlation
        for expanding circular regions.

        Parameters
        ----------
        autocorr : np.ndarray
            Normalized autocorrelogram
        center_radius : float
            Radius of central field to exclude
        autocorr_rad : float
            Radius of autocorrelogram
        half_height : int
            Half height of autocorr
        half_width : int
            Half width of autocorr

        Returns
        -------
        gridness_scores : np.ndarray
            Gridness scores at each radius
        rad_steps : np.ndarray
            Radius values tested
        """
        # Define rotation angles
        rot_angles_deg = np.array([30, 60, 90, 120, 150])
        n_rot = len(rot_angles_deg)

        # Create distance mask from center
        rr, cc = np.meshgrid(np.arange(autocorr.shape[1]), np.arange(autocorr.shape[0]))
        dist_from_center = np.sqrt((cc - (half_height - 1)) ** 2 + (rr - (half_width - 1)) ** 2)

        # Mask for excluding central field
        center_exclusion_mask = dist_from_center > center_radius

        # Define radius steps
        rad_steps = np.arange(center_radius + 1, autocorr_rad + 1)
        num_steps = len(rad_steps)

        if num_steps == 0:
            return np.array([np.nan]), np.array([center_radius])

        # Pre-compute all rotated versions
        rotated_autocorr = np.zeros((autocorr.shape[0], autocorr.shape[1], n_rot))
        for i, angle in enumerate(rot_angles_deg):
            rotated_autocorr[:, :, i] = rotate_image(
                autocorr, angle, output_shape=autocorr.shape, method="bilinear"
            )

        # Vectorize autocorr and rotated versions
        autocorr_vec = autocorr[center_exclusion_mask].ravel()
        rotated_vec = rotated_autocorr[center_exclusion_mask, :].reshape(-1, n_rot)
        dist_vec = dist_from_center[center_exclusion_mask].ravel()

        # Compute correlations at each radius
        gridness_scores = np.zeros((num_steps, 2))
        gridness_scores[:, 1] = rad_steps

        for i, radius in enumerate(rad_steps):
            # Select points within this radius
            in_radius = dist_vec < radius

            if np.sum(in_radius) < 10:  # Need minimum points
                gridness_scores[i, 0] = np.nan
                continue

            ref_circle = autocorr_vec[in_radius]
            rot_circles = rotated_vec[in_radius, :]

            # Compute Pearson correlations
            rot_corr = pearson_correlation(ref_circle[:, np.newaxis], rot_circles)

            # Gridness score: min(r_60, r_120) - max(r_30, r_90, r_150)
            # rot_corr indices: [30°, 60°, 90°, 120°, 150°] = [0, 1, 2, 3, 4]
            score = min(rot_corr[1], rot_corr[3]) - max(rot_corr[0], rot_corr[2], rot_corr[4])
            gridness_scores[i, 0] = score

        return gridness_scores, rad_steps

    def _find_optimal_gridness(
        self, gridness_scores: np.ndarray, rad_steps: np.ndarray
    ) -> tuple[float, float]:
        """
        Find the radius with maximum gridness score.

        Averages over num_gridness_radii adjacent radii for stability.

        Parameters
        ----------
        gridness_scores : np.ndarray
            Gridness scores at each radius (N x 2 array)
        rad_steps : np.ndarray
            Radius values

        Returns
        -------
        max_score : float
            Maximum gridness score
        optimal_radius : float
            Radius at maximum score
        """
        scores = gridness_scores[:, 0]
        num_steps = len(scores)

        if num_steps < self.num_gridness_radii:
            # Not enough radii, just take max
            valid_scores = scores[~np.isnan(scores)]
            if len(valid_scores) == 0:
                return np.nan, rad_steps[0] if len(rad_steps) > 0 else 0

            max_idx = np.nanargmax(scores)
            return scores[max_idx], rad_steps[max_idx]

        # Average over adjacent radii
        num_windows = num_steps - self.num_gridness_radii + 1
        mean_scores = np.zeros(num_windows)

        for i in range(num_windows):
            window = scores[i : i + self.num_gridness_radii]
            mean_scores[i] = np.nanmean(window)

        # Find maximum
        max_idx = np.nanargmax(mean_scores)
        max_score = mean_scores[max_idx]

        # Optimal radius is at center of window
        optimal_idx = max_idx + (self.num_gridness_radii - 1) // 2
        optimal_radius = rad_steps[optimal_idx]

        return float(max_score), float(optimal_radius)

    def _extract_grid_statistics(
        self, autocorr: np.ndarray, optimal_radius: float, half_height: int, half_width: int
    ) -> dict:
        """
        Extract grid field statistics (spacing, orientation, ellipse).

        Finds peaks in the autocorrelogram and fits an ellipse to them.

        Parameters
        ----------
        autocorr : np.ndarray
            Autocorrelogram
        optimal_radius : float
            Optimal radius for analysis
        half_height : int
            Half height of autocorr
        half_width : int
            Half width of autocorr

        Returns
        -------
        stats : dict
            Dictionary with spacing, orientation, ellipse parameters
        """
        # Create distance mask
        rr, cc = np.meshgrid(np.arange(autocorr.shape[1]), np.arange(autocorr.shape[0]))
        dist_from_center = np.sqrt((cc - (half_height - 1)) ** 2 + (rr - (half_width - 1)) ** 2)

        # Define search window around optimal radius
        w = optimal_radius / 4
        mask_outer = dist_from_center < (optimal_radius + w)

        # Smooth autocorr to eliminate spurious maxima
        autocorr_sm = gaussian_filter_2d(
            autocorr, sigma=optimal_radius / (2 * np.pi) / 2, mode="reflect"
        )

        # Apply mask
        masked_autocorr = mask_outer * autocorr_sm

        # Find regional maxima
        maxima_map = find_regional_maxima(masked_autocorr, connectivity=1)

        # Dilate to eliminate fragmentation
        maxima_map_dilated = dilate_image(maxima_map, selem_type="square", selem_size=3)

        # Label connected components
        labels, num_labels = label_connected_components(maxima_map_dilated, connectivity=2)

        if num_labels < 5:
            warnings.warn("Not enough grid peaks found for statistics", stacklevel=2)
            return self._create_nan_stats()

        # Get region properties
        props = regionprops(labels)

        # Extract centroids
        centroids = np.array([prop.centroid for prop in props])  # (N, 2) array of (row, col)

        # Convert to (x, y) coordinates
        centers_of_mass = centroids[:, ::-1]  # Swap to (col, row) = (x, y)

        # Compute orientations relative to center
        center_point = np.array([half_width - 1, half_height - 1])
        orientations = np.arctan2(
            centers_of_mass[:, 1] - center_point[1], centers_of_mass[:, 0] - center_point[0]
        )

        # Compute distances to center
        peaks_to_center = squared_distance(centers_of_mass.T, center_point[:, np.newaxis]).ravel()

        # Remove zero orientation (central peak if any)
        zero_idx = np.where(orientations == 0)[0]
        if len(zero_idx) > 0:
            mask = np.ones(len(orientations), dtype=bool)
            mask[zero_idx] = False
            orientations = orientations[mask]
            centers_of_mass = centers_of_mass[mask]
            peaks_to_center = peaks_to_center[mask]

        # Filter fields with similar orientations
        orient_dist_sq = circ_dist2(orientations)
        close_fields = np.abs(orient_dist_sq) < np.deg2rad(self.min_orientation)
        np.fill_diagonal(close_fields, False)
        close_fields = np.triu(close_fields)  # Keep upper triangle only

        rows, cols = np.where(close_fields)
        to_delete = []
        for row, col in zip(rows, cols, strict=True):
            # Keep the one closer to center
            if peaks_to_center[row] > peaks_to_center[col]:
                to_delete.append(row)
            else:
                to_delete.append(col)

        to_delete = np.unique(to_delete)
        if len(to_delete) > 0:
            mask = np.ones(len(orientations), dtype=bool)
            mask[to_delete] = False
            orientations = orientations[mask]
            centers_of_mass = centers_of_mass[mask]
            peaks_to_center = peaks_to_center[mask]

        if len(centers_of_mass) < 4:
            warnings.warn("Not enough grid peaks after filtering", stacklevel=2)
            return self._create_nan_stats()

        # Sort by distance to center and keep 6 closest
        sort_idx = np.argsort(peaks_to_center)
        centers_of_mass = centers_of_mass[sort_idx]
        if len(centers_of_mass) > 6:
            centers_of_mass = centers_of_mass[:6]

        # Compute final orientations and spacings
        orientations_deg = np.rad2deg(
            np.arctan2(
                centers_of_mass[:, 1] - center_point[1], centers_of_mass[:, 0] - center_point[0]
            )
        )

        spacings = np.sqrt(
            (centers_of_mass[:, 0] - center_point[0]) ** 2
            + (centers_of_mass[:, 1] - center_point[1]) ** 2
        )

        # Fit ellipse to peaks
        try:
            ellipse = fit_ellipse(centers_of_mass[:, 0], centers_of_mass[:, 1])
            ellipse_theta_deg = np.rad2deg(wrap_to_pi(ellipse[4]) + np.pi)
        except Exception:
            ellipse = np.full(5, np.nan)
            ellipse_theta_deg = np.nan

        # Select 3 orientations with smallest absolute values (closest to main axes)
        abs_orient = np.abs(orientations_deg)
        orient_sort_idx = np.argsort(abs_orient)
        orient_sort_idx2 = np.argsort(
            np.abs(orientations_deg - orientations_deg[orient_sort_idx[0]])
        )

        final_idx = orient_sort_idx2[:3]
        orientations_deg = orientations_deg[final_idx]
        spacings = spacings[final_idx]

        # Sort by orientation
        sort_idx = np.argsort(orientations_deg)
        orientations_deg = orientations_deg[sort_idx]
        spacings = spacings[sort_idx]

        return {
            "spacing": spacings,
            "orientation": orientations_deg,
            "ellipse": ellipse,
            "ellipse_theta_deg": ellipse_theta_deg,
            "peak_locations": centers_of_mass,
        }

    def _create_nan_result(self) -> GridnessResult:
        """Create a result with NaN values for failed analysis."""
        return GridnessResult(
            score=np.nan,
            spacing=np.full(3, np.nan),
            orientation=np.full(3, np.nan),
            ellipse=np.full(5, np.nan),
            ellipse_theta_deg=np.nan,
            center_radius=0,
            optimal_radius=np.nan,
        )

    def _create_nan_stats(self) -> dict:
        """Create statistics dict with NaN values."""
        return {
            "spacing": np.full(3, np.nan),
            "orientation": np.full(3, np.nan),
            "ellipse": np.full(5, np.nan),
            "ellipse_theta_deg": np.nan,
        }


def compute_2d_autocorrelation(rate_map: np.ndarray, overlap: float = 0.8) -> np.ndarray:
    """
    Compute 2D spatial autocorrelation of a firing rate map.

    This is a convenience wrapper around the autocorrelation function
    from the correlation module.

    Parameters
    ----------
    rate_map : np.ndarray
        2D firing rate map
    overlap : float, optional
        Overlap percentage (0-1). Default is 0.8.

    Returns
    -------
    autocorr : np.ndarray
        2D autocorrelogram

    Examples
    --------
    >>> rate_map = np.random.rand(50, 50)
    >>> autocorr = compute_2d_autocorrelation(rate_map)
    >>> print(autocorr.shape)

    Notes
    -----
    Based on autocorrelation.m from the MATLAB codebase.
    """
    return autocorrelation_2d(rate_map, overlap=overlap, normalize=True)


if __name__ == "__main__":
    print("Testing GridnessAnalyzer...")

    # Create a synthetic grid-like pattern
    print("\nCreating synthetic grid pattern...")
    x = np.linspace(-2, 2, 100)
    xx, yy = np.meshgrid(x, x)

    # Hexagonal grid pattern (sum of 3 cosines at 60° angles)
    theta1, theta2, theta3 = 0, np.pi / 3, 2 * np.pi / 3
    k = 2 * np.pi / 0.4  # Spatial frequency

    grid_pattern = (
        np.cos(k * (xx * np.cos(theta1) + yy * np.sin(theta1)))
        + np.cos(k * (xx * np.cos(theta2) + yy * np.sin(theta2)))
        + np.cos(k * (xx * np.cos(theta3) + yy * np.sin(theta3)))
    ) / 3

    # Make it a rate map (positive values)
    rate_map = (grid_pattern + 1.5) / 2.5 * 10  # Scale to 0-10 Hz range

    print(f"Rate map shape: {rate_map.shape}")
    print(f"Rate map range: [{rate_map.min():.2f}, {rate_map.max():.2f}] Hz")

    # Compute autocorrelation
    print("\nComputing autocorrelation...")
    autocorr = compute_2d_autocorrelation(rate_map)
    print(f"Autocorr shape: {autocorr.shape}")

    # Compute gridness score
    print("\nComputing gridness score...")
    analyzer = GridnessAnalyzer()
    result = analyzer.compute_gridness_score(autocorr)

    print("\nResults:")
    print(f"  Gridness score: {result.score:.3f}")
    print(f"  Grid spacing: {result.spacing}")
    print(f"  Grid orientation: {result.orientation}°")
    print(f"  Center radius: {result.center_radius}")
    print(f"  Optimal radius: {result.optimal_radius}")

    if not np.isnan(result.ellipse).any():
        print(f"  Ellipse center: ({result.ellipse[0]:.1f}, {result.ellipse[1]:.1f})")
        print(f"  Ellipse radii: ({result.ellipse[2]:.1f}, {result.ellipse[3]:.1f})")
        print(f"  Ellipse angle: {result.ellipse_theta_deg:.1f}°")

    print("\nGridnessAnalyzer test completed!")
