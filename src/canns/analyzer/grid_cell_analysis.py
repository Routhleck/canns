"""
Grid Cell Analysis Module

This module provides comprehensive analysis tools for Grid Cell models,
including grid score computation, hexagonal pattern detection, and
grid field parameter extraction.

Functions adapted and optimized from funcs_.py with numpy backend.
"""

import numpy as np
from scipy.ndimage import rotate
from scipy.signal import correlate2d
from scipy.spatial.distance import pdist


def compute_grid_score(
    heatmap: np.ndarray,
    method: str = "rotation",
    smoothing_sigma: float | None = None,
) -> float:
    """
    Compute grid score using the standard six-fold rotational symmetry method.

    Grid score quantifies how well a firing rate map exhibits the characteristic
    hexagonal lattice pattern of grid cells.

    Args:
        heatmap (np.ndarray): 2D firing rate map
        method (str): Method for grid score computation ("rotation" or "correlation")
        smoothing_sigma (float, optional): Gaussian smoothing parameter

    Returns:
        float: Grid score (higher values indicate stronger grid patterns)
    """
    if smoothing_sigma is not None:
        from scipy.ndimage import gaussian_filter

        heatmap = gaussian_filter(heatmap, smoothing_sigma)

    if method == "rotation":
        return _compute_grid_score_rotation(heatmap)
    elif method == "correlation":
        return _compute_grid_score_correlation(heatmap)
    else:
        raise ValueError("Method must be 'rotation' or 'correlation'")


def _compute_grid_score_rotation(heatmap: np.ndarray) -> float:
    """
    Compute grid score using rotation-based method.

    The grid score is computed as:
    score = min(corr_60, corr_120) - max(corr_30, corr_90, corr_150)

    Args:
        heatmap (np.ndarray): 2D firing rate map

    Returns:
        float: Grid score
    """
    center_x, center_y = np.array(heatmap.shape) // 2

    # Create circular mask to focus on central region
    y, x = np.ogrid[: heatmap.shape[0], : heatmap.shape[1]]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(heatmap.shape) // 3) ** 2

    masked_map = heatmap * mask

    # Compute correlations at different rotation angles
    angles = [30, 60, 90, 120, 150]
    correlations = []

    for angle in angles:
        rotated = rotate(masked_map, angle, reshape=False)

        # Compute correlation coefficient between original and rotated
        original_flat = masked_map[mask].flatten()
        rotated_flat = rotated[mask].flatten()

        if len(original_flat) > 0 and np.std(original_flat) > 0 and np.std(rotated_flat) > 0:
            corr = np.corrcoef(original_flat, rotated_flat)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0.0)

    # Grid score calculation
    corr_60 = correlations[1]  # 60 degrees
    corr_120 = correlations[3]  # 120 degrees
    corr_30 = correlations[0]  # 30 degrees
    corr_90 = correlations[2]  # 90 degrees
    corr_150 = correlations[4]  # 150 degrees

    grid_score = min(corr_60, corr_120) - max(corr_30, corr_90, corr_150)

    return grid_score


def _compute_grid_score_correlation(heatmap: np.ndarray) -> float:
    """
    Alternative grid score computation using spatial autocorrelation.

    Args:
        heatmap (np.ndarray): 2D firing rate map

    Returns:
        float: Grid score based on autocorrelation
    """
    # Compute 2D autocorrelation
    autocorr = correlate2d(heatmap, heatmap, mode="same")
    autocorr = autocorr / np.max(autocorr)

    # Extract center region
    # This is a simplified version - full implementation would be more complex
    return 0.0  # Placeholder


def detect_grid_fields(
    heatmap: np.ndarray,
    min_field_size: float = 0.1,
    threshold_factor: float = 0.3,
) -> tuple[np.ndarray, int]:
    """
    Detect individual grid fields in a firing rate map.

    Args:
        heatmap (np.ndarray): 2D firing rate map
        min_field_size (float): Minimum field size as fraction of map
        threshold_factor (float): Threshold as fraction of peak rate

    Returns:
        Tuple[np.ndarray, int]: (field_centers, num_fields)
            - field_centers: Shape (num_fields, 2) - [x, y] coordinates
            - num_fields: Number of detected fields
    """
    # Threshold the map
    threshold = threshold_factor * np.max(heatmap)
    binary_map = heatmap > threshold

    # Label connected components
    from scipy.ndimage import center_of_mass, label

    labeled_map, num_fields = label(binary_map)

    # Extract field centers
    field_centers = []
    for i in range(1, num_fields + 1):
        # Check field size
        field_size = np.sum(labeled_map == i) / labeled_map.size

        if field_size >= min_field_size:
            # Compute center of mass
            center = center_of_mass(heatmap, labeled_map, i)
            field_centers.append([center[1], center[0]])  # [x, y] format

    field_centers = np.array(field_centers) if field_centers else np.zeros((0, 2))

    return field_centers, len(field_centers)


def compute_grid_spacing(
    field_centers: np.ndarray,
    method: str = "nearest_neighbor",
) -> float:
    """
    Compute grid spacing from detected field centers.

    Args:
        field_centers (np.ndarray): Shape (N, 2) - field centers
        method (str): Method for spacing computation

    Returns:
        float: Estimated grid spacing
    """
    if len(field_centers) < 2:
        return 0.0

    if method == "nearest_neighbor":
        # Compute pairwise distances
        distances = pdist(field_centers)

        # Grid spacing is typically the most common nearest neighbor distance
        # Use histogram to find the most frequent distance
        hist, bins = np.histogram(distances, bins=20)
        max_bin_idx = np.argmax(hist)
        spacing = (bins[max_bin_idx] + bins[max_bin_idx + 1]) / 2

        return spacing

    elif method == "fft":
        # Use FFT-based method (simplified)
        if len(field_centers) >= 6:
            # Compute distances to 6 nearest neighbors (hexagonal)
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=7).fit(field_centers)  # 7 to include self
            distances, indices = nbrs.kneighbors(field_centers)

            # Average of nearest neighbor distances (excluding self)
            nearest_distances = distances[:, 1]  # First neighbor (excluding self)
            spacing = np.mean(nearest_distances)

            return spacing

    return 0.0


def compute_grid_orientation(
    field_centers: np.ndarray,
    reference_center: np.ndarray | None = None,
) -> float:
    """
    Compute grid orientation angle from field centers.

    Args:
        field_centers (np.ndarray): Shape (N, 2) - field centers
        reference_center (np.ndarray, optional): Reference point [x, y]

    Returns:
        float: Grid orientation in radians
    """
    if len(field_centers) < 3:
        return 0.0

    if reference_center is None:
        reference_center = np.mean(field_centers, axis=0)

    # Compute vectors from reference to each field
    vectors = field_centers - reference_center

    # Compute angles
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # For hexagonal lattice, we expect angles at multiples of 60 degrees
    # Find the dominant orientation
    # Simplified: return the angle of the first field relative to reference
    if len(angles) > 0:
        return angles[0] % (np.pi / 3)  # Mod 60 degrees

    return 0.0


def extract_grid_parameters(
    heatmap: np.ndarray,
    width: float = 1.0,
    height: float = 1.0,
) -> dict[str, float | np.ndarray | int]:
    """
    Extract comprehensive grid cell parameters from firing rate map.

    Args:
        heatmap (np.ndarray): 2D firing rate map
        width (float): Physical width of the environment
        height (float): Physical height of the environment

    Returns:
        Dict: Dictionary containing grid parameters
            - grid_score: Grid score value
            - spacing: Grid spacing in physical units
            - orientation: Grid orientation in radians
            - field_centers: Detected field centers
            - num_fields: Number of fields
            - regularity: Grid regularity measure
    """
    # Basic grid analysis
    grid_score = compute_grid_score(heatmap)
    field_centers, num_fields = detect_grid_fields(heatmap)

    # Convert field positions to physical coordinates
    if len(field_centers) > 0:
        physical_centers = field_centers.copy()
        physical_centers[:, 0] *= width / heatmap.shape[1]  # x-coordinates
        physical_centers[:, 1] *= height / heatmap.shape[0]  # y-coordinates

        spacing = compute_grid_spacing(physical_centers)
        orientation = compute_grid_orientation(physical_centers)
        regularity = _compute_grid_regularity(physical_centers)
    else:
        physical_centers = np.zeros((0, 2))
        spacing = 0.0
        orientation = 0.0
        regularity = 0.0

    return {
        "grid_score": grid_score,
        "spacing": spacing,
        "orientation": orientation,
        "field_centers": physical_centers,
        "num_fields": num_fields,
        "regularity": regularity,
        "peak_rate": np.max(heatmap),
        "mean_rate": np.mean(heatmap),
    }


def _compute_grid_regularity(field_centers: np.ndarray) -> float:
    """
    Compute grid regularity measure based on field spacing consistency.

    Args:
        field_centers (np.ndarray): Shape (N, 2) - field centers

    Returns:
        float: Regularity measure (1.0 = perfectly regular, 0.0 = irregular)
    """
    if len(field_centers) < 3:
        return 0.0

    # Compute all pairwise distances
    distances = pdist(field_centers)

    if len(distances) == 0:
        return 0.0

    # Regularity is inversely related to coefficient of variation
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    if mean_dist > 0:
        cv = std_dist / mean_dist
        regularity = np.exp(-cv)  # Convert to [0,1] scale
        return regularity

    return 0.0


def analyze_grid_population(
    activity: np.ndarray,
    positions: np.ndarray,
    grid_indices: np.ndarray,
    width: float = 1.0,
    height: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Analyze population-level grid cell properties.

    Args:
        activity (np.ndarray): Shape (T, N) - neural activity
        positions (np.ndarray): Shape (T, 2) - positions
        grid_indices (np.ndarray): Indices of grid cells
        width (float): Environment width
        height (float): Environment height

    Returns:
        Dict: Population analysis results
    """
    from .place_cell_analysis import compute_firing_field

    heatmaps = compute_firing_field(activity, positions, width, height)

    # Analyze each grid cell
    num_grid_cells = len(grid_indices)
    population_results = {
        "grid_scores": np.zeros(num_grid_cells),
        "spacings": np.zeros(num_grid_cells),
        "orientations": np.zeros(num_grid_cells),
        "num_fields": np.zeros(num_grid_cells, dtype=int),
        "regularities": np.zeros(num_grid_cells),
        "peak_rates": np.zeros(num_grid_cells),
    }

    field_centers_list = []

    for i, neuron_idx in enumerate(grid_indices):
        heatmap = heatmaps[:, :, neuron_idx]
        params = extract_grid_parameters(heatmap, width, height)

        population_results["grid_scores"][i] = params["grid_score"]
        population_results["spacings"][i] = params["spacing"]
        population_results["orientations"][i] = params["orientation"]
        population_results["num_fields"][i] = params["num_fields"]
        population_results["regularities"][i] = params["regularity"]
        population_results["peak_rates"][i] = params["peak_rate"]

        field_centers_list.append(params["field_centers"])

    population_results["field_centers"] = field_centers_list
    population_results["heatmaps"] = heatmaps[:, :, grid_indices]

    return population_results


def select_grid_cells(
    activity: np.ndarray,
    positions: np.ndarray,
    grid_score_threshold: float = 0.3,
    min_fields_threshold: int = 3,
    width: float = 1.0,
    height: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Select grid cells based on grid score and field number criteria.

    Args:
        activity (np.ndarray): Shape (T, N) - neural activity
        positions (np.ndarray): Shape (T, 2) - positions
        grid_score_threshold (float): Minimum grid score
        min_fields_threshold (int): Minimum number of fields
        width (float): Environment width
        height (float): Environment height

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: (grid_scores, grid_indices, num_grid_cells)
    """
    from .place_cell_analysis import compute_firing_field

    heatmaps = compute_firing_field(activity, positions, width, height)
    T, N = activity.shape

    grid_scores = np.zeros(N)
    num_fields = np.zeros(N, dtype=int)

    # Compute grid scores for all neurons
    for neuron in range(N):
        heatmap = heatmaps[:, :, neuron]
        grid_scores[neuron] = compute_grid_score(heatmap)
        _, num_fields[neuron] = detect_grid_fields(heatmap)

    # Apply selection criteria
    grid_criteria = (grid_scores >= grid_score_threshold) & (num_fields >= min_fields_threshold)

    grid_indices = np.where(grid_criteria)[0]
    num_grid_cells = len(grid_indices)

    return grid_scores, grid_indices, num_grid_cells


def compute_grid_cell_phase_relationships(
    activity: np.ndarray,
    positions: np.ndarray,
    grid_indices: np.ndarray,
    width: float = 1.0,
    height: float = 1.0,
) -> np.ndarray:
    """
    Compute phase relationships between grid cells.

    Args:
        activity (np.ndarray): Shape (T, N) - neural activity
        positions (np.ndarray): Shape (T, 2) - positions
        grid_indices (np.ndarray): Grid cell indices
        width (float): Environment width
        height (float): Environment height

    Returns:
        np.ndarray: Shape (num_grid_cells, num_grid_cells) - phase relationship matrix
    """
    from .place_cell_analysis import compute_firing_field

    heatmaps = compute_firing_field(activity, positions, width, height)
    num_grid_cells = len(grid_indices)
    phase_matrix = np.zeros((num_grid_cells, num_grid_cells))

    for i in range(num_grid_cells):
        for j in range(num_grid_cells):
            if i != j:
                heatmap_i = heatmaps[:, :, grid_indices[i]]
                heatmap_j = heatmaps[:, :, grid_indices[j]]

                # Compute spatial cross-correlation
                xcorr = correlate2d(heatmap_i, heatmap_j, mode="same")

                # Find peak correlation location relative to center
                center = np.array(xcorr.shape) // 2
                peak_loc = np.unravel_index(np.argmax(xcorr), xcorr.shape)

                # Compute phase offset (simplified)
                phase_offset = np.linalg.norm(np.array(peak_loc) - center)
                phase_matrix[i, j] = phase_offset

    return phase_matrix
