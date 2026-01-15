"""Base navigation task with geodesic distance computation capabilities."""

import copy
import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import brainpy.math as bm
import numpy as np
from canns_lib.spatial import Agent, Environment
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.path import Path
from tqdm import tqdm

from ._base import Task

# Try to import numba for JIT acceleration
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback: njit and prange become no-op
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    # prange fallback to regular range
    prange = range

__all__ = [
    "MovementCostGrid",
    "GeodesicDistanceResult",
    "BaseNavigationTask",
]

INT32_MAX = np.iinfo(np.int32).max
"""int: Maximum value for 32-bit signed integer.

Used as a sentinel value to mark impassable/blocked cells in movement cost grids.
In navigation tasks, cells with cost equal to INT32_MAX are treated as obstacles
that cannot be traversed.

Example
-------
Mark obstacles in a cost grid:

>>> import numpy as np
>>> from canns.task.navigation_base import INT32_MAX
>>> 
>>> # Create cost grid (1 = passable, INT32_MAX = obstacle)
>>> cost_grid = np.ones((10, 10), dtype=np.int32)
>>> 
>>> # Add obstacles (walls)
>>> cost_grid[5, :] = INT32_MAX  # Horizontal wall
>>> cost_grid[:, 5] = INT32_MAX  # Vertical wall
>>> 
>>> # Check if cell is passable
>>> def is_passable(cost):
...     return cost < INT32_MAX
>>> 
>>> print(f"Cell (3, 3) passable: {is_passable(cost_grid[3, 3])}")
>>> print(f"Cell (5, 5) passable: {is_passable(cost_grid[5, 5])}")

See Also
--------
MovementCostGrid : Dataclass holding cost grid and spatial information
BaseNavigationTask : Navigation task using cost grids for path planning
"""
EPSILON = 1e-12


# ============================================================================
# Dijkstra's Algorithm Implementations
# ============================================================================


def _dijkstra_python(
    costs: np.ndarray, dx: float, dy: float, start_linear_index: int
) -> np.ndarray:
    """Pure Python implementation of Dijkstra's algorithm on grid (fallback)."""
    rows, cols = costs.shape
    total_cells = rows * cols
    distances = np.full(total_cells, np.inf, dtype=np.float64)
    distances[start_linear_index] = 0.0

    heap: list[tuple[float, int]] = [(0.0, start_linear_index)]

    while heap:
        current_dist, idx = heapq.heappop(heap)
        if current_dist > distances[idx]:
            continue

        row, col = divmod(idx, cols)
        if costs[row, col] != 1:
            continue

        # Check 4 neighbors (up, down, left, right)
        neighbours = []
        if row > 0 and costs[row - 1, col] == 1:
            neighbours.append(((row - 1) * cols + col, dy))
        if row < rows - 1 and costs[row + 1, col] == 1:
            neighbours.append(((row + 1) * cols + col, dy))
        if col > 0 and costs[row, col - 1] == 1:
            neighbours.append((row * cols + (col - 1), dx))
        if col < cols - 1 and costs[row, col + 1] == 1:
            neighbours.append((row * cols + (col + 1), dx))

        for neighbour_idx, step_cost in neighbours:
            new_dist = current_dist + step_cost
            if new_dist < distances[neighbour_idx]:
                distances[neighbour_idx] = new_dist
                heapq.heappush(heap, (new_dist, neighbour_idx))

    return distances


@njit
def _dijkstra_numba(costs: np.ndarray, dx: float, dy: float, start_linear_index: int) -> np.ndarray:
    """Numba-optimized Dijkstra's algorithm using simple array-based priority queue.

    This implementation uses a simplified priority queue suitable for Numba JIT compilation.
    The speedup comes from:
    1. JIT compilation to machine code
    2. Removing Python overhead in the hot loop
    3. Direct memory access without Python object overhead
    """
    rows, cols = costs.shape
    total_cells = rows * cols
    distances = np.full(total_cells, np.inf, dtype=np.float64)
    distances[start_linear_index] = 0.0

    # Simple priority queue implementation using unsorted array
    # For each node: (distance, node_index)
    # Max queue size is total_cells
    queue_dist = np.full(total_cells, np.inf, dtype=np.float64)
    queue_node = np.full(total_cells, -1, dtype=np.int32)
    queue_size = 0

    # Initialize queue with start node
    queue_dist[0] = 0.0
    queue_node[0] = start_linear_index
    queue_size = 1

    diag_dist = np.sqrt(dx**2 + dy**2)

    while queue_size > 0:
        # Find minimum (simple linear search - still fast with JIT)
        min_idx = 0
        min_dist = queue_dist[0]
        for i in range(1, queue_size):
            if queue_dist[i] < min_dist:
                min_idx = i
                min_dist = queue_dist[i]

        # Pop minimum by replacing with last element
        current_dist = queue_dist[min_idx]
        idx = queue_node[min_idx]
        queue_size -= 1
        if queue_size > 0 and min_idx < queue_size:
            queue_dist[min_idx] = queue_dist[queue_size]
            queue_node[min_idx] = queue_node[queue_size]

        # Skip if already processed with shorter path
        if current_dist > distances[idx]:
            continue

        # Get grid coordinates
        row = idx // cols
        col = idx % cols

        if costs[row, col] != 1:
            continue

        # Check 8 neighbors
        # Up
        if row > 0 and costs[row - 1, col] == 1:
            neighbour_idx = (row - 1) * cols + col
            new_dist = current_dist + dy
            if new_dist < distances[neighbour_idx]:
                distances[neighbour_idx] = new_dist
                queue_dist[queue_size] = new_dist
                queue_node[queue_size] = neighbour_idx
                queue_size += 1

        # Down
        if row < rows - 1 and costs[row + 1, col] == 1:
            neighbour_idx = (row + 1) * cols + col
            new_dist = current_dist + dy
            if new_dist < distances[neighbour_idx]:
                distances[neighbour_idx] = new_dist
                queue_dist[queue_size] = new_dist
                queue_node[queue_size] = neighbour_idx
                queue_size += 1

        # Left
        if col > 0 and costs[row, col - 1] == 1:
            neighbour_idx = row * cols + (col - 1)
            new_dist = current_dist + dx
            if new_dist < distances[neighbour_idx]:
                distances[neighbour_idx] = new_dist
                queue_dist[queue_size] = new_dist
                queue_node[queue_size] = neighbour_idx
                queue_size += 1

        # Right
        if col < cols - 1 and costs[row, col + 1] == 1:
            neighbour_idx = row * cols + (col + 1)
            new_dist = current_dist + dx
            if new_dist < distances[neighbour_idx]:
                distances[neighbour_idx] = new_dist
                queue_dist[queue_size] = new_dist
                queue_node[queue_size] = neighbour_idx
                queue_size += 1

        # Up-Left
        if row > 0 and col > 0 and costs[row - 1, col - 1] == 1:
            neighbour_idx = (row - 1) * cols + (col - 1)
            new_dist = current_dist + diag_dist
            if new_dist < distances[neighbour_idx]:
                distances[neighbour_idx] = new_dist
                queue_dist[queue_size] = new_dist
                queue_node[queue_size] = neighbour_idx
                queue_size += 1

        # Up-Right
        if row > 0 and col < cols - 1 and costs[row - 1, col + 1] == 1:
            neighbour_idx = (row - 1) * cols + (col + 1)
            new_dist = current_dist + diag_dist
            if new_dist < distances[neighbour_idx]:
                distances[neighbour_idx] = new_dist
                queue_dist[queue_size] = new_dist
                queue_node[queue_size] = neighbour_idx
                queue_size += 1

        # Down-Left
        if row < rows - 1 and col > 0 and costs[row + 1, col - 1] == 1:
            neighbour_idx = (row + 1) * cols + (col - 1)
            new_dist = current_dist + diag_dist
            if new_dist < distances[neighbour_idx]:
                distances[neighbour_idx] = new_dist
                queue_dist[queue_size] = new_dist
                queue_node[queue_size] = neighbour_idx
                queue_size += 1

        # Down-Right
        if row < rows - 1 and col < cols - 1 and costs[row + 1, col + 1] == 1:
            neighbour_idx = (row + 1) * cols + (col + 1)
            new_dist = current_dist + diag_dist
            if new_dist < distances[neighbour_idx]:
                distances[neighbour_idx] = new_dist
                queue_dist[queue_size] = new_dist
                queue_node[queue_size] = neighbour_idx
                queue_size += 1

    return distances


@njit(parallel=True)
def _compute_all_distances_parallel(
    costs: np.ndarray,
    dx: float,
    dy: float,
    linear_indices: np.ndarray,
    progress_counter: np.ndarray,
) -> np.ndarray:
    """Compute full geodesic distance matrix using parallel Dijkstra calls.

    This function parallelizes the outer loop using Numba's prange, running
    multiple Dijkstra computations concurrently across CPU cores. Each thread
    computes distances from one starting point to all other accessible cells.

    Args:
        costs: Grid cost matrix (1 = free, INT32_MAX = blocked).
        dx: Horizontal step cost.
        dy: Vertical step cost.
        linear_indices: Flattened indices of all accessible cells.
        progress_counter: Shared array of length 1 for tracking progress.
            Each thread increments this counter after completing a cell.

    Returns:
        Distance matrix of shape (n_cells, n_cells) where n_cells is the
        number of accessible cells.
    """
    n = len(linear_indices)
    distance_matrix = np.full((n, n), np.inf, dtype=np.float64)

    # Parallel loop: each iteration is independent
    for i in prange(n):
        # Compute distances from cell i to all cells
        distances = _dijkstra_numba(costs, dx, dy, linear_indices[i])
        # Extract distances to accessible cells only
        for j in range(n):
            distance_matrix[i, j] = distances[linear_indices[j]]

        # Atomically increment progress counter
        progress_counter[0] += 1

    return distance_matrix


@dataclass(frozen=True)
class MovementCostGrid:
    """
    Dataclass representing a discretized spatial grid with movement costs.

    This immutable dataclass stores a 2D grid where each cell has an associated
    movement cost. It is used in navigation tasks to represent environments with
    obstacles and accessible regions. Cells are passable (cost=1) or blocked
    (cost=INT32_MAX).

    The grid maintains spatial information (cell edges, dimensions) and can
    optionally track accessible cells for efficient geodesic distance computations.

    Parameters
    ----------
    costs : np.ndarray
        2D array of movement costs. Shape (n_rows, n_cols).
        Values: 1 for passable cells, INT32_MAX for obstacles.
    x_edges : np.ndarray
        1D array of x-coordinate bin edges. Length n_cols + 1.
        Defines vertical boundaries of grid cells.
    y_edges : np.ndarray
        1D array of y-coordinate bin edges. Length n_rows + 1.
        Defines horizontal boundaries, typically in descending order
        (max_y to min_y) to match array row indexing.
    dx : float
        Width of each grid cell (x-direction spacing)
    dy : float
        Height of each grid cell (y-direction spacing)
    accessible_indices : np.ndarray or None, optional
        1D array of linear indices for accessible (passable) cells.
        Used for efficient geodesic distance matrix computation.
    _index_map : np.ndarray or None, optional
        2D array mapping (row, col) to index in accessible_indices.
        Value is -1 for inaccessible cells. Enables O(1) lookups.

    Attributes
    ----------
    shape : tuple[int, int] (property)
        Shape of the cost grid (n_rows, n_cols)
    x_centers : np.ndarray (property)
        X-coordinates of grid cell centers
    y_centers : np.ndarray (property)
        Y-coordinates of grid cell centers
    accessible_mask : np.ndarray (property)
        Boolean mask where True indicates passable cells

    Methods
    -------
    get_cell_index(pos)
        Get geodesic index of cell containing given position (JAX-compatible)

    Example
    -------
    Create a simple movement cost grid:

    >>> import numpy as np
    >>> from canns.task.navigation_base import MovementCostGrid, INT32_MAX
    >>> 
    >>> # Define grid edges
    >>> x_edges = np.linspace(0, 10, 11)  # 10 cells in x
    >>> y_edges = np.linspace(10, 0, 11)  # 10 cells in y (descending)
    >>> dx, dy = 1.0, 1.0
    >>> 
    >>> # Create cost grid with obstacle
    >>> costs = np.ones((10, 10), dtype=np.int32)
    >>> costs[5, :] = INT32_MAX  # Horizontal wall at y=5
    >>> 
    >>> # Create MovementCostGrid
    >>> grid = MovementCostGrid(
    ...     costs=costs,
    ...     x_edges=x_edges,
    ...     y_edges=y_edges,
    ...     dx=dx,
    ...     dy=dy
    ... )
    >>> 
    >>> # Access properties
    >>> print(f"Grid shape: {grid.shape}")
    >>> print(f"Cell centers (x): {grid.x_centers[:3]}")
    >>> print(f"Number of accessible cells: {grid.accessible_mask.sum()}")

    With accessible indices for geodesic distances:

    >>> # Identify accessible cells
    >>> accessible_mask = (costs == 1)
    >>> accessible_linear = np.where(accessible_mask.ravel())[0]
    >>> 
    >>> # Create index map
    >>> index_map = np.full((10, 10), -1, dtype=np.int32)
    >>> for i, linear_idx in enumerate(accessible_linear):
    ...     row, col = divmod(linear_idx, 10)
    ...     index_map[row, col] = i
    >>> 
    >>> grid_indexed = MovementCostGrid(
    ...     costs=costs,
    ...     x_edges=x_edges,
    ...     y_edges=y_edges,
    ...     dx=dx,
    ...     dy=dy,
    ...     accessible_indices=accessible_linear,
    ...     _index_map=index_map
    ... )
    >>> 
    >>> # Look up cell index for a position
    >>> pos = (2.5, 7.5)  # Position in continuous space
    >>> cell_idx = grid_indexed.get_cell_index(pos)
    >>> print(f"Geodesic cell index: {cell_idx}")

    See Also
    --------
    GeodesicDistanceResult : Result of geodesic distance computation
    BaseNavigationTask : Navigation task using cost grids
    INT32_MAX : Sentinel value for blocked cells

    Notes
    -----
    - Frozen dataclass ensures immutability for safe sharing
    - y_edges typically descends to align with array row indexing
    - accessible_indices and _index_map are optional but required for get_cell_index()
    - JAX-compatible methods enable use in JIT-compiled code
    """
    costs: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    dx: float
    dy: float
    accessible_indices: np.ndarray | None = None
    _index_map: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (n_rows, n_cols)."""
        return self.costs.shape

    @property
    def x_centers(self) -> np.ndarray:
        """X-coordinates of grid cell centers."""
        return self.x_edges[:-1] + self.dx / 2

    @property
    def y_centers(self) -> np.ndarray:
        """Y-coordinates of grid cell centers."""
        return self.y_edges[:-1] - self.dy / 2

    @property
    def accessible_mask(self) -> np.ndarray:
        """Boolean mask where True indicates passable cells (cost == 1)."""
        return self.costs == 1

    def get_cell_index(self, pos: Sequence[float]) -> int:
        """Get the geodesic index of the grid cell containing the given position.

        This method is JAX-compatible and can be used inside jitted functions.

        Args:
            pos: (x, y) coordinates of the position.

        Returns:
            Index of the grid cell in the accessible_indices array, or -1 if
            the position is out of bounds or in an impassable cell.

        Note:
            Returns -1 (instead of None) for JAX compatibility. The caller should
            check for negative values to detect invalid positions.
        """
        if self._index_map is None:
            raise ValueError("Index map not initialized. Grid must be built with index mapping.")

        # Import jax.numpy for JAX-compatible operations
        import jax.numpy as jnp

        x, y = pos

        # Convert arrays to JAX arrays for JAX compatibility
        x_edges_jax = jnp.asarray(self.x_edges)
        y_edges_jax = jnp.asarray(self.y_edges)
        index_map_jax = jnp.asarray(self._index_map)

        # Use jnp.searchsorted which is JAX-compatible
        # Find grid cell containing the position
        col = jnp.searchsorted(x_edges_jax, x, side="right") - 1

        # y_edges is descending (max_y to min_y), so we need special handling
        # For descending array, a cell row is defined where y_edges[row] > y >= y_edges[row+1]
        # Method: search in ascending array (reversed) and convert
        y_edges_ascending = y_edges_jax[::-1]
        # searchsorted with side='left' finds leftmost position where y_edges_ascending[idx] >= y
        idx_in_ascending = jnp.searchsorted(y_edges_ascending, y, side="left")
        # Convert to descending array index: this gives the edge index
        edge_idx_in_descending = len(y_edges_jax) - 1 - idx_in_ascending
        # The cell is one row above this edge (because y_edges[row] > y >= y_edges[row+1])
        row = edge_idx_in_descending - 1

        # Clip indices to valid range to avoid index errors
        # JAX-compatible: no if statements, just array operations
        row_clipped = jnp.clip(row, 0, self.shape[0] - 1)
        col_clipped = jnp.clip(col, 0, self.shape[1] - 1)

        # Lookup index in the precomputed map (O(1) operation)
        # _index_map[row, col] is -1 for inaccessible cells
        idx = index_map_jax[row_clipped, col_clipped]

        # Check bounds: if row/col were out of range, return -1
        # Use array operations instead of if statements
        in_bounds = (row >= 0) & (row < self.shape[0]) & (col >= 0) & (col < self.shape[1])

        # Return idx if in bounds, otherwise -1
        # This is JAX-compatible
        return jnp.where(in_bounds, idx, -1)


@dataclass(frozen=True)
class GeodesicDistanceResult:
    """
    Result of geodesic distance computation on a movement cost grid.

    This immutable dataclass stores the precomputed pairwise geodesic distances
    between all accessible cells in a spatial grid. Geodesic distances account
    for obstacles and follow the shortest navigable path between cells.

    Used in navigation tasks and spatial analysis where Euclidean distance
    would be inaccurate due to environmental constraints (walls, barriers, etc.).

    Parameters
    ----------
    distances : np.ndarray
        Pairwise distance matrix of shape (n_accessible, n_accessible).
        distances[i, j] is the geodesic distance from cell i to cell j,
        where i and j are indices into accessible_indices.
        Unreachable pairs have distance np.inf.
    accessible_indices : np.ndarray
        1D array of linear indices for accessible cells in the original grid.
        Length n_accessible. Maps matrix indices to grid positions.
    cost_grid : MovementCostGrid
        The movement cost grid used to compute distances. Contains spatial
        information and cost values.

    Example
    -------
    Access geodesic distances:

    >>> import numpy as np
    >>> from canns.task.navigation_base import (
    ...     GeodesicDistanceResult, MovementCostGrid, INT32_MAX
    ... )
    >>> 
    >>> # Assume we have computed geodesic distances
    >>> # (normally done via BaseNavigationTask.compute_geodesic_distances())
    >>> # For illustration, create dummy data
    >>> 
    >>> # Cost grid (5x5 with no obstacles)
    >>> costs = np.ones((5, 5), dtype=np.int32)
    >>> x_edges = np.linspace(0, 5, 6)
    >>> y_edges = np.linspace(5, 0, 6)
    >>> 
    >>> cost_grid = MovementCostGrid(
    ...     costs=costs,
    ...     x_edges=x_edges,
    ...     y_edges=y_edges,
    ...     dx=1.0,
    ...     dy=1.0
    ... )
    >>> 
    >>> # All cells are accessible
    >>> accessible_indices = np.arange(25)  # 5x5 = 25 cells
    >>> 
    >>> # Example distance matrix (simplified)
    >>> distances = np.random.rand(25, 25) * 10
    >>> np.fill_diagonal(distances, 0)  # Distance to self is 0
    >>> 
    >>> result = GeodesicDistanceResult(
    ...     distances=distances,
    ...     accessible_indices=accessible_indices,
    ...     cost_grid=cost_grid
    ... )
    >>> 
    >>> # Query distance between cells 0 and 24
    >>> dist = result.distances[0, 24]
    >>> print(f"Geodesic distance: {dist:.2f}")
    >>> 
    >>> # Find cells within radius 3.0 of cell 10
    >>> cell_idx = 10
    >>> nearby = np.where(result.distances[cell_idx] < 3.0)[0]
    >>> print(f"Cells within radius 3.0: {len(nearby)}")

    Using in navigation task:

    >>> from canns.task import BaseNavigationTask
    >>> # Assume task is initialized with environment
    >>> # result = task.compute_geodesic_distances(show_progress=True)
    >>> # 
    >>> # Access precomputed distances for path planning
    >>> # start_idx = task.cost_grid.get_cell_index(start_pos)
    >>> # goal_idx = task.cost_grid.get_cell_index(goal_pos)
    >>> # distance = result.distances[start_idx, goal_idx]

    See Also
    --------
    MovementCostGrid : Grid structure used for distance computation
    BaseNavigationTask : Task class that computes geodesic distances
    BaseNavigationTask.compute_geodesic_distances : Method to generate this result

    Notes
    -----
    - Distances use Dijkstra's algorithm accounting for obstacles
    - Computation can be parallelized using Numba for large grids
    - Matrix is symmetric for undirected navigation
    - Infinite distances indicate unreachable cell pairs
    - Frozen dataclass ensures thread-safe sharing
    """
    distances: np.ndarray
    accessible_indices: np.ndarray
    cost_grid: MovementCostGrid


def _point_in_rect(
    point: Sequence[float],
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> bool:
    x, y = point
    return x_left - EPSILON <= x <= x_right + EPSILON and y_bottom - EPSILON <= y <= y_top + EPSILON


def _orientation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> bool:
    return (
        min(a[0], c[0]) - EPSILON <= b[0] <= max(a[0], c[0]) + EPSILON
        and min(a[1], c[1]) - EPSILON <= b[1] <= max(a[1], c[1]) + EPSILON
    )


def _segments_intersect(
    p1: Sequence[float], p2: Sequence[float], q1: Sequence[float], q2: Sequence[float]
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    def _sign(value: float) -> int:
        if abs(value) <= EPSILON:
            return 0
        return 1 if value > 0 else -1

    s1, s2, s3, s4 = map(_sign, (o1, o2, o3, o4))

    if s1 != s2 and s3 != s4:
        return True

    if s1 == 0 and _on_segment(p1, q1, p2):
        return True
    if s2 == 0 and _on_segment(p1, q2, p2):
        return True
    if s3 == 0 and _on_segment(q1, p1, q2):
        return True
    if s4 == 0 and _on_segment(q1, p2, q2):
        return True

    return False


def _segment_intersects_rect(
    p1: Sequence[float],
    p2: Sequence[float],
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> bool:
    if (
        max(p1[0], p2[0]) < x_left - EPSILON
        or min(p1[0], p2[0]) > x_right + EPSILON
        or max(p1[1], p2[1]) < y_bottom - EPSILON
        or min(p1[1], p2[1]) > y_top + EPSILON
    ):
        return False

    if _point_in_rect(p1, x_left, x_right, y_bottom, y_top) or _point_in_rect(
        p2, x_left, x_right, y_bottom, y_top
    ):
        return True

    rect_corners = (
        (x_left, y_bottom),
        (x_right, y_bottom),
        (x_right, y_top),
        (x_left, y_top),
    )

    rect_edges = (
        (rect_corners[0], rect_corners[1]),
        (rect_corners[1], rect_corners[2]),
        (rect_corners[2], rect_corners[3]),
        (rect_corners[3], rect_corners[0]),
    )

    return any(
        _segments_intersect(p1, p2, edge_start, edge_end) for edge_start, edge_end in rect_edges
    )


def _polygon_intersects_rect(
    polygon: Sequence[Sequence[float]],
    polygon_path: Path,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> bool:
    rect_corners = (
        (x_left, y_bottom),
        (x_right, y_bottom),
        (x_right, y_top),
        (x_left, y_top),
    )

    for corner in rect_corners:
        if polygon_path.contains_point(corner, radius=1e-9):
            return True

    for vertex in polygon:
        if _point_in_rect(vertex, x_left, x_right, y_bottom, y_top):
            return True

    for idx in range(len(polygon)):
        start = polygon[idx]
        end = polygon[(idx + 1) % len(polygon)]
        if _segment_intersects_rect(start, end, x_left, x_right, y_bottom, y_top):
            return True

    return False


class BaseNavigationTask(Task):
    """
    Base class for navigation tasks with geodesic distance computation support.

    This class provides common functionality for both open-loop and closed-loop
    navigation tasks, including environment setup, agent initialization, and
    geodesic distance computation on discretized grids. It integrates with
    the canns-lib spatial navigation module to simulate realistic agent movement
    in 2D environments with obstacles.

    The task supports:
    - Custom 2D environments with boundaries, walls, holes, and objects
    - Simulated agent with realistic movement dynamics and thigmotaxis
    - Geodesic distance computation using Dijkstra's algorithm
    - Grid-based spatial discretization for CANN integration
    - Parallel computation with Numba acceleration

    Parameters
    ----------
    start_pos : tuple[float, float], default=(2.5, 2.5)
        Initial (x, y) position of the agent in meters
    width : float, default=5
        Environment width in meters
    height : float, default=5
        Environment height in meters
    dimensionality : str, default="2D"
        Spatial dimensionality ("2D" only; "1D" not supported)
    boundary_conditions : str, default="solid"
        Type of boundary ("solid" or others supported by Environment)
    scale : float or None, default=None
        Scaling factor for environment (defaults to height if None)
    dx : float, default=0.01
        Resolution for visualization
    grid_dx : float or None, default=None
        Horizontal grid cell size for cost grid (defaults to dx if None)
    grid_dy : float or None, default=None
        Vertical grid cell size for cost grid (defaults to dx if None)
    boundary : list or None, default=None
        Custom boundary specification for Environment
    walls : list or None, default=None
        List of wall specifications (each a dict with geometry)
    holes : list or None, default=None
        List of hole specifications (inaccessible regions)
    objects : list or None, default=None
        List of object specifications for the environment
    dt : float or None, default=None
        Simulation time step (uses brainpy default if None)
    speed_mean : float, default=0.04
        Mean agent speed in m/s
    speed_std : float, default=0.016
        Standard deviation of agent speed
    speed_coherence_time : float, default=0.7
        Time scale for speed persistence (seconds)
    rotational_velocity_coherence_time : float, default=0.08
        Time scale for rotational velocity persistence (seconds)
    rotational_velocity_std : float, default=120*π/180
        Standard deviation of rotational velocity (radians/s)
    head_direction_smoothing_timescale : float, default=0.15
        Time scale for head direction smoothing (seconds)
    initial_head_direction : float or None, default=None
        Initial heading angle (radians). Random if None.
    thigmotaxis : float, default=0.5
        Wall-following tendency (0=none, 1=strong)
    wall_repel_distance : float, default=0.1
        Distance threshold for wall repulsion (meters)
    wall_repel_strength : float, default=1.0
        Strength of wall repulsion force
    rng_seed : int or None, default=None
        Random seed for reproducibility
    data_class : type or None, default=None
        Custom data class for task outputs

    Attributes
    ----------
    env : Environment
        Spatial environment instance from canns-lib
    agent : Agent
        Agent instance with movement dynamics
    cost_grid : MovementCostGrid or None
        Discretized cost grid for the environment
    geodesic_result : GeodesicDistanceResult or None
        Precomputed geodesic distances (if computed)

    Methods
    -------
    set_grid_resolution(grid_dx, grid_dy)
        Update grid resolution and rebuild cost grid
    compute_geodesic_distances(show_progress=False, use_numba=True, 
                                use_parallel=True, backend="auto")
        Compute pairwise geodesic distances between accessible cells

    Example
    -------
    Create navigation task with obstacles:

    >>> import numpy as np
    >>> from canns.task import BaseNavigationTask
    >>> 
    >>> # Define walls (horizontal and vertical barriers)
    >>> walls = [
    ...     {"geometry": "line", "x0": 1.0, "y0": 2.5, "x1": 4.0, "y1": 2.5},  # Horizontal
    ...     {"geometry": "line", "x0": 2.5, "y0": 1.0, "x1": 2.5, "y1": 4.0}   # Vertical
    ... ]
    >>> 
    >>> # Create navigation task
    >>> task = BaseNavigationTask(
    ...     start_pos=(0.5, 0.5),
    ...     width=5.0,
    ...     height=5.0,
    ...     walls=walls,
    ...     grid_dx=0.1,
    ...     grid_dy=0.1,
    ...     speed_mean=0.05,
    ...     thigmotaxis=0.3,
    ...     rng_seed=42
    ... )
    >>> 
    >>> # Access environment and agent
    >>> print(f"Environment size: {task.width} x {task.height}")
    >>> print(f"Grid shape: {task.cost_grid.shape}")
    >>> print(f"Agent start: {task.start_pos}")

    Compute geodesic distances:

    >>> # Compute pairwise distances accounting for walls
    >>> result = task.compute_geodesic_distances(
    ...     show_progress=True,
    ...     use_parallel=True
    ... )
    >>> 
    >>> # Access distance matrix
    >>> print(f"Distance matrix shape: {result.distances.shape}")
    >>> print(f"Number of accessible cells: {len(result.accessible_indices)}")
    >>> 
    >>> # Query distance between two positions
    >>> pos_a = (1.0, 1.0)
    >>> pos_b = (4.0, 4.0)
    >>> idx_a = result.cost_grid.get_cell_index(pos_a)
    >>> idx_b = result.cost_grid.get_cell_index(pos_b)
    >>> if idx_a >= 0 and idx_b >= 0:
    ...     geodesic_dist = result.distances[idx_a, idx_b]
    ...     euclidean_dist = np.linalg.norm(np.array(pos_b) - np.array(pos_a))
    ...     print(f"Geodesic distance: {geodesic_dist:.2f}")
    ...     print(f"Euclidean distance: {euclidean_dist:.2f}")

    See Also
    --------
    MovementCostGrid : Grid representation of environment
    GeodesicDistanceResult : Result of distance computation
    OpenLoopNavigationTask : Subclass for open-loop navigation
    ClosedLoopNavigationTask : Subclass for closed-loop navigation

    Notes
    -----
    - Only 2D environments are supported; 1D raises NotImplementedError
    - Geodesic distances account for obstacles using Dijkstra's algorithm
    - Numba acceleration provides significant speedup for large grids
    - Parallel computation uses multiple CPU cores via Numba's prange
    - Cost grid is automatically rebuilt when grid resolution changes
    - Agent dynamics follow biologically-plausible movement patterns
    """

    def __init__(
        self,
        start_pos=(2.5, 2.5),
        # environment parameters
        width=5,
        height=5,
        dimensionality="2D",
        boundary_conditions="solid",
        scale=None,
        dx=0.01,
        grid_dx: float | None = None,
        grid_dy: float | None = None,
        boundary=None,
        walls=None,
        holes=None,
        objects=None,
        # agent parameters
        dt=None,
        speed_mean=0.04,
        speed_std=0.016,
        speed_coherence_time=0.7,
        rotational_velocity_coherence_time=0.08,
        rotational_velocity_std=120 * np.pi / 180,
        head_direction_smoothing_timescale=0.15,
        initial_head_direction: float | None = None,
        thigmotaxis=0.5,
        wall_repel_distance=0.1,
        wall_repel_strength=1.0,
        rng_seed: int | None = None,  # Add rng_seed parameter
        data_class=None,
    ):
        super().__init__(data_class=data_class)

        # time settings
        self.dt = dt if dt is not None else bm.get_dt()

        # environment settings
        self.width = width
        self.height = height
        self.aspect = width / height
        self.dimensionality = str(dimensionality).upper()
        if self.dimensionality == "1D":
            raise NotImplementedError("Navigation tasks do not support 1D environment.")
        if self.dimensionality != "2D":
            raise ValueError(f"Unsupported dimensionality '{dimensionality}'. Expected '2D'.")
        self.boundary_conditions = boundary_conditions
        self.scale = height if scale is None else scale
        self.dx = dx  # for visualization only
        self.grid_dx = dx if grid_dx is None else grid_dx
        self.grid_dy = dx if grid_dy is None else grid_dy
        self.boundary = copy.deepcopy(boundary)
        self.walls = copy.deepcopy(walls) if walls is not None else []
        self.holes = copy.deepcopy(holes) if holes is not None else []
        self.objects = copy.deepcopy(objects) if objects is not None else []
        self.cost_grid: MovementCostGrid | None = None
        self.geodesic_result: GeodesicDistanceResult | None = None
        self.set_grid_resolution(self.grid_dx, self.grid_dy)

        # agent settings
        self.speed_mean = speed_mean
        self.speed_std = speed_std
        self.speed_coherence_time = speed_coherence_time
        self.rotational_velocity_coherence_time = rotational_velocity_coherence_time
        self.rotational_velocity_std = rotational_velocity_std
        self.head_direction_smoothing_timescale = head_direction_smoothing_timescale
        self.thigmotaxis = thigmotaxis
        self.wall_repel_distance = wall_repel_distance
        self.wall_repel_strength = wall_repel_strength
        self.start_pos = start_pos
        self.initial_head_direction = initial_head_direction
        self.rng_seed = rng_seed  # Store rng_seed

        self.env_params = {
            "dimensionality": self.dimensionality,
            "boundary_conditions": self.boundary_conditions,
            "scale": self.scale,
            "aspect": self.aspect,
            "dx": self.dx,
            "boundary": self.boundary,
            "walls": copy.deepcopy(self.walls),
            "holes": copy.deepcopy(self.holes),
            "objects": copy.deepcopy(self.objects),
        }
        self.env = Environment(**self.env_params)

        self.agent_params = {
            "dt": self.dt,
            "speed_mean": self.speed_mean,
            "speed_std": self.speed_std,
            "speed_coherence_time": self.speed_coherence_time,
            "rotational_velocity_coherence_time": self.rotational_velocity_coherence_time,
            "rotational_velocity_std": self.rotational_velocity_std,
            "head_direction_smoothing_timescale": self.head_direction_smoothing_timescale,
            "thigmotaxis": self.thigmotaxis,
            "wall_repel_distance": self.wall_repel_distance,
            "wall_repel_strength": self.wall_repel_strength,
        }
        self.agent = Agent(
            environment=self.env, params=copy.deepcopy(self.agent_params), rng_seed=self.rng_seed
        )
        self.agent.set_position(np.array(start_pos))
        self.agent.dt = self.dt
        self._apply_initial_head_direction(speed_mean=self.speed_mean)

    def _apply_initial_head_direction(
        self, head_direction: float | None = None, speed_mean: float | None = None
    ) -> None:
        """Apply an initial head direction vector to the agent if provided."""

        angle = self.initial_head_direction if head_direction is None else head_direction
        if angle is None:
            return

        self.agent.head_direction = np.array([np.cos(angle), np.sin(angle)])
        velocity = self.agent.head_direction * (
            self.agent.speed_mean if speed_mean is None else speed_mean
        )
        self.agent.set_velocity(velocity)

    def build_movement_cost_grid(self, *, refresh: bool = False) -> MovementCostGrid:
        """Construct a grid-based movement cost map for the configured environment.

        A cell weight of ``1`` indicates free space, while ``INT32_MAX`` marks an
        impassable cell (intersecting a wall/hole or lying outside the boundary).

        Args:
            refresh: Force recomputation even if a cached grid is available.

        Returns:
            MovementCostGrid describing the discretised environment.
        """

        if self.grid_dx is None or self.grid_dy is None:
            raise ValueError("Grid resolution is not configured. Use set_grid_resolution().")

        if self.cost_grid is not None and not refresh:
            return self.cost_grid

        boundary_coords = self._resolve_boundary_coordinates()
        boundary_path = Path(boundary_coords) if len(boundary_coords) >= 3 else None

        min_x = float(np.min(boundary_coords[:, 0]))
        max_x = float(np.max(boundary_coords[:, 0]))
        min_y = float(np.min(boundary_coords[:, 1]))
        max_y = float(np.max(boundary_coords[:, 1]))

        n_cols = int(np.ceil((max_x - min_x) / self.grid_dx))
        n_rows = int(np.ceil((max_y - min_y) / self.grid_dy))
        if n_cols <= 0 or n_rows <= 0:
            raise ValueError("Computed grid has no cells; check boundary and resolution.")

        x_edges = min_x + np.arange(n_cols + 1, dtype=float) * self.grid_dx
        y_edges = max_y - np.arange(n_rows + 1, dtype=float) * self.grid_dy

        # Ensure the last edge covers the extremum even if dx/dy do not divide evenly.
        if x_edges[-1] < max_x:
            x_edges = np.append(x_edges, x_edges[-1] + self.grid_dx)
        if y_edges[-1] > min_y:
            y_edges = np.append(y_edges, y_edges[-1] - self.grid_dy)

        wall_segments = [np.asarray(w, dtype=float) for w in (self.walls or []) if len(w) >= 2]
        hole_polygons = [np.asarray(h, dtype=float) for h in (self.holes or []) if len(h) >= 3]
        hole_paths = [Path(poly) for poly in hole_polygons]

        costs = np.ones((len(y_edges) - 1, len(x_edges) - 1), dtype=np.int32)

        for row in range(costs.shape[0]):
            y_top = y_edges[row]
            y_bottom = y_edges[row + 1]
            center_y = (y_top + y_bottom) / 2
            for col in range(costs.shape[1]):
                x_left = x_edges[col]
                x_right = x_edges[col + 1]
                center_x = (x_left + x_right) / 2
                center = (center_x, center_y)

                if boundary_path is not None and not boundary_path.contains_point(
                    center, radius=1e-9
                ):
                    costs[row, col] = INT32_MAX
                    continue

                if self._cell_is_blocked_by_walls(wall_segments, x_left, x_right, y_bottom, y_top):
                    costs[row, col] = INT32_MAX
                    continue

                if self._cell_overlaps_hole(
                    hole_polygons,
                    hole_paths,
                    x_left,
                    x_right,
                    y_bottom,
                    y_top,
                    center,
                ):
                    costs[row, col] = INT32_MAX

        # Compute accessible indices and index map for O(1) lookup
        accessible_mask = costs == 1
        accessible_indices = np.argwhere(accessible_mask)

        # Build reverse index map: _index_map[row, col] = geodesic_index or -1
        index_map = np.full(costs.shape, -1, dtype=np.int32)
        for idx, (r, c) in enumerate(accessible_indices):
            index_map[r, c] = idx

        grid = MovementCostGrid(
            costs=costs,
            x_edges=x_edges,
            y_edges=y_edges,
            dx=self.grid_dx,
            dy=self.grid_dy,
            accessible_indices=accessible_indices,
            _index_map=index_map,
        )
        self.cost_grid = grid
        self.geodesic_result = None
        return grid

    def compute_geodesic_distance_matrix(
        self,
        dx: float | None = None,
        dy: float | None = None,
        *,
        refresh: bool = False,
    ) -> GeodesicDistanceResult:
        """Compute pairwise geodesic distances between traversable grid cells.

        The computation treats each traversable cell (weight ``1``) as a graph node
        connected to its four axis-aligned neighbours. Horizontal steps cost ``dx``
        and vertical steps cost ``dy``. Impassable cells (``INT32_MAX``) are ignored.

        When Numba is available, this method uses parallelized Dijkstra computation
        across CPU cores for significant speedup (typically 4-8x on multi-core systems).
        Without Numba, it falls back to sequential Python implementation with a
        progress bar.

        Args:
            dx: Grid cell width along the x axis. When ``None`` the existing
                ``grid_dx`` attribute is used.
            dy: Grid cell height along the y axis. When ``None`` the existing
                ``grid_dy`` attribute is used.
            refresh: Force recomputation even if cached results exist.

        Returns:
            GeodesicDistanceResult containing the distance matrix and metadata.

        Note:
            The parallel Numba implementation cannot show a progress bar during
            computation, but prints start/end messages instead.
        """
        if dx is not None:
            if dx <= 0:
                raise ValueError("dx must be a positive number.")
            self.grid_dx = dx
        if dy is not None:
            if dy <= 0:
                raise ValueError("dy must be a positive number.")
            self.grid_dy = dy

        if self.geodesic_result is not None and not refresh and dx is None and dy is None:
            return self.geodesic_result

        if self.cost_grid is None or refresh:
            grid = self.build_movement_cost_grid(refresh=refresh)
        else:
            grid = self.cost_grid
        mask = grid.accessible_mask
        accessible_indices = np.argwhere(mask)
        if accessible_indices.size == 0:
            return GeodesicDistanceResult(
                distances=np.full((0, 0), np.nan, dtype=float),
                accessible_indices=accessible_indices,
                cost_grid=grid,
            )

        rows, cols = grid.shape
        linear_indices = accessible_indices[:, 0] * cols + accessible_indices[:, 1]

        if NUMBA_AVAILABLE:
            # Use parallel Numba implementation with progress monitoring
            import threading
            import time

            n_cells = len(linear_indices)
            progress_counter = np.zeros(1, dtype=np.int64)

            # Start parallel computation in a thread
            result_holder = [None]

            def compute_parallel():
                result_holder[0] = _compute_all_distances_parallel(
                    grid.costs,
                    self.grid_dx,
                    self.grid_dy,
                    linear_indices,
                    progress_counter,
                )

            compute_thread = threading.Thread(target=compute_parallel)

            # Monitor progress with tqdm
            desc = "Computing geodesic distances (Numba parallel)"
            pbar = tqdm(total=n_cells, desc=desc, ncols=100)

            compute_thread.start()

            # Update progress bar while computation runs
            last_count = 0
            while compute_thread.is_alive():
                current_count = int(progress_counter[0])
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                time.sleep(0.1)  # Check every 100ms

            compute_thread.join()

            # Final update to ensure 100%
            current_count = int(progress_counter[0])
            if current_count > last_count:
                pbar.update(current_count - last_count)
            pbar.close()

            distance_matrix = result_holder[0]
        else:
            # Use sequential Python implementation with progress bar
            distance_matrix = np.full(
                (accessible_indices.shape[0], accessible_indices.shape[0]),
                np.inf,
                dtype=float,
            )
            desc = "Computing geodesic distances (Python)"
            for i, linear_idx in enumerate(tqdm(linear_indices, desc=desc, ncols=100)):
                distances = self._dijkstra_on_grid(
                    grid.costs, self.grid_dx, self.grid_dy, linear_idx
                )
                distance_matrix[i, :] = distances[linear_indices]

        result = GeodesicDistanceResult(
            distances=distance_matrix,
            accessible_indices=accessible_indices,
            cost_grid=grid,
        )
        self.geodesic_result = result
        return result

    def show_geodesic_distance_matrix(
        self,
        dx: float | None = None,
        dy: float | None = None,
        *,
        show: bool = True,
        save_path: str | None = None,
        cmap: str | colors.Colormap = "viridis",
        normalize: bool = False,
        colorbar: bool = True,
        refresh: bool = False,
    ) -> GeodesicDistanceResult:
        """Visualise the geodesic distance matrix for the discretised environment."""

        result = self.compute_geodesic_distance_matrix(dx=dx, dy=dy, refresh=refresh)
        distances = result.distances

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            if distances.size == 0:
                ax.text(0.5, 0.5, "No traversable cells", ha="center", va="center")
                ax.axis("off")
            else:
                matrix = self._prepare_geodesic_plot_matrix(distances, normalize=normalize)
                im = ax.imshow(matrix, cmap=cmap, interpolation="nearest")
                ax.set_title("Geodesic distances")
                ax.set_xlabel("Accessible cell index")
                ax.set_ylabel("Accessible cell index")
                if colorbar:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.savefig(save_path) if save_path else None
            plt.show() if show else None
        finally:
            plt.close(fig)

        return result

    def get_geodesic_index_by_pos(
        self, pos: Sequence[float], *, refresh: bool = False
    ) -> int | None:
        """Get the index of the grid cell containing the given position.

        Args:
            pos: (x, y) coordinates of the position.
            refresh: Recompute the cached grid before querying the index.

        Returns:
            Index of the grid cell in the geodesic distance matrix, or None if
            the position is out of bounds or in an impassable cell.
        """
        grid = self.build_movement_cost_grid(refresh=refresh)
        return grid.get_cell_index(pos)

    def set_grid_resolution(self, dx: float, dy: float) -> None:
        """Update the stored grid resolution and invalidate cached data."""

        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be positive numbers.")

        self.grid_dx = dx
        self.grid_dy = dy
        self.cost_grid = None
        self.geodesic_result = None

    def show_data(
        self,
        show: bool = True,
        save_path: str | None = None,
        *,
        overlay_movement_cost: bool = False,
        cost_grid: MovementCostGrid | None = None,
        free_color: str = "#f8f9fa",
        blocked_color: str = "#f94144",
        gridline_color: str = "#2b2d42",
        cost_alpha: float = 0.6,
        show_colorbar: bool = False,
        cost_legend_loc: str | None = None,
    ) -> None:
        """Display the agent's trajectory with optional movement cost grid overlay.

        Args:
            show: Whether to display the plot.
            save_path: Path to save the figure. If None, the figure is not saved.
            overlay_movement_cost: Whether to overlay the movement cost grid.
            cost_grid: Pre-computed cost grid. If None and overlay_movement_cost is True,
                      the grid will be built on demand.
            free_color: Color for free (accessible) cells in the cost grid.
            blocked_color: Color for blocked (inaccessible) cells in the cost grid.
            gridline_color: Color for grid lines.
            cost_alpha: Transparency of the cost grid overlay (0=transparent, 1=opaque).
            show_colorbar: Whether to show a colorbar for the cost grid.
            cost_legend_loc: Location of the legend for the cost grid (e.g., 'upper right').
                           If None, no legend is shown.
        """
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        try:
            # Check if agent has trajectory history
            trajectory_length = len(self.agent.history.get("t", []))
            if trajectory_length >= 2:
                # For both open-loop and closed-loop with sufficient history
                total_steps = getattr(self, "total_steps", trajectory_length)
                self.agent.plot_trajectory(
                    t_start=0, t_end=total_steps, fig=fig, ax=ax, color="changing"
                )
            else:
                # For closed-loop or when trajectory hasn't been generated yet
                ax.scatter(
                    self.agent.pos[0],
                    self.agent.pos[1],
                    s=30,
                    c="tab:blue",
                    label="start",
                )
                ax.legend(loc="upper right")

            # Overlay movement cost grid if requested
            if overlay_movement_cost:
                if cost_grid is None:
                    cost_grid = self.build_movement_cost_grid()
                self._plot_movement_cost_grid(
                    ax,
                    cost_grid,
                    free_color=free_color,
                    blocked_color=blocked_color,
                    gridline_color=gridline_color,
                    alpha=cost_alpha,
                    add_colorbar=show_colorbar,
                    legend_loc=cost_legend_loc,
                )

            plt.savefig(save_path) if save_path else None
            plt.show() if show else None
        finally:
            plt.close(fig)

    @staticmethod
    def _prepare_geodesic_plot_matrix(
        distances: np.ndarray, *, normalize: bool = False
    ) -> np.ndarray:
        matrix = distances.copy()
        finite_mask = np.isfinite(matrix)
        if not finite_mask.any():
            return np.zeros_like(matrix)

        if normalize:
            max_val = np.nanmax(matrix[finite_mask])
            if max_val > 0:
                matrix[finite_mask] = matrix[finite_mask] / max_val

        matrix[~finite_mask] = np.nan
        return matrix

    def _resolve_boundary_coordinates(self) -> np.ndarray:
        if self.boundary is not None and len(self.boundary) >= 3:
            return np.asarray(self.boundary, dtype=float)

        return np.asarray(
            [
                [0.0, 0.0],
                [self.width, 0.0],
                [self.width, self.height],
                [0.0, self.height],
            ],
            dtype=float,
        )

    @staticmethod
    def _cell_is_blocked_by_walls(
        wall_segments: Sequence[np.ndarray],
        x_left: float,
        x_right: float,
        y_bottom: float,
        y_top: float,
    ) -> bool:
        if not wall_segments:
            return False

        for segment in wall_segments:
            if len(segment) < 2:
                continue
            if any(
                _segment_intersects_rect(
                    segment[i], segment[i + 1], x_left, x_right, y_bottom, y_top
                )
                for i in range(len(segment) - 1)
            ):
                return True
        return False

    @staticmethod
    def _cell_overlaps_hole(
        hole_polygons: Sequence[np.ndarray],
        hole_paths: Sequence[Path],
        x_left: float,
        x_right: float,
        y_bottom: float,
        y_top: float,
        center: tuple[float, float],
    ) -> bool:
        if not hole_polygons:
            return False

        for polygon, path in zip(hole_polygons, hole_paths, strict=False):
            if path.contains_point(center, radius=1e-9):
                return True

            if _polygon_intersects_rect(
                polygon,
                path,
                x_left,
                x_right,
                y_bottom,
                y_top,
            ):
                return True
        return False

    @staticmethod
    def _dijkstra_on_grid(
        costs: np.ndarray, dx: float, dy: float, start_linear_index: int
    ) -> np.ndarray:
        """Dijkstra's algorithm on grid (with Numba JIT if available)."""
        if NUMBA_AVAILABLE:
            return _dijkstra_numba(costs, dx, dy, start_linear_index)
        else:
            return _dijkstra_python(costs, dx, dy, start_linear_index)

    @staticmethod
    def _plot_movement_cost_grid(
        ax: plt.Axes,
        grid: MovementCostGrid,
        *,
        free_color: str = "#f8f9fa",
        blocked_color: str = "#f94144",
        gridline_color: str = "#2b2d42",
        alpha: float = 0.6,
        add_colorbar: bool = False,
        legend_loc: str | None = None,
    ) -> None:
        """Overlay the movement cost grid onto an existing axes."""

        blocked_mask = grid.costs == INT32_MAX
        display = np.where(blocked_mask, 1, 0)

        cmap = colors.ListedColormap([free_color, blocked_color])
        norm = colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        x_edges = grid.x_edges
        y_edges_plot = grid.y_edges[::-1]
        display_plot = np.flipud(display)

        mesh = ax.pcolormesh(
            x_edges,
            y_edges_plot,
            display_plot,
            cmap=cmap,
            norm=norm,
            shading="auto",
            linewidth=0.5,
            edgecolors=gridline_color,
            alpha=alpha,
        )

        ymin, ymax = y_edges_plot[0], y_edges_plot[-1]
        ax.vlines(x_edges, ymin, ymax, colors=gridline_color, linewidth=0.5, alpha=0.7)
        ax.hlines(
            y_edges_plot, x_edges[0], x_edges[-1], colors=gridline_color, linewidth=0.5, alpha=0.7
        )

        ax.set_aspect("equal")

        if add_colorbar:
            cbar = plt.colorbar(
                mesh,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                boundaries=[-0.5, 0.5, 1.5],
                ticks=[0, 1],
            )
            cbar.ax.set_yticklabels(["Free", "Blocked"])

        if legend_loc:
            from matplotlib.patches import Patch

            handles = [
                Patch(facecolor=free_color, edgecolor=gridline_color, label="Free"),
                Patch(facecolor=blocked_color, edgecolor=gridline_color, label="Blocked"),
            ]
            ax.legend(handles=handles, loc=legend_loc, framealpha=0.8)
