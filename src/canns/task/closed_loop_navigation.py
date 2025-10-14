import copy
import heapq
from dataclasses import dataclass
from typing import Sequence

import brainstate
import numpy as np
import ratinabox
from matplotlib import colors, pyplot as plt
from matplotlib.path import Path
from ratinabox import Environment, Agent

from ._base import Task
from .open_loop_navigation import OpenLoopNavigationData


INT32_MAX = np.iinfo(np.int32).max
EPSILON = 1e-12


@dataclass(frozen=True)
class MovementCostGrid:
    costs: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    dx: float
    dy: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.costs.shape

    @property
    def x_centers(self) -> np.ndarray:
        return self.x_edges[:-1] + self.dx / 2

    @property
    def y_centers(self) -> np.ndarray:
        return self.y_edges[:-1] - self.dy / 2

    @property
    def accessible_mask(self) -> np.ndarray:
        return self.costs == 1


@dataclass(frozen=True)
class GeodesicDistanceResult:
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
    return (
        x_left - EPSILON <= x <= x_right + EPSILON
        and y_bottom - EPSILON <= y <= y_top + EPSILON
    )


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

    for edge_start, edge_end in rect_edges:
        if _segments_intersect(p1, p2, edge_start, edge_end):
            return True

    return False


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


class ClosedLoopNavigationTask(Task):
    def __init__(
        self,
        start_pos=(2.5, 2.5),
        # environment parameters
        width=5,
        height=5,
        dimensionality="2D",
        boundary_conditions="solid",  # "solid" or "periodic"
        scale=None,
        dx=0.01,  # for show_data only
        boundary=None,
        # coordinates [[x0,y0],[x1,y1],...] of the corners of a 2D polygon bounding the Env (if None, Env defaults to rectangular). Corners must be ordered clockwise or anticlockwise, and the polygon must be a 'simple polygon' (no holes, doesn't self-intersect).
        walls=None,
        # a list of loose walls within the environment. Each wall in the list can be defined by it's start and end coords [[x0,y0],[x1,y1]]. You can also manually add walls after init using Env.add_wall() (preferred).
        holes=None,
        # coordinates [[[x0,y0],[x1,y1],...],...] of corners of any holes inside the Env. These must be entirely inside the environment and not intersect one another. Corners must be ordered clockwise or anticlockwise. holes has 1-dimension more than boundary since there can be multiple holes
        objects=None,
        # a list of objects within the environment. Each object is defined by its position [[x0,y0],[x1,y1],...] for 2D environments and [[x0],[x1],...] for 1D environments. By default all objects are type 0, alternatively you can manually add objects after init using Env.add_object(object, type) (preferred).
        # agent parameters (they are not used in closed-loop task, we just keep them for consistency with open-loop task)
        dt=None,
        speed_mean=0.04,
        speed_std=0.016,
        speed_coherence_time=0.7,
        rotational_velocity_coherence_time=0.08,
        rotational_velocity_std=120 * np.pi / 180,
        head_direction_smoothing_timescale=0.15,
        thigmotaxis=0.5,
        wall_repel_distance=0.1,
        wall_repel_strength=1.0,
    ):
        super().__init__(data_class=OpenLoopNavigationData)

        # time settings
        self.dt = dt if dt is not None else brainstate.environ.get_dt()
        self.total_steps = 1

        # environment settings
        self.width = width
        self.height = height
        self.aspect = width / height
        self.dimensionality = str(dimensionality).upper()
        if self.dimensionality == "1D":
            raise NotImplementedError(
                "ClosedLoopNavigationTask does not support 1D environment."
            )
        if self.dimensionality != "2D":
            raise ValueError(f"Unsupported dimensionality '{dimensionality}'. Expected '2D'.")
        self.boundary_conditions = boundary_conditions
        self.scale = height if scale is None else scale
        self.dx = dx
        self.boundary = copy.deepcopy(boundary)
        self.walls = copy.deepcopy(walls) if walls is not None else []
        self.holes = copy.deepcopy(holes) if holes is not None else []
        self.objects = copy.deepcopy(objects) if objects is not None else []

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

        # ratinabox settings
        ratinabox.stylize_plots()
        ratinabox.autosave_plots = False
        ratinabox.figure_directory = "figures"

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
        self.env = Environment(params=self.env_params)

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
        self.agent = Agent(Environment=self.env, params=copy.deepcopy(self.agent_params))
        self.agent.pos = np.array(start_pos)
        self.agent.update(forced_next_position=self.agent.pos)
        self.agent.dt = self.dt

    def step_by_pos(self, new_pos):
        self.agent.update(forced_next_position = np.asarray(new_pos))

    def get_data(self):
        raise NotImplementedError("ClosedLoopNavigationTask does not have get_data method.")

    def show_data(
        self,
        show: bool = True,
        save_path: str | None = None,
        *,
        overlay_movement_cost: bool = False,
        cost_dx: float | None = None,
        cost_dy: float | None = None,
        cost_grid: MovementCostGrid | None = None,
        free_color: str = "#f8f9fa",
        blocked_color: str = "#f94144",
        gridline_color: str = "#2b2d42",
        cost_alpha: float = 0.6,
        show_colorbar: bool = False,
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        try:
            trajectory_length = len(self.agent.history.get("t", []))
            if trajectory_length >= 2:
                self.agent.plot_trajectory(
                    t_start=0, t_end=self.total_steps, fig=fig, ax=ax, color="changing"
                )
            else:
                ax.scatter(
                    self.agent.pos[0],
                    self.agent.pos[1],
                    s=30,
                    c="tab:blue",
                    label="start",
                )
                ax.legend(loc="upper right")

            if overlay_movement_cost:
                if cost_grid is None:
                    dx = cost_dx if cost_dx is not None else self.dx
                    dy = cost_dy if cost_dy is not None else self.dx
                    cost_grid = self.build_movement_cost_grid(dx=dx, dy=dy)
                self._plot_movement_cost_grid(
                    ax,
                    cost_grid,
                    free_color=free_color,
                    blocked_color=blocked_color,
                    gridline_color=gridline_color,
                    alpha=cost_alpha,
                    add_colorbar=show_colorbar,
                )

            plt.savefig(save_path) if save_path else None
            plt.show() if show else None
        finally:
            plt.close(fig)

    def show_geodesic_distance_matrix(
        self,
        dx: float,
        dy: float,
        *,
        show: bool = True,
        save_path: str | None = None,
        cmap: str | colors.Colormap = "viridis",
        normalize: bool = False,
        colorbar: bool = True,
    ) -> GeodesicDistanceResult:
        """Visualise the geodesic distance matrix for the discretised environment."""

        result = self.compute_geodesic_distance_matrix(dx=dx, dy=dy)
        distances = result.distances

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            if distances.size == 0:
                ax.text(0.5, 0.5, "No traversable cells", ha="center", va="center")
                ax.axis("off")
            else:
                matrix = distances.copy()
                finite_mask = np.isfinite(matrix)
                if not finite_mask.any():
                    matrix = np.zeros_like(matrix)
                elif normalize:
                    max_val = np.nanmax(matrix[finite_mask])
                    if max_val > 0:
                        matrix = matrix / max_val

                matrix[~finite_mask] = np.nan
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

    def build_movement_cost_grid(self, dx: float, dy: float) -> MovementCostGrid:
        """Construct a grid-based movement cost map for the configured environment.

        A cell weight of ``1`` indicates free space, while ``INT32_MAX`` marks an
        impassable cell (intersecting a wall/hole or lying outside the boundary).

        Args:
            dx: Grid cell width along the x axis.
            dy: Grid cell height along the y axis.

        Returns:
            MovementCostGrid describing the discretised environment.
        """

        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be positive numbers.")

        boundary_coords = self._resolve_boundary_coordinates()
        boundary_path = Path(boundary_coords) if len(boundary_coords) >= 3 else None

        min_x = float(np.min(boundary_coords[:, 0]))
        max_x = float(np.max(boundary_coords[:, 0]))
        min_y = float(np.min(boundary_coords[:, 1]))
        max_y = float(np.max(boundary_coords[:, 1]))

        n_cols = int(np.ceil((max_x - min_x) / dx))
        n_rows = int(np.ceil((max_y - min_y) / dy))
        if n_cols <= 0 or n_rows <= 0:
            raise ValueError("Computed grid has no cells; check boundary and resolution.")

        x_edges = min_x + np.arange(n_cols + 1, dtype=float) * dx
        y_edges = max_y - np.arange(n_rows + 1, dtype=float) * dy

        # Ensure the last edge covers the extremum even if dx/dy do not divide evenly.
        if x_edges[-1] < max_x:
            x_edges = np.append(x_edges, x_edges[-1] + dx)
        if y_edges[-1] > min_y:
            y_edges = np.append(y_edges, y_edges[-1] - dy)

        wall_segments = [
            np.asarray(w, dtype=float) for w in (self.walls or []) if len(w) >= 2
        ]
        hole_polygons = [
            np.asarray(h, dtype=float) for h in (self.holes or []) if len(h) >= 3
        ]
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

                if self._cell_is_blocked_by_walls(
                    wall_segments, x_left, x_right, y_bottom, y_top
                ):
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

        return MovementCostGrid(costs=costs, x_edges=x_edges, y_edges=y_edges, dx=dx, dy=dy)

    def compute_geodesic_distance_matrix(
        self, dx: float, dy: float
    ) -> GeodesicDistanceResult:
        """Compute pairwise geodesic distances between traversable grid cells.

        The computation treats each traversable cell (weight ``1``) as a graph node
        connected to its four axis-aligned neighbours. Horizontal steps cost ``dx``
        and vertical steps cost ``dy``. Impassable cells (``INT32_MAX``) are ignored.

        Args:
            dx: Grid cell width along the x axis.
            dy: Grid cell height along the y axis.

        Returns:
            GeodesicDistanceResult containing the distance matrix and metadata.
        """

        grid = self.build_movement_cost_grid(dx=dx, dy=dy)
        mask = grid.accessible_mask
        accessible_indices = np.argwhere(mask)
        if accessible_indices.size == 0:
            return GeodesicDistanceResult(
                distances=np.zeros((0, 0), dtype=float),
                accessible_indices=accessible_indices,
                cost_grid=grid,
            )

        rows, cols = grid.shape
        linear_indices = accessible_indices[:, 0] * cols + accessible_indices[:, 1]
        distance_matrix = np.full(
            (accessible_indices.shape[0], accessible_indices.shape[0]),
            np.inf,
            dtype=float,
        )

        for i, linear_idx in enumerate(linear_indices):
            distances = self._dijkstra_on_grid(grid.costs, dx, dy, linear_idx)
            distance_matrix[i, :] = distances[linear_indices]

        return GeodesicDistanceResult(
            distances=distance_matrix,
            accessible_indices=accessible_indices,
            cost_grid=grid,
        )

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
            if _segment_intersects_rect(
                segment[0], segment[-1], x_left, x_right, y_bottom, y_top
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

        for polygon, path in zip(hole_polygons, hole_paths):
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
        rows, cols = costs.shape
        total_cells = rows * cols
        distances = np.full(total_cells, np.inf, dtype=float)
        distances[start_linear_index] = 0.0

        heap: list[tuple[float, int]] = [(0.0, start_linear_index)]

        while heap:
            current_dist, idx = heapq.heappop(heap)
            if current_dist > distances[idx]:
                continue

            row, col = divmod(idx, cols)
            if costs[row, col] != 1:
                continue

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
        ax.hlines(y_edges_plot, x_edges[0], x_edges[-1], colors=gridline_color, linewidth=0.5, alpha=0.7)

        for row in range(grid.costs.shape[0]):
            y_top = grid.y_edges[row]
            y_bottom = grid.y_edges[row + 1]
            y_center = (y_top + y_bottom) / 2
            for col in range(grid.costs.shape[1]):
                x_left = grid.x_edges[col]
                x_right = grid.x_edges[col + 1]
                x_center = (x_left + x_right) / 2
                weight = grid.costs[row, col]
                label = "MAX" if weight == INT32_MAX else str(weight)
                text_color = "#000000" if weight != INT32_MAX else "#ffffff"
                ax.text(
                    x_center,
                    y_center,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                    alpha=0.9,
                )

        ax.set_aspect("equal")

        if add_colorbar:
            cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["Free", "Blocked"])


class TMazeClosedLoopNavigationTask(ClosedLoopNavigationTask):
    def __init__(
        self,
        w=0.3,  # corridor width
        l_s=1.0,  # stem length
        l_arm=0.75,  # arm length
        t=0.3,  # wall thickness
        start_pos=(0.0, 0.15),
        dt=None,
        **kwargs,
    ):
        hw = w / 2
        boundary = [
            [-hw, 0.0],
            [-hw, l_s],
            [-l_arm, l_s],
            [-l_arm, l_s + t],
            [l_arm, l_s + t],
            [l_arm, l_s],
            [hw, l_s],
            [hw, 0.0],
        ]
        super().__init__(
            start_pos=start_pos,
            boundary=boundary,
            dt=dt,
        )
