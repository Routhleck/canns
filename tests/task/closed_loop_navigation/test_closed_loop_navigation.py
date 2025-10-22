import os

import numpy as np
import pytest

from canns.task.closed_loop_navigation import (
    ClosedLoopNavigationTask,
    TMazeClosedLoopNavigationTask,
)
from canns.task.navigation_base import INT32_MAX


def test_tmaze_movement_cost_and_geodesic_visualisation(tmp_path):
    mpl_cache = tmp_path / "mpl"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    task = TMazeClosedLoopNavigationTask(dt=0.01)
    task.step_by_pos(task.start_pos)
    dx = dy = 0.1

    task.set_grid_resolution(dx, dy)
    grid = task.build_movement_cost_grid()

    assert grid.costs.dtype == np.int32
    assert np.any(grid.costs == 1)
    assert np.any(grid.costs == INT32_MAX)

    fig, ax = plt.subplots()
    task._plot_movement_cost_grid(ax, grid, add_colorbar=True)
    labels = {text.get_text() for text in ax.texts}
    assert (grid.costs == INT32_MAX).any()
    plt.close(fig)

    movement_path = tmp_path / "tmaze_movement_cost.png"
    task.show_data(
        show=False,
        overlay_movement_cost=True,
        cost_grid=grid,
        show_colorbar=False,
        save_path=movement_path,
    )
    assert movement_path.exists()
    assert movement_path.stat().st_size > 0

    geodesic_path = tmp_path / "tmaze_geodesic_distance.png"
    result = task.show_geodesic_distance_matrix(
        show=False,
        colorbar=False,
        save_path=geodesic_path,
    )
    assert geodesic_path.exists()
    assert geodesic_path.stat().st_size > 0
    assert result.distances.shape[0] == result.accessible_indices.shape[0]
    assert np.all(np.isfinite(result.distances.diagonal()))

    normalised_matrix = task._prepare_geodesic_plot_matrix(
        result.distances, normalize=True
    )
    finite_mask = np.isfinite(result.distances)
    if finite_mask.any():
        assert pytest.approx(np.nanmax(normalised_matrix[finite_mask]), rel=1e-9) == 1.0

    idx = task.get_geodesic_index_by_pos(task.start_pos)
    assert idx is not None


def test_geodesic_handles_no_accessible_cells(tmp_path):
    mpl_cache = tmp_path / "mpl-no-access"
    mpl_cache.mkdir(parents=True, exist_ok=True)

    task = ClosedLoopNavigationTask(
        boundary=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        holes=[
            [
                [0.05, 0.05],
                [0.95, 0.05],
                [0.95, 0.95],
                [0.05, 0.95],
            ]
        ],
        dt=0.01,
    )

    task.set_grid_resolution(0.5, 0.5)
    grid = task.build_movement_cost_grid()
    assert not grid.accessible_mask.any()

    result = task.compute_geodesic_distance_matrix()
    assert result.distances.shape == (0, 0)
    assert result.accessible_indices.size == 0


def test_geodesic_handles_single_accessible_cell(tmp_path):
    task = ClosedLoopNavigationTask(
        boundary=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        holes=None,
        dt=0.01,
    )

    task.set_grid_resolution(1.0, 1.0)
    grid = task.build_movement_cost_grid()
    assert int(grid.accessible_mask.sum()) == 1

    result = task.compute_geodesic_distance_matrix()
    assert result.distances.shape == (1, 1)
    assert np.allclose(result.distances, 0.0)


def test_geodesic_with_walls_and_holes():
    task = ClosedLoopNavigationTask(
        boundary=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        walls=[
            [
                [0.0, 0.5],
                [0.5, 0.5],
            ]
        ],
        holes=[
            [
                [0.5, 0.0],
                [1.0, 0.0],
                [1.0, 0.5],
                [0.5, 0.5],
            ]
        ],
        dt=0.01,
    )

    dx = dy = 0.25
    task.set_grid_resolution(dx, dy)
    grid = task.build_movement_cost_grid()
    assert grid.costs.shape == (4, 4)
    blocked = int((grid.costs == INT32_MAX).sum())
    assert blocked >= 1

    result = task.compute_geodesic_distance_matrix()
    accessible = result.accessible_indices.shape[0]
    assert accessible >= 2
    assert np.allclose(result.distances, result.distances.T)
    assert np.all(np.isfinite(np.diag(result.distances)))
    assert np.any((result.distances > 0) & np.isfinite(result.distances))


def test_geodesic_cache_and_default_resolution():
    task = ClosedLoopNavigationTask(dt=0.01, grid_dx=0.2, grid_dy=0.3)

    grid = task.build_movement_cost_grid()
    assert pytest.approx(grid.dx) == 0.2
    assert pytest.approx(grid.dy) == 0.3

    result = task.compute_geodesic_distance_matrix()
    assert result.cost_grid is grid

    # Second call should hit the cache and return the same object
    again = task.compute_geodesic_distance_matrix()
    assert again is result

    # get_geodesic_index_by_pos without explicit dx/dy should succeed
    idx = task.get_geodesic_index_by_pos(task.start_pos)
    assert idx is not None
