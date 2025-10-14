import os

import numpy as np
import pytest

from canns.task.closed_loop_navigation import (
    INT32_MAX,
    ClosedLoopNavigationTask,
    TMazeClosedLoopNavigationTask,
)


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

    grid = task.build_movement_cost_grid(dx, dy)

    assert grid.costs.dtype == np.int32
    assert np.any(grid.costs == 1)
    assert np.any(grid.costs == INT32_MAX)

    fig, ax = plt.subplots()
    task._plot_movement_cost_grid(ax, grid, add_colorbar=True)
    labels = {text.get_text() for text in ax.texts}
    assert "1" in labels
    assert "MAX" in labels
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
        dx,
        dy,
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

    grid = task.build_movement_cost_grid(0.5, 0.5)
    assert not grid.accessible_mask.any()

    result = task.compute_geodesic_distance_matrix(0.5, 0.5)
    assert result.distances.shape == (0, 0)
    assert result.accessible_indices.size == 0


def test_geodesic_handles_single_accessible_cell(tmp_path):
    task = ClosedLoopNavigationTask(
        boundary=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        holes=None,
        dt=0.01,
    )

    grid = task.build_movement_cost_grid(1.0, 1.0)
    assert int(grid.accessible_mask.sum()) == 1

    result = task.compute_geodesic_distance_matrix(1.0, 1.0)
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

    grid = task.build_movement_cost_grid(0.5, 0.5)
    assert grid.costs.shape == (2, 2)
    blocked = int((grid.costs == INT32_MAX).sum())
    assert blocked >= 1

    result = task.compute_geodesic_distance_matrix(0.5, 0.5)
    accessible = result.accessible_indices.shape[0]
    assert accessible >= 2
    assert np.allclose(result.distances, result.distances.T)
    assert np.all(np.isfinite(np.diag(result.distances)))
    assert np.any((result.distances > 0) & np.isfinite(result.distances))
