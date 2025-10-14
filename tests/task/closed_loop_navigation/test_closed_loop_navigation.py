import os

import numpy as np

from canns.task.closed_loop_navigation import (
    INT32_MAX,
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

    task.show_data(
        show=False,
        overlay_movement_cost=True,
        cost_grid=grid,
        show_colorbar=False,
        save_path="tmaze_movement_cost.png",
    )

    result = task.show_geodesic_distance_matrix(
        dx,
        dy,
        show=False,
        colorbar=False,
        save_path="tmaze_geodesic_distance.png",
    )
    assert result.distances.shape[0] == result.accessible_indices.shape[0]
    assert np.all(np.isfinite(result.distances.diagonal()))
