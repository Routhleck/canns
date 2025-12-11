"""Visualise a closed-loop navigation scene with walls and holes.

This script constructs a moderately complex environment, overlays the
movement-cost grid on the agent view, and renders the geodesic distance
matrix.  The resulting figures are written to ``figures/closed_loop_complex``.

Usage (from repository root)::

    uv run python examples/cann/closed_loop_complex_environment.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

from canns.task.closed_loop_navigation import ClosedLoopNavigationTask


def build_demo_task() -> ClosedLoopNavigationTask:
    """Return a ``ClosedLoopNavigationTask`` with walls and holes."""

    boundary = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
    ]

    walls = [
        [[0.5, 0.0], [0.5, 1.0]],
        [[0.5, 1.0], [1.0, 1.0]],
        [[1.5, 2.0], [1.5, 1.0]],
        [[1.5, 1.0], [1.0, 1.0]],
        [[0.2, 0.2], [1.6, 1.6]],
    ]

    holes = [
        [
            [0.7, 0.7],
            [1.3, 0.7],
            [1.3, 1.3],
            [0.7, 1.3],
        ],
        [
            [0.0, 1.25],
            [0.8, 1.25],
            [0.8, 2.0],
            [0.0, 2.0],
        ],
    ]

    return ClosedLoopNavigationTask(
        boundary=boundary,
        walls=walls,
        holes=holes,
        dt=0.01,
    )


def main(output_dir: str = "figures/closed_loop_complex", dx: float = 0.1, dy: float = 0.1) -> None:
    matplotlib.use("Agg")

    task = build_demo_task()
    task.set_grid_resolution(dx, dy)
    task.start_pos = (0.1, 0.1)  # type: ignore[assignment]
    task.step_by_pos((0.1, 0.1))
    grid = task.build_movement_cost_grid()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    movement_path = output_path / "movement_cost_overlay.png"
    task.show_data(
        show=False,
        overlay_movement_cost=True,
        cost_grid=grid,
        show_colorbar=False,
        cost_legend_loc="upper right",
        save_path=movement_path,
    )

    geodesic_path = output_path / "geodesic_distance_matrix.png"
    task.show_geodesic_distance_matrix(
        show=False,
        normalize=True,
        save_path=geodesic_path,
    )

    accessible = int(grid.accessible_mask.sum())
    total_cells = grid.costs.size
    print(f"Movement-cost overlay saved to {movement_path}")
    print(f"Geodesic distance heatmap saved to {geodesic_path}")
    print(f"Accessible cells: {accessible}/{total_cells}")


if __name__ == "__main__":
    main()
