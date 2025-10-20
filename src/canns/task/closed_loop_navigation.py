import numpy as np
from matplotlib import pyplot as plt

from .navigation_base import BaseNavigationTask, MovementCostGrid
from .open_loop_navigation import OpenLoopNavigationData


class ClosedLoopNavigationTask(BaseNavigationTask):
    """
    Closed-loop navigation task that incorporates real-time feedback from a controller.

    In this task, the agent's movement is controlled step-by-step by external commands
    rather than following a pre-generated trajectory.
    """

    def __init__(
        self,
        start_pos=(2.5, 2.5),
        # environment parameters
        width=5,
        height=5,
        dimensionality="2D",
        boundary_conditions="solid",  # "solid" or "periodic"
        scale=None,
        dx=0.01,
        grid_dx: float | None = None,
        grid_dy: float | None = None,
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
        super().__init__(
            start_pos=start_pos,
            width=width,
            height=height,
            dimensionality=dimensionality,
            boundary_conditions=boundary_conditions,
            scale=scale,
            dx=dx,
            grid_dx=grid_dx,
            grid_dy=grid_dy,
            boundary=boundary,
            walls=walls,
            holes=holes,
            objects=objects,
            dt=dt,
            speed_mean=speed_mean,
            speed_std=speed_std,
            speed_coherence_time=speed_coherence_time,
            rotational_velocity_coherence_time=rotational_velocity_coherence_time,
            rotational_velocity_std=rotational_velocity_std,
            head_direction_smoothing_timescale=head_direction_smoothing_timescale,
            thigmotaxis=thigmotaxis,
            wall_repel_distance=wall_repel_distance,
            wall_repel_strength=wall_repel_strength,
            data_class=OpenLoopNavigationData,
        )

        # Closed-loop specific settings
        self.total_steps = 1

        # Update agent with forced position
        self.agent.update(forced_next_position=self.agent.pos)

    def step_by_pos(self, new_pos):
        self.agent.update(forced_next_position=np.asarray(new_pos))

    def get_data(self):
        # TODO: should implement, but currently not used anywhere
        raise NotImplementedError("ClosedLoopNavigationTask does not have get_data method.")

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
            **kwargs,
        )
