import numpy as np

from .navigation_base import BaseNavigationTask
from .open_loop_navigation import OpenLoopNavigationData

__all__ = [
    "ClosedLoopNavigationTask",
    "TMazeClosedLoopNavigationTask",
    "TMazeRecessClosedLoopNavigationTask",
]


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


class TMazeClosedLoopNavigationTask(ClosedLoopNavigationTask):
    """
    Closed-loop navigation task in a T-maze environment.

    This subclass configures the environment with a T-maze boundary, which is useful
    for studying decision-making and spatial navigation in a controlled setting.
    """

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
        """
        Initialize T-maze closed-loop navigation task.

        Args:
            w: Width of the corridor (default: 0.3)
            l_s: Length of the stem (default: 1.0)
            l_arm: Length of each arm (default: 0.75)
            t: Thickness of the walls (default: 0.3)
            start_pos: Starting position of the agent (default: (0.0, 0.15))
            dt: Time step (default: None, uses bm.get_dt())
            **kwargs: Additional keyword arguments passed to ClosedLoopNavigationTask
        """
        hw = w / 2

        # Build simple T-maze boundary (8 vertices)
        boundary = [
            [-hw, 0.0],  # Bottom left of stem
            [-hw, l_s],  # Top left of stem
            [-l_arm, l_s],  # Inner edge of left arm
            [-l_arm, l_s + t],  # Outer edge of left arm
            [l_arm, l_s + t],  # Outer edge of right arm
            [l_arm, l_s],  # Inner edge of right arm
            [hw, l_s],  # Top right of stem
            [hw, 0.0],  # Bottom right of stem
        ]

        super().__init__(
            start_pos=start_pos,
            boundary=boundary,
            dt=dt,
            **kwargs,
        )


class TMazeRecessClosedLoopNavigationTask(TMazeClosedLoopNavigationTask):
    """
    Closed-loop navigation task in a T-maze environment with recesses at stem-arm junctions.

    This variant adds small rectangular indentations at the T-junction, creating
    additional spatial features that may be useful for studying spatial navigation
    and decision-making.
    """

    def __init__(
        self,
        w=0.3,  # corridor width
        l_s=1.0,  # stem length
        l_arm=0.75,  # arm length
        t=0.3,  # wall thickness
        recess_width=None,  # width of recesses at stem-arm junctions (default: t/4)
        recess_depth=None,  # depth of recesses extending downward (default: t/4)
        start_pos=(0.0, 0.15),
        dt=None,
        **kwargs,
    ):
        """
        Initialize T-maze with recesses closed-loop navigation task.

        Args:
            w: Width of the corridor (default: 0.3)
            l_s: Length of the stem (default: 1.0)
            l_arm: Length of each arm (default: 0.75)
            t: Thickness of the walls (default: 0.3)
            recess_width: Width of recesses at stem-arm junctions (default: t/4)
            recess_depth: Depth of recesses extending downward (default: t/4)
            start_pos: Starting position of the agent (default: (0.0, 0.15))
            dt: Time step (default: None, uses bm.get_dt())
            **kwargs: Additional keyword arguments passed to ClosedLoopNavigationTask
        """
        hw = w / 2

        # Set default recess dimensions
        if recess_width is None:
            recess_width = t / 4
        if recess_depth is None:
            recess_depth = t / 4

        # Build boundary with recesses at stem-arm junctions (12 vertices)
        # Remove [-hw, l_s] and [hw, l_s], add recess points instead
        boundary = [
            [-hw, 0.0],  # 0: Bottom left of stem
            [-hw, l_s - recess_depth],  # 1: Left side of stem, bottom of left recess
            [
                -hw - recess_width,
                l_s - recess_depth,
            ],  # 2: Outer left corner of left recess (bottom)
            [-hw - recess_width, l_s],  # 3: Outer left corner of left recess (top)
            [-l_arm, l_s],  # 4: Inner edge of left arm
            [-l_arm, l_s + t],  # 5: Outer edge of left arm (top)
            [l_arm, l_s + t],  # 6: Outer edge of right arm (top)
            [l_arm, l_s],  # 7: Inner edge of right arm
            [hw + recess_width, l_s],  # 8: Outer right corner of right recess (top)
            [
                hw + recess_width,
                l_s - recess_depth,
            ],  # 9: Outer right corner of right recess (bottom)
            [hw, l_s - recess_depth],  # 10: Right side of stem, bottom of right recess
            [hw, 0.0],  # 11: Bottom right of stem
        ]

        # Skip parent's __init__ and call ClosedLoopNavigationTask directly
        ClosedLoopNavigationTask.__init__(
            self,
            start_pos=start_pos,
            boundary=boundary,
            dt=dt,
            **kwargs,
        )
