import copy
from dataclasses import dataclass

import brainstate
import brainunit as u
import numpy as np
import ratinabox
import seaborn as sns
from matplotlib import pyplot as plt
from ratinabox.Agent import Agent
from ratinabox.Environment import Environment
from tqdm import tqdm

from ._base import Task

__all__ = ["map2pi", "SpatialNavigationTask"]


def map2pi(a):
    b = u.math.where(a > np.pi, a - np.pi * 2, a)
    c = u.math.where(b < -np.pi, b + np.pi * 2, b)
    return c


@dataclass
class SpatialNavigationData:
    """
    A dataclass to hold the inputs for the spatial navigation task.
    It contains the position, velocity, speed, movement direction, head direction, and rotational velocity of the agent.

    Additional fields for theta sweep analysis:
    - ang_velocity: Angular velocity calculated using unwrap method
    - linear_speed_gains: Normalized linear speed gains [0,1]
    - ang_speed_gains: Normalized angular speed gains [-1,1]
    """

    position: np.ndarray
    velocity: np.ndarray
    speed: np.ndarray
    movement_direction: np.ndarray  # Direction of movement (from velocity)
    hd_angle: np.ndarray  # Head direction (orientation the agent is facing)
    rot_vel: np.ndarray

    # Additional fields for theta sweep analysis
    ang_velocity: np.ndarray | None = None  # Angular velocity (unwrap method)
    linear_speed_gains: np.ndarray | None = None  # Normalized linear speed [0,1]
    ang_speed_gains: np.ndarray | None = None  # Normalized angular speed [-1,1]


class SpatialNavigationTask(Task):
    """
    A base class for spatial navigation tasks, inheriting from BaseTask.
    This class is intended to be extended for specific spatial navigation tasks.
    """

    def __init__(
        self,
        duration=20.0,
        start_pos=(2.5, 2.5),
        initial_head_direction=None,  # Initial head direction in radians (None for random)
        progress_bar=True,
        # environment parameters
        width=5,
        height=5,
        dimensionality="2D",
        boundary_conditions="solid",  # "solid" or "periodic"
        scale=None,
        dx=0.01,  # for show_data only
        boundary=None,  # coordinates [[x0,y0],[x1,y1],...] of the corners of a 2D polygon bounding the Env (if None, Env defaults to rectangular). Corners must be ordered clockwise or anticlockwise, and the polygon must be a 'simple polygon' (no holes, doesn't self-intersect).
        walls=None,  # a list of loose walls within the environment. Each wall in the list can be defined by it's start and end coords [[x0,y0],[x1,y1]]. You can also manually add walls after init using Env.add_wall() (preferred).
        holes=None,  # coordinates [[[x0,y0],[x1,y1],...],...] of corners of any holes inside the Env. These must be entirely inside the environment and not intersect one another. Corners must be ordered clockwise or anticlockwise. holes has 1-dimension more than boundary since there can be multiple holes
        objects=None,  # a list of objects within the environment. Each object is defined by its position [[x0,y0],[x1,y1],...] for 2D environments and [[x0],[x1],...] for 1D environments. By default all objects are type 0, alternatively you can manually add objects after init using Env.add_object(object, type) (preferred).
        # agent parameters
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
        super().__init__(data_class=SpatialNavigationData)

        # --- task settings ---
        # time settings
        self.duration = duration
        self.dt = dt if dt is not None else brainstate.environ.get_dt()
        self.total_steps = int(self.duration / self.dt)
        self.run_steps = np.arange(self.total_steps)
        # environment settings
        self.width = width
        self.height = height
        self.aspect = width / height
        self.dimensionality = str(dimensionality).upper()
        if self.dimensionality == "1D":
            raise NotImplementedError(
                "SpatialNavigationTask currently supports only 2D environments."
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
        self.initial_head_direction = initial_head_direction

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
        self.agent.dt = self.dt

        # Set initial movement direction if specified
        if self.initial_head_direction is not None:
            # Set initial velocity in the specified direction
            initial_speed = self.speed_mean
            initial_velocity = np.array(
                [
                    initial_speed * np.cos(self.initial_head_direction),
                    initial_speed * np.sin(self.initial_head_direction),
                ]
            )
            self.agent.velocity = initial_velocity
            # Also set head direction to match
            self.agent.head_direction = np.array(
                [np.cos(self.initial_head_direction), np.sin(self.initial_head_direction)]
            )

        self.progress_bar = progress_bar

    def calculate_theta_sweep_data(self):
        """
        Calculate additional fields needed for theta sweep analysis.
        This should be called after get_data() to add ang_velocity,
        linear_speed_gains, and ang_speed_gains to the data.
        """
        if self.data is None:
            raise ValueError("No trajectory data available. Please call get_data() first.")

        # Calculate angular velocity using unwrap method (more suitable for theta sweep)
        direction_unwrapped = np.unwrap(self.data.hd_angle)
        ang_velocity = np.diff(direction_unwrapped) / self.dt
        ang_velocity = np.insert(ang_velocity, 0, 0)  # Insert 0 for first time step

        # Calculate normalized speed gains
        linear_speed_gains = (
            self.data.speed / np.max(self.data.speed)
            if np.max(self.data.speed) > 0
            else np.zeros_like(self.data.speed)
        )
        ang_speed_gains = (
            ang_velocity / np.max(np.abs(ang_velocity))
            if np.max(np.abs(ang_velocity)) > 0
            else np.zeros_like(ang_velocity)
        )

        # Update the data object
        self.data.ang_velocity = ang_velocity
        self.data.linear_speed_gains = linear_speed_gains
        self.data.ang_speed_gains = ang_speed_gains

    def reset(self):
        """
        Resets the agent's position to the starting position.
        """
        self.agent_params["dt"] = self.dt
        self.agent = Agent(Environment=self.env, params=copy.deepcopy(self.agent_params))
        self.agent.pos = np.array(self.start_pos)
        self.agent.dt = self.dt

        # Set initial movement direction if specified
        if self.initial_head_direction is not None:
            # Set initial velocity in the specified direction
            initial_speed = self.speed_mean
            initial_velocity = np.array(
                [
                    initial_speed * np.cos(self.initial_head_direction),
                    initial_speed * np.sin(self.initial_head_direction),
                ]
            )
            self.agent.velocity = initial_velocity
            # Also set head direction to match
            self.agent.head_direction = np.array(
                [np.cos(self.initial_head_direction), np.sin(self.initial_head_direction)]
            )

    def get_data(self):
        """Generates the inputs for the agent based on its current position."""

        for _ in tqdm(
            range(self.total_steps),
            disable=not self.progress_bar,
            desc=f"<{type(self).__name__}>Generating Task data",
        ):
            self.agent.update(dt=self.dt)

        position = np.array(self.agent.history["pos"])
        velocity = np.array(self.agent.history["vel"])
        speed = np.linalg.norm(velocity, axis=1)

        # Movement direction (from velocity)
        movement_direction = np.where(speed == 0, 0, np.angle(velocity[:, 0] + velocity[:, 1] * 1j))

        # Head direction (from agent's orientation)
        head_direction_xy = np.array(self.agent.history["head_direction"])
        hd_angle = np.arctan2(head_direction_xy[:, 1], head_direction_xy[:, 0])

        rot_vel = np.zeros_like(hd_angle)
        rot_vel[1:] = map2pi(np.diff(hd_angle))

        self.data = SpatialNavigationData(
            position=position,
            velocity=velocity,
            speed=speed,
            movement_direction=movement_direction,
            hd_angle=hd_angle,
            rot_vel=rot_vel,
        )

    def show_data(
        self,
        show=True,
        save_path=None,
    ):
        """
        Displays the trajectory of the agent in the environment.
        """
        # self.reset()
        # self.generate_trajectory()
        if self.data is None:
            raise ValueError("No trajectory data available. Please generate the data first.")
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        try:
            self.agent.plot_trajectory(
                t_start=0, t_end=self.total_steps, fig=fig, ax=ax, color="changing"
            )
            plt.savefig(save_path) if save_path else None
            plt.show() if show else None
        finally:
            plt.close(fig)

    def show_trajectory_analysis(
        self,
        show: bool = True,
        save_path: str | None = None,
        figsize: tuple[int, int] = (12, 3),
        **kwargs,
    ):
        """
        Display comprehensive trajectory analysis including position, speed, and direction changes.

        Args:
            show: Whether to display the plot
            save_path: Path to save the figure
            figsize: Figure size (width, height)
            **kwargs: Additional matplotlib parameters
        """
        if self.data is None:
            raise ValueError("No trajectory data available. Please call get_data() first.")

        # Ensure theta sweep data is calculated if needed
        if self.data.ang_velocity is None:
            self.calculate_theta_sweep_data()

        fig, axs = plt.subplots(1, 3, figsize=figsize, width_ratios=[1, 2, 2])

        try:
            # Plot 1: Trajectory
            ax = axs[0]
            ax.plot(
                self.data.position[:, 0], self.data.position[:, 1], lw=1, color="black", **kwargs
            )
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([0, self.width])
            ax.set_yticks([0, self.height])
            ax.set_title("Animal Trajectory")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

            # Plot 2: Speed over time
            ax = axs[1]
            sns.despine(ax=ax)
            time_array = self.run_steps * self.dt
            ax.plot(time_array, self.data.speed, lw=1, color="#009FB9", **kwargs)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (m/s)")
            ax.set_title("Movement Speed")

            # Plot 3: Direction over time (handle wrapping)
            ax = axs[2]
            sns.despine(ax=ax)

            # Handle direction wrapping for plotting
            direction = self.data.hd_angle
            jumps = np.where(np.abs(np.diff(direction)) > np.pi)[0]
            direction_plot = direction.copy()
            direction_plot[jumps + 1] = np.nan

            ax.plot(time_array, direction_plot, lw=1, color="#009FB9", **kwargs)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Direction (rad)")
            ax.set_title("Head Direction")

            # Add y-tick labels for clarity
            ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Trajectory analysis saved to: {save_path}")

            if show:
                plt.show()

        finally:
            if not show:
                plt.close(fig)

    def get_empty_trajectory(self) -> SpatialNavigationData:
        """
        Returns an empty trajectory data structure with the same shape as the generated trajectory.
        This is useful for initializing the trajectory data structure without any actual data.
        """
        return SpatialNavigationData(
            position=np.zeros((self.total_steps, 2)),
            velocity=np.zeros((self.total_steps, 2)),
            speed=np.zeros(self.total_steps),
            movement_direction=np.zeros(self.total_steps),
            hd_angle=np.zeros(self.total_steps),
            rot_vel=np.zeros(self.total_steps),
            ang_velocity=np.zeros(self.total_steps),
            linear_speed_gains=np.zeros(self.total_steps),
            ang_speed_gains=np.zeros(self.total_steps),
        )
