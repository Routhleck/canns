import copy
from dataclasses import dataclass

import brainpy.math as bm
import numpy as np
import seaborn as sns
from canns_lib.spatial import Agent, Environment
from matplotlib import pyplot as plt
from tqdm import tqdm

from .navigation_base import BaseNavigationTask

__all__ = [
    "map2pi",
    "OpenLoopNavigationTask",
    "OpenLoopNavigationData",
    "TMazeOpenLoopNavigationTask",
    "TMazeRecessOpenLoopNavigationTask",
]


def map2pi(a):
    b = bm.where(a > np.pi, a - np.pi * 2, a)
    c = bm.where(b < -np.pi, b + np.pi * 2, b)
    return c


@dataclass
class OpenLoopNavigationData:
    """
    Container for the inputs recorded during the open-loop navigation task.
    It contains the position, velocity, speed, movement direction, head direction,
    and rotational velocity of the agent.

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


class OpenLoopNavigationTask(BaseNavigationTask):
    """
    Open-loop spatial navigation task that synthesises trajectories without
    incorporating real-time feedback from a controller.
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
        dx=0.01,
        grid_dx: float | None = None,  # Grid resolution for geodesic computation
        grid_dy: float | None = None,  # Grid resolution for geodesic computation
        boundary=None,
        # coordinates [[x0,y0],[x1,y1],...] of the corners of a 2D polygon bounding the Env (if None, Env defaults to rectangular). Corners must be ordered clockwise or anticlockwise, and the polygon must be a 'simple polygon' (no holes, doesn't self-intersect).
        walls=None,
        # a list of loose walls within the environment. Each wall in the list can be defined by it's start and end coords [[x0,y0],[x1,y1]]. You can also manually add walls after init using Env.add_wall() (preferred).
        holes=None,
        # coordinates [[[x0,y0],[x1,y1],...],...] of corners of any holes inside the Env. These must be entirely inside the environment and not intersect one another. Corners must be ordered clockwise or anticlockwise. holes has 1-dimension more than boundary since there can be multiple holes
        objects=None,
        # a list of objects within the environment. Each object is defined by its position [[x0,y0],[x1,y1],...] for 2D environments and [[x0],[x1],...] for 1D environments. By default all objects are type 0, alternatively you can manually add objects after init using Env.add_object(object, type) (preferred).
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
            initial_head_direction=initial_head_direction,
            thigmotaxis=thigmotaxis,
            wall_repel_distance=wall_repel_distance,
            wall_repel_strength=wall_repel_strength,
            data_class=OpenLoopNavigationData,
        )

        # Open-loop specific settings
        self.duration = duration
        self.total_steps = int(self.duration / self.dt)
        self.run_steps = np.arange(self.total_steps)
        self.progress_bar = progress_bar

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
        self.agent = Agent(Environment=self.env, params=copy.deepcopy(self.agent_params))
        self.agent.pos = np.array(self.start_pos)
        self.agent.dt = self.dt
        self._apply_initial_head_direction()

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

        self.data = OpenLoopNavigationData(
            position=position,
            velocity=velocity,
            speed=speed,
            movement_direction=movement_direction,
            hd_angle=hd_angle,
            rot_vel=rot_vel,
        )

    def show_trajectory_analysis(
        self,
        show: bool = True,
        save_path: str | None = None,
        figsize: tuple[int, int] = (12, 3),
        smooth_window: int = 50,
        **kwargs,
    ):
        """
        Display comprehensive trajectory analysis including position, speed, and direction changes.

        Args:
            show: Whether to display the plot
            save_path: Path to save the figure
            figsize: Figure size (width, height)
            smooth_window: Window size for smoothing speed and direction plots (set to 0 to disable smoothing)
            **kwargs: Additional matplotlib parameters
        """
        if self.data is None:
            raise ValueError("No trajectory data available. Please call get_data() first.")

        # Ensure theta sweep data is calculated if needed
        if self.data.ang_velocity is None:
            self.calculate_theta_sweep_data()

        def smooth_data(data, window_size):
            """Apply moving average smoothing to data"""
            if window_size <= 1 or len(data) < window_size:
                return data

            # Use convolution for moving average
            kernel = np.ones(window_size) / window_size

            # Handle NaN values by padding
            valid_mask = ~np.isnan(data)
            if not np.any(valid_mask):
                return data

            # For simplicity, use simple moving average
            smoothed = np.convolve(data, kernel, mode="same")

            # Handle edges by using smaller windows
            half_window = window_size // 2
            for i in range(half_window):
                start_idx = max(0, i - half_window)
                end_idx = min(len(data), i + half_window + 1)
                smoothed[i] = np.nanmean(data[start_idx:end_idx])

                start_idx = max(0, len(data) - i - half_window - 1)
                end_idx = min(len(data), len(data) - i + half_window)
                smoothed[len(data) - i - 1] = np.nanmean(data[start_idx:end_idx])

            return smoothed

        def smooth_circular_data(angles, window_size):
            """Apply smoothing to circular data (angles) handling wrapping"""
            if window_size <= 1 or len(angles) < window_size:
                return angles

            # Convert to complex representation for circular averaging
            complex_angles = np.exp(1j * angles)

            # Smooth in complex domain
            smoothed_complex = smooth_data(complex_angles.real, window_size) + 1j * smooth_data(
                complex_angles.imag, window_size
            )

            # Convert back to angles
            smoothed_angles = np.angle(smoothed_complex)

            return smoothed_angles

        fig, axs = plt.subplots(1, 3, figsize=figsize, width_ratios=[1, 2, 2])

        try:
            # Plot 1: Trajectory with time-based coloring
            ax = axs[0]

            # Create time-based color mapping
            time_array = self.run_steps * self.dt
            scatter = ax.scatter(
                self.data.position[:, 0],
                self.data.position[:, 1],
                c=time_array,
                cmap="viridis",
                s=1,
                alpha=0.7,
                **kwargs,
            )

            # Add start and end markers
            ax.plot(
                self.data.position[0, 0],
                self.data.position[0, 1],
                "go",
                markersize=8,
                label="Start",
            )
            ax.plot(
                self.data.position[-1, 0],
                self.data.position[-1, 1],
                "ro",
                markersize=8,
                label="End",
            )

            # Add colorbar for time
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Time (s)", fontsize=10)

            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([0, self.width])
            ax.set_yticks([0, self.height])
            ax.set_title("Animal Trajectory")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.legend(fontsize=8, loc="upper right")

            # Plot 2: Speed over time
            ax = axs[1]
            sns.despine(ax=ax)
            time_array = self.run_steps * self.dt

            # Plot original data (if smoothing is enabled)
            if smooth_window > 1:
                ax.plot(
                    time_array,
                    self.data.speed,
                    lw=0.5,
                    color="#009FB9",
                    alpha=0.3,
                    label="Raw",
                    **kwargs,
                )
                # Plot smoothed data
                smoothed_speed = smooth_data(self.data.speed, smooth_window)
                ax.plot(
                    time_array, smoothed_speed, lw=2, color="#009FB9", label="Smoothed", **kwargs
                )
                ax.legend(fontsize=8)
            else:
                ax.plot(time_array, self.data.speed, lw=1, color="#009FB9", **kwargs)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (m/s)")
            title = f"Movement Speed{f' (smoothed, window={smooth_window})' if smooth_window > 1 else ''}"
            ax.set_title(title)

            # Plot 3: Direction over time (handle wrapping)
            ax = axs[2]
            sns.despine(ax=ax)

            # Handle direction wrapping for plotting
            direction = self.data.hd_angle

            if smooth_window > 1:
                # Plot original data
                jumps = np.where(np.abs(np.diff(direction)) > np.pi)[0]
                direction_plot = direction.copy()
                direction_plot[jumps + 1] = np.nan
                ax.plot(
                    time_array,
                    direction_plot,
                    lw=0.5,
                    color="#009FB9",
                    alpha=0.3,
                    label="Raw",
                    **kwargs,
                )

                # Plot smoothed data
                smoothed_direction = smooth_circular_data(direction, smooth_window)
                jumps_smooth = np.where(np.abs(np.diff(smoothed_direction)) > np.pi)[0]
                smoothed_direction_plot = smoothed_direction.copy()
                smoothed_direction_plot[jumps_smooth + 1] = np.nan
                ax.plot(
                    time_array,
                    smoothed_direction_plot,
                    lw=2,
                    color="#009FB9",
                    label="Smoothed",
                    **kwargs,
                )
                ax.legend(fontsize=8)
            else:
                jumps = np.where(np.abs(np.diff(direction)) > np.pi)[0]
                direction_plot = direction.copy()
                direction_plot[jumps + 1] = np.nan
                ax.plot(time_array, direction_plot, lw=1, color="#009FB9", **kwargs)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Direction (rad)")
            title = f"Head Direction{f' (smoothed, window={smooth_window})' if smooth_window > 1 else ''}"
            ax.set_title(title)

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

    def get_empty_trajectory(self) -> OpenLoopNavigationData:
        """
        Returns an empty trajectory data structure with the same shape as the generated trajectory.
        This is useful for initializing the trajectory data structure without any actual data.
        """
        return OpenLoopNavigationData(
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

    def import_data(
        self,
        position_data: np.ndarray,
        times: np.ndarray = None,
        dt: float = None,
        head_direction: np.ndarray = None,
        initial_pos: np.ndarray = None,
    ) -> None:
        """
        Import external position coordinates and calculate derived features.

        This method allows importing external trajectory data (e.g., from experimental
        recordings or other simulations) instead of using the built-in random motion model.
        The imported data will be processed to calculate velocity, speed, movement direction,
        head direction, and rotational velocity.

        Args:
            position_data (np.ndarray): Array of position coordinates with shape (n_steps, 2)
                                       for 2D trajectories or (n_steps, 1) for 1D trajectories.
            times (np.ndarray, optional): Array of time points corresponding to position_data.
                                         If None, uniform time steps with dt will be assumed.
            dt (float, optional): Time step between consecutive positions. If None, uses
                                 self.dt. Required if times is None.
            head_direction (np.ndarray, optional): Array of head direction angles in radians
                                                  with shape (n_steps,). If None, head direction
                                                  will be derived from movement direction.
            initial_pos (np.ndarray, optional): Initial position for the agent. If None,
                                               uses the first position from position_data.

        Raises:
            ValueError: If position_data has invalid dimensions or if required parameters
                       are missing.

        Example:
            ```python
            # Import experimental trajectory data
            positions = np.array([[0, 0], [0.1, 0.05], [0.2, 0.1], ...])  # shape: (n_steps, 2)
            times = np.array([0, 0.1, 0.2, ...])  # shape: (n_steps,)

            task = OpenLoopNavigationTask(...)
            task.import_data(position_data=positions, times=times)

            # Or with uniform time steps
            task.import_data(position_data=positions, dt=0.1)
            ```
        """
        # Input validation
        if not isinstance(position_data, np.ndarray):
            raise ValueError("position_data must be a numpy array")

        if position_data.ndim != 2:
            raise ValueError("position_data must be a 2D array with shape (n_steps, n_dims)")

        n_steps, n_dims = position_data.shape

        if n_dims not in [1, 2]:
            raise ValueError("position_data must have 1 or 2 spatial dimensions")

        expected_dims = 2 if self.dimensionality == "2D" else 1
        if expected_dims != n_dims:
            raise ValueError(
                f"position_data dimensions ({n_dims}) must match task dimensionality ({self.dimensionality})"
            )

        if n_steps < 2:
            raise ValueError("position_data must contain at least 2 time steps")

        # Handle time array
        if times is None:
            if dt is None:
                dt = self.dt
            times = np.arange(n_steps) * dt
        else:
            if not isinstance(times, np.ndarray):
                raise ValueError("times must be a numpy array")
            if times.shape[0] != n_steps:
                raise ValueError("times array length must match position_data length")
            if dt is None:
                dt = np.mean(np.diff(times)) if len(times) > 1 else self.dt

        # Set up environment and agent if not already done (but don't use RatInABox processing)
        if not hasattr(self, "env") or self.env is None:
            self.env = Environment(params=self.env_params)

        if not hasattr(self, "agent") or self.agent is None:
            self.agent = Agent(Environment=self.env, params=copy.deepcopy(self.agent_params))

        # Set initial position
        if initial_pos is None:
            initial_pos = position_data[0]
        self.agent.pos = np.array(initial_pos)
        self.agent.dt = dt

        # Note: We skip agent.import_trajectory() to avoid RatInABox's internal processing
        # which can modify our clean imported data

        # Calculate derived features
        position = position_data.copy()

        # Calculate velocity from position differences
        velocity = np.zeros_like(position)
        if n_steps > 1:
            time_diffs = np.diff(times)
            pos_diffs = np.diff(position, axis=0)
            velocity[1:] = pos_diffs / time_diffs[:, np.newaxis]
            velocity[0] = velocity[1]  # Use second velocity for first step

        # Calculate speed
        speed = np.linalg.norm(velocity, axis=1)

        # Calculate movement direction from velocity
        if n_dims == 2:
            movement_direction = np.where(
                speed == 0, 0, np.angle(velocity[:, 0] + velocity[:, 1] * 1j)
            )
        else:  # 1D case
            movement_direction = np.where(speed == 0, 0, np.where(velocity[:, 0] >= 0, 0, np.pi))

        # Handle head direction
        if head_direction is None:
            # Use movement direction as head direction if not provided
            hd_angle = movement_direction.copy()
            # For stationary points, maintain previous head direction
            for i in range(1, len(hd_angle)):
                if speed[i] == 0:
                    hd_angle[i] = hd_angle[i - 1]
        else:
            if not isinstance(head_direction, np.ndarray):
                raise ValueError("head_direction must be a numpy array")
            if head_direction.shape[0] != n_steps:
                raise ValueError("head_direction array length must match position_data length")
            hd_angle = head_direction.copy()

        # Calculate rotational velocity
        rot_vel = np.zeros_like(hd_angle)
        if n_steps > 1:
            rot_vel[1:] = map2pi(np.diff(hd_angle))

        # Update total_steps to match imported data
        self.total_steps = n_steps
        self.run_steps = np.arange(self.total_steps)

        # Create OpenLoopNavigationData object
        self.data = OpenLoopNavigationData(
            position=position,
            velocity=velocity,
            speed=speed,
            movement_direction=movement_direction,
            hd_angle=hd_angle,
            rot_vel=rot_vel,
        )

        print(f"Successfully imported trajectory data with {n_steps} time steps")
        print(f"Spatial dimensions: {n_dims}D")
        print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} s")
        print(f"Mean speed: {np.mean(speed):.3f} units/s")


class TMazeOpenLoopNavigationTask(OpenLoopNavigationTask):
    """
    Open-loop navigation task in a T-maze environment.

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
        duration=20.0,
        dt=None,
        **kwargs,
    ):
        """
        Initialize T-maze open-loop navigation task.

        Args:
            w: Width of the corridor (default: 0.3)
            l_s: Length of the stem (default: 1.0)
            l_arm: Length of each arm (default: 0.75)
            t: Thickness of the walls (default: 0.3)
            start_pos: Starting position of the agent (default: (0.0, 0.15))
            duration: Duration of the trajectory in seconds (default: 20.0)
            dt: Time step (default: None, uses bm.get_dt())
            **kwargs: Additional keyword arguments passed to OpenLoopNavigationTask
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
            duration=duration,
            dt=dt,
            **kwargs,
        )


class TMazeRecessOpenLoopNavigationTask(TMazeOpenLoopNavigationTask):
    """
    Open-loop navigation task in a T-maze environment with recesses at stem-arm junctions.

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
        duration=20.0,
        dt=None,
        **kwargs,
    ):
        """
        Initialize T-maze with recesses open-loop navigation task.

        Args:
            w: Width of the corridor (default: 0.3)
            l_s: Length of the stem (default: 1.0)
            l_arm: Length of each arm (default: 0.75)
            t: Thickness of the walls (default: 0.3)
            recess_width: Width of recesses at stem-arm junctions (default: t/4)
            recess_depth: Depth of recesses extending downward (default: t/4)
            start_pos: Starting position of the agent (default: (0.0, 0.15))
            duration: Duration of the trajectory in seconds (default: 20.0)
            dt: Time step (default: None, uses bm.get_dt())
            **kwargs: Additional keyword arguments passed to OpenLoopNavigationTask
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

        # Skip parent's __init__ and call OpenLoopNavigationTask directly
        OpenLoopNavigationTask.__init__(
            self,
            start_pos=start_pos,
            boundary=boundary,
            duration=duration,
            dt=dt,
            **kwargs,
        )
