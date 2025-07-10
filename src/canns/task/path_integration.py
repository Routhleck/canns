from dataclasses import dataclass

import brainstate
import brainunit as u
import numpy as np
import ratinabox
from matplotlib import pyplot as plt
from ratinabox.Agent import Agent
from ratinabox.Environment import Environment
from tqdm import tqdm

from ._base import Task

__all__ = ["map2pi", "PathIntegrationTask"]


def map2pi(a):
    b = u.math.where(a > np.pi, a - np.pi * 2, a)
    c = u.math.where(b < -np.pi, b + np.pi * 2, b)
    return c


@dataclass
class TrajectoryData:
    """
    A dataclass to hold the inputs for the path integration task.
    It contains the position, velocity, speed, heading angle, and rotational velocity of the agent.
    """

    position: np.ndarray
    velocity: np.ndarray
    speed: np.ndarray
    hd_angle: np.ndarray
    rot_vel: np.ndarray


class PathIntegrationTask(Task):
    """
    A base class for path integration tasks, inheriting from BaseTask.
    This class is intended to be extended for specific path integration tasks.
    """

    def __init__(
        self,
        width=5,
        height=5,
        speed_mean=0.04,
        speed_std=0.016,
        duration=20.0,
        dt=None,
        start_pos=(2.5, 2.5),
        progress_bar=True,
    ):
        super().__init__()

        # --- task settings ---
        # time settings
        self.duration = duration
        self.dt = dt if dt is not None else brainstate.environ.get_dt()
        self.total_steps = int(self.duration / self.dt)
        # environment settings
        self.width = width
        self.height = height
        self.aspect = width / height
        # agent settings
        self.speed_mean = speed_mean
        self.speed_std = speed_std
        self.start_pos = start_pos

        # ratinabox settings
        ratinabox.stylize_plots()
        ratinabox.autosave_plots = False
        ratinabox.figure_directory = "figures"

        self.env = Environment(params={"aspect": self.aspect, "scale": self.height})

        self.agent = Agent(
            Environment=self.env,
            params={
                "speed_mean": self.speed_mean,
                "speed_std": self.speed_std,
            },
        )
        self.agent.pos = np.array(start_pos)
        self.agent.dt = self.dt

        self.progress_bar = progress_bar

    def reset(self):
        """
        Resets the agent's position to the starting position.
        """
        self.agent = Agent(
            Environment=self.env,
            params={
                "speed_mean": self.speed_mean,
                "speed_std": self.speed_std,
            },
        )
        self.agent.pos = np.array(self.start_pos)

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
        hd_angle = np.where(speed == 0, 0, np.angle(velocity[:, 0] + velocity[:, 1] * 1j))
        rot_vel = np.zeros_like(hd_angle)
        rot_vel[1:] = map2pi(np.diff(hd_angle))

        self.data = TrajectoryData(
            position=position, velocity=velocity, speed=speed, hd_angle=hd_angle, rot_vel=rot_vel
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
        self.agent.plot_trajectory(
            t_start=0, t_end=self.total_steps, fig=fig, ax=ax, color="changing"
        )
        plt.savefig(save_path) if save_path else None
        plt.show() if show else None

    def get_empty_trajectory(self) -> TrajectoryData:
        """
        Returns an empty trajectory data structure with the same shape as the generated trajectory.
        This is useful for initializing the trajectory data structure without any actual data.
        """
        return TrajectoryData(
            position=np.zeros((self.total_steps, 2)),
            velocity=np.zeros((self.total_steps, 2)),
            speed=np.zeros(self.total_steps),
            hd_angle=np.zeros(self.total_steps),
            rot_vel=np.zeros(self.total_steps),
        )
