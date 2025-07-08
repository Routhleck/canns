from dataclasses import dataclass

import brainstate
import brainunit as u
import numpy as np
import ratinabox
from ratinabox.Agent import Agent
from ratinabox.Environment import Environment
from tqdm import tqdm

from ._base import BaseTask

__all__ = [
    "map2pi",
    "PathIntegrationTask"
]

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

class PathIntegrationTask(BaseTask):
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
        duration=20.,
        dt=None,
        start_pos=(2.5, 2.5),
        progress_bar=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # --- task settings ---
        # time settings
        self.duration = duration
        self.dt = dt if dt is not None else brainstate.environ.get_dt()
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
        ratinabox.figure_directory = 'figures'

        self.env = Environment(params={"aspect": self.aspect, "scale": self.height})

        self.agent = Agent(
            Environment=self.env,
            params={
                "speed_mean": self.speed_mean,
                "speed_std": self.speed_std,
            }
        )
        self.agent.pos = np.array(start_pos)
        self.agent.dt = self.dt

        self.progress_bar = progress_bar


    def generate_trajectory(self) -> TrajectoryData:
        """Generates the inputs for the agent based on its current position."""

        for _ in tqdm(range(int(self.duration / self.dt)), disable=not self.progress_bar):
            self.agent.update(dt=self.dt)

        position = self.agent.history['pos']
        velocity = np.array(self.agent.history['vel'])
        speed = np.linalg.norm(velocity, axis=1)
        hd_angle = np.where(speed == 0, 0,
                            np.angle(velocity[:, 0] + velocity[:, 1] * 1j))
        rot_vel = map2pi(np.diff(hd_angle))

        return TrajectoryData(
            position=position,
            velocity=velocity,
            speed=speed,
            hd_angle=hd_angle,
            rot_vel=rot_vel
        )


def map2pi(a):
    """
    Maps an angle 'a' to the interval [-pi, pi] using the modulo operator.

    Args:
        a: The input angle in radians.

    Returns:
        The angle mapped to the interval [-pi, pi].
    """
    # Normalize to [0, 2*pi]
    b = np.fmod(a + u.math.pi, 2 * u.math.pi)
    # Map to [-pi, pi]
    c = b - np.pi
    return c
