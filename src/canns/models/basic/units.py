"""
Basic neural units for hierarchical models.

This module contains the fundamental building blocks used in hierarchical
path integration models, including recurrent and non-recurrent units.
"""

import brainstate
import brainunit as u
from brainstate.nn import exp_euler_step

from typing import Any
from ...task.path_integration import map2pi
from ._base import BasicModel
from ...typing import ArrayLike

__all__ = [
    "GaussRecUnits",
    "NonRecUnits",
]


class GaussRecUnits(BasicModel):
    """A model of recurrently connected units with Gaussian connectivity.

    This class implements a 1D continuous attractor neural network (CANN). The network
    maintains a stable "bump" of activity that can represent a continuous variable,
    such as heading direction. The connectivity between neurons is Gaussian, and the
    network dynamics include divisive normalization.

    Attributes:
        size (int): The number of neurons in the network.
        tau (float): The time constant for the synaptic input `u`.
        k (float): The inhibition strength for divisive normalization.
        a (float): The width of the Gaussian connection profile.
        noise_0 (float): The standard deviation of the Gaussian noise added to the system.
        z_min (float): The minimum value of the encoded feature space.
        z_max (float): The maximum value of the encoded feature space.
        z_range (float): The range of the feature space (z_max - z_min).
        x (brainunit.math.ndarray): The preferred feature values for each neuron.
        rho (float): The neural density (number of neurons per unit of feature space).
        dx (float): The stimulus density (feature space range per neuron).
        J (float): The final connection strength, scaled by J0.
        conn_mat (brainunit.math.ndarray): The connection matrix.
        r (brainstate.HiddenState): The firing rates of the neurons.
        u (brainstate.HiddenState): The synaptic inputs to the neurons.
        center (brainstate.State): The decoded center of the activity bump.
        input (brainstate.State): The external input to the network.
    """

    def __init__(
        self,
        size: int,
        tau: float = 1.0,
        k: float = 0.1,
        a: float = 0.5,
        noise_0: float = 0.0,
        J0: float = 4.0,
        z_min: float = -u.math.pi,
        z_max: float = u.math.pi,
        **kwargs,
    ):
        """Initialize the GaussRecUnits model.

        Args:
            size: The number of neurons in the network.
            tau: The time constant for the synaptic input `u`.
            k: The inhibition strength for divisive normalization.
            a: The width of the Gaussian connection profile.
            noise_0: The standard deviation of the Gaussian noise.
            J0: The baseline connection strength.
            z_min: The minimum value of the encoded feature space.
            z_max: The maximum value of the encoded feature space.
            **kwargs: Additional arguments passed to BasicModel.
        """
        super().__init__(size, **kwargs)

        # Model parameters
        self.size = size
        self.tau = tau
        self.k = k
        self.a = a
        self.noise_0 = noise_0

        # Feature space setup
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = u.math.linspace(z_min, z_max, size)
        self.rho = size / self.z_range
        self.dx = self.z_range / size

        # Connection strength
        self.J = J0 / (u.math.sqrt(2 * u.math.pi) * self.a * self.rho)

        # Initialize connectivity matrix
        self.conn_mat = self.make_conn()

        # Initialize states
        self.r = brainstate.HiddenState(u.math.zeros(self.varshape))
        self.u = brainstate.HiddenState(u.math.zeros(self.varshape))
        self.center = brainstate.State(0.0)
        self.input = brainstate.State(u.math.zeros(self.varshape))

    def dist(self, d: ArrayLike) -> ArrayLike:
        """Calculate the distance with periodic boundary conditions."""
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > self.z_range / 2, d - self.z_range, d)
        return d

    def make_conn(self) -> ArrayLike:
        """Create the connectivity matrix."""
        x_left = u.math.reshape(self.x, (-1, 1))
        x_right = u.math.repeat(self.x.reshape((1, -1)), len(self.x), axis=0)
        d = self.dist(x_left - x_right)
        return self.J * u.math.exp(-0.5 * u.math.square(d / self.a))

    def update(self) -> None:
        """Update the network state for one time step."""
        # Compute firing rates
        self.r.value = u.math.square(self.u.value)

        # Apply divisive normalization
        r_sum = u.math.sum(self.r.value)
        self.r.value = self.r.value / (1.0 + self.k * r_sum)

        # Compute synaptic input from recurrent connections
        Irec = u.math.dot(self.conn_mat, self.r.value)

        # Update synaptic input
        du = (-self.u.value + Irec + self.input.value) / self.tau
        if self.noise_0 > 0:
            du += self.noise_0 * u.math.random.randn(*self.u.value.shape)
        
        self.u.value = exp_euler_step(self.u.value, du)

        # Decode the center of activity
        self.center.value = self.get_center()

    def get_center(self) -> float:
        """Decode the population vector to get the center of activity."""
        exppos = u.math.exp(1j * self.x)
        self.center.value = u.math.angle(u.math.sum(self.r.value * exppos))
        return self.center.value

    def get_stimulus_by_pos(self, pos: float) -> ArrayLike:
        """Generate external stimulus centered at the given position."""
        return u.math.exp(-0.25 * u.math.square(self.dist(self.x - pos) / self.a))


class NonRecUnits(BasicModel):
    """A model of non-recurrently connected units.

    This class implements a simpler network without recurrent connections,
    typically used as input or output layers in hierarchical models.

    Attributes:
        size (int): The number of neurons in the network.
        tau (float): The time constant for the synaptic input.
        noise_0 (float): The standard deviation of the Gaussian noise.
        z_min (float): The minimum value of the encoded feature space.
        z_max (float): The maximum value of the encoded feature space.
        z_range (float): The range of the feature space.
        x (brainunit.math.ndarray): The preferred feature values for each neuron.
        rho (float): The neural density.
        dx (float): The stimulus density.
        r (brainstate.HiddenState): The firing rates of the neurons.
        u (brainstate.HiddenState): The synaptic inputs to the neurons.
        center (brainstate.State): The decoded center of activity.
        input (brainstate.State): The external input to the network.
    """

    def __init__(
        self,
        size: int,
        tau: float = 1.0,
        noise_0: float = 0.0,
        z_min: float = -u.math.pi,
        z_max: float = u.math.pi,
        **kwargs,
    ):
        """Initialize the NonRecUnits model.

        Args:
            size: The number of neurons in the network.
            tau: The time constant for the synaptic input.
            noise_0: The standard deviation of the Gaussian noise.
            z_min: The minimum value of the encoded feature space.
            z_max: The maximum value of the encoded feature space.
            **kwargs: Additional arguments passed to BasicModel.
        """
        super().__init__(size, **kwargs)

        # Model parameters
        self.size = size
        self.tau = tau
        self.noise_0 = noise_0

        # Feature space setup
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = u.math.linspace(z_min, z_max, size)
        self.rho = size / self.z_range
        self.dx = self.z_range / size

        # Initialize states
        self.r = brainstate.HiddenState(u.math.zeros(self.varshape))
        self.u = brainstate.HiddenState(u.math.zeros(self.varshape))
        self.center = brainstate.State(0.0)
        self.input = brainstate.State(u.math.zeros(self.varshape))

    def dist(self, d: ArrayLike) -> ArrayLike:
        """Calculate the distance with periodic boundary conditions."""
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > self.z_range / 2, d - self.z_range, d)
        return d

    def update(self) -> None:
        """Update the network state for one time step."""
        # Update synaptic input (no recurrent connections)
        du = (-self.u.value + self.input.value) / self.tau
        if self.noise_0 > 0:
            du += self.noise_0 * u.math.random.randn(*self.u.value.shape)
        
        self.u.value = exp_euler_step(self.u.value, du)

        # Compute firing rates
        self.r.value = u.math.square(self.u.value)

        # Decode the center of activity
        self.center.value = self.get_center()

    def get_center(self) -> float:
        """Decode the population vector to get the center of activity."""
        exppos = u.math.exp(1j * self.x)
        self.center.value = u.math.angle(u.math.sum(self.r.value * exppos))
        return self.center.value

    def get_stimulus_by_pos(self, pos: float) -> ArrayLike:
        """Generate external stimulus centered at the given position."""
        return u.math.exp(-0.25 * u.math.square(self.dist(self.x - pos) / self.a))