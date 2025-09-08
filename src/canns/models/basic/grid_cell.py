import math

import brainstate
import brainunit as u
import jax

from ...typing import time_type
from ._base import BasicModel

__all__ = [
    # Base Model
    "BaseGridCell",
    "BaseGridCell1D",
    "BaseGridCell2D",
    # Grid Cell 1D Models
    "GridCell1D",
    "GridCell1D_CAN",
    # Grid Cell 2D Models
    "GridCell2D",
    "GridCell2D_CAN",
]


class BaseGridCell(BasicModel):
    """
    Base class for Grid Cell models.
    Grid cells are neurons in the medial entorhinal cortex that fire at regular
    intervals as an animal navigates, forming hexagonal lattice patterns in 2D
    or periodic patterns in 1D environments.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...],
        **kwargs,
    ):
        """
        Initializes the base Grid Cell model.

        Args:
            shape (int or tuple): The number of neurons in the network.
            **kwargs: Additional keyword arguments passed to the parent BasicModel.
        """
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            raise TypeError("shape must be an int or a tuple of ints")
        super().__init__(math.prod(self.shape), **kwargs)

    def integrate_velocity(self, velocity, dt):
        """
        Integrates velocity input for path integration.
        This method should be implemented in subclasses.

        Args:
            velocity (Array): Velocity vector.
            dt (float): Time step.

        Returns:
            Array: Updated position or activity bump.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class BaseGridCell1D(BaseGridCell):
    """
    Base class for 1D Grid Cell models.
    Models grid cells on linear tracks where firing fields are periodic
    but not hexagonally organized (appears as slices through 2D patterns).
    """

    def __init__(
        self,
        num: int,
        tau: time_type = 1.0,
        spacing: float = 0.3,
        phase: float = 0.0,
        orientation: float = 0.0,
        k: float = 8.1,
        A: float = 1.0,
        x_min: float = -u.math.pi,
        x_max: float = u.math.pi,
        **kwargs,
    ):
        """
        Initializes the 1D Grid Cell model.

        Args:
            num (int): Number of grid cells.
            tau (float): Membrane time constant.
            spacing (float): Grid spacing (distance between firing fields).
            phase (float): Phase offset of the grid pattern.
            orientation (float): Orientation angle (for 1D slice of 2D pattern).
            k (float): Global inhibition strength.
            A (float): Input amplitude.
            x_min, x_max (float): Spatial boundaries.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(num, **kwargs)

        # --- Model Parameters ---
        self.tau = tau
        self.spacing = spacing
        self.phase = phase
        self.orientation = orientation
        self.k = k
        self.A = A

        # --- Spatial Properties ---
        self.x_min = x_min
        self.x_max = x_max
        self.x_range = x_max - x_min
        self.x = u.math.linspace(x_min, x_max, num)
        self.dx = self.x_range / num

    def get_grid_input(self, pos):
        """
        Generates grid-like input based on current position.
        Creates periodic firing pattern characteristic of grid cells.

        Args:
            pos (float): Current position along the track.

        Returns:
            Array: Grid-modulated input for each cell.
        """
        # Create periodic grid pattern (cosine waves)
        grid_pattern = u.math.cos(2 * u.math.pi * (pos - self.phase) / self.spacing)
        # Apply to all cells with position-dependent modulation
        return self.A * u.math.maximum(
            0.0, grid_pattern + u.math.cos(2 * u.math.pi * self.x / self.spacing)
        )


class GridCell1D(BaseGridCell1D):
    """
    Standard 1D Grid Cell model implementing basic periodic firing.
    Creates regular firing fields along a linear track, representing
    a 1D slice through the 2D hexagonal grid pattern.

    Reference:
        Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005).
        Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.
    """

    def init_state(self, *args, **kwargs):
        """Initializes the state variables."""
        # --- State Variables ---
        self.r = brainstate.HiddenState(u.math.zeros(self.varshape))
        self.u = brainstate.HiddenState(u.math.zeros(self.varshape))

        # --- Inputs ---
        self.inp = brainstate.State(u.math.zeros(self.varshape))

    def update(self, pos_input):
        """
        Updates grid cell activity based on current position.

        Args:
            pos_input (float): Current position along the track.
        """
        # Get grid-patterned input
        grid_input = self.get_grid_input(pos_input)
        self.inp.value = grid_input

        # Update membrane potential
        self.u.value += (-self.u.value + self.inp.value) / self.tau * brainstate.environ.get_dt()

        # Apply global inhibition and compute firing rate
        total_activity = u.math.sum(u.math.square(self.u.value))
        self.r.value = u.math.square(self.u.value) / (1.0 + self.k * total_activity)


class GridCell1D_CAN(BaseGridCell1D):
    """
    1D Grid Cell model with Continuous Attractor Network (CAN) dynamics.
    Implements path integration through recurrent connectivity and
    velocity-driven activity bump movement.

    Reference:
        Burak, Y., & Fiete, I. R. (2009). Accurate path integration in
        continuous attractor network models of grid cells. PLoS computational biology, 5(2).
    """

    def __init__(
        self,
        num: int,
        tau: time_type = 1.0,
        spacing: float = 0.3,
        phase: float = 0.0,
        orientation: float = 0.0,
        k: float = 8.1,
        A: float = 1.0,
        x_min: float = -u.math.pi,
        x_max: float = u.math.pi,
        beta: float = 3.0,
        **kwargs,
    ):
        """
        Initializes the 1D Grid Cell CAN model.

        Args:
            beta (float): Velocity integration strength.
            (Other parameters inherited from BaseGridCell1D)
        """
        super().__init__(num, tau, spacing, phase, orientation, k, A, x_min, x_max, **kwargs)
        self.beta = beta

        # Create recurrent connectivity matrix for 1D grid
        self.conn_mat = self.make_conn()

    def make_conn(self):
        """
        Creates recurrent connectivity matrix for 1D grid pattern.
        Uses periodic connectivity to maintain grid-like activity bumps.

        Returns:
            Array: Connectivity matrix.
        """
        # Create distance matrix with periodic boundaries
        i_indices = u.math.arange(self.varshape[0])[:, None]
        j_indices = u.math.arange(self.varshape[0])[None, :]

        # Calculate periodic distances
        diff = i_indices - j_indices
        diff = (
            u.math.remainder(diff + self.varshape[0] // 2, self.varshape[0]) - self.varshape[0] // 2
        )
        distances = u.math.abs(diff * self.dx)

        # Create Mexican hat connectivity (excitatory center, inhibitory surround)
        sigma_ex = self.spacing / 4  # Excitatory radius
        sigma_in = self.spacing / 2  # Inhibitory radius

        excitation = u.math.exp(-0.5 * u.math.square(distances / sigma_ex))
        inhibition = 0.5 * u.math.exp(-0.5 * u.math.square(distances / sigma_in))

        return excitation - inhibition

    def init_state(self, *args, **kwargs):
        """Initializes the state variables including activity bump."""
        # --- State Variables ---
        self.r = brainstate.HiddenState(u.math.zeros(self.varshape))
        self.u = brainstate.HiddenState(u.math.zeros(self.varshape))

        # Initialize with a small activity bump
        center = self.varshape[0] // 2
        width = max(1, int(self.spacing / self.dx))
        bump = u.math.exp(-0.5 * u.math.square(u.math.arange(self.varshape[0]) - center) / width**2)
        self.u = brainstate.HiddenState(bump * 0.1)

        # --- Inputs ---
        self.inp = brainstate.State(u.math.zeros(self.varshape))

    def integrate_velocity(self, velocity, dt):
        """
        Integrates velocity to shift activity bump.

        Args:
            velocity (float): Current velocity along the track.
            dt (float): Time step.
        """
        # Calculate shift amount based on velocity
        shift_amount = velocity * dt / self.dx

        # Shift activity pattern using circular convolution
        # This is a simplified version - in practice, more sophisticated methods are used
        if abs(shift_amount) > 0.01:  # Only shift if movement is significant
            shift_indices = u.math.round(shift_amount).astype(int)
            self.u.value = u.math.roll(self.u.value, shift_indices)

    def update(self, velocity_input):
        """
        Updates grid cell activity with path integration.

        Args:
            velocity_input (float): Current velocity for path integration.
        """
        dt = brainstate.environ.get_dt()

        # Path integration - shift activity bump based on velocity
        self.integrate_velocity(velocity_input, dt)

        # Recurrent dynamics
        recurrent_input = u.math.dot(self.conn_mat, self.r.value)

        # Update membrane potential
        self.u.value += (-self.u.value + recurrent_input) / self.tau * dt

        # Compute firing rate with global inhibition
        total_activity = u.math.sum(u.math.square(self.u.value))
        self.r.value = u.math.square(self.u.value) / (1.0 + self.k * total_activity)


class BaseGridCell2D(BaseGridCell):
    """
    Base class for 2D Grid Cell models.
    Models the classic hexagonal grid cell firing pattern in 2D environments.
    """

    def __init__(
        self,
        length: int,
        tau: time_type = 1.0,
        spacing: float = 0.3,
        orientation: float = 0.0,
        phase: tuple[float, float] = (0.0, 0.0),
        k: float = 8.1,
        A: float = 1.0,
        x_min: float = -1.0,
        x_max: float = 1.0,
        y_min: float = -1.0,
        y_max: float = 1.0,
        **kwargs,
    ):
        """
        Initializes the 2D Grid Cell model.

        Args:
            length (int): Number of neurons per dimension.
            tau (float): Membrane time constant.
            spacing (float): Grid spacing (distance between firing fields).
            orientation (float): Orientation of the hexagonal grid.
            phase (tuple): Phase offsets (x, y) for the grid pattern.
            k (float): Global inhibition strength.
            A (float): Input amplitude.
            x_min, x_max, y_min, y_max (float): Spatial boundaries.
            **kwargs: Additional keyword arguments.
        """
        self.length = length
        super().__init__((length, length), **kwargs)

        # --- Model Parameters ---
        self.tau = tau
        self.spacing = spacing
        self.orientation = orientation
        self.phase = phase
        self.k = k
        self.A = A

        # --- Spatial Properties ---
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.x_range = x_max - x_min
        self.y_range = y_max - y_min

        # Create spatial coordinate grids
        x_coords = u.math.linspace(x_min, x_max, length)
        y_coords = u.math.linspace(y_min, y_max, length)
        self.X, self.Y = u.math.meshgrid(x_coords, y_coords)
        self.dx = self.x_range / length
        self.dy = self.y_range / length

    def get_hexagonal_input(self, pos):
        """
        Generates hexagonal grid input based on current position.
        Creates the characteristic hexagonal firing pattern of grid cells.

        Args:
            pos (Array): Current position [x, y].

        Returns:
            Array: 2D array of hexagonal grid input.
        """
        pos = u.math.asarray(pos)
        assert pos.shape == (2,), "Input position must be a 2D coordinate [x, y]."

        x, y = pos[0] - self.phase[0], pos[1] - self.phase[1]

        # Create hexagonal lattice using three cosine waves at 60° angles
        # This is the standard method for generating hexagonal patterns
        cos1 = u.math.cos(
            4
            * u.math.pi
            / (u.math.sqrt(3) * self.spacing)
            * (
                u.math.cos(self.orientation) * (self.X - x)
                + u.math.sin(self.orientation) * (self.Y - y)
            )
        )

        cos2 = u.math.cos(
            4
            * u.math.pi
            / (u.math.sqrt(3) * self.spacing)
            * (
                u.math.cos(self.orientation + u.math.pi / 3) * (self.X - x)
                + u.math.sin(self.orientation + u.math.pi / 3) * (self.Y - y)
            )
        )

        cos3 = u.math.cos(
            4
            * u.math.pi
            / (u.math.sqrt(3) * self.spacing)
            * (
                u.math.cos(self.orientation + 2 * u.math.pi / 3) * (self.X - x)
                + u.math.sin(self.orientation + 2 * u.math.pi / 3) * (self.Y - y)
            )
        )

        # Combine the three waves to create hexagonal pattern
        hexagonal_pattern = (cos1 + cos2 + cos3) / 3.0

        # Apply threshold and scaling
        return self.A * u.math.maximum(0.0, hexagonal_pattern)


class GridCell2D(BaseGridCell2D):
    """
    Standard 2D Grid Cell model implementing hexagonal firing patterns.
    Creates the characteristic triangular/hexagonal lattice of firing fields
    observed in medial entorhinal cortex.

    Reference:
        Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005).
        Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.
    """

    def init_state(self, *args, **kwargs):
        """Initializes the state variables."""
        # --- State Variables ---
        self.r = brainstate.HiddenState(u.math.zeros((self.length, self.length)))
        self.u = brainstate.HiddenState(u.math.zeros((self.length, self.length)))

        # --- Inputs ---
        self.inp = brainstate.State(u.math.zeros((self.length, self.length)))

    def update(self, pos_input):
        """
        Updates 2D grid cell activity based on current position.

        Args:
            pos_input (Array): Current position [x, y].
        """
        # Get hexagonal grid input
        hex_input = self.get_hexagonal_input(pos_input)
        self.inp.value = hex_input

        # Update membrane potential
        self.u.value += (-self.u.value + self.inp.value) / self.tau * brainstate.environ.get_dt()

        # Apply global inhibition and compute firing rate
        total_activity = u.math.sum(u.math.square(self.u.value))
        self.r.value = u.math.square(self.u.value) / (1.0 + self.k * total_activity)


class GridCell2D_CAN(BaseGridCell2D):
    """
    2D Grid Cell model with Continuous Attractor Network dynamics.
    Implements hexagonal activity bumps that move through path integration,
    maintaining stable grid patterns during navigation.

    Reference:
        Burak, Y., & Fiete, I. R. (2009). Accurate path integration in
        continuous attractor network models of grid cells. PLoS computational biology, 5(2).
    """

    def __init__(
        self,
        length: int,
        tau: time_type = 1.0,
        spacing: float = 0.3,
        orientation: float = 0.0,
        phase: tuple[float, float] = (0.0, 0.0),
        k: float = 8.1,
        A: float = 1.0,
        x_min: float = -1.0,
        x_max: float = 1.0,
        y_min: float = -1.0,
        y_max: float = 1.0,
        beta: float = 3.0,
        **kwargs,
    ):
        """
        Initializes the 2D Grid Cell CAN model.

        Args:
            beta (float): Velocity integration strength.
            (Other parameters inherited from BaseGridCell2D)
        """
        super().__init__(
            length, tau, spacing, orientation, phase, k, A, x_min, x_max, y_min, y_max, **kwargs
        )
        self.beta = beta

        # Create hexagonal recurrent connectivity
        self.conn_mat = self.make_conn()

    def make_conn(self):
        """
        Creates 2D recurrent connectivity matrix for hexagonal grid pattern.
        Uses periodic boundaries and Mexican hat connectivity.

        Returns:
            Array: 4D connectivity tensor reshaped to matrix.
        """
        # Create coordinate arrays for all neuron pairs
        flat_coords = u.math.stack([self.X.flatten(), self.Y.flatten()]).T

        @jax.vmap
        def compute_conn_row(target_coord):
            # Calculate distances to all neurons (with periodic boundaries)
            dx = flat_coords[:, 0] - target_coord[0]
            dy = flat_coords[:, 1] - target_coord[1]

            # Apply periodic boundary conditions
            dx = u.math.remainder(dx + self.x_range / 2, self.x_range) - self.x_range / 2
            dy = u.math.remainder(dy + self.y_range / 2, self.y_range) - self.y_range / 2

            distances = u.math.sqrt(dx**2 + dy**2)

            # Mexican hat connectivity (excitatory center, inhibitory surround)
            sigma_ex = self.spacing / 4
            sigma_in = self.spacing / 2

            excitation = u.math.exp(-0.5 * u.math.square(distances / sigma_ex))
            inhibition = 0.5 * u.math.exp(-0.5 * u.math.square(distances / sigma_in))

            return excitation - inhibition

        return compute_conn_row(flat_coords)

    def init_state(self, *args, **kwargs):
        """Initializes state with hexagonal activity pattern."""
        # --- State Variables ---
        self.r = brainstate.HiddenState(u.math.zeros((self.length, self.length)))
        self.u = brainstate.HiddenState(u.math.zeros((self.length, self.length)))

        # Initialize with hexagonal activity bump
        init_pattern = self.get_hexagonal_input([0.0, 0.0])
        self.u = brainstate.HiddenState(init_pattern * 0.1)

        # --- Inputs ---
        self.inp = brainstate.State(u.math.zeros((self.length, self.length)))

    def integrate_velocity(self, velocity, dt):
        """
        Integrates 2D velocity to shift hexagonal activity pattern.

        Args:
            velocity (Array): Velocity vector [vx, vy].
            dt (float): Time step.
        """
        velocity = u.math.asarray(velocity)
        assert velocity.shape == (2,), "Velocity must be a 2D vector [vx, vy]."

        # Calculate shift amounts
        shift_x = velocity[0] * dt / self.dx
        shift_y = velocity[1] * dt / self.dy

        # Shift activity pattern (simplified - real implementation would use more sophisticated methods)
        if u.math.sqrt(shift_x**2 + shift_y**2) > 0.01:
            shift_x_int = u.math.round(shift_x).astype(int)
            shift_y_int = u.math.round(shift_y).astype(int)

            # Apply 2D circular shift
            self.u.value = u.math.roll(
                u.math.roll(self.u.value, shift_x_int, axis=1), shift_y_int, axis=0
            )

    def update(self, velocity_input):
        """
        Updates 2D grid cell activity with path integration.

        Args:
            velocity_input (Array): Velocity vector [vx, vy].
        """
        dt = brainstate.environ.get_dt()

        # Path integration
        self.integrate_velocity(velocity_input, dt)

        # Recurrent dynamics
        recurrent_flat = u.math.dot(self.conn_mat, self.r.value.flatten())
        recurrent_input = recurrent_flat.reshape((self.length, self.length))

        # Update membrane potential
        self.u.value += (-self.u.value + recurrent_input) / self.tau * dt

        # Compute firing rate with global inhibition
        total_activity = u.math.sum(u.math.square(self.u.value))
        self.r.value = u.math.square(self.u.value) / (1.0 + self.k * total_activity)
