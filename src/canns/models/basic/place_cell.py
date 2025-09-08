import math

import brainstate
import brainunit as u

from ...typing import time_type
from ._base import BasicModel

__all__ = [
    # Base Model
    "BasePlaceCell",
    "BasePlaceCell1D",
    "BasePlaceCell2D",
    # Place Cell 1D Models
    "PlaceCell1D",
    "PlaceCell1D_Theta",
    # Place Cell 2D Models
    "PlaceCell2D",
    "PlaceCell2D_Theta",
]


class BasePlaceCell(BasicModel):
    """
    Base class for Place Cell models.
    Place cells are pyramidal neurons in the hippocampus that become active
    when an animal enters a particular place in its environment (place field).
    They collectively form a cognitive map of spatial locations.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...],
        **kwargs,
    ):
        """
        Initializes the base Place Cell model.

        Args:
            shape (int or tuple): The number of neurons in the network. If an int is provided,
                                  it will be converted to a single-element tuple. If a tuple is provided,
                                  it defines the shape of the network (e.g., (length, length) for 2D).
            **kwargs: Additional keyword arguments passed to the parent BasicModel.
        """
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            raise TypeError("shape must be an int or a tuple of ints")
        super().__init__(math.prod(self.shape), **kwargs)

    def get_stimulus_by_pos(self, pos):
        """
        Generates an external stimulus based on a given position in the spatial environment.
        This method should be implemented in subclasses to define how place-specific
        stimuli are generated.

        Args:
            pos (float or Array): The position in the spatial environment.

        Returns:
            Array: An array of stimulus values for each neuron.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class BasePlaceCell1D(BasePlaceCell):
    """
    Base class for 1D Place Cell models.
    Models place cells on linear tracks where each neuron has a Gaussian
    tuning curve centered at a specific location along the track.
    """

    def __init__(
        self,
        num: int,
        tau: time_type = 1.0,
        sigma: float = 0.3,
        A: float = 1.0,
        baseline: float = 0.0,
        x_min: float = 0.0,
        x_max: float = 1.0,
        **kwargs,
    ):
        """
        Initializes the 1D Place Cell model.

        Args:
            num (int): The number of place cells.
            tau (float): The membrane time constant.
            sigma (float): The width of the place field (standard deviation of Gaussian).
            A (float): The maximum firing rate amplitude.
            baseline (float): The baseline firing rate.
            x_min (float): The minimum position of the spatial environment.
            x_max (float): The maximum position of the spatial environment.
            **kwargs: Additional keyword arguments passed to the parent BasicModel.
        """
        super().__init__(num, **kwargs)

        # --- Model Parameters ---
        self.tau = tau  # Membrane time constant
        self.sigma = sigma  # Place field width
        self.A = A  # Maximum firing rate
        self.baseline = baseline  # Baseline firing rate

        # --- Spatial Properties ---
        self.x_min = x_min
        self.x_max = x_max
        self.x_range = x_max - x_min
        # Place field centers evenly distributed along the track
        self.x_centers = u.math.linspace(x_min, x_max, num)

    def get_stimulus_by_pos(self, pos):
        """
        Generates place-specific input based on current position.
        Each place cell receives maximum input when the animal is at its preferred location.

        Args:
            pos (float): The current position along the track.

        Returns:
            Array: Gaussian-shaped input for each place cell.
        """
        # Calculate distance from current position to each place field center
        distances = u.math.abs(pos - self.x_centers)
        # Generate Gaussian input profile
        return self.A * u.math.exp(-0.5 * u.math.square(distances / self.sigma)) + self.baseline


class PlaceCell1D(BasePlaceCell1D):
    """
    Standard 1D Place Cell model implementing basic place field dynamics.
    Each neuron has a Gaussian spatial tuning curve and integrates spatial inputs
    over time with membrane dynamics.

    Reference:
        O'Keefe, J., & Nadel, L. (1978). The hippocampus as a cognitive map.
        Oxford University Press.
    """

    def init_state(self, *args, **kwargs):
        """Initializes the state variables of the model."""
        # --- State Variables ---
        # Firing rate of place cells
        self.r = brainstate.HiddenState(u.math.zeros(self.varshape))
        # Membrane potential
        self.u = brainstate.HiddenState(u.math.zeros(self.varshape))

        # --- Inputs ---
        # Spatial input based on current position
        self.inp = brainstate.State(u.math.zeros(self.varshape))

    def update(self, pos_input):
        """
        Updates place cell activity based on current position.

        Args:
            pos_input (float): Current position along the track.
        """
        # Get place-specific input
        spatial_input = self.get_stimulus_by_pos(pos_input)
        self.inp.value = spatial_input

        # Update membrane potential with leaky integration
        self.u.value += (-self.u.value + self.inp.value) / self.tau * brainstate.environ.get_dt()

        # Compute firing rate with rectification
        self.r.value = u.math.maximum(0.0, self.u.value)


class PlaceCell1D_Theta(BasePlaceCell1D):
    """
    1D Place Cell model with theta rhythm modulation.
    Implements the Position-Theta-Phase (PTP) model where firing rate
    is modulated by both spatial location and theta phase.

    Reference:
        O'Keefe, J., & Recce, M. L. (1993). Phase relationship between hippocampal
        place units and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
    """

    def __init__(
        self,
        num: int,
        tau: time_type = 1.0,
        sigma: float = 0.3,
        A: float = 1.0,
        baseline: float = 0.0,
        x_min: float = 0.0,
        x_max: float = 1.0,
        theta_freq: float = 8.0,
        theta_amp: float = 0.5,
        **kwargs,
    ):
        """
        Initializes the 1D Place Cell model with theta modulation.

        Args:
            theta_freq (float): Theta rhythm frequency (Hz).
            theta_amp (float): Amplitude of theta modulation.
            (Other parameters inherited from BasePlaceCell1D)
        """
        super().__init__(num, tau, sigma, A, baseline, x_min, x_max, **kwargs)
        self.theta_freq = theta_freq
        self.theta_amp = theta_amp

    def init_state(self, *args, **kwargs):
        """Initializes the state variables including theta phase."""
        # --- State Variables ---
        self.r = brainstate.HiddenState(u.math.zeros(self.varshape))
        self.u = brainstate.HiddenState(u.math.zeros(self.varshape))
        # Theta phase for each cell (can vary for phase precession)
        self.theta_phase = brainstate.HiddenState(u.math.zeros(self.varshape))

        # --- Inputs ---
        self.inp = brainstate.State(u.math.zeros(self.varshape))

    def update(self, pos_input, time_input):
        """
        Updates place cell activity with theta modulation.

        Args:
            pos_input (float): Current position along the track.
            time_input (float): Current time for theta rhythm.
        """
        # Get spatial input
        spatial_input = self.get_stimulus_by_pos(pos_input)

        # Add theta modulation
        theta_modulation = 1.0 + self.theta_amp * u.math.cos(
            2 * u.math.pi * self.theta_freq * time_input + self.theta_phase.value
        )
        self.inp.value = spatial_input * theta_modulation

        # Update membrane potential
        self.u.value += (-self.u.value + self.inp.value) / self.tau * brainstate.environ.get_dt()

        # Compute firing rate
        self.r.value = u.math.maximum(0.0, self.u.value)


class BasePlaceCell2D(BasePlaceCell):
    """
    Base class for 2D Place Cell models.
    Models place cells in open field environments where each neuron has
    a 2D Gaussian place field.
    """

    def __init__(
        self,
        length: int,
        tau: time_type = 1.0,
        sigma: float = 0.3,
        A: float = 1.0,
        baseline: float = 0.0,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
        **kwargs,
    ):
        """
        Initializes the 2D Place Cell model.

        Args:
            length (int): Number of neurons per dimension (total neurons = length^2).
            tau (float): The membrane time constant.
            sigma (float): Place field width.
            A (float): Maximum firing rate.
            baseline (float): Baseline firing rate.
            x_min, x_max (float): X-axis boundaries.
            y_min, y_max (float): Y-axis boundaries.
            **kwargs: Additional keyword arguments.
        """
        self.length = length
        super().__init__((length, length), **kwargs)

        # --- Model Parameters ---
        self.tau = tau
        self.sigma = sigma
        self.A = A
        self.baseline = baseline

        # --- Spatial Properties ---
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.x_range = x_max - x_min
        self.y_range = y_max - y_min

        # Place field centers arranged in a grid
        x_coords = u.math.linspace(x_min, x_max, length)
        y_coords = u.math.linspace(y_min, y_max, length)
        self.x_centers, self.y_centers = u.math.meshgrid(x_coords, y_coords)

    def get_stimulus_by_pos(self, pos):
        """
        Generates 2D place-specific input based on current position.

        Args:
            pos (Array): Current position [x, y].

        Returns:
            Array: 2D array of place cell inputs.
        """
        pos = u.math.asarray(pos)
        assert pos.shape == (2,), "Input position must be a 2D coordinate [x, y]."

        x, y = pos[0], pos[1]
        # Calculate 2D Gaussian place fields
        dx = x - self.x_centers
        dy = y - self.y_centers
        distances_sq = dx**2 + dy**2

        return self.A * u.math.exp(-0.5 * distances_sq / (self.sigma**2)) + self.baseline


class PlaceCell2D(BasePlaceCell2D):
    """
    Standard 2D Place Cell model for open field navigation.
    Each neuron represents a specific location in 2D space with Gaussian tuning.

    Reference:
        Muller, R. U., & Kubie, J. L. (1987). The effects of changes in the environment
        on the spatial firing of hippocampal complex-spike cells.
    """

    def init_state(self, *args, **kwargs):
        """Initializes the state variables of the 2D model."""
        # --- State Variables ---
        self.r = brainstate.HiddenState(u.math.zeros((self.length, self.length)))
        self.u = brainstate.HiddenState(u.math.zeros((self.length, self.length)))

        # --- Inputs ---
        self.inp = brainstate.State(u.math.zeros((self.length, self.length)))

    def update(self, pos_input):
        """
        Updates 2D place cell activity based on current position.

        Args:
            pos_input (Array): Current position [x, y].
        """
        # Get spatial input
        spatial_input = self.get_stimulus_by_pos(pos_input)
        self.inp.value = spatial_input

        # Update membrane potential
        self.u.value += (-self.u.value + self.inp.value) / self.tau * brainstate.environ.get_dt()

        # Compute firing rate
        self.r.value = u.math.maximum(0.0, self.u.value)


class PlaceCell2D_Theta(BasePlaceCell2D):
    """
    2D Place Cell model with theta rhythm modulation.
    Extends 2D place cells with theta phase modulation for more realistic
    hippocampal dynamics.
    """

    def __init__(
        self,
        length: int,
        tau: time_type = 1.0,
        sigma: float = 0.3,
        A: float = 1.0,
        baseline: float = 0.0,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
        theta_freq: float = 8.0,
        theta_amp: float = 0.5,
        **kwargs,
    ):
        """
        Initializes the 2D Place Cell model with theta modulation.

        Args:
            theta_freq (float): Theta rhythm frequency (Hz).
            theta_amp (float): Amplitude of theta modulation.
            (Other parameters inherited from BasePlaceCell2D)
        """
        super().__init__(length, tau, sigma, A, baseline, x_min, x_max, y_min, y_max, **kwargs)
        self.theta_freq = theta_freq
        self.theta_amp = theta_amp

    def init_state(self, *args, **kwargs):
        """Initializes state variables including 2D theta phases."""
        # --- State Variables ---
        self.r = brainstate.HiddenState(u.math.zeros((self.length, self.length)))
        self.u = brainstate.HiddenState(u.math.zeros((self.length, self.length)))
        self.theta_phase = brainstate.HiddenState(u.math.zeros((self.length, self.length)))

        # --- Inputs ---
        self.inp = brainstate.State(u.math.zeros((self.length, self.length)))

    def update(self, pos_input, time_input):
        """
        Updates 2D place cell activity with theta modulation.

        Args:
            pos_input (Array): Current position [x, y].
            time_input (float): Current time for theta rhythm.
        """
        # Get spatial input
        spatial_input = self.get_stimulus_by_pos(pos_input)

        # Add theta modulation
        theta_modulation = 1.0 + self.theta_amp * u.math.cos(
            2 * u.math.pi * self.theta_freq * time_input + self.theta_phase.value
        )
        self.inp.value = spatial_input * theta_modulation

        # Update membrane potential
        self.u.value += (-self.u.value + self.inp.value) / self.tau * brainstate.environ.get_dt()

        # Compute firing rate
        self.r.value = u.math.maximum(0.0, self.u.value)
