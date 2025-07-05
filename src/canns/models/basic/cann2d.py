import brainstate
import brainunit as u
import jax
import matplotlib.pyplot as plt
from brainunit import Quantity

from ._base import BasicModel


class BaseCANN2D(BasicModel):
    """
    Base class for 2D Continuous Attractor Neural Network (CANN) models.
    This class sets up the fundamental properties of the network, including
    neuronal properties, feature space, and the connectivity matrix, which
    are shared by different CANN model variations.
    """

    def __init__(
        self,
        length: int,
        tau: Quantity | float = 1.0,
        k: float = 8.1,
        a: float = 0.5,
        A: float = 10,
        J0: float = 4.0,
        z_min: float = -u.math.pi,
        z_max: float = u.math.pi,
        **kwargs,
    ):
        """
        Initializes the base 2D CANN model.

        Args:
            length (int): The number of neurons in one dimension of the network (the network is square).
            tau (float): The synaptic time constant, controlling how quickly the membrane potential changes.
            k (float): A parameter controlling the strength of the global inhibition.
            a (float): The half-width of the excitatory connection range. It defines the "spread" of local connections.
            A (float): The magnitude (amplitude) of the external stimulus.
            J0 (float): The maximum connection strength between neurons.
            z_min (float): The minimum value of the feature space (e.g., -pi for an angle).
            z_max (float): The maximum value of the feature space (e.g., +pi for an angle).
            **kwargs: Additional keyword arguments passed to the parent BasicModel.
        """
        self.length = length
        self.num = length ** 2  # Total number of neurons in the 2D grid.
        super().__init__(self.num, **kwargs)

        # --- Model Parameters ---
        self.tau = tau  # Synaptic time constant.
        self.k = k  # Degree of the rescaled inhibition.
        self.a = a  # Half-width of the range of excitatory connections.
        self.A = A  # Magnitude of the external input.
        self.J0 = J0  # Maximum connection value (amplitude of the connectivity kernel).

        # --- Feature Space Properties ---
        self.z_min = z_min  # Minimum of the feature space.
        self.z_max = z_max  # Maximum of the feature space.
        self.z_range = z_max - z_min  # The total range of the feature space.
        # An array representing the preferred feature value for each neuron.
        self.x = u.math.linspace(z_min, z_max, length)
        self.rho = length / self.z_range  # The neural density.
        self.dx = self.z_range / length  # The stimulus density

        # --- Connectivity Matrix ---
        # The connection matrix, defining the strength of synapses between all pairs of neurons.
        self.conn_mat = self.make_conn()

    def show_conn(self):
        """
        Displays the connectivity matrix as an image.
        This method visualizes the connection strengths between neurons in the 2D feature space.
        """
        plt.imshow(u.math.as_numpy(self.conn_mat))
        plt.colorbar()
        plt.show()

    def dist(self, d):
        """
        Calculates the shortest distance vector between two points in a 2D feature space
        with periodic boundary conditions (a torus).

        Args:
            d (Array): The difference vector between two positions, e.g., [dx, dy].

        Returns:
            Array: The shortest distance vector, with each component wrapped around
                   the periodic boundary.
        """
        # Define the size of the periodic box for each dimension.
        box_size = u.math.asarray([self.z_range, self.z_range])
        # Apply the periodic boundary condition to each component of the vector
        # using the remainder. This wraps the differences into the [0, box_size) interval.
        d = u.math.remainder(d, box_size)
        # Ensure each component of the distance vector is the shortest path.
        # For example, in a dimension of size 360, the distance between 350 and 10
        # should be -20 (magnitude 20), not 340.
        d = u.math.where(d > box_size / 2, d - box_size, d)
        return d

    def make_conn(self):
        """
        Constructs the connectivity matrix for a 2D grid of neurons based on a
        Gaussian profile. The connection strength between two neurons depends on the
        Euclidean distance between their preferred feature coordinates in a 2D
        toroidal space (space with periodic boundaries in both dimensions).

        Returns:
            Array: A ((num*num) x (num*num)) connectivity matrix.
        """
        # Create a 2D grid of coordinates for all neurons.
        x1, x2 = u.math.meshgrid(self.x, self.x)
        # Reshape the grid into a list of coordinate pairs.
        all_coords = u.math.stack([x1.flatten(), x2.flatten()]).T

        # Define a function to compute connectivity from one neuron to all others.
        @jax.vmap
        def get_conn_for_one_neuron(source_coord):
            # Calculate the difference vectors from this source neuron to all other neurons.
            # The self.dist function handles the periodic boundary conditions for each vector component.
            diff_vectors = self.dist(source_coord - all_coords)

            # Calculate the scalar Euclidean distance (L2 norm) for each difference vector.
            # This gives the true shortest distance in the 2D toroidal space.
            scalar_distances = u.math.linalg.norm(diff_vectors, axis=1)

            # Compute connection strengths using the same Gaussian (normal distribution) function.
            # Neurons with closer coordinates will have stronger excitatory connections.
            conn_strengths = (
                self.J0
                * u.math.exp(-0.5 * u.math.square(scalar_distances / self.a))
                / (u.math.sqrt(2 * u.math.pi) * self.a)
            )
            return conn_strengths

        return get_conn_for_one_neuron(all_coords)

    def get_stimulus_by_pos(self, pos):
        """
        Generates a Gaussian-shaped external stimulus centered at a given
        coordinate on the 2D neural grid.

        Args:
            pos (Array, tuple): The center coordinate [x, y] of the stimulus
                                in the feature space.

        Returns:
            Array: A 2D array (grid) of stimulus values for each neuron.
        """
        # Validate that the input position is two-dimensional.
        pos = u.math.asarray(pos)
        assert pos.shape == (2,), "Input position must be a 2D coordinate, e.g., [x, y]."
        # Create a 2D grid of coordinates for all neurons.
        x1, x2 = u.math.meshgrid(self.x, self.x)
        all_coords = u.math.stack([x1.flatten(), x2.flatten()]).T
        # Calculate the distance from the stimulus center to every neuron.
        diff_vectors = self.dist(all_coords - pos)
        scalar_distances = u.math.linalg.norm(diff_vectors, axis=1)
        # Calculate the stimulus intensity using a Gaussian function.
        stimulus_flat = self.A * u.math.exp(-0.25 * u.math.square(scalar_distances / self.a))
        # Reshape the flat stimulus array back into a 2D grid.
        num_neurons_per_dim = self.x.shape[0]
        stimulus_grid = stimulus_flat.reshape((num_neurons_per_dim, num_neurons_per_dim))
        return stimulus_grid


class CANN2D(BaseCANN2D):
    """
    A 2D Continuous Attractor Neural Network (CANN) model.
    This model extends the base CANN2D class to include specific dynamics
    and properties for a 2D neural network.

    Reference:
        Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
        Neural computation, 20(4), 994-1025.
    """

    def init_state(self, *args, **kwargs):
        """
        Initializes the state variables of the model.
        """
        # --- State Variables ---
        # Firing rate of the neurons.
        self.r = brainstate.HiddenState(u.math.zeros((self.length, self.length)))
        # Synaptic input to the neurons.
        self.u = brainstate.HiddenState(u.math.zeros((self.length, self.length)))

        # --- Inputs ---
        # External input to the neurons
        self.inp = brainstate.State(u.math.zeros((self.length, self.length)))

    def update(self, inp):
        """
        The main update function, defining the dynamics of the network for one time step.

        Args:
            inp (Array): The external input to the network, which can be a stimulus or other driving force.
        """
        self.inp.value = inp
        # The numerator for the firing rate calculation (a non-linear activation function).
        r1 = u.math.square(self.u.value)
        # The denominator, which implements global divisive inhibition.
        r2 = 1.0 + self.k * u.math.sum(r1)
        # Calculate the firing rate of each neuron using divisive normalization.
        self.r.value = r1 / r2
        # Calculate the recurrent input from other neurons in the network.
        Irec = (self.r.value.flatten() @ self.conn_mat).reshape((self.length, self.length))
        # Update the synaptic input based on the recurrent input and external input.
        self.u.value += (
            (-self.u.value + Irec + self.inp.value) / self.tau * brainstate.environ.get_dt()
        )


class CANN2D_SFA(BaseCANN2D):
    """
    A 2D Continuous Attractor Neural Network (CANN) model with a specific
    implementation of the Synaptic Firing Activity (SFA) dynamics.
    This model extends the base CANN2D class to include SFA-specific dynamics.
    """

    def __init__(
        self,
        length: int,
        tau: Quantity | float = 1.0,
        tau_v: Quantity | float = 50.0,
        k: float = 8.1,
        a: float = 0.3,
        A: float = 0.2,
        J0: float = 1.0,
        z_min: float = -u.math.pi,
        z_max: float = u.math.pi,
        m: float = 0.3,
        **kwargs,
    ):
        """
        Initializes the 2D CANN model with SFA dynamics.
        """
        super().__init__(length=length, tau=tau, k=k, a=a, A=A, J0=J0, z_min=z_min, z_max=z_max, **kwargs)
        # --- SFA-specific Parameters ---
        self.tau_v = tau_v  # Time Constant of the adaptation variable.
        self.m = m  # Strength of the adaptation.

    def init_state(self, *args, **kwargs):
        """Initializes the state variables of the model, including the adaptation variable."""
        # --- State Variables ---
        self.r = brainstate.HiddenState(u.math.zeros(self.varshape))  # Firing rate.
        self.u = brainstate.HiddenState(u.math.zeros(self.varshape))  # Synaptic input.
        # self.v: The adaptation variable, which tracks the synaptic inputs 'u' and provides negative feedback.
        self.v = brainstate.HiddenState(u.math.zeros(self.varshape))

        # --- Inputs ---
        self.inp = brainstate.State(u.math.zeros(self.varshape))  # External input.

    def update(self, inp):
        """
        The main update function for the SFA model. It includes dynamics for both
        the membrane potential and the adaptation variable.

        Args:
            inp (Array): The external input for the current time step.
        """
        self.inp.value = inp
        # Firing rate calculation is the same as the standard CANN model.
        r1 = u.math.square(self.u.value)
        r2 = 1.0 + self.k * u.math.sum(r1)
        self.r.value = r1 / r2
        # Calculate recurrent input.
        Irec = (self.r.value.flatten() @ self.conn_mat).reshape((self.length, self.length))
        # Update the synaptic input. Note the additional '- self.v.value' term,
        self.u.value += (
            (-self.u.value + Irec + self.inp.value - self.v.value) / self.tau * brainstate.environ.get_dt()
        )
        # Update the adaptation variable 'v'. It slowly tracks the membrane potential 'u'
        # and has its own decay, creating a slow negative feedback loop.
        self.v.value += (
            (-self.v.value + self.m * self.u.value) / self.tau_v * brainstate.environ.get_dt()
        )
