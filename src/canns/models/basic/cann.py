import brainpy.math as bm
import jax
from matplotlib import pyplot as plt

from ...typing import time_type
from ._base import BasicModel

__all__ = [
    # Base Model
    "BaseCANN",
    "BaseCANN1D",
    "BaseCANN2D",
    # CANN 1D Models
    "CANN1D",
    "CANN1D_SFA",
    # CANN 2D Models
    "CANN2D",
    "CANN2D_SFA",
]


class BaseCANN(BasicModel):
    """
    Base class for Continuous Attractor Neural Network (CANN) models.
    This class sets up the fundamental properties of the network, including
    neuronal properties, feature space, and the connectivity matrix, which
    are shared by different CANN model variations.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...],
        **kwargs,
    ):
        """
        Initializes the base CANN model.

        Args:
            shape (int or tuple): The number of neurons in the network. If an int is provided,
                                  it will be converted to a single-element tuple. If a tuple is provided,
                                  it defines the shape of the network (e.g., (length, length) for 2D).
                                  Internally, shape is always stored as a tuple.
            **kwargs: Additional keyword arguments passed to the parent BasicModel.
        """
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            raise TypeError("shape must be an int or a tuple of ints")
        super().__init__(**kwargs)

    def make_conn(self):
        """
        Constructs the connectivity matrix for the CANN model.
        This method should be implemented in subclasses to define how neurons
        are connected based on their feature preferences.

        Returns:
            Array: A connectivity matrix defining the synaptic strengths between neurons.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_stimulus_by_pos(self, pos):
        """
        Generates an external stimulus based on a given position in the feature space.
        This method should be implemented in subclasses to define how the stimulus is shaped.

        Args:
            pos (float or Array): The position in the feature space where the stimulus is centered.

        Returns:
            Array: An array of stimulus values for each neuron.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class BaseCANN1D(BaseCANN):
    """
    Base class for 1D Continuous Attractor Neural Network (CANN) models.
    This class sets up the fundamental properties of the network, including
    neuronal properties, feature space, and the connectivity matrix, which
    are shared by different CANN model variations.
    """

    def __init__(
        self,
        num: int,
        tau: time_type = 1.0,
        k: float = 8.1,
        a: float = 0.5,
        A: float = 10,
        J0: float = 4.0,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
        **kwargs,
    ):
        """
        Initializes the base 1D CANN model.

        Args:
            num (int): The number of neurons in the network.
            tau (float): The synaptic time constant, controlling how quickly the membrane potential changes.
            k (float): A parameter controlling the strength of the global inhibition.
            a (float): The half-width of the excitatory connection range. It defines the "spread" of local connections.
            A (float): The magnitude (amplitude) of the external stimulus.
            J0 (float): The maximum connection strength between neurons.
            z_min (float): The minimum value of the feature space (e.g., -pi for an angle).
            z_max (float): The maximum value of the feature space (e.g., +pi for an angle).
            **kwargs: Additional keyword arguments passed to the parent BasicModel.
        """
        super().__init__(num, **kwargs)

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
        self.x = bm.linspace(z_min, z_max, num)
        self.rho = num / self.z_range  # The neural density.
        self.dx = self.z_range / num  # The stimulus density

        # --- Connectivity Matrix ---
        # The connection matrix, defining the strength of synapses between all pairs of neurons.
        self.conn_mat = self.make_conn()

    def dist(self, d):
        """
        Calculates the shortest distance between two points in a circular feature space
        with periodic boundary conditions.

        Args:
            d (Array): The difference between two positions.

        Returns:
            Array: The shortest distance, wrapped around the periodic boundary.
        """
        # Apply periodic boundary condition using the remainder.
        d = bm.remainder(d, self.z_range)
        # Ensure the distance is the shortest path (e.g., the distance between 350 and 10 degrees is 20, not 340).
        d = bm.where(d > self.z_range / 2, d - self.z_range, d)
        return d

    def make_conn(self):
        """
        Constructs the connectivity matrix based on a Gaussian-like profile.
        The connection strength between two neurons depends on the distance
        between their preferred feature values in the circular space.

        Returns:
            Array: A (num x num) connectivity matrix.
        """
        # Prepare coordinate arrays to compute pairwise distances.
        x_left = bm.reshape(self.x, (-1, 1))
        x_right = bm.repeat(self.x.reshape((1, -1)), len(self.x), axis=0)
        # Calculate the pairwise distance matrix with periodic boundaries.
        d = self.dist(x_left - x_right)
        # Compute the connection strengths using a Gaussian (normal distribution) function.
        # Neurons with similar feature preferences will have stronger excitatory connections.
        return self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)

    def get_stimulus_by_pos(self, pos):
        """
        Generates a Gaussian-shaped external stimulus centered at a given position.

        Args:
            pos (float): The center position of the stimulus in the feature space.

        Returns:
            Array: An array of stimulus values for each neuron.
        """
        # The stimulus is a "bump" of activity, modeled by a Gaussian function.
        return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))


class CANN1D(BaseCANN1D):
    """
    A standard 1D Continuous Attractor Neural Network (CANN) model.
    This model implements the core dynamics where a localized "bump" of activity
    can be sustained and moved by external inputs.

    Reference:
        Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
        Neural computation, 20(4), 994-1025.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the 1D CANN model.

        Args:
            (Parameters are inherited from BaseCANN1D)
        """
        super().__init__(*args, **kwargs)

        # Firing rate of the neurons.
        self.r = bm.Variable(bm.zeros(self.shape))
        # Synaptic input to the neurons.
        self.u = bm.Variable(bm.zeros(self.shape))

        # --- Inputs ---
        # External input to the network.
        self.inp = bm.Variable(bm.zeros(self.shape))

    def update(self, inp):
        """
        The main update function, defining the dynamics of the network for one time step.

        Args:
            inp (Array): The external input for the current time step.
        """
        self.inp.value = inp
        # The numerator for the firing rate calculation (a non-linear activation function).
        r1 = bm.square(self.u.value)
        # The denominator, which implements global divisive inhibition.
        r2 = 1.0 + self.k * bm.sum(r1)
        # Calculate the firing rate of each neuron using divisive normalization.
        self.r.value = r1 / r2
        # Calculate the recurrent input from other neurons in the network.
        Irec = bm.dot(self.conn_mat, self.r.value)
        # Update the synaptic inputs using Euler's method. The change depends on a leak
        # current (-u), recurrent input (Irec), and external input (inp).
        self.u.value += (-self.u.value + Irec + self.inp.value) / self.tau * bm.get_dt()


class CANN1D_SFA(BaseCANN1D):
    """
    A 1D CANN model that incorporates Spike-Frequency Adaptation (SFA).
    SFA is a slow negative feedback mechanism that causes neurons to fire less
    over time for a sustained input, which can induce anticipative tracking behavior.

    Reference:
        Mi, Y., Fung, C. C., Wong, K. Y., & Wu, S. (2014). Spike frequency adaptation
        implements anticipative tracking in continuous attractor neural networks.
        Advances in neural information processing systems, 27.
    """

    def __init__(
        self,
        num: int,
        tau: time_type = 1.0,
        tau_v: time_type = 50.0,
        k: float = 8.1,
        a: float = 0.3,
        A: float = 0.2,
        J0: float = 1.0,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
        m: float = 0.3,
        **kwargs,
    ):
        """
        Initializes the 1D CANN model with SFA.

        Args:
            tau_v (float): The time constant for the adaptation variable 'v'. A larger value means slower adaptation.
            m (float): The strength of the adaptation, coupling the membrane potential 'u' to the adaptation variable 'v'.
            (Other parameters are inherited from BaseCANN1D)
        """
        super().__init__(num, tau, k, a, A, J0, z_min, z_max, **kwargs)
        # --- SFA-specific Parameters ---
        self.tau_v = tau_v  # Time constant of the adaptation variable.
        self.m = m  # Strength of the adaptation.

        self.r = bm.Variable(bm.zeros(self.shape))  # Firing rate.
        self.u = bm.Variable(bm.zeros(self.shape))  # Synaptic inputs.
        # self.v: The adaptation variable, which tracks the synaptic inputs 'u' and provides negative feedback.
        self.v = bm.Variable(bm.zeros(self.shape))

        # --- Inputs ---
        self.inp = bm.Variable(bm.zeros(self.shape))  # External input.

    def update(self, inp):
        """
        The main update function for the SFA model. It includes dynamics for both
        the membrane potential and the adaptation variable.

        Args:
            inp (Array): The external input for the current time step.
        """
        self.inp.value = inp
        # Firing rate calculation is the same as the standard CANN model.
        r1 = bm.square(self.u.value)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        # Calculate recurrent input.
        Irec = bm.dot(self.conn_mat, self.r.value)
        # Update the synaptic input. Note the additional '- self.v.value' term,
        self.u.value += (
            (-self.u.value + Irec + self.inp.value - self.v.value) / self.tau * bm.get_dt()
        )
        # Update the adaptation variable 'v'. It slowly tracks the membrane potential 'u'
        # and has its own decay, creating a slow negative feedback loop.
        self.v.value += (-self.v.value + self.m * self.u.value) / self.tau_v * bm.get_dt()


class BaseCANN2D(BaseCANN):
    """
    Base class for 2D Continuous Attractor Neural Network (CANN) models.
    This class sets up the fundamental properties of the network, including
    neuronal properties, feature space, and the connectivity matrix, which
    are shared by different CANN model variations.
    """

    def __init__(
        self,
        length: int,
        tau: float = 1.0,
        k: float = 8.1,
        a: float = 0.5,
        A: float = 10,
        J0: float = 4.0,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
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
        super().__init__((self.length,) * 2, **kwargs)  # square network of neurons

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
        self.x = bm.linspace(z_min, z_max, length)
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
        plt.imshow(bm.as_numpy(self.conn_mat))
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
        box_size = bm.asarray([self.z_range, self.z_range])
        # Apply the periodic boundary condition to each component of the vector
        # using the remainder. This wraps the differences into the [0, box_size) interval.
        d = bm.remainder(d, box_size)
        # Ensure each component of the distance vector is the shortest path.
        # For example, in a dimension of size 360, the distance between 350 and 10
        # should be -20 (magnitude 20), not 340.
        d = bm.where(d > box_size / 2, d - box_size, d)
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
        x1, x2 = bm.meshgrid(self.x, self.x)
        # Reshape the grid into a list of coordinate pairs.
        all_coords = bm.stack([x1.flatten(), x2.flatten()]).T

        # Define a function to compute connectivity from one neuron to all others.
        @jax.vmap
        def get_conn_for_one_neuron(source_coord):
            # Calculate the difference vectors from this source neuron to all other neurons.
            # The self.dist function handles the periodic boundary conditions for each vector component.
            diff_vectors = self.dist(source_coord - all_coords)

            # Calculate the scalar Euclidean distance (L2 norm) for each difference vector.
            # This gives the true shortest distance in the 2D toroidal space.
            scalar_distances = bm.linalg.norm(diff_vectors, axis=1)

            # Compute connection strengths using the same Gaussian (normal distribution) function.
            # Neurons with closer coordinates will have stronger excitatory connections.
            conn_strengths = (
                self.J0
                * bm.exp(-0.5 * bm.square(scalar_distances / self.a))
                / (bm.sqrt(2 * bm.pi) * self.a)
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
        pos = bm.asarray(pos)
        assert pos.shape == (2,), "Input position must be a 2D coordinate, e.g., [x, y]."
        # Create a 2D grid of coordinates for all neurons.
        x1, x2 = bm.meshgrid(self.x, self.x)
        all_coords = bm.stack([x1.flatten(), x2.flatten()]).T
        # Calculate the distance from the stimulus center to every neuron.
        diff_vectors = self.dist(all_coords - pos)
        scalar_distances = bm.linalg.norm(diff_vectors, axis=1)
        # Calculate the stimulus intensity using a Gaussian function.
        stimulus_flat = self.A * bm.exp(-0.25 * bm.square(scalar_distances / self.a))
        # Reshape the flat stimulus array back into a 2D grid.
        num_neurons_per_dim = self.x.shape[0]
        return stimulus_flat.reshape((num_neurons_per_dim, num_neurons_per_dim))


class CANN2D(BaseCANN2D):
    """
    A 2D Continuous Attractor Neural Network (CANN) model.
    This model extends the base CANN2D class to include specific dynamics
    and properties for a 2D neural network.

    Reference:
        Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
        Neural computation, 20(4), 994-1025.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the 2D CANN model.

        Args:
            (Parameters are inherited from BaseCANN2D)
        """
        super().__init__(*args, **kwargs)

        # Firing rate of the neurons.
        self.r = bm.Variable(bm.zeros((self.length, self.length)))
        # Synaptic input to the neurons.
        self.u = bm.Variable(bm.zeros((self.length, self.length)))

        # --- Inputs ---
        # External input to the neurons
        self.inp = bm.Variable(bm.zeros((self.length, self.length)))

    def update(self, inp):
        """
        The main update function, defining the dynamics of the network for one time step.

        Args:
            inp (Array): The external input to the network, which can be a stimulus or other driving force.
        """
        self.inp.value = inp
        # The numerator for the firing rate calculation (a non-linear activation function).
        r1 = bm.square(self.u.value)
        # The denominator, which implements global divisive inhibition.
        r2 = 1.0 + self.k * bm.sum(r1)
        # Calculate the firing rate of each neuron using divisive normalization.
        self.r.value = r1 / r2
        # Calculate the recurrent input from other neurons in the network.
        Irec = (self.r.value.flatten() @ self.conn_mat).reshape((self.length, self.length))
        # Update the synaptic input based on the recurrent input and external input.
        self.u.value += (-self.u.value + Irec + self.inp.value) / self.tau * bm.get_dt()


class CANN2D_SFA(BaseCANN2D):
    """
    A 2D Continuous Attractor Neural Network (CANN) model with a specific
    implementation of the Synaptic Firing Activity (SFA) dynamics.
    This model extends the base CANN2D class to include SFA-specific dynamics.
    """

    def __init__(
        self,
        length: int,
        tau: float = 1.0,
        tau_v: float = 50.0,
        k: float = 8.1,
        a: float = 0.3,
        A: float = 0.2,
        J0: float = 1.0,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
        m: float = 0.3,
        **kwargs,
    ):
        """
        Initializes the 2D CANN model with SFA dynamics.
        """
        super().__init__(
            length=length, tau=tau, k=k, a=a, A=A, J0=J0, z_min=z_min, z_max=z_max, **kwargs
        )
        # --- SFA-specific Parameters ---
        self.tau_v = tau_v  # Time Constant of the adaptation variable.
        self.m = m  # Strength of the adaptation.

        self.r = bm.Variable(bm.zeros((self.length, self.length)))  # Firing rate.
        self.u = bm.Variable(bm.zeros((self.length, self.length)))  # Synaptic input.
        # self.v: The adaptation variable, which tracks the synaptic inputs 'u' and provides negative feedback.
        self.v = bm.Variable(bm.zeros((self.length, self.length)))

        # --- Inputs ---
        self.inp = bm.Variable(bm.zeros((self.length, self.length)))  # External input.

    def update(self, inp):
        """
        The main update function for the SFA model. It includes dynamics for both
        the membrane potential and the adaptation variable.

        Args:
            inp (Array): The external input for the current time step.
        """
        self.inp.value = inp
        # Firing rate calculation is the same as the standard CANN model.
        r1 = bm.square(self.u.value)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        # Calculate recurrent input.
        Irec = (self.r.value.flatten() @ self.conn_mat).reshape((self.length, self.length))
        # Update the synaptic input. Note the additional '- self.v.value' term,
        self.u.value += (
            (-self.u.value + Irec + self.inp.value - self.v.value) / self.tau * bm.get_dt()
        )
        # Update the adaptation variable 'v'. It slowly tracks the membrane potential 'u'
        # and has its own decay, creating a slow negative feedback loop.
        self.v.value += (-self.v.value + self.m * self.u.value) / self.tau_v * bm.get_dt()
