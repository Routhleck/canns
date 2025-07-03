import brainstate as bst
import brainunit as u

from ._base import BasicModel


class BaseCANN1D(BasicModel):
    """
    Base class for 1D Continuous Attractor Neural Network (CANN) models.
    This class sets up the fundamental properties of the network, including
    neuronal properties, feature space, and the connectivity matrix, which
    are shared by different CANN model variations.
    """

    def __init__(
        self,
        num,
        tau=1.0,
        k=8.1,
        a=0.5,
        A=10,
        J0=4.0,
        z_min=-u.math.pi,
        z_max=u.math.pi,
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
        self.x = u.math.linspace(z_min, z_max, num)
        self.rho = num / self.z_range  # The neural density.
        self.dx = self.z_range / num  # The discretization step of the feature space.

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
        d = u.math.remainder(d, self.z_range)
        # Ensure the distance is the shortest path (e.g., the distance between 350 and 10 degrees is 20, not 340).
        d = u.math.where(d > self.z_range / 2, d - self.z_range, d)
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
        x_left = u.math.reshape(self.x, (-1, 1))
        x_right = u.math.repeat(self.x.reshape((1, -1)), len(self.x), axis=0)
        # Calculate the pairwise distance matrix with periodic boundaries.
        d = self.dist(x_left - x_right)
        # Compute the connection strengths using a Gaussian (normal distribution) function.
        # Neurons with similar feature preferences will have stronger excitatory connections.
        conn = (
            self.J0
            * u.math.exp(-0.5 * u.math.square(d / self.a))
            / (u.math.sqrt(2 * u.math.pi) * self.a)
        )
        return conn

    def get_stimulus_by_pos(self, pos):
        """
        Generates a Gaussian-shaped external stimulus centered at a given position.

        Args:
            pos (float): The center position of the stimulus in the feature space.

        Returns:
            Array: An array of stimulus values for each neuron.
        """
        # The stimulus is a "bump" of activity, modeled by a Gaussian function.
        return self.A * u.math.exp(-0.25 * u.math.square(self.dist(self.x - pos) / self.a))


class CANN1D(BaseCANN1D):
    """
    A standard 1D Continuous Attractor Neural Network (CANN) model.
    This model implements the core dynamics where a localized "bump" of activity
    can be sustained and moved by external inputs.

    Reference:
        Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
        Neural computation, 20(4), 994-1025.
    """

    def init_state(self, *args, **kwargs):
        """Initializes the state variables of the model."""
        # --- State Variables ---
        # Firing rate of the neurons.
        self.r = bst.HiddenState(u.math.zeros(self.varshape))
        # Synaptic input to the neurons.
        self.u = bst.HiddenState(u.math.zeros(self.varshape))

        # --- Inputs ---
        # External input to the network.
        self.inp = bst.State(u.math.zeros(self.varshape))

    def update(self, inp):
        """
        The main update function, defining the dynamics of the network for one time step.

        Args:
            inp (Array): The external input for the current time step.
        """
        self.inp.value = inp
        # The numerator for the firing rate calculation (a non-linear activation function).
        r1 = u.math.square(self.u.value)
        # The denominator, which implements global divisive inhibition.
        r2 = 1.0 + self.k * u.math.sum(r1)
        # Calculate the firing rate of each neuron using divisive normalization.
        self.r.value = r1 / r2
        # Calculate the recurrent input from other neurons in the network.
        Irec = u.math.dot(self.conn_mat, self.r.value)
        # Update the synaptic inputs using Euler's method. The change depends on a leak
        # current (-u), recurrent input (Irec), and external input (inp).
        self.u.value += (
            (-self.u.value + Irec + self.inp.value) / self.tau * bst.environ.get_dt()
        )


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
        num,
        tau=1.0,
        tau_v=50.0,
        k=8.1,
        a=0.3,
        A=0.2,
        J0=1.0,
        z_min=-u.math.pi,
        z_max=u.math.pi,
        m=0.3,
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

    def init_state(self, *args, **kwargs):
        """Initializes the state variables of the model, including the adaptation variable."""
        # --- State Variables ---
        self.r = bst.HiddenState(u.math.zeros(self.varshape))  # Firing rate.
        self.u = bst.HiddenState(u.math.zeros(self.varshape))  # Synaptic inputs.
        # self.v: The adaptation variable, which provides a slow hyperpolarizing current.
        self.v = bst.HiddenState(u.math.zeros(self.varshape))

        # --- Inputs ---
        self.inp = bst.State(u.math.zeros(self.varshape))  # External input.

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
        Irec = u.math.dot(self.conn_mat, self.r.value)
        # Update the synaptic input. Note the additional '- self.v.value' term,
        self.u.value += (
            (-self.u.value + Irec + self.inp.value - self.v.value) / self.tau * bst.environ.get_dt()
        )
        # Update the adaptation variable 'v'. It slowly tracks the membrane potential 'u'
        # and has its own decay, creating a slow negative feedback loop.
        self.v.value += (-self.v.value + self.m * self.u.value) / self.tau_v * bst.environ.get_dt()
