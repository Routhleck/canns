import brainstate
import brainunit as u
import jax

from ...task.path_integration import map2pi
from ._base import BasicModel

__all__ = ["BandCell"]


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
        J0: float = 1.1,
        k: float = 5e-4,
        a: float = 2 / 9 * u.math.pi,
        z_min: float = -u.math.pi,
        z_max: float = u.math.pi,
        noise: float = 2.0,
    ):
        """Initializes the GaussRecUnits model.

        Args:
            size (int): The number of neurons in the network.
            tau (float, optional): The time constant of the neurons. Defaults to 1.0.
            J0 (float, optional): A scaling factor for the critical connection strength. Defaults to 1.1.
            k (float, optional): The strength of the global inhibition. Defaults to 5e-4.
            a (float, optional): The width of the Gaussian connection profile. Defaults to 2/9*pi.
            z_min (float, optional): The minimum value of the feature space. Defaults to -pi.
            z_max (float, optional): The maximum value of the feature space. Defaults to pi.
            noise (float, optional): The level of noise in the system. Defaults to 2.0.
        """
        self.size = size
        super().__init__(size)
        self.tau = tau  # The time constant
        self.k = k  # The inhibition strength
        self.a = a  # The width of the Gaussian connection
        self.noise_0 = noise  # The noise level

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = u.math.linspace(z_min, z_max, size, endpoint=False)  # The encoded feature values
        self.rho = size / self.z_range  # The neural density
        self.dx = self.z_range / size  # The stimulus density

        self.J = J0 * self.Jc()  # The connection strength
        self.conn_mat = self.make_conn()  # The connection matrix

    def init_state(self):
        self.r = brainstate.HiddenState(u.math.zeros(self.size))  # The neural firing rate
        self.u = brainstate.HiddenState(u.math.zeros(self.size))  # The neural synaptic input
        self.center = brainstate.State(
            u.math.zeros(
                1,
            )
        )  # The center of the bump

        self.input = brainstate.State(u.math.zeros(self.size))  # The external input

        # initialize the neural activity
        self.u.value = (
            10.0
            * u.math.exp(-0.5 * u.math.square((self.x - 0) / self.a))
            / (u.math.sqrt(2 * u.math.pi) * self.a)
        )
        self.r.value = (
            30.0
            * u.math.exp(-0.5 * u.math.square((self.x - 0) / self.a))
            / (u.math.sqrt(2 * u.math.pi) * self.a)
        )

    # make the connection matrix
    def make_conn(self):
        """Constructs the periodic Gaussian connection matrix.

        The connection strength between two neurons depends on the periodic distance
        between their preferred feature values, following a Gaussian profile.
        """
        dis = self.x[:, None] - self.x[None, :]
        d = self.dist(dis)
        return (
            self.J
            * u.math.exp(-0.5 * u.math.square(d / self.a))
            / (u.math.sqrt(2 * u.math.pi) * self.a)
        )

    # critical connection strength
    def Jc(self):
        """Calculates the critical connection strength.

        This is the minimum connection strength required to sustain a stable
        activity bump in the attractor network.
        """
        return u.math.sqrt(8 * u.math.sqrt(2 * u.math.pi) * self.k * self.a / self.rho)

    # truncate the distance into the range of feature space
    def dist(self, d):
        """Calculates the periodic distance in the feature space.

        This function wraps distances to ensure they fall within the periodic
        boundaries of the feature space, i.e., [-z_range/2, z_range/2].

        Args:
            d (brainunit.math.ndarray): The array of distances.
        """
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    # decode the neural activity
    def decode(self, r, axis=0):
        """Decodes the center of the activity bump.

        This method uses a population vector average to compute the center of the
        neural activity bump from the firing rates.

        Args:
            r (Array): The firing rates of the neurons.
            axis (int, optional): The axis along which to perform the decoding. Defaults to 0.

        Returns:
            float: The angle representing the decoded center of the bump.
        """
        expo_r = u.math.exp(1j * self.x) * r
        return u.math.angle(u.math.sum(expo_r, axis=axis) / u.math.sum(r, axis=axis))

    # update the neural activity
    def update(self, input):
        self.input.value = input
        r1 = u.math.square(self.u.value)
        r2 = 1.0 + self.k * u.math.sum(r1)
        self.r.value = r1 / r2
        Irec = u.math.dot(self.conn_mat, self.r.value)
        self.u.value = (
            self.u.value
            + (-self.u.value + Irec + self.input.value) / self.tau * brainstate.environ.get_dt()
        )
        self.center = self.decode(self.u.value)


class NonRecUnits(BasicModel):
    """A model of non-recurrently connected units.

    This class implements a simple leaky integrator model for a population of
    neurons that do not have recurrent connections among themselves. They respond
    to external inputs and have a non-linear activation function.

    Attributes:
        size (int): The number of neurons.
        noise_0 (float): The standard deviation of the Gaussian noise.
        tau (float): The time constant for the synaptic input `u`.
        z_min (float): The minimum value of the encoded feature space.
        z_max (float): The maximum value of the encoded feature space.
        z_range (float): The range of the feature space.
        x (brainunit.math.ndarray): The preferred feature values for each neuron.
        rho (float): The neural density.
        dx (float): The stimulus density.
        r (brainstate.State): The firing rates of the neurons.
        u (brainstate.State): The synaptic inputs to the neurons.
        input (brainstate.State): The external input to the neurons.
    """

    def __init__(
        self,
        size: int,
        tau: float = 0.1,
        z_min: float = -u.math.pi,
        z_max: float = u.math.pi,
        noise: float = 2.0,
    ):
        """Initializes the NonRecUnits model.

        Args:
            size (int): The number of neurons.
            tau (float, optional): The time constant of the neurons. Defaults to 0.1.
            z_min (float, optional): The minimum value of the feature space. Defaults to -pi.
            z_max (float, optional): The maximum value of the feature space. Defaults to pi.
            noise (float, optional): The level of noise in the system. Defaults to 2.0.
        """
        super().__init__(size)
        self.size = size
        self.noise_0 = noise  # The noise level

        self.tau = tau  # The time constant

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = u.math.linspace(z_min, z_max, size, endpoint=False)  # The encoded feature values
        self.rho = size / self.z_range  # The neural density
        self.dx = self.z_range / size  # The stimulus density

    def init_state(self):
        self.r = brainstate.State(u.math.zeros(self.size))  # The neural firing rate
        self.u = brainstate.State(u.math.zeros(self.size))  # The neural synaptic input
        self.input = brainstate.State(u.math.zeros(self.size))  # The external input

    # choose the activation function
    def activate(self, x):
        """Applies an activation function to the input.

        Args:
            x (Array): The input to the activation function (e.g., synaptic input `u`).

        Returns:
            Array: The result of the activation function (ReLU).
        """
        return u.math.relu(x)

    def dist(self, d):
        """Calculates the periodic distance in the feature space.

        This function wraps distances to ensure they fall within the periodic
        boundaries of the feature space.

        Args:
            d (Array): The array of distances.

        Returns:
            Array: The wrapped distances.
        """
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def update(self, input):
        self.input.value = input
        self.r.value = u.math.where(
            self.noise_0 != 0.0,
            self.activate(self.u.value) + self.noise_0 * brainstate.random.randn(self.size),
            self.activate(self.u.value),
        )
        # self.r.value = self.activate(self.u.value) + self.noise_0 * brainstate.random.randn(
        #     self.size
        # )
        self.u.value = (
            self.u.value
            + (-self.u.value + self.input.value) / self.tau * brainstate.environ.get_dt()
        )


class BandCell(BasicModel):
    """A model of a band cell module for path integration.

    This model represents a set of neurons whose receptive fields form parallel bands
    across a 2D space. It is composed of a central `GaussRecUnits` attractor network
    (the band cells proper) that represents a 1D phase, and two `NonRecUnits`
    populations (left and right) that help shift the activity in the attractor
    network based on velocity input. This mechanism allows the module to integrate
    the component of velocity along its preferred direction.

    Attributes:
        size (int): The number of neurons in each sub-population.
        spacing (float): The spacing between the bands in the 2D environment.
        angle (float): The orientation angle of the bands.
        proj_k (brainunit.math.ndarray): The projection vector for converting 2D position/velocity to 1D phase.
        band_cells (GaussRecUnits): The core recurrent network representing the phase.
        left (NonRecUnits): A population of non-recurrent units for positive shifts.
        right (NonRecUnits): A population of non-recurrent units for negative shifts.
        w_L2S (float): Connection weight from band cells to left/right units.
        w_S2L (float): Connection weight from left/right units to band cells.
        gain (float): A gain factor for velocity-modulated input.
        center_ideal (brainstate.State): The ideal, noise-free center based on velocity integration.
        center (brainstate.State): The actual decoded center of the band cell activity bump.
    """

    def __init__(
        self,
        angle,
        spacing,
        size=180,
        z_min=-u.math.pi,
        z_max=u.math.pi,
        noise=2.0,
        w_L2S=0.2,
        w_S2L=1.0,
        gain=0.2,
        **kwargs,
    ):
        """Initializes the BandCell model.

        Args:
            angle (float): The orientation angle of the bands.
            spacing (float): The spacing between the bands.
            size (int, optional): The number of neurons in each group. Defaults to 180.
            z_min (float, optional): The minimum value of the feature space (phase). Defaults to -pi.
            z_max (float, optional): The maximum value of the feature space (phase). Defaults to pi.
            noise (float, optional): The noise level for the neuron groups. Defaults to 2.0.
            w_L2S (float, optional): Weight from band cells to shifter units. Defaults to 0.2.
            w_S2L (float, optional): Weight from shifter units to band cells. Defaults to 1.0.
            gain (float, optional): A gain factor for the velocity signal. Defaults to 0.2.
            **kwargs: Additional keyword arguments for the base class.
        """
        self.size = size  # The number of neurons in each neuron group except DN
        super().__init__(size, **kwargs)

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = u.math.linspace(z_min, z_max, size, endpoint=False)  # The encoded feature values
        self.rho = size / self.z_range  # The neural density
        self.dx = self.z_range / size  # The stimulus density
        self.spacing = spacing
        self.angle = angle
        self.proj_k = (
            u.math.array([u.math.cos(angle - u.math.pi / 2), u.math.sin(angle - u.math.pi / 2)])
            * 2
            * u.math.pi
            / spacing
        )

        # shifts
        self.phase_shift = 1 / 9 * u.math.pi * 0.76  # the shift of the connection from PEN to EPG
        # self.PFL3_shift = 3/8*u.math.pi # the shift of the connection from EPG to PFL3
        # self.PEN_shift_num = int(self.PEN_shift / self.dx) # the number of interval shifted
        # self.PFL3_shift_num = int(self.PFL3_shift / self.dx) # the number of interval shifted

        # neurons
        self.band_cells = GaussRecUnits(size=size, noise=noise)  # heading direction
        self.left = NonRecUnits(size=size, noise=noise)
        self.right = NonRecUnits(size=size, noise=noise)

        # weights
        self.w_L2S = w_L2S
        self.w_S2L = w_S2L
        self.gain = gain
        self.synapses()

    def init_state(self, *args, **kwargs):
        self.center_ideal = brainstate.State(
            u.math.zeros(
                1,
            )
        )  # The center of v-
        self.center = brainstate.State(
            u.math.zeros(
                1,
            )
        )  # The center of v-

        # init heading direction
        self.band_cells.init_state()
        # init left and right neurons
        self.left.init_state()
        self.right.init_state()

    # define the synapses
    def synapses(self):
        """Defines the synaptic connections between the neuron groups.

        This method sets up the shifted connections from the left/right shifter
        populations to the central band cell attractor network, as well as the
        one-to-one connections from the band cells to the shifters.
        """
        self.W_PENl2EPG = self.w_S2L * self.make_conn(self.phase_shift)
        self.W_PENr2EPG = self.w_S2L * self.make_conn(-self.phase_shift)
        # synapses
        self.syn_Band2Left = brainstate.nn.OneToOne(self.size, self.w_L2S)
        self.syn_Band2Right = brainstate.nn.OneToOne(self.size, self.w_L2S)
        self.syn_Left2Band = brainstate.nn.Linear(self.size, self.size, self.W_PENl2EPG)
        self.syn_Right2Band = brainstate.nn.Linear(self.size, self.size, self.W_PENr2EPG)

    def dist(self, d):
        """Calculates the periodic distance in the feature space.

        Args:
            d (Array): The array of distances.

        Returns:
            Array: The wrapped distances.
        """
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self, shift):
        """Creates a shifted Gaussian connection profile.

        This is used to create the connections from the left/right shifter units
        to the band cells, which implements the bump-shifting mechanism.

        Args:
            shift (float): The amount to shift the connection profile by.

        Returns:
            Array: The shifted connection matrix.
        """
        d = self.dist(self.x[:, None] - self.x[None, :] + shift)
        return u.math.exp(-0.5 * u.math.square(d / self.band_cells.a)) / (
            u.math.sqrt(2 * u.math.pi) * self.band_cells.a
        )

    def Postophase(self, pos):
        """Projects a 2D position to a 1D phase.

        This function converts a 2D coordinate in the environment into a 1D phase
        value based on the band cell's preferred angle and spacing.

        Args:
            pos (Array): The 2D position vector.

        Returns:
            float: The corresponding 1D phase.
        """
        phase = u.math.mod(u.math.dot(pos, self.proj_k), 2 * u.math.pi) - u.math.pi
        return phase

    def get_stimulus_by_pos(self, pos):
        """Generates a stimulus input based on a 2D position.

        This creates a Gaussian bump of input centered on the phase corresponding
        to the given position, which can be used to anchor the network's activity.

        Args:
            pos (Array): The 2D position vector.

        Returns:
            Array: The stimulus input vector for the band cells.
        """
        phase = self.Postophase(pos)
        d = self.dist(phase - self.x)
        return u.math.exp(-0.25 * u.math.square(d / self.band_cells.a))

    # move the heading direction representation (for testing)
    def move_heading(self, shift):
        """Manually shifts the activity bump in the band cells.

        This is a utility function for testing purposes.

        Args:
            shift (int): The number of neurons to roll the activity by.
        """
        self.band_cells.r.value = u.math.roll(self.band_cells.r, shift)
        self.band_cells.u.value = u.math.roll(self.band_cells.u, shift)

    def get_center(self):
        """Decodes and updates the current center of the band cell activity."""
        exppos = u.math.exp(1j * self.x)
        r = self.band_cells.r.value
        self.center.value = u.math.angle(u.math.atleast_1d(u.math.sum(exppos * r)))

    def reset(self):
        """Resets the synaptic inputs of the left and right shifter units."""
        self.left.u.value = u.math.zeros(self.size)
        self.right.u.value = u.math.zeros(self.size)

    def update(self, velocity, loc, loc_input_stre):
        """Updates the BandCell module for one time step.

        It integrates the component of `velocity` along the module's preferred
        direction to update the phase representation. The activity bump is shifted
        by modulating the inputs from the left/right shifter populations. It can
        also incorporate a direct location-based input.

        Args:
            velocity (Array): The 2D velocity vector.
            loc (Array): The current 2D location.
            loc_input_stre (float): The strength of the location-based input.
        """
        loc_input = jax.lax.cond(
            loc_input_stre != 0.0,
            lambda op: self.get_stimulus_by_pos(op[0]) * op[1],
            lambda op: u.math.zeros(self.size, dtype=float),
            operand=(loc, loc_input_stre),
        )
        # if loc_input_stre != 0.:
        #     loc_input = self.get_stimulus_by_pos(loc) * loc_input_stre
        # else:
        #     loc_input = u.math.zeros(self.size)

        v_phi = u.math.dot(velocity, self.proj_k)
        center_ideal = self.center_ideal.value + v_phi * brainstate.environ.get_dt()
        self.center_ideal.value = map2pi(center_ideal)
        # EPG output last time step
        Band_output = self.band_cells.r.value
        # PEN input
        left_input = self.syn_Band2Left(Band_output)
        right_input = self.syn_Band2Right(Band_output)
        # PEN output and gain
        self.left(left_input)
        self.right(right_input)
        self.left.r.value = (self.gain + v_phi) * self.left.r.value
        self.right.r.value = (self.gain - v_phi) * self.right.r.value
        # EPG input
        Band_input = self.syn_Left2Band(self.left.r.value) + self.syn_Right2Band(self.right.r.value)
        # EPG output
        self.band_cells(Band_input + loc_input)
        # self.Band_cells.update(loc_input)
        self.get_center()