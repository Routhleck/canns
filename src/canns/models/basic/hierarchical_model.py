import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp

from ...task.open_loop_navigation import map2pi
from ._base import BasicModel, BasicModelGroup

__all__ = [
    # Base Units
    "GaussRecUnits",
    "NonRecUnits",
    # Band Cell and Grid Cell Models
    "BandCell",
    "GridCell",
    # Hierarchical Path Integration Model
    "HierarchicalPathIntegrationModel",
    # Hierarchical Network
    "HierarchicalNetwork",
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
        x (bm.math.ndarray): The preferred feature values for each neuron.
        rho (float): The neural density (number of neurons per unit of feature space).
        dx (float): The stimulus density (feature space range per neuron).
        J (float): The final connection strength, scaled by J0.
        conn_mat (bm.math.ndarray): The connection matrix.
        r (bm.Variable): The firing rates of the neurons.
        u (bm.Variable): The synaptic inputs to the neurons.
        center (bm.Variable): The decoded center of the activity bump.
        input (bm.Variable): The external input to the network.
    """

    def __init__(
        self,
        size: int,
        tau: float = 1.0,
        J0: float = 1.1,
        k: float = 5e-4,
        a: float = 2 / 9 * bm.pi,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
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
        super().__init__()
        self.tau = tau  # The time constant
        self.k = k  # The inhibition strength
        self.a = a  # The width of the Gaussian connection
        self.noise_0 = noise  # The noise level

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, size, endpoint=False)  # The encoded feature values
        self.rho = size / self.z_range  # The neural density
        self.dx = self.z_range / size  # The stimulus density

        self.J = J0 * self.Jc()  # The connection strength
        self.conn_mat = self.make_conn()  # The connection matrix

        self.r = bm.Variable(bm.zeros(self.size))  # The neural firing rate
        self.u = bm.Variable(bm.zeros(self.size))  # The neural synaptic input
        self.center = bm.Variable(
            bm.zeros(
                1,
            )
        )  # The center of the bump

        self.input = bm.Variable(bm.zeros(self.size))  # The external input

        # initialize the neural activity
        self.u.value = (
            10.0 * bm.exp(-0.5 * bm.square((self.x - 0) / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
        )
        self.r.value = (
            30.0 * bm.exp(-0.5 * bm.square((self.x - 0) / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
        )

    # make the connection matrix
    def make_conn(self):
        """Constructs the periodic Gaussian connection matrix.

        The connection strength between two neurons depends on the periodic distance
        between their preferred feature values, following a Gaussian profile.
        """
        dis = self.x[:, None] - self.x[None, :]
        d = self.dist(dis)
        return self.J * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)

    # critical connection strength
    def Jc(self):
        """Calculates the critical connection strength.

        This is the minimum connection strength required to sustain a stable
        activity bump in the attractor network.
        """
        return bm.sqrt(8 * bm.sqrt(2 * bm.pi) * self.k * self.a / self.rho)

    # truncate the distance into the range of feature space
    def dist(self, d):
        """Calculates the periodic distance in the feature space.

        This function wraps distances to ensure they fall within the periodic
        boundaries of the feature space, i.e., [-z_range/2, z_range/2].

        Args:
            d (bm.math.ndarray): The array of distances.
        """
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
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
        expo_r = bm.exp(1j * self.x) * r
        return bm.angle(bm.sum(expo_r, axis=axis) / bm.sum(r, axis=axis))

    # update the neural activity
    def update(self, input):
        self.input.value = input
        r1 = bm.square(self.u.value)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        Irec = bm.dot(self.conn_mat, self.r.value)
        self.u.value = (
            self.u.value + (-self.u.value + Irec + self.input.value) / self.tau * bm.get_dt()
        )
        self.input.value = self.input.value.at[:].set(0.0)
        self.center.value = self.center.value.at[0].set(self.decode(self.u.value))


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
        x (bm.ndarray): The preferred feature values for each neuron.
        rho (float): The neural density.
        dx (float): The stimulus density.
        r (bm.Variable): The firing rates of the neurons.
        u (bm.Variable): The synaptic inputs to the neurons.
        input (bm.Variable): The external input to the neurons.
    """

    def __init__(
        self,
        size: int,
        tau: float = 0.1,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
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
        super().__init__()
        self.size = size
        self.noise_0 = noise  # The noise level

        self.tau = tau  # The time constant

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, size, endpoint=False)  # The encoded feature values
        self.rho = size / self.z_range  # The neural density
        self.dx = self.z_range / size  # The stimulus density

        self.r = bm.Variable(bm.zeros(self.size))  # The neural firing rate
        self.u = bm.Variable(bm.zeros(self.size))  # The neural synaptic input
        self.input = bm.Variable(bm.zeros(self.size))  # The external input

    # choose the activation function
    def activate(self, x):
        """Applies an activation function to the input.

        Args:
            x (Array): The input to the activation function (e.g., synaptic input `u`).

        Returns:
            Array: The result of the activation function (ReLU).
        """
        return bm.relu(x)

    def dist(self, d):
        """Calculates the periodic distance in the feature space.

        This function wraps distances to ensure they fall within the periodic
        boundaries of the feature space.

        Args:
            d (Array): The array of distances.

        Returns:
            Array: The wrapped distances.
        """
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def update(self, input):
        self.input.value = input
        self.r.value = bm.where(
            self.noise_0 != 0.0,
            self.activate(self.u.value) + self.noise_0 * bm.random.randn(self.size),
            self.activate(self.u.value),
        )
        # self.r.value = self.activate(self.u.value) + self.noise_0 * bm.random.randn(
        #     self.size
        # )
        self.u.value = self.u.value + (-self.u.value + self.input.value) / self.tau * bm.get_dt()
        self.input.value = self.input.value.at[:].set(0.0)
        return self.r.value


# the intact networks contains a group of EPG neurons (recurrent units), two P-EN neurons (non-recurrent units), one group of
# FC2 (recurrent units), two PFL3 (non-recurrent units) and two DN neurons (non-recurrent units)


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
        proj_k (bm.math.ndarray): The projection vector for converting 2D position/velocity to 1D phase.
        band_cells (GaussRecUnits): The core recurrent network representing the phase.
        left (NonRecUnits): A population of non-recurrent units for positive shifts.
        right (NonRecUnits): A population of non-recurrent units for negative shifts.
        w_L2S (float): Connection weight from band cells to left/right units.
        w_S2L (float): Connection weight from left/right units to band cells.
        gain (float): A gain factor for velocity-modulated input.
        center_ideal (bm.Variable): The ideal, noise-free center based on velocity integration.
        center (bm.Variable): The actual decoded center of the band cell activity bump.
    """

    def __init__(
        self,
        angle,
        spacing,
        size=180,
        z_min=-bm.pi,
        z_max=bm.pi,
        noise=2.0,
        w_L2S=0.2,
        w_S2L=1.0,
        gain=0.2,
        # GaussRecUnits configuration
        gauss_tau=1.0,
        gauss_J0=1.1,
        gauss_k=5e-4,
        gauss_a=2 / 9 * bm.pi,
        # NonRecUnits configuration
        nonrec_tau=0.1,
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
            gauss_tau (float, optional): Time constant for GaussRecUnits. Defaults to 1.0.
            gauss_J0 (float, optional): Connection strength scaling factor for GaussRecUnits. Defaults to 1.1.
            gauss_k (float, optional): Global inhibition strength for GaussRecUnits. Defaults to 5e-4.
            gauss_a (float, optional): Gaussian connection width for GaussRecUnits. Defaults to 2/9*pi.
            nonrec_tau (float, optional): Time constant for NonRecUnits. Defaults to 0.1.
            **kwargs: Additional keyword arguments for the base class.
        """
        self.size = size  # The number of neurons in each neuron group except DN
        super().__init__(**kwargs)

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.linspace(z_min, z_max, size, endpoint=False)  # The encoded feature values
        self.rho = size / self.z_range  # The neural density
        self.dx = self.z_range / size  # The stimulus density
        self.spacing = spacing
        self.angle = angle
        self.proj_k = (
            bm.array([bm.cos(angle - bm.pi / 2), bm.sin(angle - bm.pi / 2)]) * 2 * bm.pi / spacing
        )

        # shifts
        self.phase_shift = 1 / 9 * bm.pi * 0.76  # the shift of the connection from PEN to EPG
        # self.PFL3_shift = 3/8*bm.pi # the shift of the connection from EPG to PFL3
        # self.PEN_shift_num = int(self.PEN_shift / self.dx) # the number of interval shifted
        # self.PFL3_shift_num = int(self.PFL3_shift / self.dx) # the number of interval shifted

        # neurons - create with custom parameters
        self.band_cells = GaussRecUnits(
            size=size,
            tau=gauss_tau,
            J0=gauss_J0,
            k=gauss_k,
            a=gauss_a,
            z_min=z_min,
            z_max=z_max,
            noise=noise,
        )  # heading direction
        self.left = NonRecUnits(size=size, tau=nonrec_tau, z_min=z_min, z_max=z_max, noise=noise)
        self.right = NonRecUnits(size=size, tau=nonrec_tau, z_min=z_min, z_max=z_max, noise=noise)

        # weights
        self.w_L2S = w_L2S
        self.w_S2L = w_S2L
        self.gain = gain
        self.synapses()

        self.center_ideal = bm.Variable(
            bm.zeros(
                1,
            )
        )  # The center of v-
        self.center = bm.Variable(
            bm.zeros(
                1,
            )
        )  # The center of v-

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
        self.syn_Band2Left = bp.dnn.OneToOne(self.size, self.w_L2S)
        self.syn_Band2Right = bp.dnn.OneToOne(self.size, self.w_L2S)
        self.syn_Left2Band = bp.dnn.Linear(self.size, self.size, self.W_PENl2EPG)
        self.syn_Right2Band = bp.dnn.Linear(self.size, self.size, self.W_PENr2EPG)

    def dist(self, d):
        """Calculates the periodic distance in the feature space.

        Args:
            d (Array): The array of distances.

        Returns:
            Array: The wrapped distances.
        """
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
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
        return bm.exp(-0.5 * bm.square(d / self.band_cells.a)) / (
            bm.sqrt(2 * bm.pi) * self.band_cells.a
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
        phase = bm.mod(bm.dot(pos, self.proj_k), 2 * bm.pi) - bm.pi
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
        return bm.exp(-0.25 * bm.square(d / self.band_cells.a))

    # move the heading direction representation (for testing)
    def move_heading(self, shift):
        """Manually shifts the activity bump in the band cells.

        This is a utility function for testing purposes.

        Args:
            shift (int): The number of neurons to roll the activity by.
        """
        self.band_cells.r.value = bm.roll(self.band_cells.r, shift)
        self.band_cells.u.value = bm.roll(self.band_cells.u, shift)

    def get_center(self):
        """Decodes and updates the current center of the band cell activity."""
        exppos = bm.exp(1j * self.x)
        r = self.band_cells.r.value
        self.center.value = bm.angle(bm.atleast_1d(bm.sum(exppos * r)))

    def reset(self):
        """Resets the synaptic inputs of the left and right shifter units."""
        self.left.u.value = bm.zeros(self.size)
        self.right.u.value = bm.zeros(self.size)

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
            lambda op: bm.zeros(self.size, dtype=float).value,
            operand=(loc, loc_input_stre),
        )
        # if loc_input_stre != 0.:
        #     loc_input = self.get_stimulus_by_pos(loc) * loc_input_stre
        # else:
        #     loc_input = bm.zeros(self.size)

        v_phi = bm.dot(velocity, self.proj_k)
        center_ideal = self.center_ideal.value + v_phi * bm.get_dt()
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


# Grid cell model modules
class GridCell(BasicModel):
    """A model of a grid cell module using a 2D continuous attractor network.

    This class implements a 2D continuous attractor network on a toroidal manifold
    to model the firing patterns of grid cells. The network dynamics include
    synaptic depression or adaptation, which helps stabilize the activity bumps.
    The connectivity is defined on a hexagonal grid structure.

    Attributes:
        num (int): The total number of neurons (num_side x num_side).
        tau (float): The synaptic time constant for `u`.
        tau_v (float): The time constant for the adaptation variable `v`.
        k (float): The degree of rescaled inhibition.
        a (float): The half-width of the excitatory connection range.
        A (float): The magnitude of the external input.
        J0 (float): The maximum connection value.
        m (float): The strength of the adaptation.
        angle (float): The orientation of the grid.
        value_grid (bm.math.ndarray): The (x, y) preferred phase coordinates for each neuron.
        conn_mat (bm.math.ndarray): The connection matrix.
        r (bm.Variable): The firing rates of the neurons.
        u (bm.Variable): The synaptic inputs to the neurons.
        v (bm.Variable): The adaptation variables for the neurons.
        center (bm.Variable): The decoded 2D center of the activity bump.
    """

    def __init__(
        self,
        num,
        angle,
        spacing,
        tau=0.1,
        tau_v=10.0,
        k=5e-3,
        a=bm.pi / 9,
        A=1.0,
        J0=1.0,
        mbar=1.0,
    ):
        """Initializes the GridCell model.

        Args:
            num (int): The number of neurons along one dimension of the square grid.
            angle (float): The orientation angle of the grid pattern.
            spacing (float): The spacing of the grid pattern.
            tau (float, optional): The synaptic time constant. Defaults to 0.1.
            tau_v (float, optional): The adaptation time constant. Defaults to 10.0.
            k (float, optional): The strength of global inhibition. Defaults to 5e-3.
            a (float, optional): The width of the connection profile. Defaults to pi/9.
            A (float, optional): The magnitude of external input. Defaults to 1.0.
            J0 (float, optional): The maximum connection strength. Defaults to 1.0.
            mbar (float, optional): The base strength of adaptation. Defaults to 1.0.
        """
        self.num = num**2
        super().__init__()
        # dynamics parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        # self.w_max = w_max
        self.ratio = bm.pi * 2 / spacing
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.m = mbar * tau / tau_v
        self.angle = angle

        # feature space
        self.x_range = 2 * bm.pi
        self.x = bm.linspace(-bm.pi, bm.pi, num, endpoint=False)
        x_grid, y_grid = bm.meshgrid(self.x, self.x)
        self.x_grid = x_grid.flatten()
        self.y_grid = y_grid.flatten()
        self.value_grid = bm.stack([self.x_grid, self.y_grid]).T
        self.rho = self.num / (self.x_range**2)  # The neural density
        self.dxy = 1 / self.rho  # The stimulus density
        self.coor_transform = bm.array([[1, -1 / bm.sqrt(3)], [0, 2 / bm.sqrt(3)]])
        self.rot = bm.array(
            [
                [bm.cos(self.angle), -bm.sin(self.angle)],
                [bm.sin(self.angle), bm.cos(self.angle)],
            ]
        )

        # initialize conn matrix
        self.conn_mat = self.make_conn()

        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))

        self.input = bm.Variable(bm.zeros(self.num))
        self.center = bm.Variable(
            bm.zeros(
                2,
            )
        )

        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, Irec: (-u + Irec + self.input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def reset_state(self, *args, **kwargs):
        """Resets the state variables of the model to zeros."""
        self.r.value = bm.zeros(self.num)
        self.u.value = bm.zeros(self.num)
        self.v.value = bm.zeros(self.num)

        self.input.value = bm.zeros(self.num)
        self.center.value = bm.zeros(
            2,
        )

    def dist(self, d):
        """Calculates the distance on the hexagonal grid.

        It first maps the periodic difference vector `d` into a Cartesian
        coordinate system that reflects the hexagonal lattice structure and then
        computes the Euclidean distance.

        Args:
            d (Array): An array of difference vectors in the phase space.

        Returns:
            Array: The corresponding distances on the hexagonal lattice.
        """
        d = map2pi(d)
        delta_x = d[:, 0]
        delta_y = (d[:, 1] - 1 / 2 * d[:, 0]) * 2 / bm.sqrt(3)
        return bm.sqrt(delta_x**2 + delta_y**2)

    def make_conn(self):
        """Constructs the connection matrix for the 2D attractor network.

        The connection strength between two neurons is a Gaussian function of the
        hexagonal distance between their preferred phases.

        Returns:
            Array: The connection matrix (num x num).
        """

        @jax.vmap
        def get_J(v):
            d = self.dist(v - self.value_grid)
            Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
            return Jxx

        return get_J(self.value_grid)

    def circle_period(self, d):
        """Wraps values into the periodic range [-pi, pi].

        Args:
            d (Array): The input values.

        Returns:
            Array: The wrapped values.
        """
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2 * bm.pi, d)
        return d

    def get_center(self):
        """Decodes and updates the 2D center of the activity bump.

        It uses a population vector average for both the x and y dimensions of the
        phase space.
        """
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        r = bm.where(self.r.value > bm.max(self.r.value) * 0.1, self.r.value, 0)
        self.center.value = bm.asarray(
            [bm.angle(bm.sum(exppos_x * r)), bm.angle(bm.sum(exppos_y * r))]
        )

    def update(self, input):
        self.input.value = input
        Irec = bm.dot(self.conn_mat, self.r.value)
        # Update neural state
        _u, _v = self.integral(self.u, self.v, bp.share["t"], Irec, bm.dt)

        self.u.value = bm.where(_u > 0, _u, 0)
        self.v.value = _v
        # self.u.value += (
        #     (-self.u.value + Irec + self.input.value - self.v.value)
        #     / self.tau
        #     * bm.get_dt()
        # )
        # self.u.value = bm.where(self.u.value > 0, self.u.value, 0)
        # self.v.value += (
        #     (-self.v.value + self.m * self.u.value) / self.tau_v * bm.get_dt()
        # )
        r1 = bm.square(self.u.value)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        self.get_center()


class HierarchicalPathIntegrationModel(BasicModelGroup):
    """A hierarchical model combining band cells and grid cells for path integration.

    This model forms a single grid module. It consists of three `BandCell` modules,
    each with a different preferred orientation (separated by 60 degrees), and one
    `GridCell` module. The band cells integrate velocity along their respective
    directions, and their combined outputs provide the input to the `GridCell`
    network, effectively driving the grid cell's activity bump. The model can
    also project its grid cell activity to a population of place cells.

    Attributes:
        band_cell_x (BandCell): The first band cell module (orientation `angle`).
        band_cell_y (BandCell): The second band cell module (orientation `angle` + 60 deg).
        band_cell_z (BandCell): The third band cell module (orientation `angle` + 120 deg).
        grid_cell (GridCell): The grid cell module driven by the band cells.
        place_center (bm.math.ndarray): The center locations of the target place cells.
        Wg2p (bm.math.ndarray): The connection weights from grid cells to place cells.
        grid_output (bm.Variable): The activity of the place cells.
    """

    def __init__(
        self,
        spacing,
        angle,
        place_center=None,
        # BandCell configuration
        band_size=180,
        band_noise=0.0,
        band_w_L2S=0.2,
        band_w_S2L=1.0,
        band_gain=0.2,
        # GridCell configuration
        grid_num=20,
        grid_tau=0.1,
        grid_tau_v=10.0,
        grid_k=5e-3,
        grid_a=bm.pi / 9,
        grid_A=1.0,
        grid_J0=1.0,
        grid_mbar=1.0,
        # GaussRecUnits configuration (for BandCell)
        gauss_tau=1.0,
        gauss_J0=1.1,
        gauss_k=5e-4,
        gauss_a=2 / 9 * bm.pi,
        # NonRecUnits configuration (for BandCell)
        nonrec_tau=0.1,
    ):
        """Initializes the HierarchicalPathIntegrationModel.

        Args:
            spacing (float): The spacing of the grid pattern for this module.
            angle (float): The base orientation angle for the module.
            place_center (bm.math.ndarray, optional): The center locations of the
                target place cell population. Defaults to a random distribution.
            band_size (int, optional): Number of neurons in each BandCell group. Defaults to 180.
            band_noise (float, optional): Noise level for BandCells. Defaults to 0.0.
            band_w_L2S (float, optional): Weight from band cells to shifter units. Defaults to 0.2.
            band_w_S2L (float, optional): Weight from shifter units to band cells. Defaults to 1.0.
            band_gain (float, optional): Gain factor for velocity signal in BandCells. Defaults to 0.2.
            grid_num (int, optional): Number of neurons per dimension for GridCell. Defaults to 20.
            grid_tau (float, optional): Synaptic time constant for GridCell. Defaults to 0.1.
            grid_tau_v (float, optional): Adaptation time constant for GridCell. Defaults to 10.0.
            grid_k (float, optional): Global inhibition strength for GridCell. Defaults to 5e-3.
            grid_a (float, optional): Connection width for GridCell. Defaults to pi/9.
            grid_A (float, optional): External input magnitude for GridCell. Defaults to 1.0.
            grid_J0 (float, optional): Maximum connection strength for GridCell. Defaults to 1.0.
            grid_mbar (float, optional): Base adaptation strength for GridCell. Defaults to 1.0.
            gauss_tau (float, optional): Time constant for GaussRecUnits in BandCells. Defaults to 1.0.
            gauss_J0 (float, optional): Connection strength scaling for GaussRecUnits. Defaults to 1.1.
            gauss_k (float, optional): Global inhibition for GaussRecUnits. Defaults to 5e-4.
            gauss_a (float, optional): Connection width for GaussRecUnits. Defaults to 2/9*pi.
            nonrec_tau (float, optional): Time constant for NonRecUnits in BandCells. Defaults to 0.1.
        """
        super().__init__()
        # Create BandCell instances with custom parameters
        self.band_cell_x = BandCell(
            angle=angle,
            spacing=spacing,
            size=band_size,
            noise=band_noise,
            w_L2S=band_w_L2S,
            w_S2L=band_w_S2L,
            gain=band_gain,
            gauss_tau=gauss_tau,
            gauss_J0=gauss_J0,
            gauss_k=gauss_k,
            gauss_a=gauss_a,
            nonrec_tau=nonrec_tau,
        )
        self.band_cell_y = BandCell(
            angle=angle + bm.pi / 3,
            spacing=spacing,
            size=band_size,
            noise=band_noise,
            w_L2S=band_w_L2S,
            w_S2L=band_w_S2L,
            gain=band_gain,
            gauss_tau=gauss_tau,
            gauss_J0=gauss_J0,
            gauss_k=gauss_k,
            gauss_a=gauss_a,
            nonrec_tau=nonrec_tau,
        )
        self.band_cell_z = BandCell(
            angle=angle + bm.pi / 3 * 2,
            spacing=spacing,
            size=band_size,
            noise=band_noise,
            w_L2S=band_w_L2S,
            w_S2L=band_w_S2L,
            gain=band_gain,
            gauss_tau=gauss_tau,
            gauss_J0=gauss_J0,
            gauss_k=gauss_k,
            gauss_a=gauss_a,
            nonrec_tau=nonrec_tau,
        )
        # Create GridCell instance with custom parameters
        self.grid_cell = GridCell(
            num=grid_num,
            angle=angle,
            spacing=spacing,
            tau=grid_tau,
            tau_v=grid_tau_v,
            k=grid_k,
            a=grid_a,
            A=grid_A,
            J0=grid_J0,
            mbar=grid_mbar,
        )
        self.proj_k_x = self.band_cell_x.proj_k
        self.proj_k_y = self.band_cell_y.proj_k
        self.place_center = (
            place_center if place_center is not None else 10 * bm.random.rand(512, 2)
        )
        self.make_conn()
        self.make_Wg2p()
        self.num_place = place_center.shape[0]
        self.coor_transform = bm.array([[1, -1 / bm.sqrt(3)], [0, 2 / bm.sqrt(3)]])

        self.grid_output = bm.Variable(bm.zeros(self.num_place))

    def make_conn(self):
        """Creates the connection matrices from the band cells to the grid cells.

        The connection from a band cell to a grid cell is strong if the grid cell's
        preferred phase along the band cell's direction matches the band cell's
        preferred phase.
        """

        value_grid = self.grid_cell.value_grid
        band_x = self.band_cell_x.x
        band_y = self.band_cell_y.x
        band_z = self.band_cell_z.x
        J0 = self.grid_cell.J0 * 0.1
        grid_x = value_grid[:, 0]
        grid_y = value_grid[:, 1]
        # Calculate the distance between each grid cell and band cell
        grid_vector = bm.zeros(value_grid.shape)
        grid_vector = grid_vector.at[:, 0].set(value_grid[:, 0])
        grid_vector = grid_vector.at[:, 1].set(
            (value_grid[:, 1] - 1 / 2 * value_grid[:, 0]) * 2 / bm.sqrt(3)
        )
        z_vector = bm.array([-1 / 2, bm.sqrt(3) / 2])
        grid_phase_z = bm.dot(grid_vector, z_vector)
        dis_x = self.band_cell_x.dist(grid_x[:, None] - band_x[None, :])
        dis_y = self.band_cell_y.dist(grid_y[:, None] - band_y[None, :])
        dis_z = self.band_cell_z.dist(grid_phase_z[:, None] - band_z[None, :])
        self.W_x_grid = (
            J0
            * bm.exp(-0.5 * bm.square(dis_x / self.band_cell_x.band_cells.a))
            / (bm.sqrt(2 * bm.pi) * self.band_cell_x.band_cells.a)
        )
        self.W_y_grid = (
            J0
            * bm.exp(-0.5 * bm.square(dis_y / self.band_cell_y.band_cells.a))
            / (bm.sqrt(2 * bm.pi) * self.band_cell_y.band_cells.a)
        )
        self.W_z_grid = (
            J0
            * bm.exp(-0.5 * bm.square(dis_z / self.band_cell_z.band_cells.a))
            / (bm.sqrt(2 * bm.pi) * self.band_cell_z.band_cells.a)
        )

    def Postophase(self, pos):
        """Projects a 2D position to the 2D phase space of the grid module.

        Args:
            pos (Array): The 2D position vector.

        Returns:
            Array: The corresponding 2D phase vector.
        """
        phase_x = bm.mod(bm.dot(pos, self.proj_k_x), 2 * bm.pi) - bm.pi
        phase_y = bm.mod(bm.dot(pos, self.proj_k_y), 2 * bm.pi) - bm.pi
        return bm.array([phase_x, phase_y]).transpose()

    def make_Wg2p(self):
        """Creates the connection weights from grid cells to place cells.

        The connection strength is determined by the proximity of a place cell's
        center to a grid cell's firing field, calculated in the phase domain.
        """
        phase_place = self.Postophase(self.place_center)
        phase_grid = self.grid_cell.value_grid
        d = phase_place[:, jnp.newaxis, :] - phase_grid[jnp.newaxis, :, :]
        d = map2pi(d)
        delta_x = d[:, :, 0]
        delta_y = (d[:, :, 1] - 1 / 2 * d[:, :, 0]) * 2 / bm.sqrt(3)
        # delta_x = d[:,:,0] + d[:,:,1]/2
        # delta_y = d[:,:,1] * bm.sqrt(3) / 2
        dis = bm.sqrt(delta_x**2 + delta_y**2)
        Wg2p = bm.exp(-0.5 * bm.square(dis / self.band_cell_x.band_cells.a)) / (
            bm.sqrt(2 * bm.pi) * self.band_cell_x.band_cells.a
        )
        self.Wg2p = Wg2p

    def dist(self, d):
        """Calculates the distance on the hexagonal grid.

        Args:
            d (Array): An array of difference vectors in the phase space.

        Returns:
            Array: The corresponding distances on the hexagonal lattice.
        """
        d = map2pi(d)
        delta_x = d[:, 0]
        delta_y = (d[:, 1] - 1 / 2 * d[:, 0]) * 2 / bm.sqrt(3)
        return bm.sqrt(delta_x**2 + delta_y**2)

    def get_input(self, Phase):
        """Generates a stimulus input for the grid cell based on a 2D phase.

        Args:
            Phase (Array): The 2D phase vector.

        Returns:
            Array: The stimulus input vector for the grid cells.
        """
        dis = self.dist(Phase - self.grid_cell.value_grid)
        return bm.exp(-0.5 * bm.square(dis / self.band_cell_x.band_cells.a)) / (
            bm.sqrt(2 * bm.pi) * self.band_cell_x.band_cells.a
        )

    def update(self, velocity, loc, loc_input_stre=0.0):
        self.band_cell_x(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
        self.band_cell_y(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
        self.band_cell_z(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
        band_output = (
            self.W_x_grid @ self.band_cell_x.band_cells.r.value
            + self.W_y_grid @ self.band_cell_y.band_cells.r.value
            + self.W_z_grid @ self.band_cell_z.band_cells.r.value
        )
        # band_output = (self.W_x_grid @ self.band_cell_x.Band_cells.r + self.W_y_grid @ self.band_cell_y.Band_cells.r)
        max_output = bm.max(band_output)
        band_output = bm.where(band_output > max_output / 2, band_output - max_output / 2, 0)
        phase_x = self.band_cell_x.center.value
        phase_y = self.band_cell_y.center.value
        Phase = bm.array([phase_x, phase_y]).transpose()
        # Phase = self.Postophase(loc)
        loc_input = self.get_input(Phase) * 5000
        self.grid_cell.update(input=loc_input)
        grid_fr = self.grid_cell.r.value
        # self.grid_output = bm.dot(self.Wg2p, grid_fr-bm.max(grid_fr)/2)
        self.grid_output.value = bm.dot(self.Wg2p, grid_fr)

        # band_cell_x_states = self.band_cell_x.states()
        # band_cell_y_states = self.band_cell_y.states()
        # band_cell_z_states = self.band_cell_z.states()
        # gird_cell_states = self.grid_cell.states()
        #
        # return {
        #     'band_cell_x': band_cell_x_states,
        #     'band_cell_y': band_cell_y_states,
        #     'band_cell_z': band_cell_z_states,
        #     'grid_cell': gird_cell_states,
        #
        #     'gird_fr': gird_cell_states['r'],
        #     'band_x_fr': band_cell_x_states['band_cells']['r'],
        #     'band_y_fr': band_cell_y_states['band_cells']['r'],
        #     'grid_output': self.grid_output,
        # }


class HierarchicalNetwork(BasicModelGroup):
    """A full hierarchical network composed of multiple grid modules.

    This class creates and manages a collection of `HierarchicalPathIntegrationModel`
    modules, each with a different grid spacing. By combining the outputs of these
    modules, the network can represent position unambiguously over a large area.
    The final output is a population of place cells whose activities are used to
    decode the animal's estimated position.

    Attributes:
        num_module (int): The number of grid modules in the network.
        num_place (int): The number of place cells in the output layer.
        place_center (bm.math.ndarray): The center locations of the place cells.
        MEC_model_list (list): A list containing all the `HierarchicalPathIntegrationModel` instances.
        grid_fr (bm.Variable): The firing rates of the grid cell population.
        band_x_fr (bm.Variable): The firing rates of the x-oriented band cell population.
        band_y_fr (bm.Variable): The firing rates of the y-oriented band cell population.
        place_fr (bm.Variable): The firing rates of the place cell population.
        decoded_pos (bm.Variable): The final decoded 2D position.

    References:
        Anonymous Author(s) "Unfolding the Black Box of Recurrent Neural Networks for Path Integration" (under review).
    """

    def __init__(
        self,
        num_module,
        num_place,
        # Module spacing configuration
        spacing_min=2.0,
        spacing_max=5.0,
        module_angle=0.0,
        # BandCell configuration
        band_size=180,
        band_noise=0.0,
        band_w_L2S=0.2,
        band_w_S2L=1.0,
        band_gain=0.2,
        # GridCell configuration
        grid_num=20,
        grid_tau=0.1,
        grid_tau_v=10.0,
        grid_k=5e-3,
        grid_a=bm.pi / 9,
        grid_A=1.0,
        grid_J0=1.0,
        grid_mbar=1.0,
        # GaussRecUnits configuration (for BandCell)
        gauss_tau=1.0,
        gauss_J0=1.1,
        gauss_k=5e-4,
        gauss_a=2 / 9 * bm.pi,
        # NonRecUnits configuration (for BandCell)
        nonrec_tau=0.1,
    ):
        """Initializes the HierarchicalNetwork.

        Args:
            num_module (int): The number of grid modules to create.
            num_place (int): The number of place cells along one dimension of a square grid.
            spacing_min (float, optional): Minimum spacing for grid modules. Defaults to 2.0.
            spacing_max (float, optional): Maximum spacing for grid modules. Defaults to 5.0.
            module_angle (float, optional): Base orientation angle for all modules. Defaults to 0.0.
            band_size (int, optional): Number of neurons in each BandCell group. Defaults to 180.
            band_noise (float, optional): Noise level for BandCells. Defaults to 0.0.
            band_w_L2S (float, optional): Weight from band cells to shifter units. Defaults to 0.2.
            band_w_S2L (float, optional): Weight from shifter units to band cells. Defaults to 1.0.
            band_gain (float, optional): Gain factor for velocity signal in BandCells. Defaults to 0.2.
            grid_num (int, optional): Number of neurons per dimension for GridCell. Defaults to 20.
            grid_tau (float, optional): Synaptic time constant for GridCell. Defaults to 0.1.
            grid_tau_v (float, optional): Adaptation time constant for GridCell. Defaults to 10.0.
            grid_k (float, optional): Global inhibition strength for GridCell. Defaults to 5e-3.
            grid_a (float, optional): Connection width for GridCell. Defaults to pi/9.
            grid_A (float, optional): External input magnitude for GridCell. Defaults to 1.0.
            grid_J0 (float, optional): Maximum connection strength for GridCell. Defaults to 1.0.
            grid_mbar (float, optional): Base adaptation strength for GridCell. Defaults to 1.0.
            gauss_tau (float, optional): Time constant for GaussRecUnits in BandCells. Defaults to 1.0.
            gauss_J0 (float, optional): Connection strength scaling for GaussRecUnits. Defaults to 1.1.
            gauss_k (float, optional): Global inhibition for GaussRecUnits. Defaults to 5e-4.
            gauss_a (float, optional): Connection width for GaussRecUnits. Defaults to 2/9*pi.
            nonrec_tau (float, optional): Time constant for NonRecUnits in BandCells. Defaults to 0.1.
        """
        super().__init__()
        self.num_module = num_module
        self.num_place = num_place**2
        # randomly sample num_place place field centers from a square arena (5m x 5m)
        x = bm.linspace(0, 5, num_place)
        X, Y = bm.meshgrid(x, x)
        self.place_center = bm.stack([X.flatten(), Y.flatten()]).T
        # self.place_center = 5 * bm.random.rand(num_place,2)

        # load heatmaps_grid from heatmaps_grid.npz
        # data = np.load('heatmaps_grid.npz', allow_pickle=True)
        # heatmaps_grid = data['heatmaps_grid']
        # print(heatmaps_grid.shape)

        MEC_model_list = []
        # self.W_g2p_list = []
        spacing = bm.linspace(spacing_min, spacing_max, num_module)
        for i in range(num_module):
            MEC_model_list.append(
                HierarchicalPathIntegrationModel(
                    spacing=spacing[i],
                    angle=module_angle,
                    place_center=self.place_center,
                    band_size=band_size,
                    band_noise=band_noise,
                    band_w_L2S=band_w_L2S,
                    band_w_S2L=band_w_S2L,
                    band_gain=band_gain,
                    grid_num=grid_num,
                    grid_tau=grid_tau,
                    grid_tau_v=grid_tau_v,
                    grid_k=grid_k,
                    grid_a=grid_a,
                    grid_A=grid_A,
                    grid_J0=grid_J0,
                    grid_mbar=grid_mbar,
                    gauss_tau=gauss_tau,
                    gauss_J0=gauss_J0,
                    gauss_k=gauss_k,
                    gauss_a=gauss_a,
                    nonrec_tau=nonrec_tau,
                )
            )
            # W_g2p = self.W_place2grid(heatmaps_grid[i*400:(i+1)*400])
            # self.W_g2p_list.append(W_g2p)
        self.MEC_model_list = MEC_model_list

        self.place_fr = bm.Variable(bm.zeros(self.num_place))
        self.grid_fr = bm.Variable(bm.zeros((self.num_module, 20**2)))
        self.band_x_fr = bm.Variable(bm.zeros((self.num_module, 180)))
        self.band_y_fr = bm.Variable(bm.zeros((self.num_module, 180)))
        self.decoded_pos = bm.Variable(bm.zeros(2))

    def update(self, velocity, loc, loc_input_stre=0.0):
        grid_output = bm.zeros(self.num_place)
        for i in range(self.num_module):
            # update the band cell module
            self.MEC_model_list[i](velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
            self.grid_fr.value = self.grid_fr.value.at[i].set(
                self.MEC_model_list[i].grid_cell.r.value
            )
            self.band_x_fr.value = self.band_x_fr.value.at[i].set(
                self.MEC_model_list[i].band_cell_x.band_cells.r.value
            )
            self.band_y_fr.value = self.band_y_fr.value.at[i].set(
                self.MEC_model_list[i].band_cell_y.band_cells.r.value
            )
            grid_output_module = self.MEC_model_list[i].grid_output.value
            # W_g2p = self.W_g2p_list[i]
            # grid_fr = self.MEC_model_list[i].Grid_cell.r
            # grid_output_module = bm.dot(W_g2p, grid_fr)
            grid_output += grid_output_module
        # update the place cell module
        grid_output = bm.where(grid_output > 0, grid_output, 0)
        u_place = bm.where(
            grid_output > bm.max(grid_output) / 2, grid_output - bm.max(grid_output) / 2, 0
        )
        # grid_output = grid_output**2/(1+bm.sum(grid_output**2))
        # max_id = bm.argmax(grid_output)
        # center = self.place_center[max_id]
        center = bm.sum(self.place_center * u_place[:, jnp.newaxis], axis=0) / (
            1e-5 + bm.sum(u_place)
        )
        self.decoded_pos.value = center
        self.place_fr.value = u_place**2 / (1 + bm.sum(u_place**2))
        # self.place_fr = softmax(grid_output)

    # the optimized run function is not run well(the performance is not good enough, as the original one),
    '''
    def run(self, indices, velocities, positions, loc_input_stre=0.0, pbar=None):
        """Runs the hierarchical network for a series of time steps.

        Args:
            indices (Array): The indices of the time steps to run.
            velocities (Array): The 2D velocity vectors at each time step.
            positions (Array): The 2D position vectors at each time step.
            loc_input_stre (Array): The strength of the location-based input.
            p_bar (ProgressBar): A progress bar for tracking the simulation progress.
        """

        band_x_r = bm.zeros((indices.shape[0], self.num_module, 180))
        band_y_r = bm.zeros((indices.shape[0], self.num_module, 180))
        grid_r = bm.zeros((indices.shape[0], self.num_module, 20**2))
        grid_output = bm.zeros((indices.shape[0], self.num_place))
        loc_input_stre = bm.ones((indices.shape[0],)) * loc_input_stre

        for i, model in enumerate(self.MEC_model_list):
            def run_single_module(velocity, loc, loc_input_stre):
                model(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
                return (
                    model.band_cell_x.band_cells.r.value,
                    model.band_cell_y.band_cells.r.value,
                    model.grid_cell.r.value,
                    model.grid_output.value,
                )

            single_band_x_r, single_band_y_r, single_grid_r, single_grid_output = (
                bm.for_loop(
                    run_single_module,
                    velocities,
                    positions,
                    loc_input_stre,
                )
            )
            band_x_r = band_x_r.at[:, i, :].set(single_band_x_r)
            band_y_r = band_y_r.at[:, i, :].set(single_band_y_r)
            grid_r = grid_r.at[:, i, :].set(single_grid_r)
            grid_output += single_grid_output

        grid_output = bm.where(grid_output > 0, grid_output, 0)
        u_place = bm.where(
            grid_output > bm.max(grid_output, axis=1, keepdims=True) / 2,
            grid_output - bm.max(grid_output, axis=1, keepdims=True) / 2,
            0,
        )

        place_r = u_place**2 / (1 + bm.sum(u_place**2, axis=1, keepdims=True))

        return band_x_r, band_y_r, grid_r, place_r
    '''
