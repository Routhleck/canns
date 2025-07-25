import brainstate
import brainunit as u
import jax
from brainstate.nn import exp_euler_step

from ...task.path_integration import map2pi
from ._base import BasicModel

__all__ = [
    "GridCell",
]


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
        value_grid (brainunit.math.ndarray): The (x, y) preferred phase coordinates for each neuron.
        conn_mat (brainunit.math.ndarray): The connection matrix.
        r (brainstate.HiddenState): The firing rates of the neurons.
        u (brainstate.HiddenState): The synaptic inputs to the neurons.
        v (brainstate.HiddenState): The adaptation variables for the neurons.
        center (brainstate.State): The decoded 2D center of the activity bump.
    """

    def __init__(
        self,
        num,
        angle,
        spacing,
        tau=0.1,
        tau_v=10.0,
        k=5e-3,
        a=u.math.pi / 9,
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
        super().__init__(self.num)
        # dynamics parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        # self.w_max = w_max
        self.ratio = u.math.pi * 2 / spacing
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.m = mbar * tau / tau_v
        self.angle = angle

        # feature space
        self.x_range = 2 * u.math.pi
        self.x = u.math.linspace(-u.math.pi, u.math.pi, num, endpoint=False)
        x_grid, y_grid = u.math.meshgrid(self.x, self.x)
        self.x_grid = x_grid.flatten()
        self.y_grid = y_grid.flatten()
        self.value_grid = u.math.stack([self.x_grid, self.y_grid]).T
        self.rho = self.num / (self.x_range**2)  # The neural density
        self.dxy = 1 / self.rho  # The stimulus density
        self.coor_transform = u.math.array([[1, -1 / u.math.sqrt(3)], [0, 2 / u.math.sqrt(3)]])
        self.rot = u.math.array(
            [
                [u.math.cos(self.angle), -u.math.sin(self.angle)],
                [u.math.sin(self.angle), u.math.cos(self.angle)],
            ]
        )

        # initialize conn matrix
        self.conn_mat = self.make_conn()

    def init_state(self, *args, **kwargs):
        self.r = brainstate.HiddenState(u.math.zeros(self.num))
        self.u = brainstate.HiddenState(u.math.zeros(self.num))
        self.v = brainstate.HiddenState(u.math.zeros(self.num))

        self.input = brainstate.State(u.math.zeros(self.num))
        self.center = brainstate.State(
            u.math.zeros(
                2,
            )
        )

    def reset_state(self, *args, **kwargs):
        """Resets the state variables of the model to zeros."""
        self.r.value = u.math.zeros(self.num)
        self.u.value = u.math.zeros(self.num)
        self.v.value = u.math.zeros(self.num)

        self.input.value = u.math.zeros(self.num)
        self.center.value = u.math.zeros(
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
        delta_y = (d[:, 1] - 1 / 2 * d[:, 0]) * 2 / u.math.sqrt(3)
        return u.math.sqrt(delta_x**2 + delta_y**2)

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
            Jxx = (
                self.J0
                * u.math.exp(-0.5 * u.math.square(d / self.a))
                / (u.math.sqrt(2 * u.math.pi) * self.a)
            )
            return Jxx

        return get_J(self.value_grid)

    def circle_period(self, d):
        """Wraps values into the periodic range [-pi, pi].

        Args:
            d (Array): The input values.

        Returns:
            Array: The wrapped values.
        """
        d = u.math.where(d > u.math.pi, d - 2 * u.math.pi, d)
        d = u.math.where(d < -u.math.pi, d + 2 * u.math.pi, d)
        return d

    def get_center(self):
        """Decodes and updates the 2D center of the activity bump.

        It uses a population vector average for both the x and y dimensions of the
        phase space.
        """
        exppos_x = u.math.exp(1j * self.x_grid)
        exppos_y = u.math.exp(1j * self.y_grid)
        r = u.math.where(self.r.value > u.math.max(self.r.value) * 0.1, self.r.value, 0)
        self.center.value = u.math.asarray(
            [u.math.angle(u.math.sum(exppos_x * r)), u.math.angle(u.math.sum(exppos_y * r))]
        )

    def update(self, input):
        self.input.value = input
        Irec = u.math.dot(self.conn_mat, self.r.value)
        # Update neural state
        self.v.value = exp_euler_step(
            lambda v: (-v + self.m * self.u.value) / self.tau_v,
            self.v.value,
        )
        self.u.value = exp_euler_step(
            lambda u, Irec: (-u + Irec + self.input.value - self.v.value) / self.tau,
            self.u.value,
            Irec,
        )
        self.u.value = u.math.where(self.u.value > 0, self.u.value, 0)
        # self.u.value += (
        #     (-self.u.value + Irec + self.input.value - self.v.value)
        #     / self.tau
        #     * brainstate.environ.get_dt()
        # )
        # self.u.value = u.math.where(self.u.value > 0, self.u.value, 0)
        # self.v.value += (
        #     (-self.v.value + self.m * self.u.value) / self.tau_v * brainstate.environ.get_dt()
        # )
        r1 = u.math.square(self.u.value)
        r2 = 1.0 + self.k * u.math.sum(r1)
        self.r.value = r1 / r2
        self.get_center()