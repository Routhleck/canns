import brainstate
import brainunit as u
import jax
import numpy as np
from brainstate.nn import exp_euler_step

from ...task.path_integration import map2pi_jnp
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
        self.center = brainstate.State(u.math.zeros(1,))  # The center of the bump

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
        dis = self.x[:, None] - self.x[None, :]
        d = self.dist(dis)
        return (
            self.J
            * u.math.exp(-0.5 * u.math.square(d / self.a))
            / (u.math.sqrt(2 * u.math.pi) * self.a)
        )

    # critical connection strength
    def Jc(self):
        return u.math.sqrt(8 * u.math.sqrt(2 * u.math.pi) * self.k * self.a / self.rho)

    # truncate the distance into the range of feature space
    def dist(self, d):
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    # decode the neural activity
    def decode(self, r, axis=0):
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
            self.u.value + (-self.u.value + Irec + self.input.value) / self.tau * brainstate.environ.get_dt()
        )
        self.center = self.decode(self.u.value)


class NonRecUnits(BasicModel):
    def __init__(
        self,
        size: int,
        tau: float = 0.1,
        z_min: float = -u.math.pi,
        z_max: float = u.math.pi,
        noise: float = 2.0,
    ):
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
        return u.math.relu(x)

    def dist(self, d):
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def update(self, input):
        self.input.value = input
        self.r.value = self.activate(self.u.value) + self.noise_0 * brainstate.random.randn(self.size)
        self.u.value = self.u.value + (-self.u.value + self.input.value) / self.tau * brainstate.environ.get_dt()


# the intact networks contains a group of EPG neurons (recurrent units), two P-EN neurons (non-recurrent units), one group of
# FC2 (recurrent units), two PFL3 (non-recurrent units) and two DN neurons (non-recurrent units)


class BandCell(BasicModel):
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
        self.Band_cells = GaussRecUnits(size=size, noise=noise)  # heading direction
        self.left = NonRecUnits(size=size, noise=noise)
        self.right = NonRecUnits(size=size, noise=noise)

        # weights
        self.w_L2S = w_L2S
        self.w_S2L = w_S2L
        self.gain = gain
        self.synapses()

    def init_state(self, *args, **kwargs):
        self.center_ideal = brainstate.State(u.math.zeros(1,))  # The center of v-
        self.center = brainstate.State(u.math.zeros(1,))  # The center of v-

        # init heading direction
        self.Band_cells.init_state()
        # init left and right neurons
        self.left.init_state()
        self.right.init_state()

    # define the synapses
    def synapses(self):
        self.W_PENl2EPG = self.w_S2L * self.make_conn(self.phase_shift)
        self.W_PENr2EPG = self.w_S2L * self.make_conn(-self.phase_shift)
        # synapses
        self.syn_Band2Left = brainstate.nn.OneToOne(self.size, self.w_L2S)
        self.syn_Band2Right = brainstate.nn.OneToOne(self.size, self.w_L2S)
        self.syn_Left2Band = brainstate.nn.Linear(self.size, self.size, self.W_PENl2EPG)
        self.syn_Right2Band = brainstate.nn.Linear(self.size, self.size, self.W_PENr2EPG)

    def dist(self, d):
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self, shift):
        d = self.dist(self.x[:, None] - self.x[None, :] + shift)
        return u.math.exp(-0.5 * u.math.square(d / self.Band_cells.a)) / (
            u.math.sqrt(2 * u.math.pi) * self.Band_cells.a
        )

    def Postophase(self, pos):
        phase = u.math.mod(u.math.dot(pos, self.proj_k), 2 * u.math.pi) - u.math.pi
        return phase

    def get_stimulus_by_pos(self, pos):
        phase = self.Postophase(pos)
        d = self.dist(phase - self.x)
        return u.math.exp(-0.25 * u.math.square(d / self.Band_cells.a))

    # move the heading direction representation (for testing)
    def move_heading(self, shift):
        self.Band_cells.r.value = u.math.roll(self.Band_cells.r, shift)
        self.Band_cells.u.value = u.math.roll(self.Band_cells.u, shift)

    def get_center(self):
        exppos = u.math.exp(1j * self.x)
        r = self.Band_cells.r.value
        self.center.value = u.math.angle(u.math.atleast_1d(u.math.sum(exppos * r)))

    def reset(self):
        self.left.u.value = u.math.zeros(self.size)
        self.right.u.value = u.math.zeros(self.size)

    def update(self, velocity, loc, loc_input_stre):
        # location input
        loc_input = self.get_stimulus_by_pos(loc) * loc_input_stre

        v_phi = u.math.dot(velocity, self.proj_k)
        center_ideal = self.center_ideal.value + v_phi * brainstate.environ.get_dt()
        self.center_ideal.value = map2pi_jnp(center_ideal)
        # EPG output last time step
        Band_output = self.Band_cells.r.value
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
        self.Band_cells(Band_input + loc_input)
        # self.Band_cells.update(loc_input)
        self.get_center()


# Grid cell model modules
class GridCell(BasicModel):
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
        self.center = brainstate.State(u.math.zeros(2,))

    def reset_state(self, *args, **kwargs):
        self.r.value = u.math.zeros(self.num)
        self.u.value = u.math.zeros(self.num)
        self.v.value = u.math.zeros(self.num)

        self.input.value = u.math.zeros(self.num)
        self.center.value = u.math.zeros(2,)

    def dist(self, d):
        d = map2pi_jnp(d)
        delta_x = d[:, 0]
        delta_y = (d[:, 1] - 1 / 2 * d[:, 0]) * 2 / u.math.sqrt(3)
        return u.math.sqrt(delta_x**2 + delta_y**2)

    def make_conn(self):
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
        d = u.math.where(d > u.math.pi, d - 2 * u.math.pi, d)
        d = u.math.where(d < -u.math.pi, d + 2 * u.math.pi, d)
        return d

    def get_center(self):
        exppos_x = u.math.exp(1j * self.x_grid)
        exppos_y = u.math.exp(1j * self.y_grid)
        r = u.math.where(self.r.value > u.math.max(self.r.value) * 0.1, self.r.value, 0)
        self.center.value = u.math.asarray([u.math.angle(u.math.sum(exppos_x * r)), u.math.angle(u.math.sum(exppos_y * r))])

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


class HierarchicalPathIntegrationModel(BasicModelGroup):
    def __init__(self, spacing, angle, place_center=None):
        super().__init__()
        self.band_cell_x = BandCell(angle=angle, spacing=spacing, noise=0.0)
        self.band_cell_y = BandCell(angle=angle + u.math.pi / 3, spacing=spacing, noise=0.0)
        self.band_cell_z = BandCell(angle=angle + u.math.pi / 3 * 2, spacing=spacing, noise=0.0)
        self.Grid_cell = GridCell(num=20, angle=angle, spacing=spacing)
        self.proj_k_x = self.band_cell_x.proj_k
        self.proj_k_y = self.band_cell_y.proj_k
        self.place_center = (
            place_center if place_center is not None else 10 * brainstate.random.rand(512, 2)
        )
        self.make_conn()
        self.make_Wg2p()
        self.num_place = place_center.shape[0]
        self.coor_transform = u.math.array([[1, -1 / u.math.sqrt(3)], [0, 2 / u.math.sqrt(3)]])

    def init_state(self, *args, **kwargs):
        self.grid_output = brainstate.State(u.math.zeros(self.num_place))

        self.band_cell_x.init_state()
        self.band_cell_y.init_state()
        self.band_cell_z.init_state()
        self.Grid_cell.init_state()

    def make_conn(self):
        value_grid = self.Grid_cell.value_grid
        band_x = self.band_cell_x.x
        band_y = self.band_cell_y.x
        band_z = self.band_cell_z.x
        J0 = self.Grid_cell.J0 * 0.1
        grid_x = value_grid[:, 0]
        grid_y = value_grid[:, 1]
        # Calculate the distance between each grid cell and band cell
        grid_vector = u.math.zeros(value_grid.shape)
        grid_vector = grid_vector.at[:, 0].set(value_grid[:, 0])
        grid_vector = grid_vector.at[:, 1].set((value_grid[:, 1] - 1 / 2 * value_grid[:, 0]) * 2 / u.math.sqrt(3))
        z_vector = u.math.array([-1 / 2, u.math.sqrt(3) / 2])
        grid_phase_z = u.math.dot(grid_vector, z_vector)
        dis_x = self.band_cell_x.dist(grid_x[:, None] - band_x[None, :])
        dis_y = self.band_cell_y.dist(grid_y[:, None] - band_y[None, :])
        dis_z = self.band_cell_z.dist(grid_phase_z[:, None] - band_z[None, :])
        self.W_x_grid = (
            J0
            * u.math.exp(-0.5 * u.math.square(dis_x / self.band_cell_x.Band_cells.a))
            / (u.math.sqrt(2 * u.math.pi) * self.band_cell_x.Band_cells.a)
        )
        self.W_y_grid = (
            J0
            * u.math.exp(-0.5 * u.math.square(dis_y / self.band_cell_y.Band_cells.a))
            / (u.math.sqrt(2 * u.math.pi) * self.band_cell_y.Band_cells.a)
        )
        self.W_z_grid = (
            J0
            * u.math.exp(-0.5 * u.math.square(dis_z / self.band_cell_z.Band_cells.a))
            / (u.math.sqrt(2 * u.math.pi) * self.band_cell_z.Band_cells.a)
        )

    def Postophase(self, pos):
        phase_x = u.math.mod(u.math.dot(pos, self.proj_k_x), 2 * u.math.pi) - u.math.pi
        phase_y = u.math.mod(u.math.dot(pos, self.proj_k_y), 2 * u.math.pi) - u.math.pi
        return u.math.array([phase_x, phase_y]).transpose()

    def make_Wg2p(self):
        phase_place = self.Postophase(self.place_center)
        phase_grid = self.Grid_cell.value_grid
        d = phase_place[:, u.math.newaxis, :] - phase_grid[u.math.newaxis, :, :]
        d = map2pi_jnp(d)
        delta_x = d[:, :, 0]
        delta_y = (d[:, :, 1] - 1 / 2 * d[:, :, 0]) * 2 / u.math.sqrt(3)
        # delta_x = d[:,:,0] + d[:,:,1]/2
        # delta_y = d[:,:,1] * u.math.sqrt(3) / 2
        dis = u.math.sqrt(delta_x**2 + delta_y**2)
        Wg2p = u.math.exp(-0.5 * u.math.square(dis / self.band_cell_x.Band_cells.a)) / (
            u.math.sqrt(2 * u.math.pi) * self.band_cell_x.Band_cells.a
        )
        self.Wg2p = Wg2p

    def dist(self, d):
        d = map2pi_jnp(d)
        delta_x = d[:, 0]
        delta_y = (d[:, 1] - 1 / 2 * d[:, 0]) * 2 / u.math.sqrt(3)
        return u.math.sqrt(delta_x**2 + delta_y**2)

    def get_input(self, Phase):
        dis = self.dist(Phase - self.Grid_cell.value_grid)
        return u.math.exp(-0.5 * u.math.square(dis / self.band_cell_x.Band_cells.a)) / (
            u.math.sqrt(2 * u.math.pi) * self.band_cell_x.Band_cells.a
        )

    def update(self, velocity, loc, loc_input_stre=0.0):
        self.band_cell_x(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
        self.band_cell_y(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
        self.band_cell_z(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
        band_output = (
            self.W_x_grid @ self.band_cell_x.Band_cells.r.value
            + self.W_y_grid @ self.band_cell_y.Band_cells.r.value
            + self.W_z_grid @ self.band_cell_z.Band_cells.r.value
        )
        # band_output = (self.W_x_grid @ self.band_cell_x.Band_cells.r + self.W_y_grid @ self.band_cell_y.Band_cells.r)
        max_output = u.math.max(band_output)
        band_output = u.math.where(band_output > max_output / 2, band_output - max_output / 2, 0)
        phase_x = self.band_cell_x.center.value
        phase_y = self.band_cell_y.center.value
        Phase = u.math.array([phase_x, phase_y]).transpose()
        # Phase = self.Postophase(loc)
        loc_input = self.get_input(Phase) * 5000
        self.Grid_cell.update(input=loc_input)
        grid_fr = self.Grid_cell.r.value
        # self.grid_output = u.math.dot(self.Wg2p, grid_fr-u.math.max(grid_fr)/2)
        self.grid_output = u.math.dot(self.Wg2p, grid_fr)


class HierarchicalNetwork(BasicModelGroup):
    def __init__(self, num_module, num_place):
        super().__init__()
        self.num_module = num_module
        self.num_place = num_place**2
        # randomly sample num_place place field centers from a square arena (5m x 5m)
        x = u.math.linspace(0, 5, num_place)
        X, Y = u.math.meshgrid(x, x)
        self.place_center = u.math.stack([X.flatten(), Y.flatten()]).T
        # self.place_center = 5 * u.math.random.rand(num_place,2)

        # load heatmaps_grid from heatmaps_grid.npz
        # data = np.load('heatmaps_grid.npz', allow_pickle=True)
        # heatmaps_grid = data['heatmaps_grid']
        # print(heatmaps_grid.shape)

        MEC_model_list = []
        # self.W_g2p_list = []
        spacing = u.math.linspace(2, 5, num_module)
        for i in range(num_module):
            MEC_model_list.append(
                HierarchicalPathIntegrationModel(
                    spacing=spacing[i], angle=0.0, place_center=self.place_center
                )
            )
            # W_g2p = self.W_place2grid(heatmaps_grid[i*400:(i+1)*400])
            # self.W_g2p_list.append(W_g2p)
        self.MEC_model_list = MEC_model_list

    def init_state(self, *args, **kwargs):
        self.place_fr = brainstate.HiddenState(u.math.zeros(self.num_place))
        self.grid_fr = brainstate.HiddenState(u.math.zeros((self.num_module, 20**2)))
        self.band_x_fr = brainstate.HiddenState(u.math.zeros((self.num_module, 180)))
        self.band_y_fr = brainstate.HiddenState(u.math.zeros((self.num_module, 180)))
        self.decoded_pos = brainstate.State(u.math.zeros(2))

        for i in range(self.num_module):
            self.MEC_model_list[i].init_state()

    def update(self, velocity, loc, loc_input_stre=0.0):
        grid_output = u.math.zeros(self.num_place)
        for i in range(self.num_module):
            # update the band cell module
            self.MEC_model_list[i](velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
            self.grid_fr.value = self.grid_fr.value.at[i].set(self.MEC_model_list[i].Grid_cell.u.value)
            self.band_x_fr.value = self.band_x_fr.value.at[i].set(self.MEC_model_list[i].band_cell_x.Band_cells.r.value)
            self.band_y_fr.value = self.band_y_fr.value.at[i].set(self.MEC_model_list[i].band_cell_y.Band_cells.r.value)
            grid_output_module = self.MEC_model_list[i].grid_output
            # W_g2p = self.W_g2p_list[i]
            # grid_fr = self.MEC_model_list[i].Grid_cell.r
            # grid_output_module = u.math.dot(W_g2p, grid_fr)
            grid_output += grid_output_module
        # update the place cell module
        grid_output = u.math.where(grid_output > 0, grid_output, 0)
        u_place = u.math.where(
            grid_output > u.math.max(grid_output) / 2, grid_output - u.math.max(grid_output) / 2, 0
        )
        # grid_output = grid_output**2/(1+u.math.sum(grid_output**2))
        # max_id = u.math.argmax(grid_output)
        # center = self.place_center[max_id]
        center = u.math.sum(self.place_center * u_place[:, u.math.newaxis], axis=0) / (
            1e-5 + u.math.sum(u_place)
        )
        self.decoded_pos.value = center
        self.place_fr.value = u_place**2 / (1 + u.math.sum(u_place**2))
        # self.place_fr = softmax(grid_output)
