import brainstate
import brainunit as u
import jax
from brainstate.nn import exp_euler_step

from ...task.path_integration import map2pi
from ._base import BasicModel, BasicModelGroup
from .band_cell import BandCell
from .grid_cell import GridCell

__all__ = [
    # Hierarchical Path Integration Model
    "HierarchicalPathIntegrationModel",
    # Hierarchical Network
    "HierarchicalNetwork",
]


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
        place_center (brainunit.math.ndarray): The center locations of the target place cells.
        Wg2p (brainunit.math.ndarray): The connection weights from grid cells to place cells.
        grid_output (brainstate.State): The activity of the place cells.
    """

    def __init__(self, spacing, angle, place_center=None):
        """Initializes the HierarchicalPathIntegrationModel.

        Args:
            spacing (float): The spacing of the grid pattern for this module.
            angle (float): The base orientation angle for the module.
            place_center (brainunit.math.ndarray, optional): The center locations of the
                target place cell population. Defaults to a random distribution.
        """
        super().__init__()
        self.band_cell_x = BandCell(angle=angle, spacing=spacing, noise=0.0)
        self.band_cell_y = BandCell(angle=angle + u.math.pi / 3, spacing=spacing, noise=0.0)
        self.band_cell_z = BandCell(angle=angle + u.math.pi / 3 * 2, spacing=spacing, noise=0.0)
        self.grid_cell = GridCell(num=20, angle=angle, spacing=spacing)
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
        self.grid_cell.init_state()

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
        grid_vector = u.math.zeros(value_grid.shape)
        grid_vector = grid_vector.at[:, 0].set(value_grid[:, 0])
        grid_vector = grid_vector.at[:, 1].set(
            (value_grid[:, 1] - 1 / 2 * value_grid[:, 0]) * 2 / u.math.sqrt(3)
        )
        z_vector = u.math.array([-1 / 2, u.math.sqrt(3) / 2])
        grid_phase_z = u.math.dot(grid_vector, z_vector)
        dis_x = self.band_cell_x.dist(grid_x[:, None] - band_x[None, :])
        dis_y = self.band_cell_y.dist(grid_y[:, None] - band_y[None, :])
        dis_z = self.band_cell_z.dist(grid_phase_z[:, None] - band_z[None, :])
        self.W_x_grid = (
            J0
            * u.math.exp(-0.5 * u.math.square(dis_x / self.band_cell_x.band_cells.a))
            / (u.math.sqrt(2 * u.math.pi) * self.band_cell_x.band_cells.a)
        )
        self.W_y_grid = (
            J0
            * u.math.exp(-0.5 * u.math.square(dis_y / self.band_cell_y.band_cells.a))
            / (u.math.sqrt(2 * u.math.pi) * self.band_cell_y.band_cells.a)
        )
        self.W_z_grid = (
            J0
            * u.math.exp(-0.5 * u.math.square(dis_z / self.band_cell_z.band_cells.a))
            / (u.math.sqrt(2 * u.math.pi) * self.band_cell_z.band_cells.a)
        )

    def Postophase(self, pos):
        """Projects a 2D position to the 2D phase space of the grid module.

        Args:
            pos (Array): The 2D position vector.

        Returns:
            Array: The corresponding 2D phase vector.
        """
        phase_x = u.math.mod(u.math.dot(pos, self.proj_k_x), 2 * u.math.pi) - u.math.pi
        phase_y = u.math.mod(u.math.dot(pos, self.proj_k_y), 2 * u.math.pi) - u.math.pi
        return u.math.array([phase_x, phase_y]).transpose()

    def make_Wg2p(self):
        """Creates the connection weights from grid cells to place cells.

        The connection strength is determined by the proximity of a place cell's
        center to a grid cell's firing field, calculated in the phase domain.
        """
        phase_place = self.Postophase(self.place_center)
        phase_grid = self.grid_cell.value_grid
        d = phase_place[:, u.math.newaxis, :] - phase_grid[u.math.newaxis, :, :]
        d = map2pi(d)
        delta_x = d[:, :, 0]
        delta_y = (d[:, :, 1] - 1 / 2 * d[:, :, 0]) * 2 / u.math.sqrt(3)
        # delta_x = d[:,:,0] + d[:,:,1]/2
        # delta_y = d[:,:,1] * u.math.sqrt(3) / 2
        dis = u.math.sqrt(delta_x**2 + delta_y**2)
        Wg2p = u.math.exp(-0.5 * u.math.square(dis / self.band_cell_x.band_cells.a)) / (
            u.math.sqrt(2 * u.math.pi) * self.band_cell_x.band_cells.a
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
        delta_y = (d[:, 1] - 1 / 2 * d[:, 0]) * 2 / u.math.sqrt(3)
        return u.math.sqrt(delta_x**2 + delta_y**2)

    def get_input(self, Phase):
        """Generates a stimulus input for the grid cell based on a 2D phase.

        Args:
            Phase (Array): The 2D phase vector.

        Returns:
            Array: The stimulus input vector for the grid cells.
        """
        dis = self.dist(Phase - self.grid_cell.value_grid)
        return u.math.exp(-0.5 * u.math.square(dis / self.band_cell_x.band_cells.a)) / (
            u.math.sqrt(2 * u.math.pi) * self.band_cell_x.band_cells.a
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
        max_output = u.math.max(band_output)
        band_output = u.math.where(band_output > max_output / 2, band_output - max_output / 2, 0)
        phase_x = self.band_cell_x.center.value
        phase_y = self.band_cell_y.center.value
        Phase = u.math.array([phase_x, phase_y]).transpose()
        # Phase = self.Postophase(loc)
        loc_input = self.get_input(Phase) * 5000
        self.grid_cell.update(input=loc_input)
        grid_fr = self.grid_cell.r.value
        # self.grid_output = u.math.dot(self.Wg2p, grid_fr-u.math.max(grid_fr)/2)
        self.grid_output.value = u.math.dot(self.Wg2p, grid_fr)

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
        place_center (brainunit.math.ndarray): The center locations of the place cells.
        MEC_model_list (list): A list containing all the `HierarchicalPathIntegrationModel` instances.
        grid_fr (brainstate.HiddenState): The firing rates of the grid cell population.
        band_x_fr (brainstate.HiddenState): The firing rates of the x-oriented band cell population.
        band_y_fr (brainstate.HiddenState): The firing rates of the y-oriented band cell population.
        place_fr (brainstate.HiddenState): The firing rates of the place cell population.
        decoded_pos (brainstate.State): The final decoded 2D position.

    References:
        Anonymous Author(s) "Unfolding the Black Box of Recurrent Neural Networks for Path Integration" (under review).
    """

    def __init__(self, num_module, num_place):
        """Initializes the HierarchicalNetwork.

        Args:
            num_module (int): The number of grid modules to create.
            num_place (int): The number of place cells along one dimension of a square grid.
        """
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

        band_x_r = u.math.zeros((indices.shape[0], self.num_module, 180))
        band_y_r = u.math.zeros((indices.shape[0], self.num_module, 180))
        grid_r = u.math.zeros((indices.shape[0], self.num_module, 20**2))
        grid_output = u.math.zeros((indices.shape[0], self.num_place))
        loc_input_stre = u.math.ones((indices.shape[0],)) * loc_input_stre

        for i, model in enumerate(self.MEC_model_list):
            if pbar is None:
                pbar = brainstate.compile.ProgressBar(
                    total=indices.shape[0], desc=f"Module {i + 1}/{self.num_module}"
                )

            def run_single_module(velocity, loc, loc_input_stre):
                model(velocity=velocity, loc=loc, loc_input_stre=loc_input_stre)
                return (
                    model.band_cell_x.band_cells.r.value,
                    model.band_cell_y.band_cells.r.value,
                    model.grid_cell.r.value,
                    model.grid_output.value,
                )

            single_band_x_r, single_band_y_r, single_grid_r, single_grid_output = (
                brainstate.compile.for_loop(
                    run_single_module,
                    velocities,
                    positions,
                    loc_input_stre,
                    pbar=pbar,
                )
            )
            band_x_r = band_x_r.at[:, i, :].set(single_band_x_r)
            band_y_r = band_y_r.at[:, i, :].set(single_band_y_r)
            grid_r = grid_r.at[:, i, :].set(single_grid_r)
            grid_output += single_grid_output

        grid_output = u.math.where(grid_output > 0, grid_output, 0)
        u_place = u.math.where(
            grid_output > u.math.max(grid_output, axis=1, keepdims=True) / 2,
            grid_output - u.math.max(grid_output, axis=1, keepdims=True) / 2,
            0,
        )

        place_r = u_place**2 / (1 + u.math.sum(u_place**2, axis=1, keepdims=True))

        return band_x_r, band_y_r, grid_r, place_r
    '''