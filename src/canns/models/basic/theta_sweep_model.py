import brainstate
import brainunit as u
import jax
import numpy as np
from brainstate.nn import exp_euler_step

from ._base import BasicModel


def calculate_theta_modulation(
    time_step: int,
    linear_gain: float,
    ang_gain: float,
    theta_strength_hd: float = 0.0,
    theta_strength_gc: float = 0.0,
    theta_cycle_len: float = 100.0,
    dt: float = None,
) -> tuple[float, float, float]:
    """
    Calculate theta oscillation phase and modulation factors for direction and grid cell networks.

    Args:
        time_step: Current time step index
        linear_gain: Normalized linear speed gain [0,1]
        ang_gain: Normalized angular speed gain [-1,1]
        theta_strength_hd: Theta modulation strength for head direction cells
        theta_strength_gc: Theta modulation strength for grid cells
        theta_cycle_len: Length of theta cycle in time units
        dt: Time step size (if None, uses brainstate.environ.get_dt())

    Returns:
        tuple: (theta_phase, theta_modulation_hd, theta_modulation_gc)
            - theta_phase: Current theta phase [-π, π]
            - theta_modulation_hd: Theta modulation for direction cells
            - theta_modulation_gc: Theta modulation for grid cells
    """
    if dt is None:
        dt = brainstate.environ.get_dt()

    # Calculate current time and theta phase
    t = time_step * dt
    theta_phase = u.math.mod(t, theta_cycle_len) / theta_cycle_len
    theta_phase = theta_phase * 2 * u.math.pi - u.math.pi

    # Calculate theta modulation for both networks
    # HD network: theta modulation scales with angular speed
    theta_modulation_hd = 1 + theta_strength_hd * (0.5 + ang_gain) * u.math.cos(theta_phase)

    # GC network: theta modulation scales with linear speed
    theta_modulation_gc = 1 + theta_strength_gc * (0.5 + linear_gain) * u.math.cos(theta_phase)

    return theta_phase, theta_modulation_hd, theta_modulation_gc


class DirectionCellNetwork(BasicModel):
    """
    1D continuous-attractor direction cell network
    References:
        Ji, Z., Lomi, E., Jeffery, K., Mitchell, A. S., & Burgess, N. (2025). Phase Precession Relative to Turning Angle in Theta‐Modulated Head Direction Cells. Hippocampus, 35(2), e70008.
    """

    def __init__(
        self,
        num: int,
        tau: float = 10.0,
        tau_v: float = 100.0,
        noise_strength: float = 0.1,
        k: float = 0.2,
        adaptation_strength: float = 15.0,
        a: float = 0.7,
        A: float = 3.0,
        J0: float = 1.0,
        g: float = 1.0,
        z_min: float = -u.math.pi,
        z_max: float = u.math.pi,
        conn_noise: float = 0.0,
    ):
        super().__init__(in_size=num)
        self.num = num

        self.tau = tau
        self.tau_v = tau_v
        self.noise_strength = noise_strength
        self.k = k
        self.adaptation_strength = adaptation_strength
        self.a = a
        self.A = A
        self.J0 = J0
        self.g = g
        self.conn_noise = conn_noise

        # derived parameters
        self.m = adaptation_strength * tau / tau_v

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        x1 = u.math.linspace(z_min, z_max, num + 1)
        self.x = x1[:-1]

        # connectivity
        base_connection = self.make_connection()
        noise_connection = np.random.normal(0, conn_noise, size=(num, num))
        self.conn_mat = base_connection + noise_connection

    def init_state(self, *args, **kwargs):
        self.r = brainstate.HiddenState(u.math.zeros(self.num))
        self.v = brainstate.HiddenState(u.math.zeros(self.num))
        self.u = brainstate.HiddenState(u.math.zeros(self.num))
        self.center = brainstate.State(u.math.zeros(1))
        self.centerI = brainstate.State(u.math.zeros(1))

    def update(self, head_direction, theta_input):
        self.center.value = self.get_bump_center(r=self.r, x=self.x)
        Iext = theta_input * self.input_bump(head_direction)
        Irec = self.conn_mat @ self.r.value
        noise = brainstate.random.randn(self.num) * self.noise_strength
        input_total = Iext + Irec + noise

        _u = exp_euler_step(
            lambda u, input: (-u + input - self.v.value) / self.tau,
            self.u.value,
            input_total,
        )

        _v = exp_euler_step(
            lambda v: (-v + self.m * self.u.value) / self.tau_v,
            self.v.value,
        )
        self.u.value = u.math.where(_u > 0, _u, 0)
        self.v.value = _v

        u_sq = u.math.square(self.u.value)
        self.r.value = self.g * u_sq / (1.0 + self.k * u.math.sum(u_sq))

    @staticmethod
    def handle_periodic_condition(A):
        B = u.math.where(A > u.math.pi, A - 2 * u.math.pi, A)
        B = u.math.where(B < -u.math.pi, B + 2 * u.math.pi, B)
        return B

    def calculate_dist(self, d):
        d = self.handle_periodic_condition(d)
        d = u.math.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_connection(self):
        @jax.vmap
        def get_J(xbins):
            d = self.calculate_dist(xbins - self.x)
            Jxx = (
                self.J0
                * u.math.exp(-0.5 * u.math.square(d / self.a))
                / (u.math.sqrt(2 * u.math.pi) * self.a)
            )
            return Jxx

        return get_J(self.x)

    @staticmethod
    def get_bump_center(r, x):
        exppos = u.math.exp(1j * x)
        center = u.math.angle(u.math.sum(exppos * r.value))
        return center.reshape(
            -1,
        )

    def input_bump(self, head_direction):
        return self.A * u.math.exp(
            -0.5 * u.math.square(self.calculate_dist(self.x - head_direction) / self.a)
        )


class GridCellNetwork(BasicModel):
    """
    2D continuous-attractor grid cell network
    References:
        Ji, Z., Chu, T., Wu, S., & Burgess, N. (2025). A systems model of alternating theta sweeps via firing rate adaptation. Current Biology, 35(4), 709-722.
    """

    def __init__(
        self,
        num_dc: int = 100,
        num_gc_x: int = 100,
        # dynamics
        tau: float = 10.0,
        tau_v: float = 100.0,
        noise_strength: float = 0.1,  # activity noise
        conn_noise: float = 0.0,  # connectivity noise
        k: float = 1.0,
        adaptation_strength: float = 15.0,  # (mbar)
        # connectivity / input
        a: float = 0.8,
        A: float = 3.0,
        J0: float = 5.0,
        g: float = 1000.0,  # scale the firing rate to make it reasonable, no biological meaning
        # controlling grid spacing, larger means smaller spacing
        mapping_ratio: float = 1,
        # cntrolling offset length from conjunctive gc layer to gc layer, this is the key to drive the bump to move
        phase_offset: float = 1.0 / 20,  # relative to -pi~pi range
    ):
        self.num = num_gc_x * num_gc_x
        super().__init__(in_size=self.num)

        self.num_dc = num_dc
        self.num_gc_1side = num_gc_x

        self.tau = tau
        self.tau_v = tau_v
        self.noise_strength = noise_strength
        self.k = k
        self.adaptation_strength = adaptation_strength
        self.a = a
        self.A = A
        self.J0 = J0
        self.g = g
        self.conn_noise = conn_noise
        self.mapping_ratio = mapping_ratio
        self.phase_offset = phase_offset

        # derived parameters
        self.m = adaptation_strength * tau / tau_v
        self.Lambda = 2 * u.math.pi / mapping_ratio  # grid spacing

        # coordinate transforms (hex -> rect)
        # Note that coor_transform is to map a parallelogram with a 60-degree angle back to a square
        # The logic is to partition the 2D space into parallelograms, each of which contains one lattice of grid cells, and repeat the parallelogram to tile the whole space
        self.coor_transform = u.math.array(
            [[1.0, -1.0 / u.math.sqrt(3.0)], [0.0, 2.0 / u.math.sqrt(3.0)]]
        )

        # inverse, which is u.math.array([[1.0, 1.0 / 2],[0.0,  u.math.sqrt(3.0) / 2]])
        # Note that coor_transform_inv is to map a square to a parallelogram with a 60-degree angle
        self.coor_transform_inv = u.linalg.inv(self.coor_transform)

        # feature space
        x_bins = u.math.linspace(-u.math.pi, u.math.pi, num_gc_x + 1)
        x_grid, y_grid = u.math.meshgrid(x_bins[:-1], x_bins[:-1])
        self.x_grid = x_grid.reshape(-1)
        self.y_grid = y_grid.reshape(-1)

        # positions in (x, y) space and transformed space
        self.value_grid = u.math.stack([self.x_grid, self.y_grid], axis=1)  # (num, 2)
        self.value_bump = self.value_grid * 4
        # candidate centers (for center snapping)
        self.candidate_centers = self.make_candidate_centers(self.Lambda)

        # connectivity
        base_connection = self.make_connection()
        noise_connection = np.random.normal(0, conn_noise, size=(self.num, self.num))
        self.conn_mat = base_connection + noise_connection

    def init_state(self, *args, **kwargs):
        self.r = brainstate.HiddenState(u.math.zeros(self.num))
        self.v = brainstate.HiddenState(u.math.zeros(self.num))
        self.u = brainstate.HiddenState(u.math.zeros(self.num))
        self.gc_bump = brainstate.State(u.math.zeros(self.num))
        self.conj_input = brainstate.State(u.math.zeros(self.num))
        self.center_phase = brainstate.State(u.math.zeros(2))
        self.center_position = brainstate.State(u.math.zeros(2))

    def make_connection(self):
        @jax.vmap
        def kernel(v):
            # v: (2,) location in (x,y)
            d = self.calculate_dist(v - self.value_grid)  # (N,)
            return (
                (self.J0 / self.g)
                * u.math.exp(-0.5 * u.math.square(d / self.a))
                / (u.math.sqrt(2.0 * u.math.pi) * self.a)
            )

        return kernel(self.value_grid)  # (N, N)

    def calculate_dist(self, d):
        """
        d: (..., 2) displacement in original (x,y).
        Return Euclidean distance after transform (hex/rect).
        """
        # consider the periodic boundary condition
        d = self.handle_periodic_condition(d)
        # transform to lattice axes
        dist = (
            u.math.matmul(self.coor_transform_inv, d.T)
        ).T  # This means the bump on the parallelogram lattice is a Gaussian, while in the square space it is a twisted Gaussian
        return u.math.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2)

    def handle_periodic_condition(self, d):
        d = u.math.where(d > u.math.pi, d - 2.0 * u.math.pi, d)
        d = u.math.where(d < -u.math.pi, d + 2.0 * u.math.pi, d)
        return d

    def make_candidate_centers(self, Lambda):
        N_c = 32
        cc = u.math.zeros((N_c, N_c, 2))

        for i in range(N_c):
            for j in range(N_c):
                cc = cc.at[i, j, 0].set((-N_c // 2 + i) * Lambda)
                cc = cc.at[i, j, 1].set((-N_c // 2 + j) * Lambda)

        cc_tranformed = u.math.dot(self.coor_transform_inv, cc.reshape(N_c * N_c, 2).T).T

        return cc_tranformed

    def update(self, animal_posistion, direction_activity, theta_modulation):
        # get bump activity in real space info from network activity on the manifold ---
        center_phase, center_position, gc_bump = self.get_unique_activity_bump(
            self.r.value, animal_posistion
        )
        self.center_phase.value = center_phase
        self.center_position.value = center_position
        self.gc_bump.value = gc_bump

        # get external input to grid cell layer from conjunctive grid cell layer
        # note that this conjunctive input will be theta modulated. When speed is high, theta modulation is high, thus input is stronger
        # This is how we get longer theta sweeps when speed is high
        conj_input = self.calculate_input_from_conjgc(
            animal_posistion, direction_activity, theta_modulation
        )
        self.conj_input.value = conj_input

        # recurrent + noise
        Irec = u.math.matmul(self.conn_mat, self.r.value)
        input_noise = brainstate.random.randn(self.num) * self.noise_strength
        total_net_input = Irec + conj_input + input_noise

        # integrate
        _u = exp_euler_step(
            lambda u, input: (-u + input - self.v.value) / self.tau,
            self.u.value,
            total_net_input,
        )
        _v = exp_euler_step(
            lambda v: (-v + self.m * self.u.value) / self.tau_v,
            self.v.value,
        )
        self.u.value = u.math.where(_u > 0.0, _u, 0.0)
        self.v.value = _v

        # get neuron firing by global inhibition
        u_sq = u.math.square(self.u.value)
        self.r.value = self.g * u_sq / (1.0 + self.k * u.math.sum(u_sq))

    def get_unique_activity_bump(self, network_activity, animal_posistion):
        """
        Estimate a unique bump (activity peak) from the current network state,
        given the animal's actual position.

        Returns:
            center_phase : (2,) array
                Phase coordinates of bump center on the manifold.
            center_position : (2,) array
                Real-space position of the bump (nearest candidate).
            bump : (N,) array
                Gaussian bump template centered at center_position.
        """

        # find bump center in phase space
        exppos_x = u.math.exp(1j * self.x_grid)
        exppos_y = u.math.exp(1j * self.y_grid)
        activity_masked = u.math.where(
            network_activity > u.math.max(network_activity) * 0.1, network_activity, 0.0
        )

        center_phase = u.math.zeros((2,))
        center_phase = center_phase.at[0].set(u.math.angle(u.math.sum(exppos_x * activity_masked)))
        center_phase = center_phase.at[1].set(u.math.angle(u.math.sum(exppos_y * activity_masked)))

        # --- map back to real space, snap to nearest candidate ---
        center_pos_residual = (
            u.math.matmul(self.coor_transform_inv, center_phase) / self.mapping_ratio
        )
        candidate_pos_all = self.candidate_centers + center_pos_residual
        distances = u.math.linalg.norm(candidate_pos_all - animal_posistion, axis=1)
        center_position = candidate_pos_all[u.math.argmin(distances)]

        # --- build Gaussian bump template ---
        d = u.math.asarray(center_position) - self.value_bump
        dist = u.math.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        gc_bump = self.A * u.math.exp(-u.math.square(dist / self.a))

        return center_phase, center_position, gc_bump

    def calculate_input_from_conjgc(self, animal_pos, direction_activity, theta_modulation):
        assert u.math.size(animal_pos) == 2
        num_dc = self.num_dc
        num_gc = self.num
        direction_bin = u.math.linspace(-u.math.pi, u.math.pi, num_dc)

        # # lag relative to head direction
        # lagvec = -u.math.array([u.math.cos(head_direction), u.math.sin(head_direction)]) * self.params.phase_offset * 1.4
        # offset = u.math.array([u.math.cos(direction_bin), u.math.sin(direction_bin)]) * self.params.phase_offset + lagvec.reshape(-1, 1)

        offset = (
            u.math.array([u.math.cos(direction_bin), u.math.sin(direction_bin)]) * self.phase_offset
        )

        center_conj = self.position2phase(animal_pos.reshape(-1, 1) + offset.reshape(-1, num_dc))

        conj_input = u.math.zeros((num_dc, num_gc))
        for i in range(num_dc):
            d = self.calculate_dist(u.math.asarray(center_conj[:, i]) - self.value_grid)
            conj_input = conj_input.at[i].set(self.A * u.math.exp(-0.5 * u.math.square(d / self.a)))

        # weighting by direction bump activity: keep top one-third (by max) then normalize, I thinking using all direction_activity should also be fine
        weight = u.math.where(
            direction_activity > u.math.max(direction_activity) / 3.0, direction_activity, 0.0
        )
        weight = weight / (u.math.sum(weight) + 1e-12)  # avoid div-by-zero, dim: (num_dc,)

        return (
            u.math.matmul(conj_input.T, weight).reshape(-1) * theta_modulation
        )  # dim: (num_gc, num_dc) x (num_dc,) -> (num_gc,)

    def position2phase(self, position):
        """
        map position->phase; phase is wrapped to [-pi, pi] per-axis
        """
        mapped_pos = position * self.mapping_ratio
        phase = u.math.matmul(self.coor_transform, mapped_pos) + u.math.pi
        px = u.math.mod(phase[0], 2.0 * u.math.pi) - u.math.pi
        py = u.math.mod(phase[1], 2.0 * u.math.pi) - u.math.pi
        return u.math.array([px, py])
