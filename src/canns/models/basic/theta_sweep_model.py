import brainpy as bp
import brainpy.math as bm
import jax
import numpy as np

from ._base import BasicModel

# TODO: exp_euler_step should be implemented in BrainPy


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
        dt: Time step size (if None, uses bm.get_dt())

    Returns:
        tuple: (theta_phase, theta_modulation_hd, theta_modulation_gc)
            - theta_phase: Current theta phase [-π, π]
            - theta_modulation_hd: Theta modulation for direction cells
            - theta_modulation_gc: Theta modulation for grid cells
    """
    if dt is None:
        dt = bm.get_dt()

    # Calculate current time and theta phase
    t = time_step * dt
    theta_phase = bm.mod(t, theta_cycle_len) / theta_cycle_len
    theta_phase = theta_phase * 2 * bm.pi - bm.pi

    # Calculate theta modulation for both networks
    # HD network: theta modulation scales with angular speed
    theta_modulation_hd = 1 + theta_strength_hd * (0.5 + ang_gain) * bm.cos(theta_phase)

    # GC network: theta modulation scales with linear speed
    theta_modulation_gc = 1 + theta_strength_gc * (0.5 + linear_gain) * bm.cos(theta_phase)

    return theta_phase, theta_modulation_hd, theta_modulation_gc


class DirectionCellNetwork(BasicModel):
    """
    1D continuous-attractor direction (head direction) cell network.

    This network implements a ring attractor model for representing head direction
    with theta-modulated dynamics and spike-frequency adaptation (SFA). The model
    exhibits key properties of biological head direction cells including:
    - Persistent activity bumps encoding current heading
    - Theta phase precession relative to turning angle
    - Anticipative tracking through adaptation mechanisms

    The network dynamics include:
    - Membrane potential (u) with recurrent excitation and global inhibition
    - Adaptation variable (v) implementing slow negative feedback
    - Firing rate (r) computed via divisive normalization
    - External input modulated by theta oscillations

    Args:
        num: Number of neurons in the network (resolution of head direction representation)
        tau: Membrane time constant (ms). Controls speed of neural dynamics.
        tau_v: Adaptation time constant (ms). Larger values = slower adaptation.
        noise_strength: Standard deviation of Gaussian noise added to inputs
        k: Global inhibition strength for divisive normalization
        adaptation_strength: Strength of adaptation coupling (dimensionless)
        a: Width of connectivity kernel (radians). Determines bump width.
        A: Amplitude of external input bump
        J0: Peak recurrent connection strength
        g: Gain parameter for firing rate transformation
        z_min: Minimum value of feature space (default: -π)
        z_max: Maximum value of feature space (default: π)
        conn_noise: Standard deviation of Gaussian noise added to connectivity matrix

    Attributes:
        num (int): Number of neurons
        x (Array): Preferred directions of neurons, uniformly distributed in [z_min, z_max)
        conn_mat (Array): Recurrent connectivity matrix with Gaussian profile
        r (HiddenState): Firing rates of neurons
        u (HiddenState): Membrane potentials
        v (HiddenState): Adaptation variables
        center (State): Current bump center position
        m (float): Effective adaptation strength (adaptation_strength * tau / tau_v)

    Example:
        >>> import brainpy.math as bm
        >>> from canns.models.basic.theta_sweep_model import DirectionCellNetwork
        >>>
        >>> bm.set_dt(1.)  # 1ms time step
        >>> dc_net = DirectionCellNetwork(num=60)
        >>>
        >>> # Update with head direction and theta modulation
        >>> head_direction = 0.5  # radians
        >>> theta_modulation = 1.2  # theta phase-dependent gain
        >>> dc_net.update(head_direction, theta_modulation)

    References:
        Ji, Z., Lomi, E., Jeffery, K., Mitchell, A. S., & Burgess, N. (2025).
        Phase Precession Relative to Turning Angle in Theta‐Modulated Head Direction Cells.
        Hippocampus, 35(2), e70008.
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
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
        conn_noise: float = 0.0,
    ):
        super().__init__()
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
        x1 = bm.linspace(z_min, z_max, num + 1)
        self.x = x1[:-1]

        # connectivity
        base_connection = self.make_connection()
        noise_connection = np.random.normal(0, conn_noise, size=(num, num))
        self.conn_mat = base_connection + noise_connection

        """
        Initialize network state variables.

        Creates and initializes:
        - r: Firing rates (all zeros)
        - u: Membrane potentials (all zeros)
        - v: Adaptation variables (all zeros)
        - center: Current bump center (zero)
        - centerI: Input bump center (zero)
        """
        self.r = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.center = bm.Variable(bm.zeros(1))
        self.centerI = bm.Variable(bm.zeros(1))

        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def update(self, head_direction, theta_input):
        """
        Single time-step update of network dynamics.

        Args:
            head_direction: Target head direction in radians [-π, π]
            theta_input: Theta modulation factor (typically 1.0 ± theta_strength)
        """
        self.center.value = self.get_bump_center(r=self.r, x=self.x)
        Iext = theta_input * self.input_bump(head_direction)
        Irec = self.conn_mat @ self.r.value
        noise = bm.random.randn(self.num) * self.noise_strength
        input_total = Iext + Irec + noise

        _u, _v = self.integral(self.u, self.v, bp.share["t"], input_total, bm.dt)
        self.u.value = bm.where(_u > 0, _u, 0)
        self.v.value = _v

        u_sq = bm.square(self.u.value)
        self.r.value = self.g * u_sq / (1.0 + self.k * bm.sum(u_sq))

    @staticmethod
    def handle_periodic_condition(A):
        B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
        B = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
        return B

    def calculate_dist(self, d):
        """
        Calculate distance on circular feature space with periodic boundary.

        Args:
            d: Raw angular difference

        Returns:
            Shortest angular distance considering periodicity
        """
        d = self.handle_periodic_condition(d)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_connection(self):
        """
        Generate recurrent connectivity matrix with Gaussian profile.

        Creates a circulant connectivity matrix where connection strength
        decreases with distance according to a Gaussian kernel.

        Returns:
            Array of shape (num, num): Connectivity matrix
        """

        @jax.vmap
        def get_J(xbins):
            d = self.calculate_dist(xbins - self.x)
            Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
            return Jxx

        return get_J(self.x)

    @staticmethod
    def get_bump_center(r, x):
        """
        Decode bump center from population activity using circular mean.

        Args:
            r: Firing rate vector
            x: Preferred direction vector

        Returns:
            Decoded center position in radians
        """
        exppos = bm.exp(1j * x)
        center = bm.angle(bm.sum(exppos * r.value))
        return center.reshape(
            -1,
        )

    def input_bump(self, head_direction):
        """
        Generate Gaussian-shaped external input centered on target direction.

        Args:
            head_direction: Center of input bump in radians

        Returns:
            Input vector of shape (num,)
        """
        return self.A * bm.exp(
            -0.5 * bm.square(self.calculate_dist(self.x - head_direction) / self.a)
        )


class GridCellNetwork(BasicModel):
    """
    2D continuous-attractor grid cell network with hexagonal lattice structure.

    This network implements a twisted torus topology that generates grid cell-like
    spatial representations with hexagonal periodicity. The model combines:
    - 2D continuous attractor dynamics on a twisted manifold
    - Spike-frequency adaptation for theta modulation
    - Integration of direction cell inputs via conjunctive cells
    - Phase offset mechanism for theta sweeps

    The network operates in a transformed coordinate system where grid cells form
    a hexagonal lattice, enabling realistic grid field spacing and orientation.

    Args:
        num_dc: Number of direction cells providing heading input
        num_gc_x: Number of grid cells along one dimension (total = num_gc_x^2)
        tau: Membrane time constant (ms)
        tau_v: Adaptation time constant (ms). Larger = slower adaptation.
        noise_strength: Standard deviation of activity noise
        conn_noise: Standard deviation of connectivity noise
        k: Global inhibition strength for divisive normalization
        adaptation_strength: Coupling strength between u and v
        a: Width of connectivity kernel. Determines bump width.
        A: Amplitude of external input
        J0: Peak recurrent connection strength
        g: Firing rate gain factor (scales to biological range)
        mapping_ratio: Controls grid spacing (larger = smaller spacing).
            Grid spacing λ = 2π / mapping_ratio
        phase_offset: Phase shift for conjunctive input, drives theta sweeps.
            Expressed as fraction of [-π, π] range (default: 1/20)

    Attributes:
        num (int): Total number of grid cells (num_gc_x^2)
        x_grid, y_grid (Array): Grid cell preferred phases in [-π, π]
        value_grid (Array): Neuron positions in phase space, shape (num, 2)
        Lambda (float): Grid spacing in real space
        coor_transform (Array): Hexagonal to rectangular coordinate transform
        conn_mat (Array): Recurrent connectivity matrix
        candidate_centers (Array): Grid of candidate bump centers for decoding
        r (HiddenState): Firing rates
        u (HiddenState): Membrane potentials
        v (HiddenState): Adaptation variables
        center_phase (State): Decoded bump center in phase space
        center_position (State): Decoded position in real space
        gc_bump (State): Grid cell bump activity pattern

    Example:
        >>> import brainpy.math as bm
        >>> from canns.models.basic.theta_sweep_model import GridCellNetwork
        >>>
        >>> bm.set_dt(1.0)
        >>> gc_net = GridCellNetwork(num_dc=60, num_gc_x=30, mapping_ratio=1.5)
        >>>
        >>> # Update with position, direction activity, and theta modulation
        >>> position = [0.5, 0.3]  # animal position
        >>> dir_activity = np.random.rand(60)  # direction cell firing
        >>> theta_mod = 1.2  # theta phase modulation
        >>> gc_net.update(position, dir_activity, theta_mod)

    References:
        Ji, Z., Chu, T., Wu, S., & Burgess, N. (2025).
        A systems model of alternating theta sweeps via firing rate adaptation.
        Current Biology, 35(4), 709-722.
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
        super().__init__()

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
        self.Lambda = 2 * bm.pi / mapping_ratio  # grid spacing

        # coordinate transforms (hex -> rect)
        # Note that coor_transform is to map a parallelogram with a 60-degree angle back to a square
        # The logic is to partition the 2D space into parallelograms, each of which contains one lattice of grid cells, and repeat the parallelogram to tile the whole space
        self.coor_transform = bm.array([[1.0, -1.0 / bm.sqrt(3.0)], [0.0, 2.0 / bm.sqrt(3.0)]])

        # inverse, which is bm.array([[1.0, 1.0 / 2],[0.0,  bm.sqrt(3.0) / 2]])
        # Note that coor_transform_inv is to map a square to a parallelogram with a 60-degree angle
        self.coor_transform_inv = bm.linalg.inv(self.coor_transform)

        # feature space
        x_bins = bm.linspace(-bm.pi, bm.pi, num_gc_x + 1)
        x_grid, y_grid = bm.meshgrid(x_bins[:-1], x_bins[:-1])
        self.x_grid = x_grid.reshape(-1)
        self.y_grid = y_grid.reshape(-1)

        # positions in (x, y) space and transformed space
        self.value_grid = bm.stack([self.x_grid, self.y_grid], axis=1)  # (num, 2)
        self.value_bump = self.value_grid * 4
        # candidate centers (for center snapping)
        self.candidate_centers = self.make_candidate_centers(self.Lambda)

        # connectivity
        base_connection = self.make_connection()
        noise_connection = np.random.normal(0, conn_noise, size=(self.num, self.num))
        self.conn_mat = base_connection + noise_connection

        """
        Initialize network state variables.

        Creates and initializes:
        - r: Firing rates (shape: num)
        - u: Membrane potentials (shape: num)
        - v: Adaptation variables (shape: num)
        - gc_bump: Grid cell bump pattern (shape: num)
        - conj_input: Conjunctive cell input (shape: num)
        - center_phase: Bump center in phase space (shape: 2)
        - center_position: Decoded position in real space (shape: 2)
        """
        self.r = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.gc_bump = bm.Variable(bm.zeros(self.num))
        self.conj_input = bm.Variable(bm.zeros(self.num))
        self.center_phase = bm.Variable(bm.zeros(2))
        self.center_position = bm.Variable(bm.zeros(2))

        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def make_connection(self):
        """
        Generate recurrent connectivity matrix with 2D Gaussian kernel.

        Uses hexagonal lattice geometry via coordinate transformation.
        Connection strength decays with distance in transformed space.

        Returns:
            Array of shape (num, num): Recurrent connectivity matrix
        """

        @jax.vmap
        def kernel(v):
            # v: (2,) location in (x,y)
            d = self.calculate_dist(v - self.value_grid)  # (N,)
            return (
                (self.J0 / self.g)
                * bm.exp(-0.5 * bm.square(d / self.a))
                / (bm.sqrt(2.0 * bm.pi) * self.a)
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
            bm.matmul(self.coor_transform_inv, d.T)
        ).T  # This means the bump on the parallelogram lattice is a Gaussian, while in the square space it is a twisted Gaussian
        return bm.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2)

    def handle_periodic_condition(self, d):
        """
        Apply periodic boundary conditions to wrap phases into [-π, π].

        Args:
            d: Phase values (any shape)

        Returns:
            Wrapped phase values in [-π, π]
        """
        d = bm.where(d > bm.pi, d - 2.0 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2.0 * bm.pi, d)
        return d

    def make_candidate_centers(self, Lambda):
        """
        Generate grid of candidate bump centers for decoding.

        Creates a regular lattice of potential activity bump locations
        used for disambiguating position from grid cell phases.

        Args:
            Lambda: Grid spacing in real space

        Returns:
            Array of shape (N_c*N_c, 2): Candidate centers in transformed coordinates
        """
        N_c = 32
        cc = bm.zeros((N_c, N_c, 2))

        for i in range(N_c):
            for j in range(N_c):
                cc = cc.at[i, j, 0].set((-N_c // 2 + i) * Lambda)
                cc = cc.at[i, j, 1].set((-N_c // 2 + j) * Lambda)

        cc_tranformed = bm.dot(self.coor_transform_inv, cc.reshape(N_c * N_c, 2).T).T

        return cc_tranformed

    def update(self, animal_posistion, direction_activity, theta_modulation):
        """
        Single time-step update of grid cell network dynamics.

        Integrates conjunctive inputs from direction cells, applies theta modulation,
        and updates grid cell activity via continuous attractor dynamics with adaptation.

        Args:
            animal_posistion: Current position [x, y] for disambiguating grid phase
            direction_activity: Direction cell firing rates (shape: num_dc)
            theta_modulation: Theta phase-dependent gain factor
        """
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
        Irec = bm.matmul(self.conn_mat, self.r.value)
        input_noise = bm.random.randn(self.num) * self.noise_strength
        total_net_input = Irec + conj_input + input_noise

        # integrate
        _u, _v = self.integral(self.u, self.v, bp.share["t"], total_net_input, bm.dt)
        self.u.value = bm.where(_u > 0.0, _u, 0.0)
        self.v.value = _v

        # get neuron firing by global inhibition
        u_sq = bm.square(self.u.value)
        self.r.value = self.g * u_sq / (1.0 + self.k * bm.sum(u_sq))

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
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        activity_masked = bm.where(
            network_activity > bm.max(network_activity) * 0.1, network_activity, 0.0
        )

        center_phase = bm.zeros((2,))
        center_phase = center_phase.at[0].set(bm.angle(bm.sum(exppos_x * activity_masked)))
        center_phase = center_phase.at[1].set(bm.angle(bm.sum(exppos_y * activity_masked)))

        # --- map back to real space, snap to nearest candidate ---
        center_pos_residual = bm.matmul(self.coor_transform_inv, center_phase) / self.mapping_ratio
        candidate_pos_all = self.candidate_centers + center_pos_residual
        distances = bm.linalg.norm(candidate_pos_all - animal_posistion, axis=1)
        center_position = candidate_pos_all[bm.argmin(distances)]

        # --- build Gaussian bump template ---
        d = bm.asarray(center_position) - self.value_bump
        dist = bm.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        gc_bump = self.A * bm.exp(-bm.square(dist / self.a))

        return center_phase, center_position, gc_bump

    def calculate_input_from_conjgc(self, animal_pos, direction_activity, theta_modulation):
        """
        Calculate external input to grid cells from conjunctive grid cells.

        Conjunctive cells integrate position and direction to generate grid cell inputs
        with phase offsets. This drives theta sweeps when modulated by theta oscillations.

        Args:
            animal_pos: Current position [x, y]
            direction_activity: Direction cell firing rates (shape: num_dc)
            theta_modulation: Theta phase-dependent gain factor

        Returns:
            Array of shape (num_gc,): Weighted conjunctive input to grid cells
        """
        assert bm.size(animal_pos) == 2
        num_dc = self.num_dc
        num_gc = self.num
        direction_bin = bm.linspace(-bm.pi, bm.pi, num_dc)

        # # lag relative to head direction
        # lagvec = -bm.array([bm.cos(head_direction), bm.sin(head_direction)]) * self.params.phase_offset * 1.4
        # offset = bm.array([bm.cos(direction_bin), bm.sin(direction_bin)]) * self.params.phase_offset + lagvec.reshape(-1, 1)

        offset = bm.array([bm.cos(direction_bin), bm.sin(direction_bin)]) * self.phase_offset

        center_conj = self.position2phase(animal_pos.reshape(-1, 1) + offset.reshape(-1, num_dc))

        conj_input = bm.zeros((num_dc, num_gc))
        for i in range(num_dc):
            d = self.calculate_dist(bm.asarray(center_conj[:, i]) - self.value_grid)
            conj_input = conj_input.at[i].set(self.A * bm.exp(-0.5 * bm.square(d / self.a)))

        # weighting by direction bump activity: keep top one-third (by max) then normalize, I thinking using all direction_activity should also be fine
        weight = bm.where(
            direction_activity > bm.max(direction_activity) / 3.0, direction_activity, 0.0
        )
        weight = weight / (bm.sum(weight) + 1e-12)  # avoid div-by-zero, dim: (num_dc,)

        return (
            bm.matmul(conj_input.T, weight).reshape(-1) * theta_modulation
        )  # dim: (num_gc, num_dc) x (num_dc,) -> (num_gc,)

    def position2phase(self, position):
        """
        Convert real-space position to grid cell phase coordinates.

        Applies coordinate transformation and wraps to periodic boundaries.
        Each grid cell's preferred phase is determined by the animal's position
        on the hexagonal lattice.

        Args:
            position: Real-space coordinates, shape (2,) or (2, N)

        Returns:
            Array of shape (2,) or (2, N): Phase coordinates in [-π, π] per axis
        """
        mapped_pos = position * self.mapping_ratio
        phase = bm.matmul(self.coor_transform, mapped_pos) + bm.pi
        px = bm.mod(phase[0], 2.0 * bm.pi) - bm.pi
        py = bm.mod(phase[1], 2.0 * bm.pi) - bm.pi
        return bm.array([px, py])


class PlaceCellNetwork(BasicModel):
    """
    Graph-based continuous-attractor place cell network using environment geodesic distances.

    This network implements a place cell representation where neurons are tuned to discrete
    locations in a navigation environment. Connectivity is based on geodesic (shortest path)
    distances within the environment, allowing the network to adapt to complex non-convex spaces
    with obstacles.

    Key features:
    - Connectivity matrix based on geodesic distances (not Euclidean)
    - Replaces NetworkX graph representation with grid-based geodesic computation
    - Uses GeodesicDistanceResult for environment definition and distance computation
    - Continuous attractor dynamics with spike-frequency adaptation
    - Supports arbitrary environment shapes (rectangular, T-maze, complex polygons with holes/walls)

    Args:
        geodesic_result: Geodesic distance computation result from navigation task
        tau: Membrane time constant (ms). Controls speed of neural dynamics.
        tau_v: Adaptation time constant (ms). Larger values = slower adaptation.
        noise_strength: Standard deviation of Gaussian noise added to inputs
        k: Global inhibition strength for divisive normalization
        m: Strength of adaptation coupling (dimensionless)
        a: Width of connectivity kernel. Determines bump width in grid units.
        A: Amplitude of external input bump
        J0: Peak recurrent connection strength
        g: Gain parameter for firing rate transformation
        conn_noise: Standard deviation of Gaussian noise added to connectivity matrix

    Attributes:
        geodesic_result (GeodesicDistanceResult): Geodesic distance computation result
        cell_num (int): Number of place cells (= number of accessible grid cells)
        D (Array): Geodesic distance matrix of shape (cell_num, cell_num)
        accessible_indices (Array): Grid indices of accessible cells (cell_num, 2)
        cost_grid (MovementCostGrid): Grid cost information for position lookups
        conn_mat (Array): Recurrent connectivity matrix with Gaussian profile
        r (HiddenState): Firing rates of place cells
        u (HiddenState): Membrane potentials
        v (HiddenState): Adaptation variables
        center (State): Current decoded bump center
        m (float): Effective adaptation strength (adaptation_strength * tau / tau_v)
    """

    def __init__(
        self,
        geodesic_result,
        tau: float = 10.0,
        tau_v: float = 100.0,
        noise_strength: float = 0.0,
        k: float = 0.2,
        m: float = 3.0,
        a: float = 0.2,
        A: float = 5.0,
        J0: float = 1.0,
        g: float = 1.0,
        conn_noise: float = 0.0,
    ):
        self.geodesic_result = geodesic_result
        self.cost_grid = geodesic_result.cost_grid

        # Extract geodesic distances and accessible indices
        self.D = bm.asarray(geodesic_result.distances)  # (cell_num, cell_num)
        self.accessible_indices = geodesic_result.accessible_indices  # (cell_num, 2)
        self.cell_num = len(self.accessible_indices)
        self.dx, self.dy = geodesic_result.cost_grid.dx, geodesic_result.cost_grid.dy

        # assume square grid cells
        self.x = bm.arange(self.cell_num) * self.dx

        super().__init__()

        # Store parameters
        self.tau = tau
        self.tau_v = tau_v
        self.noise_strength = noise_strength
        self.k = k
        self.a = a
        self.A = A
        self.J0 = J0
        self.g = g
        self.conn_noise = conn_noise

        # Derived parameters
        self.m = m

        # Build connectivity based on geodesic distance
        base_connection = self.make_connection()
        noise_connection = np.random.normal(0, conn_noise, size=(self.cell_num, self.cell_num))
        self.conn_mat = base_connection + noise_connection

        """
        Initialize network state variables.

        Creates and initializes:
        - r: Firing rates (all zeros)
        - u: Membrane potentials (all zeros)
        - v: Adaptation variables (all zeros)
        - center: Current bump center (zero)
        """
        self.r = bm.Variable(bm.zeros(self.cell_num))
        self.v = bm.Variable(bm.zeros(self.cell_num))
        self.u = bm.Variable(bm.zeros(self.cell_num))
        self.center = bm.Variable(bm.zeros(1))

        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def make_connection(self):
        """
        Generate recurrent connectivity matrix with Gaussian profile based on geodesic distance.

        Connection strength between place cells decays with geodesic distance according
        to a normalized Gaussian kernel.

        Returns:
            Array of shape (cell_num, cell_num): Connectivity matrix
        """

        @jax.vmap
        def kernel_row(d):
            # d: (cell_num,) distances from one cell to all others
            return self.J0 * bm.exp(-d / (2 * self.a**2)) / ((2 * bm.pi) * self.a**2)

        return kernel_row(self.D)

    def get_bump_center(self, r, x):
        """
        Decode bump center from population activity.

        Uses weighted average of cell indices, normalized by total activity.

        Args:
            r: Firing rate vector (cell_num,)

        Returns:
            Decoded center index (scalar)
        """
        denom = bm.sum(r) + 1e-12
        center_idx = bm.sum(r * x) / denom
        return center_idx.reshape(
            -1,
        )

    def get_geodesic_index_by_pos(self, pos):
        """
        Get the geodesic index of the grid cell containing the given position.

        Args:
            pos: (x, y) coordinates of the position

        Returns:
            Index of the grid cell in the geodesic distance matrix, or None if
            the position is out of bounds or in an impassable cell.
        """
        return self.cost_grid.get_cell_index(pos)

    def input_bump(self, animal_pos):
        """
        Generate Gaussian bump external input centered on the animal's current position.

        Args:
            animal_pos: Current position (x, y) tuple or array

        Returns:
            Input vector of shape (cell_num,)
        """
        # Get the cell index corresponding to the animal's position
        # Returns -1 if position is out of bounds or inaccessible
        cell_idx = self.get_geodesic_index_by_pos(animal_pos)

        # Use geodesic distances from the distance matrix
        # If cell_idx is -1 (invalid), clip to 0 to avoid index error
        # The result will be near-zero anyway due to large distances
        cell_idx_safe = bm.maximum(cell_idx, 0)
        d = self.D[cell_idx_safe]

        # Zero out the input if the position was invalid (cell_idx < 0)
        # This is JAX-compatible
        is_valid = cell_idx >= 0
        bump = self.A * bm.exp(-d / (2 * self.a**2))
        return bm.where(is_valid, bump, bm.zeros_like(bump))

    def update(self, animal_pos, theta_input):
        """
        Single time-step update of network dynamics.

        Args:
            animal_pos: Current position (x, y) tuple or array
            theta_input: Theta modulation factor (typically 1.0 ± theta_strength)
        """
        self.center.value = self.get_bump_center(r=self.r.value, x=self.x)
        Iext = theta_input * self.input_bump(animal_pos)
        Irec = bm.matmul(self.conn_mat, self.r.value)
        noise = bm.random.randn(self.cell_num) * self.noise_strength
        input_total = Iext + Irec + noise

        _u, _v = self.integral(self.u, self.v, bp.share["t"], input_total, bm.dt)
        self.u.value = bm.where(_u > 0, _u, 0)
        self.v.value = _v

        u_sq = bm.square(self.u.value)
        self.r.value = self.g * u_sq / (1.0 + self.k * bm.sum(u_sq))
