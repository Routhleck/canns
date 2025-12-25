"""
Simplified 2D grid cell network with hexagonal lattice structure.

This module implements a standard grid cell model,
suitable for basic spatial navigation and path integration tasks.
"""

import brainpy.math as bm
import jax

from ._base import BasicModel


class GridCell2D(BasicModel):
    """
    Simplified 2D continuous-attractor grid cell network with hexagonal lattice structure.

    This network implements a twisted torus topology that generates grid cell-like
    spatial representations with hexagonal periodicity.

    The network operates in a transformed coordinate system where grid cells form
    a hexagonal lattice, enabling realistic grid field spacing and orientation.

    Args:
        length: Number of grid cells along one dimension (total = length^2). Default: 30
        tau: Membrane time constant (ms). Default: 10.0
        k: Global inhibition strength for divisive normalization. Default: 1.0
        a: Width of connectivity kernel. Determines bump width. Default: 0.8
        A: Amplitude of external input. Default: 3.0
        J0: Peak recurrent connection strength. Default: 5.0
        mapping_ratio: Controls grid spacing (larger = smaller spacing).
            Grid spacing λ = 2π / mapping_ratio. Default: 1.5
        noise_strength: Standard deviation of activity noise. Default: 0.1
        conn_noise: Standard deviation of connectivity noise. Default: 0.0
        g: Firing rate gain factor (scales to biological range). Default: 1.0

    Attributes:
        num (int): Total number of grid cells (length^2)
        x_grid, y_grid (Array): Grid cell preferred phases in [-π, π]
        value_grid (Array): Neuron positions in phase space, shape (num, 2)
        Lambda (float): Grid spacing in real space
        coor_transform (Array): Hexagonal to rectangular coordinate transform
        coor_transform_inv (Array): Rectangular to hexagonal coordinate transform
        conn_mat (Array): Recurrent connectivity matrix
        candidate_centers (Array): Grid of candidate bump centers for decoding
        r (Variable): Firing rates (shape: num)
        u (Variable): Membrane potentials (shape: num)
        center_phase (Variable): Decoded bump center in phase space (shape: 2)
        center_position (Variable): Decoded position in real space (shape: 2)
        inp (Variable): External input for tracking (shape: num)
        gc_bump (Variable): Grid cell bump activity pattern (shape: num)

    Example:
        >>> import brainpy.math as bm
        >>> from canns.models.basic import GridCell2D
        >>>
        >>> bm.set_dt(1.0)
        >>> model = GridCell2D(length=30, mapping_ratio=1.5)
        >>>
        >>> # Update with 2D position
        >>> position = [0.5, 0.3]
        >>> model.update(position)
        >>>
        >>> # Access decoded position
        >>> decoded_pos = model.center_position.value
        >>> print(f"Decoded position: {decoded_pos}")

    References:
        Burak, Y., & Fiete, I. R. (2009).
        Accurate path integration in continuous attractor network models of grid cells.
        PLoS Computational Biology, 5(2), e1000291.
    """

    def __init__(
        self,
        length: int = 30,
        tau: float = 10.0,
        k: float = 1.0,
        a: float = 0.8,
        A: float = 3.0,
        J0: float = 5.0,
        mapping_ratio: float = 1.5,
        noise_strength: float = 0.1,
        conn_noise: float = 0.0,
        g: float = 1.0,
    ):
        """Initialize the simplified grid cell network."""
        self.num = length * length
        super().__init__()

        # Store parameters
        self.length = length
        self.tau = tau
        self.k = k
        self.a = a
        self.A = A
        self.J0 = J0
        self.g = g
        self.noise_strength = noise_strength
        self.conn_noise = conn_noise
        self.mapping_ratio = mapping_ratio

        # Grid spacing in real space
        self.Lambda = 2 * bm.pi / mapping_ratio

        # Coordinate transformation matrices (hexagonal <-> rectangular)
        # coor_transform maps parallelogram (60° angle) to square
        # This partitions 2D space into parallelograms, each containing one lattice of grid cells
        self.coor_transform = bm.array([[1.0, -1.0 / bm.sqrt(3.0)], [0.0, 2.0 / bm.sqrt(3.0)]])

        # coor_transform_inv maps square to parallelogram (60° angle)
        # Equivalent to: bm.array([[1.0, 1.0/2], [0.0, bm.sqrt(3.0)/2]])
        self.coor_transform_inv = bm.linalg.inv(self.coor_transform)

        # Feature space: phase coordinates in [-π, π]
        x_bins = bm.linspace(-bm.pi, bm.pi, length + 1)
        x_grid, y_grid = bm.meshgrid(x_bins[:-1], x_bins[:-1])
        self.x_grid = x_grid.reshape(-1)
        self.y_grid = y_grid.reshape(-1)

        # Neuron positions in phase space, shape (num, 2)
        self.value_grid = bm.stack([self.x_grid, self.y_grid], axis=1)
        # Scaled positions for bump template generation
        self.value_bump = self.value_grid * 4

        # Candidate centers for position decoding (disambiguates periodic grid)
        self.candidate_centers = self.make_candidate_centers(self.Lambda)

        # Build connectivity matrix with optional noise
        base_connection = self.make_connection()
        noise_connection = bm.random.randn(self.num, self.num) * conn_noise
        self.conn_mat = base_connection + noise_connection

        # Initialize state variables
        self.r = bm.Variable(bm.zeros(self.num))  # Firing rates
        self.u = bm.Variable(bm.zeros(self.num))  # Membrane potentials
        self.inp = bm.Variable(bm.zeros(self.num))  # External input (for tracking)
        self.center_phase = bm.Variable(bm.zeros(2))  # Decoded bump center (phase)
        self.center_position = bm.Variable(bm.zeros(2))  # Decoded position (real space)
        self.gc_bump = bm.Variable(bm.zeros(self.num))  # Bump activity pattern

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
            # v: (2,) location in (x,y) phase space
            d = self.calculate_dist(v - self.value_grid)  # (N,) distances
            return (
                (self.J0 / self.g)
                * bm.exp(-0.5 * bm.square(d / self.a))
                / (bm.sqrt(2.0 * bm.pi) * self.a)
            )

        return kernel(self.value_grid)  # (N, N)

    def calculate_dist(self, d):
        """
        Calculate Euclidean distance after hexagonal coordinate transformation.

        Applies periodic boundary conditions and transforms displacement vectors
        from phase space to hexagonal lattice coordinates.

        Args:
            d: Displacement vectors in phase space, shape (..., 2)

        Returns:
            Array of shape (...,): Euclidean distances in hexagonal space
        """
        # Apply periodic boundary conditions
        d = self.handle_periodic_condition(d)
        # Transform to lattice axes (hex/rect)
        # This means the bump on the parallelogram lattice is a Gaussian,
        # while in the square space it is a twisted Gaussian
        dist = bm.matmul(self.coor_transform_inv, d.T).T
        return bm.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2)

    def handle_periodic_condition(self, d):
        """
        Apply periodic boundary conditions to wrap phases into [-π, π].

        Args:
            d: Phase values (any shape with last dimension = 2)

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

        cc_transformed = bm.dot(self.coor_transform_inv, cc.reshape(N_c * N_c, 2).T).T

        return cc_transformed

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

    def get_unique_activity_bump(self, network_activity, animal_position):
        """
        Decode unique bump location from network activity and animal position.

        Estimates the activity bump center in phase space using population vector
        decoding, then maps it to real space and snaps to the nearest candidate
        center to resolve periodic ambiguity.

        Args:
            network_activity: Grid cell firing rates, shape (num,)
            animal_position: Current animal position for disambiguation, shape (2,)

        Returns:
            center_phase: Phase coordinates of bump center, shape (2,)
            center_position: Real-space position of bump (nearest candidate), shape (2,)
            bump: Gaussian bump template centered at center_position, shape (num,)
        """
        # Decode bump center in phase space using population vector
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        activity_masked = bm.where(
            network_activity > bm.max(network_activity) * 0.1, network_activity, 0.0
        )

        center_phase = bm.zeros((2,))
        center_phase = center_phase.at[0].set(bm.angle(bm.sum(exppos_x * activity_masked)))
        center_phase = center_phase.at[1].set(bm.angle(bm.sum(exppos_y * activity_masked)))

        # Map back to real space, snap to nearest candidate center
        center_pos_residual = bm.matmul(self.coor_transform_inv, center_phase) / self.mapping_ratio
        candidate_pos_all = self.candidate_centers + center_pos_residual
        distances = bm.linalg.norm(candidate_pos_all - animal_position, axis=1)
        center_position = candidate_pos_all[bm.argmin(distances)]

        # Build Gaussian bump template
        d = bm.asarray(center_position) - self.value_bump
        dist = bm.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        gc_bump = self.A * bm.exp(-bm.square(dist / self.a))

        return center_phase, center_position, gc_bump

    def get_stimulus_by_pos(self, position):
        """
        Generate Gaussian stimulus centered at given position.

        Useful for previewing input patterns without calling update.

        Args:
            position: 2D position [x, y] in real space

        Returns:
            Array of shape (num,): External input pattern
        """
        position = bm.asarray(position)
        phase = self.position2phase(position)
        d = self.calculate_dist(phase - self.value_grid)
        return self.A * bm.exp(-0.5 * bm.square(d / self.a))

    def update(self, position):
        """
        Single time-step update of grid cell network dynamics.

        Updates network activity using continuous attractor dynamics with
        direct position-based external input. No adaptation or theta modulation.

        Args:
            position: Current 2D position [x, y] in real space, shape (2,)
        """
        # Convert position to array if needed
        position = bm.asarray(position)

        # 1. Decode current bump center for tracking
        center_phase, center_position, gc_bump = self.get_unique_activity_bump(
            self.r.value, position
        )
        self.center_phase.value = center_phase
        self.center_position.value = center_position
        self.gc_bump.value = gc_bump

        # 2. Calculate external input directly from position
        phase = self.position2phase(position)
        d = self.calculate_dist(phase - self.value_grid)
        Iext = self.A * bm.exp(-0.5 * bm.square(d / self.a))
        self.inp.value = Iext  # store for debugging/analysis

        # 3. Calculate recurrent input
        Irec = bm.matmul(self.conn_mat, self.r.value)

        # 4. Add activity noise
        noise = bm.random.randn(self.num) * self.noise_strength

        # 5. Update membrane potential (Euler integration)
        total_input = Irec + Iext + noise
        self.u.value += (-self.u.value + total_input) / self.tau * bm.get_dt()
        # Apply ReLU non-linearity
        self.u.value = bm.where(self.u.value > 0.0, self.u.value, 0.0)

        # 6. Calculate firing rates with divisive normalization
        u_sq = bm.square(self.u.value)
        self.r.value = self.g * u_sq / (1.0 + self.k * bm.sum(u_sq))


class GridCell2D_SFA(GridCell2D):
    """
    GridCell2D with Spike-Frequency Adaptation (SFA).

    Extends GridCell2D with slow negative feedback adaptation for
    anticipative tracking behavior.

    Args:
        All GridCell2D parameters, plus:
        tau_v: Adaptation time constant (much slower than tau). Default: 50.0
        m: Coupling strength from membrane potential to adaptation. Default: 0.3

    Additional Attributes:
        v (Variable): Adaptation variable (shape: num)

    Example:
        >>> import brainpy.math as bm
        >>> from canns.models.basic import GridCell2D_SFA
        >>>
        >>> bm.set_dt(1.0)
        >>> model = GridCell2D_SFA(length=30, mapping_ratio=1.5)
        >>>
        >>> # Same interface as GridCell2D
        >>> position = [0.5, 0.3]
        >>> model.update(position)

    References:
        Adaptation mechanism enables anticipative tracking in moving
        input scenarios.
    """

    def __init__(
        self,
        length: int = 30,
        tau: float = 10.0,
        k: float = 1.0,
        a: float = 0.8,
        A: float = 3.0,
        J0: float = 5.0,
        mapping_ratio: float = 1.5,
        noise_strength: float = 0.1,
        conn_noise: float = 0.0,
        g: float = 1.0,
        tau_v: float = 50.0,
        m: float = 0.3,
    ):
        """Initialize GridCell2D with SFA adaptation."""
        # Initialize parent GridCell2D
        super().__init__(
            length=length,
            tau=tau,
            k=k,
            a=a,
            A=A,
            J0=J0,
            mapping_ratio=mapping_ratio,
            noise_strength=noise_strength,
            conn_noise=conn_noise,
            g=g,
        )

        # Store SFA parameters
        self.tau_v = tau_v
        self.m = m

        # Initialize adaptation variable
        self.v = bm.Variable(bm.zeros(self.num))

    def update(self, position):
        """
        Single time-step update with SFA adaptation.

        Same as GridCell2D.update() but with slow negative feedback
        from adaptation variable v.

        Args:
            position: Current 2D position [x, y] in real space, shape (2,)
        """
        # Convert position to array if needed
        position = bm.asarray(position)

        # 1. Decode current bump center for tracking
        center_phase, center_position, gc_bump = self.get_unique_activity_bump(
            self.r.value, position
        )
        self.center_phase.value = center_phase
        self.center_position.value = center_position
        self.gc_bump.value = gc_bump

        # 2. Calculate external input directly from position
        phase = self.position2phase(position)
        d = self.calculate_dist(phase - self.value_grid)
        Iext = self.A * bm.exp(-0.5 * bm.square(d / self.a))
        self.inp.value = Iext

        # 3. Calculate recurrent input
        Irec = bm.matmul(self.conn_mat, self.r.value)

        # 4. Add activity noise
        noise = bm.random.randn(self.num) * self.noise_strength

        # 5. Update membrane potential with SFA (KEY MODIFICATION)
        total_input = Irec + Iext + noise - self.v.value  # Subtract adaptation
        self.u.value += (-self.u.value + total_input) / self.tau * bm.get_dt()
        # Apply ReLU non-linearity
        self.u.value = bm.where(self.u.value > 0.0, self.u.value, 0.0)

        # 6. Calculate firing rates with divisive normalization
        u_sq = bm.square(self.u.value)
        self.r.value = self.g * u_sq / (1.0 + self.k * bm.sum(u_sq))

        # 7. Update adaptation variable (NEW)
        self.v.value += (-self.v.value + self.m * self.u.value) / self.tau_v * bm.get_dt()
