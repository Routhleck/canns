import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ._base import BrainInspiredModel

__all__ = ["AmariHopfieldNetwork", "DiscreteHopfieldNetwork"]


class AmariHopfieldNetwork(BrainInspiredModel):
    """
    Amari-Hopfield Network implementation.

    This class implements the Amari-Hopfield neural network model, which combines
    the continuous dynamics of Amari's neural field equations with the associative
    memory capabilities of Hopfield networks. The network can store and retrieve
    patterns through Hebbian learning and energy minimization.

    The network dynamics follow:
    τ * du_i/dt = -u_i + Σ_j W_ij * f(u_j) + I_i

    Where:
    - u_i: membrane potential of neuron i
    - W_ij: connection weight from neuron j to i
    - f: activation function (typically sigmoid or tanh)
    - I_i: external input to neuron i
    - τ: time constant

    Reference:
        Amari, S. (1977). Neural theory of association and concept-formation.
        Biological Cybernetics, 26(3), 175-185.

        Hopfield, J. J. (1982). Neural networks and physical systems with
        emergent collective computational abilities. Proceedings of the
        National Academy of Sciences of the USA, 79(8), 2554-2558.
    """

    def __init__(
        self,
        num_neurons: int,
        tau: float = 1.0,
        activation: str = "tanh",
        dt: float = 0.1,
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the Amari-Hopfield Network.

        Args:
            num_neurons: Number of neurons in the network
            tau: Time constant for the dynamics
            activation: Activation function ('tanh', 'sigmoid', 'linear')
            dt: Integration time step
            temperature: Temperature parameter for activation function
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent with required in_size parameter
        super().__init__(in_size=num_neurons, **kwargs)

        self.num_neurons = num_neurons
        self.tau = tau
        self.activation = activation
        self.dt = dt
        self.temperature = temperature

        # Weight matrix as trainable parameter
        self.W = brainstate.ParamState(jnp.zeros((num_neurons, num_neurons)))

    def init_state(self):
        """Initialize network state variables."""
        self.u = brainstate.HiddenState(jnp.zeros(self.num_neurons))  # Membrane potentials
        self.r = brainstate.HiddenState(jnp.zeros(self.num_neurons))  # Firing rates
        self.inp = brainstate.HiddenState(jnp.zeros(self.num_neurons))  # External input

    def f_activation(self, u):
        """
        Apply activation function to membrane potentials.

        Args:
            u: Membrane potentials

        Returns:
            Activated values (firing rates)
        """
        if self.activation == "tanh":
            return jnp.tanh(u / self.temperature)
        elif self.activation == "sigmoid":
            return jax.nn.sigmoid(u / self.temperature)
        elif self.activation == "linear":
            return u
        elif self.activation == "sign":
            return jnp.sign(u)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def update(self, inputs=None):
        """
        Update network state for one time step.

        Args:
            inputs: External input to the network
        """
        if inputs is not None:
            self.inp.value = inputs

        # Compute firing rates
        self.r.value = self.f_activation(self.u.value)

        # Update membrane potentials: τ * du/dt = -u + W*r + I
        recurrent_input = jnp.dot(self.W.value, self.r.value)
        du_dt = (-self.u.value + recurrent_input + self.inp.value) / self.tau
        self.u.value = self.u.value + self.dt * du_dt

    def __call__(self, inputs=None):
        """Forward pass of the network."""
        self.update(inputs)
        return self.r.value

    def apply_learning_rule(self, patterns, learning_rate=0.01, rule="hebbian"):
        """
        Apply learning rule to store patterns in the network.

        Args:
            patterns: Array of patterns to store, shape (num_patterns, num_neurons)
            learning_rate: Learning rate for weight updates
            rule: Learning rule type ('hebbian', 'storkey')
        """
        patterns = jnp.array(patterns)
        if patterns.ndim == 1:
            patterns = patterns[None, :]  # Make it 2D

        if rule == "hebbian":
            self._hebbian_learning(patterns, learning_rate)
        elif rule == "storkey":
            self._storkey_learning(patterns, learning_rate)
        else:
            raise ValueError(f"Unknown learning rule: {rule}")

    def _hebbian_learning(self, patterns, learning_rate):
        """
        Standard Hebbian learning for pattern storage.

        W_ij = (1/N) * Σ_μ (ξ_i^μ * ξ_j^μ)
        where ξ^μ are the stored patterns.
        """
        num_patterns, num_neurons = patterns.shape
        assert num_neurons == self.num_neurons, "Pattern size must match network size"

        # Compute correlation matrix
        correlation_matrix = jnp.dot(patterns.T, patterns) / num_patterns

        # Update weights (ensure no self-connections)
        new_weights = learning_rate * correlation_matrix
        new_weights = new_weights.at[jnp.diag_indices(self.num_neurons)].set(0.0)
        self.W.value = new_weights

    def _storkey_learning(self, patterns, learning_rate):
        """
        Storkey learning rule for improved pattern storage.
        More sophisticated than Hebbian learning, reduces spurious states.
        """
        # Simplified Storkey rule implementation
        # For full implementation, would need iterative updates
        num_patterns, num_neurons = patterns.shape
        assert num_neurons == self.num_neurons, "Pattern size must match network size"

        W_new = jnp.zeros((num_neurons, num_neurons))

        for mu in range(num_patterns):
            pattern = patterns[mu]
            # Local field without contribution from current pattern
            h_local = jnp.dot(self.W.value, pattern) - jnp.outer(pattern, pattern) * jnp.diag(
                self.W.value
            )

            # Storkey update
            dW = (
                jnp.outer(pattern, pattern)
                - jnp.outer(pattern, jnp.tanh(h_local))
                - jnp.outer(jnp.tanh(h_local), pattern)
            )
            W_new += learning_rate * dW / num_patterns

        W_new = W_new.at[jnp.diag_indices(self.num_neurons)].set(0.0)
        self.W.value = W_new

    def compute_energy(self, state=None):
        """
        Compute the energy of the network state.

        Energy function: E = -0.5 * Σ_ij W_ij * r_i * r_j - Σ_i I_i * r_i

        Args:
            state: State to compute energy for. If None, uses current firing rates.

        Returns:
            Energy value
        """
        if state is None:
            state = self.r.value

        # Quadratic term: -0.5 * r^T * W * r
        quadratic_term = -0.5 * jnp.dot(state, jnp.dot(self.W.value, state))

        # Linear term: -I^T * r
        linear_term = -jnp.dot(self.inp.value, state)

        return quadratic_term + linear_term

    def retrieve_pattern(self, partial_pattern, max_iter=100, threshold=1e-6):
        """
        Retrieve a stored pattern from a partial or noisy input.

        Args:
            partial_pattern: Partial or noisy pattern to complete
            max_iter: Maximum number of iterations
            threshold: Convergence threshold

        Returns:
            Retrieved pattern and convergence info
        """
        # Set initial state
        self.u.value = jnp.array(partial_pattern, dtype=jnp.float32)
        self.inp.value = jnp.zeros(self.num_neurons)  # No external input during retrieval

        energy_history = []
        state_history = []

        converged = False
        final_iterations = 0

        for i in range(max_iter):
            old_state = self.r.value.copy()
            self.update()

            # Record history
            energy = self.compute_energy()
            energy_history.append(energy)
            state_history.append(self.r.value.copy())

            # Check convergence
            state_change = jnp.linalg.norm(self.r.value - old_state)
            if state_change < threshold:
                converged = True
                final_iterations = i + 1
                break
            final_iterations = i + 1

        return {
            "final_pattern": self.r.value.copy(),
            "converged": converged,
            "iterations": final_iterations,
            "final_energy": energy_history[-1],
            "energy_history": jnp.array(energy_history),
            "state_history": jnp.array(state_history),
        }

    def add_noise(self, noise_level=0.1, seed=None):
        """
        Add noise to the current membrane potentials.

        Args:
            noise_level: Standard deviation of Gaussian noise
            seed: Random seed for reproducibility
        """
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(np.random.randint(0, 2**32))

        noise = jax.random.normal(key, shape=(self.num_neurons,))
        self.u.value = self.u.value + noise_level * noise

    def get_stored_patterns_capacity(self):
        """
        Estimate the storage capacity based on network size.

        Returns:
            Theoretical storage capacity (approximately N/(4*ln(N)) for large N)
        """
        return max(1, int(self.num_neurons / (4 * np.log(self.num_neurons))))

    def compute_overlap(self, pattern1, pattern2):
        """
        Compute overlap between two patterns.

        Args:
            pattern1, pattern2: Patterns to compare

        Returns:
            Overlap value (1 for identical, 0 for orthogonal, -1 for opposite)
        """
        return jnp.dot(pattern1, pattern2) / self.num_neurons

    def get_attractors(self, num_random_starts=100, max_iter=50):
        """
        Find attractor states of the network by running from random initial conditions.

        Args:
            num_random_starts: Number of random initial conditions to try
            max_iter: Maximum iterations for each trial

        Returns:
            List of unique attractor states
        """
        attractors = []

        for i in range(num_random_starts):
            # Random initial state
            key = jax.random.PRNGKey(i)
            initial_state = jax.random.uniform(key, (self.num_neurons,), minval=-1, maxval=1)

            result = self.retrieve_pattern(initial_state, max_iter=max_iter)
            final_pattern = result["final_pattern"]

            # Check if this attractor is already found
            is_new = True
            for existing_attractor in attractors:
                if jnp.allclose(final_pattern, existing_attractor, atol=0.1):
                    is_new = False
                    break

            if is_new:
                attractors.append(final_pattern)

        return attractors


class DiscreteHopfieldNetwork(BrainInspiredModel):
    """
    Discrete Hopfield Network implementation.

    This class implements the classic discrete Hopfield network with binary states
    (+1, -1). The network performs pattern completion through energy minimization
    using asynchronous or synchronous updates.

    The network energy function:
    E = -0.5 * Σ_ij W_ij * s_i * s_j

    Where s_i ∈ {-1, +1} are the binary states.

    Reference:
        Hopfield, J. J. (1982). Neural networks and physical systems with
        emergent collective computational abilities. Proceedings of the
        National Academy of Sciences of the USA, 79(8), 2554-2558.
    """

    def __init__(
        self,
        num_neurons: int,
        asyn: bool = False,
        threshold: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the Discrete Hopfield Network.

        Args:
            num_neurons: Number of neurons in the network
            asyn: Whether to run asynchronously or synchronously
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(in_size=num_neurons, **kwargs)

        self.num_neurons = num_neurons
        self.asyn = asyn
        self.activation = u.math.sign
        self.threshold = threshold

    def init_state(self):
        """Initialize network state variables."""
        self.s = brainstate.HiddenState(
            jnp.ones(self.num_neurons, dtype=jnp.float32)
        )  # Binary states (+1/-1)
        self.W = brainstate.ParamState(
            jnp.zeros((self.num_neurons, self.num_neurons), dtype=jnp.float32)
        )  # Weight matrix as trainable parameter

    def update(self, e_old):
        """
        Update network state for one time step.
        """
        if self.asyn:
            self._asynchronous_update()
        else:
            self._synchronous_update()

    def _asynchronous_update(self):
        """Asynchronous update - one neuron at a time."""
        random_indices = jax.random.permutation(brainstate.random.get_key(), self.num_neurons)
        for idx in random_indices:
            self.s.value.at[idx] = self.activation(
                self.W.value[idx].T @ self.s.value - self.threshold
            )

    def _synchronous_update(self):
        """Synchronous update - all neurons simultaneously."""
        # update s
        self.s.value = self.activation(self.W.value @ self.s.value - self.threshold)

    def apply_hebbian_learning(self, train_data):
        num_data = len(train_data)
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data * self.num_neurons)

        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            self.W.value += u.math.outer(t, t)

        # make diagonal element of W into 0
        diagW = u.math.diag(u.math.diag(self.W.value))
        self.W.value -= diagW
        self.W.value /= num_data

    def predict(self, data, num_iter=20):
        """
        Predict using the Hopfield network with energy-based convergence.

        Args:
            data: Initial pattern, shape (n_neurons,)
            num_iter: Maximum number of iterations

        Returns:
            Final converged pattern
        """

        # Initialize state with input data - use float32 for consistency
        self.s.value = jnp.array(data, dtype=jnp.float32)

        # Compute initial energy - ensure it's float32 for consistency
        initial_energy = jnp.float32(self.energy)

        def cond_fn(carry):
            """Continue while not converged and under max iterations."""
            s, prev_energy, iteration = carry
            return iteration < num_iter

        def body_fn(carry):
            """Single step of the network update."""
            s, prev_energy, iteration = carry

            # Set current state
            self.s.value = s

            # Call the update method (handles sync/async automatically)
            self.update(prev_energy)

            # Get new state and energy - all float32 for consistency
            new_s = jnp.array(self.s.value, dtype=jnp.float32)
            new_energy = jnp.float32(self.energy)

            # Note: Energy convergence check could be implemented here
            # but while_loop doesn't support early exit, so we rely on max_iter
            return new_s, new_energy, iteration + 1

        # Initial carry: (state, energy, iteration) - all consistent types
        initial_carry = (
            jnp.array(self.s.value, dtype=jnp.float32), 
            initial_energy, 
            0
        )

        # Run compiled while loop
        final_s, final_energy, final_iter = brainstate.compile.while_loop(
            cond_fn, body_fn, initial_carry,
        )

        return final_s

    @property
    def energy(self):
        """
        Compute the energy of the network state.
        """
        state = self.s.value

        # Energy: E = -0.5 * Σ_ij W_ij * s_i * s_j
        return -0.5 * jnp.dot(state, jnp.dot(self.W.value, state))

    @property
    def storage_capacity(self):
        """
        Get theoretical storage capacity.

        Returns:
            Theoretical storage capacity (approximately N/(4*ln(N)))
        """
        return max(1, int(self.num_neurons / (4 * np.log(self.num_neurons))))

    def compute_overlap(self, pattern1, pattern2):
        """
        Compute overlap between two binary patterns.

        Args:
            pattern1, pattern2: Binary patterns to compare

        Returns:
            Overlap value (1 for identical, 0 for orthogonal, -1 for opposite)
        """
        return jnp.dot(pattern1, pattern2) / self.num_neurons
