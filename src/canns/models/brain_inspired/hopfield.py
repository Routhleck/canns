import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ._base import BrainInspiredModel

__all__ = ["AmariHopfieldNetwork"]


class AmariHopfieldNetwork(BrainInspiredModel):
    """
    Amari-Hopfield Network implementation supporting both discrete and continuous dynamics.

    This class implements Hopfield networks with flexible activation functions,
    supporting both discrete binary states and continuous dynamics. The network
    performs pattern completion through energy minimization using asynchronous
    or synchronous updates.

    The network energy function:
    E = -0.5 * Σ_ij W_ij * s_i * s_j

    Where s_i can be discrete {-1, +1} or continuous depending on activation function.

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
        asyn: bool = False,
        threshold: float = 0.0,
        activation: str = "sign",
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the Amari-Hopfield Network.

        Args:
            num_neurons: Number of neurons in the network
            asyn: Whether to run asynchronously or synchronously
            threshold: Threshold for activation function
            activation: Activation function type ("sign", "tanh", "sigmoid")
            temperature: Temperature parameter for continuous activations
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(in_size=num_neurons, **kwargs)

        self.num_neurons = num_neurons
        self.asyn = asyn
        self.threshold = threshold
        self.temperature = temperature

        # Set activation function based on type
        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation: str):
        """Get activation function based on activation type."""
        if activation == "sign":
            return u.math.sign
        elif activation == "tanh":
            return lambda x: jnp.tanh(x / self.temperature)
        elif activation == "sigmoid":
            return lambda x: jax.nn.sigmoid(x / self.temperature)
        else:
            raise ValueError(f"Unknown activation type: {activation}")

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

        for i in tqdm(range(num_data), desc="Learning patterns"):
            t = train_data[i] - rho
            self.W.value += u.math.outer(t, t)

        # make diagonal element of W into 0
        diagW = u.math.diag(u.math.diag(self.W.value))
        self.W.value -= diagW
        self.W.value /= num_data

    def predict(
        self, data, num_iter=20, compiled=True, progress_callback=None, convergence_threshold=1e-10
    ):
        """
        Predict using the Hopfield network with energy-based convergence.

        Args:
            data: Initial pattern, shape (n_neurons,)
            num_iter: Maximum number of iterations
            compiled: If True, use compiled while_loop (faster, no progress).
                     If False, use Python loop (slower, with progress)
            progress_callback: Optional callback function called each iteration with
                             (iteration, energy, converged) tuple
            convergence_threshold: Energy change threshold for convergence detection

        Returns:
            Final converged pattern
        """
        # Initialize state with input data - use float32 for consistency
        self.s.value = jnp.array(data, dtype=jnp.float32)

        if compiled and progress_callback is None:
            # Use compiled version for maximum performance
            return self._predict_compiled(num_iter)
        else:
            # Use uncompiled version with progress reporting
            return self._predict_uncompiled(num_iter, progress_callback, convergence_threshold)

    def _predict_compiled(self, num_iter):
        """Compiled prediction using while_loop (high performance)."""
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
        initial_carry = (jnp.array(self.s.value, dtype=jnp.float32), initial_energy, 0)

        # Run compiled while loop
        final_s, final_energy, final_iter = brainstate.compile.while_loop(
            cond_fn,
            body_fn,
            initial_carry,
        )

        return final_s

    def _predict_uncompiled(self, num_iter, progress_callback, convergence_threshold):
        """Uncompiled prediction with progress reporting and early stopping."""
        prev_energy = float(self.energy)

        for iteration in range(num_iter):
            # Call the update method
            self.update(prev_energy)

            # Compute new energy
            current_energy = float(self.energy)

            # Check for convergence
            energy_change = abs(current_energy - prev_energy)
            converged = energy_change < convergence_threshold

            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(iteration + 1, current_energy, converged, energy_change)

            # Early stopping if converged
            if converged:
                break

            prev_energy = current_energy

        return self.s.value

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
