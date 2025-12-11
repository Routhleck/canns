import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np

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
        super().__init__(**kwargs)

        self.num_neurons = num_neurons
        self.asyn = asyn
        self.threshold = threshold
        self.temperature = temperature

        # Set activation function based on type
        self.activation = self._get_activation_fn(activation)

        self.s = bm.Variable(jnp.ones(self.num_neurons, dtype=jnp.float32))  # Binary states (+1/-1)
        self.W = bm.Variable(
            jnp.zeros((self.num_neurons, self.num_neurons), dtype=jnp.float32)
        )  # Weight matrix as trainable parameter

    def _get_activation_fn(self, activation: str):
        """Get activation function based on activation type."""
        if activation == "sign":
            return bm.sign
        elif activation == "tanh":
            return lambda x: jnp.tanh(x / self.temperature)
        elif activation == "sigmoid":
            return lambda x: jax.nn.sigmoid(x / self.temperature)
        else:
            raise ValueError(f"Unknown activation type: {activation}")

    def update(self, e_old):
        """
        Update network state for one time step.
        """
        if self.asyn:
            self._asynchronous_update()
        else:
            self._synchronous_update()

    def _asynchronous_update(self):
        """Asynchronous update - one neuron at a time.

        Implemented with JAX-friendly primitives so it can be used in compiled
        prediction loops. Avoid Python-side mutation of traced indices.
        """
        key = bm.random.get_key()
        idxs = jax.random.permutation(key, self.num_neurons)

        def body(i, s):
            idx = idxs[i]
            # Update a single randomly-chosen neuron based on current state s
            val = self.activation(self.W.value[idx].T @ s - self.threshold)
            return s.at[idx].set(val)

        self.s.value = jax.lax.fori_loop(0, self.num_neurons, body, self.s.value)

    def _synchronous_update(self):
        """Synchronous update - all neurons simultaneously."""
        # update s
        self.s.value = self.activation(self.W.value @ self.s.value - self.threshold)

    # Hebbian learning is handled by HebbianTrainer; no model-specific method needed.

    def resize(self, num_neurons: int, preserve_submatrix: bool = True):
        """Resize the network dimension and state/weights.

        Args:
            num_neurons: New neuron count (N)
            preserve_submatrix: If True, copy the top-left min(old, N) block of W into
                the new matrix; otherwise reinitialize W with zeros.
        """
        old_n = getattr(self, "num_neurons", None)
        old_W = getattr(self, "W", None)
        old_s = getattr(self, "s", None)

        self.num_neurons = int(num_neurons)

        # Prepare new arrays
        N = self.num_neurons
        new_W = jnp.zeros((N, N), dtype=jnp.float32)
        if (
            preserve_submatrix
            and old_n is not None
            and old_W is not None
            and hasattr(old_W, "value")
        ):
            m = min(old_n, N)
            new_W = new_W.at[:m, :m].set(jnp.asarray(old_W.value)[:m, :m])
        # Zero diagonal for stability
        new_W = new_W - jnp.diag(jnp.diag(new_W))

        new_s = jnp.ones((N,), dtype=jnp.float32)

        # Assign back
        if old_W is not None and hasattr(old_W, "value"):
            old_W.value = new_W
        else:
            # In case resize called before init_state
            self.W = bm.Variable(new_W)

        if old_s is not None and hasattr(old_s, "value"):
            old_s.value = new_s
        else:
            self.s = bm.Variable(new_s)

    # Predict methods intentionally removed: use HebbianTrainer.predict for unified API.

    @property
    def energy(self):
        """
        Compute the energy of the network state.
        """
        state = self.s.value
        # Energy with threshold term: E = -0.5 * s^T W s + Σ_i s_i * threshold
        quad = -0.5 * jnp.dot(state, jnp.dot(self.W.value, state))
        thr = jnp.float32(self.threshold) * jnp.sum(state)
        return quad + thr

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
