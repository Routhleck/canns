"""BCM (Bienenstock-Cooper-Munro) layer with sliding threshold plasticity."""

from __future__ import annotations

import brainstate
import jax.numpy as jnp

from ._base import BrainInspiredModel

__all__ = ["BCMLayer"]


class BCMLayer(BrainInspiredModel):
    """
    BCM (Bienenstock-Cooper-Munro) layer with sliding threshold plasticity.
    
    The BCM rule uses a dynamic postsynaptic threshold to switch between
    potentiation and depression based on recent activity, yielding stable
    receptive-field development and experience-dependent refinement.
    
    Learning Rule:
        ΔW_ij = η * y_i * (y_i - θ_i) * x_j
    
    where:
        - W_ij is the weight from input j to neuron i
        - x_j is the presynaptic activity
        - y_i is the postsynaptic activity (y = W @ x)
        - θ_i is the modification threshold for neuron i
        - θ evolves as a sliding average of y^2: θ = <y^2>
    
    The threshold divides the learning rule into two regimes:
        - If y > θ: potentiation (strengthen synapses)
        - If y < θ: depression (weaken synapses)
    
    Reference:
        Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982).
        Theory for the development of neuron selectivity: orientation specificity
        and binocular interaction in visual cortex. Journal of Neuroscience, 2(1), 32-48.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        threshold_tau: float = 100.0,
        **kwargs,
    ):
        """
        Initialize the BCM layer.
        
        Args:
            input_size: Dimensionality of input vectors
            output_size: Number of output neurons
            threshold_tau: Time constant for threshold sliding average (higher = slower)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(in_size=input_size, **kwargs)
        
        self.input_size = input_size
        self.output_size = output_size
        self.threshold_tau = threshold_tau
    
    def init_state(self):
        """Initialize layer parameters and state variables."""
        # Weight matrix W: (output_size, input_size)
        self.W = brainstate.ParamState(
            jnp.zeros((self.output_size, self.input_size), dtype=jnp.float32)
        )
        # Input state
        self.x = brainstate.HiddenState(
            jnp.zeros(self.input_size, dtype=jnp.float32)
        )
        # Output state
        self.y = brainstate.HiddenState(
            jnp.zeros(self.output_size, dtype=jnp.float32)
        )
        # Sliding threshold (theta) - initialized to small positive values
        self.theta = brainstate.HiddenState(
            jnp.ones(self.output_size, dtype=jnp.float32) * 0.1
        )
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            x: Input vector of shape (input_size,)
        
        Returns:
            Output vector of shape (output_size,)
        """
        self.x.value = jnp.asarray(x, dtype=jnp.float32)
        self.y.value = self.W.value @ self.x.value
        return self.y.value
    
    def update_threshold(self):
        """Update the sliding threshold based on recent activity."""
        # θ <- θ + (1/τ) * (y^2 - θ)
        y_squared = self.y.value ** 2
        alpha = 1.0 / self.threshold_tau if self.threshold_tau > 0 else 1.0
        self.theta.value = self.theta.value + alpha * (y_squared - self.theta.value)
    
    def update(self, prev_energy):
        """Update method for trainer compatibility."""
        pass
    
    @property
    def energy(self) -> float:
        """Energy for trainer compatibility (0 for feedforward layer)."""
        return 0.0
    
    @property
    def weight_attr(self) -> str:
        """Name of weight parameter for generic training."""
        return "W"
    
    @property
    def predict_state_attr(self) -> str:
        """Name of output state for prediction."""
        return "y"
    
    def resize(self, input_size: int, output_size: int | None = None, preserve_submatrix: bool = True):
        """
        Resize layer dimensions.
        
        Args:
            input_size: New input dimension
            output_size: New output dimension (if None, keep current)
            preserve_submatrix: Whether to preserve existing weights
        """
        if output_size is None:
            output_size = self.output_size
        
        old_W = self.W.value if hasattr(self, "W") else None
        old_theta = self.theta.value if hasattr(self, "theta") else None
        
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        
        # Create new weight matrix
        new_W = jnp.zeros((self.output_size, self.input_size), dtype=jnp.float32)
        
        if preserve_submatrix and old_W is not None:
            min_out = min(old_W.shape[0], self.output_size)
            min_in = min(old_W.shape[1], self.input_size)
            new_W = new_W.at[:min_out, :min_in].set(old_W[:min_out, :min_in])
        
        # Create new threshold vector
        new_theta = jnp.ones(self.output_size, dtype=jnp.float32) * 0.1
        if preserve_submatrix and old_theta is not None:
            min_out = min(old_theta.shape[0], self.output_size)
            new_theta = new_theta.at[:min_out].set(old_theta[:min_out])
        
        # Update parameters
        if hasattr(self, "W"):
            self.W.value = new_W
        else:
            self.W = brainstate.ParamState(new_W)
        
        if hasattr(self, "theta"):
            self.theta.value = new_theta
        else:
            self.theta = brainstate.HiddenState(new_theta)
        
        if hasattr(self, "x"):
            self.x.value = jnp.zeros(self.input_size, dtype=jnp.float32)
        else:
            self.x = brainstate.HiddenState(jnp.zeros(self.input_size, dtype=jnp.float32))
        
        if hasattr(self, "y"):
            self.y.value = jnp.zeros(self.output_size, dtype=jnp.float32)
        else:
            self.y = brainstate.HiddenState(jnp.zeros(self.output_size, dtype=jnp.float32))
