"""Restricted Boltzmann Machine (RBM) for energy-based learning."""

from __future__ import annotations

import brainstate
import jax
import jax.numpy as jnp

from ._base import BrainInspiredModel

__all__ = ["RestrictedBoltzmannModel"]


class RestrictedBoltzmannModel(BrainInspiredModel):
    """
    Restricted Boltzmann Machine (RBM) for unsupervised feature learning.
    
    An RBM is a generative stochastic neural network with two layers:
    - Visible layer (v): represents observed data
    - Hidden layer (h): represents latent features
    
    Energy function:
        E(v, h) = -v^T W h - b^T v - c^T h
    
    where W is the weight matrix, b is visible bias, c is hidden bias.
    
    Training uses Contrastive Divergence to approximate maximum likelihood.
    
    Reference:
        Hinton, G. E. (2002). Training products of experts by minimizing
        contrastive divergence. Neural Computation, 14(8), 1771-1800.
    """
    
    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        Initialize Restricted Boltzmann Machine.
        
        Args:
            num_visible: Number of visible units
            num_hidden: Number of hidden units
            temperature: Temperature for Gibbs sampling (default: 1.0)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(in_size=num_visible, **kwargs)
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.temperature = temperature
    
    def init_state(self):
        """Initialize RBM parameters and state variables."""
        # Weight matrix connecting visible to hidden
        self.W = brainstate.ParamState(
            jnp.zeros((self.num_hidden, self.num_visible), dtype=jnp.float32)
        )
        
        # Visible bias
        self.b = brainstate.ParamState(
            jnp.zeros(self.num_visible, dtype=jnp.float32)
        )
        
        # Hidden bias
        self.c = brainstate.ParamState(
            jnp.zeros(self.num_hidden, dtype=jnp.float32)
        )
        
        # Current visible state
        self.v = brainstate.HiddenState(
            jnp.zeros(self.num_visible, dtype=jnp.float32)
        )
        
        # Current hidden state
        self.h = brainstate.HiddenState(
            jnp.zeros(self.num_hidden, dtype=jnp.float32)
        )
    
    def sample_hidden(self, visible: jnp.ndarray) -> jnp.ndarray:
        """
        Sample hidden units given visible units.
        
        Args:
            visible: Visible unit activations
        
        Returns:
            Binary hidden unit samples
        """
        # Compute hidden activation probabilities
        h_prob = self.hidden_prob(visible)
        
        # Sample from Bernoulli distribution
        key = brainstate.random.get_key()
        h_sample = (jax.random.uniform(key, h_prob.shape) < h_prob).astype(jnp.float32)
        
        return h_sample
    
    def sample_visible(self, hidden: jnp.ndarray) -> jnp.ndarray:
        """
        Sample visible units given hidden units.
        
        Args:
            hidden: Hidden unit activations
        
        Returns:
            Binary visible unit samples
        """
        # Compute visible activation probabilities
        v_prob = self.visible_prob(hidden)
        
        # Sample from Bernoulli distribution
        key = brainstate.random.get_key()
        v_sample = (jax.random.uniform(key, v_prob.shape) < v_prob).astype(jnp.float32)
        
        return v_sample
    
    def hidden_prob(self, visible: jnp.ndarray) -> jnp.ndarray:
        """
        Compute probability of hidden units being active.
        
        Args:
            visible: Visible unit values
        
        Returns:
            Hidden unit probabilities
        """
        activation = self.W.value @ visible + self.c.value
        return jax.nn.sigmoid(activation / self.temperature)
    
    def visible_prob(self, hidden: jnp.ndarray) -> jnp.ndarray:
        """
        Compute probability of visible units being active.
        
        Args:
            hidden: Hidden unit values
        
        Returns:
            Visible unit probabilities
        """
        activation = self.W.value.T @ hidden + self.b.value
        return jax.nn.sigmoid(activation / self.temperature)
    
    def gibbs_step(self, visible: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform one step of Gibbs sampling.
        
        Args:
            visible: Current visible state
        
        Returns:
            Tuple of (new_visible, new_hidden)
        """
        # Sample hidden given visible
        hidden = self.sample_hidden(visible)
        
        # Sample visible given hidden
        new_visible = self.sample_visible(hidden)
        
        return new_visible, hidden
    
    def free_energy(self, visible: jnp.ndarray) -> float:
        """
        Compute free energy of visible configuration.
        
        Free energy F(v) = -log(sum_h exp(-E(v,h)))
        
        For RBM: F(v) = -b^T v - sum_i log(1 + exp(W_i^T v + c_i))
        
        Args:
            visible: Visible unit configuration
        
        Returns:
            Free energy value
        """
        visible_term = -jnp.dot(self.b.value, visible)
        
        wx_b = self.W.value @ visible + self.c.value
        hidden_term = -jnp.sum(jnp.log(1.0 + jnp.exp(wx_b)))
        
        return float(visible_term + hidden_term)
    
    @property
    def energy(self) -> float:
        """Compute energy of current state."""
        return self.free_energy(self.v.value)
    
    def update(self, prev_energy):
        """Update method for trainer compatibility."""
        pass
    
    @property
    def weight_attr(self) -> str:
        """Name of weight parameter for generic training."""
        return "W"
    
    @property
    def predict_state_attr(self) -> str:
        """Name of visible state for prediction."""
        return "v"
    
    def resize(self, num_visible: int, num_hidden: int | None = None, preserve_submatrix: bool = True):
        """
        Resize RBM dimensions.
        
        Args:
            num_visible: New number of visible units
            num_hidden: New number of hidden units (if None, keep current)
            preserve_submatrix: Whether to preserve existing weights
        """
        if num_hidden is None:
            num_hidden = self.num_hidden
        
        old_W = self.W.value if hasattr(self, "W") else None
        old_b = self.b.value if hasattr(self, "b") else None
        old_c = self.c.value if hasattr(self, "c") else None
        
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        
        # Create new parameters
        new_W = jnp.zeros((self.num_hidden, self.num_visible), dtype=jnp.float32)
        new_b = jnp.zeros(self.num_visible, dtype=jnp.float32)
        new_c = jnp.zeros(self.num_hidden, dtype=jnp.float32)
        
        if preserve_submatrix:
            if old_W is not None:
                min_h = min(old_W.shape[0], self.num_hidden)
                min_v = min(old_W.shape[1], self.num_visible)
                new_W = new_W.at[:min_h, :min_v].set(old_W[:min_h, :min_v])
            if old_b is not None:
                min_v = min(old_b.shape[0], self.num_visible)
                new_b = new_b.at[:min_v].set(old_b[:min_v])
            if old_c is not None:
                min_h = min(old_c.shape[0], self.num_hidden)
                new_c = new_c.at[:min_h].set(old_c[:min_h])
        
        # Update parameters
        if hasattr(self, "W"):
            self.W.value = new_W
        else:
            self.W = brainstate.ParamState(new_W)
        
        if hasattr(self, "b"):
            self.b.value = new_b
        else:
            self.b = brainstate.ParamState(new_b)
        
        if hasattr(self, "c"):
            self.c.value = new_c
        else:
            self.c = brainstate.ParamState(new_c)
        
        # Update state variables
        if hasattr(self, "v"):
            self.v.value = jnp.zeros(self.num_visible, dtype=jnp.float32)
        else:
            self.v = brainstate.HiddenState(jnp.zeros(self.num_visible, dtype=jnp.float32))
        
        if hasattr(self, "h"):
            self.h.value = jnp.zeros(self.num_hidden, dtype=jnp.float32)
        else:
            self.h = brainstate.HiddenState(jnp.zeros(self.num_hidden, dtype=jnp.float32))
