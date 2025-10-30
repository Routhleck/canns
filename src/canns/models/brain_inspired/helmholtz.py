"""Helmholtz Machine for wake-sleep learning."""

from __future__ import annotations

import brainstate
import jax
import jax.numpy as jnp

from ._base import BrainInspiredModel

__all__ = ["HelmholtzMachine"]


class HelmholtzMachine(BrainInspiredModel):
    """
    Helmholtz Machine for unsupervised learning with wake-sleep algorithm.
    
    A Helmholtz machine consists of two networks:
    - Recognition network (bottom-up): maps data to latent representations
    - Generative network (top-down): maps latent representations to data
    
    The wake-sleep algorithm trains these networks in two phases:
    - Wake phase: uses recognition network to infer latents, trains generative network
    - Sleep phase: generates data from prior, trains recognition network
    
    Architecture:
        visible (v) <-> hidden (h)
        
        Recognition: h = f_rec(v)
        Generation: v = f_gen(h)
    
    Reference:
        Hinton, G. E., Dayan, P., Frey, B. J., & Neal, R. M. (1995).
        The "wake-sleep" algorithm for unsupervised neural networks.
        Science, 268(5214), 1158-1161.
    """
    
    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        **kwargs,
    ):
        """
        Initialize Helmholtz Machine.
        
        Args:
            num_visible: Number of visible units
            num_hidden: Number of hidden (latent) units
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(in_size=num_visible, **kwargs)
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
    
    def init_state(self):
        """Initialize recognition and generative network parameters."""
        # Recognition network weights (bottom-up): visible -> hidden
        self.W_rec = brainstate.ParamState(
            jnp.zeros((self.num_hidden, self.num_visible), dtype=jnp.float32)
        )
        self.b_rec = brainstate.ParamState(
            jnp.zeros(self.num_hidden, dtype=jnp.float32)
        )
        
        # Generative network weights (top-down): hidden -> visible
        self.W_gen = brainstate.ParamState(
            jnp.zeros((self.num_visible, self.num_hidden), dtype=jnp.float32)
        )
        self.b_gen = brainstate.ParamState(
            jnp.zeros(self.num_visible, dtype=jnp.float32)
        )
        
        # Current visible state
        self.v = brainstate.HiddenState(
            jnp.zeros(self.num_visible, dtype=jnp.float32)
        )
        
        # Current hidden state
        self.h = brainstate.HiddenState(
            jnp.zeros(self.num_hidden, dtype=jnp.float32)
        )
    
    def recognize(self, visible: jnp.ndarray, stochastic: bool = True) -> jnp.ndarray:
        """
        Recognition network: infer hidden state from visible.
        
        Args:
            visible: Visible unit values
            stochastic: Whether to sample or use probabilities
        
        Returns:
            Hidden unit activations (sampled or probabilities)
        """
        # Compute hidden probabilities
        h_logits = self.W_rec.value @ visible + self.b_rec.value
        h_prob = jax.nn.sigmoid(h_logits)
        
        if stochastic:
            # Sample from Bernoulli
            key = brainstate.random.get_key()
            return (jax.random.uniform(key, h_prob.shape) < h_prob).astype(jnp.float32)
        else:
            return h_prob
    
    def generate(self, hidden: jnp.ndarray, stochastic: bool = True) -> jnp.ndarray:
        """
        Generative network: generate visible from hidden.
        
        Args:
            hidden: Hidden unit values
            stochastic: Whether to sample or use probabilities
        
        Returns:
            Visible unit activations (sampled or probabilities)
        """
        # Compute visible probabilities
        v_logits = self.W_gen.value @ hidden + self.b_gen.value
        v_prob = jax.nn.sigmoid(v_logits)
        
        if stochastic:
            # Sample from Bernoulli
            key = brainstate.random.get_key()
            return (jax.random.uniform(key, v_prob.shape) < v_prob).astype(jnp.float32)
        else:
            return v_prob
    
    def forward(self, visible: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: recognize then generate.
        
        Args:
            visible: Input visible pattern
        
        Returns:
            Reconstructed visible pattern
        """
        self.v.value = jnp.asarray(visible, dtype=jnp.float32)
        self.h.value = self.recognize(self.v.value, stochastic=False)
        reconstruction = self.generate(self.h.value, stochastic=False)
        return reconstruction
    
    def sample_from_prior(self) -> jnp.ndarray:
        """
        Sample hidden units from prior (uniform Bernoulli).
        
        Returns:
            Sampled hidden state
        """
        key = brainstate.random.get_key()
        return (jax.random.uniform(key, (self.num_hidden,)) < 0.5).astype(jnp.float32)
    
    @property
    def energy(self) -> float:
        """Energy based on reconstruction error."""
        reconstruction = self.forward(self.v.value)
        return float(jnp.mean((self.v.value - reconstruction) ** 2))
    
    def update(self, prev_energy):
        """Update method for trainer compatibility."""
        pass
    
    @property
    def weight_attr(self) -> str:
        """Name of recognition weight parameter."""
        return "W_rec"
    
    @property
    def predict_state_attr(self) -> str:
        """Name of visible state for prediction."""
        return "v"
    
    def resize(
        self,
        num_visible: int,
        num_hidden: int | None = None,
        preserve_submatrix: bool = True
    ):
        """
        Resize Helmholtz Machine dimensions.
        
        Args:
            num_visible: New number of visible units
            num_hidden: New number of hidden units (if None, keep current)
            preserve_submatrix: Whether to preserve existing weights
        """
        if num_hidden is None:
            num_hidden = self.num_hidden
        
        old_W_rec = self.W_rec.value if hasattr(self, "W_rec") else None
        old_W_gen = self.W_gen.value if hasattr(self, "W_gen") else None
        old_b_rec = self.b_rec.value if hasattr(self, "b_rec") else None
        old_b_gen = self.b_gen.value if hasattr(self, "b_gen") else None
        
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        
        # Create new parameters
        new_W_rec = jnp.zeros((self.num_hidden, self.num_visible), dtype=jnp.float32)
        new_W_gen = jnp.zeros((self.num_visible, self.num_hidden), dtype=jnp.float32)
        new_b_rec = jnp.zeros(self.num_hidden, dtype=jnp.float32)
        new_b_gen = jnp.zeros(self.num_visible, dtype=jnp.float32)
        
        if preserve_submatrix:
            if old_W_rec is not None:
                min_h = min(old_W_rec.shape[0], self.num_hidden)
                min_v = min(old_W_rec.shape[1], self.num_visible)
                new_W_rec = new_W_rec.at[:min_h, :min_v].set(old_W_rec[:min_h, :min_v])
            if old_W_gen is not None:
                min_v = min(old_W_gen.shape[0], self.num_visible)
                min_h = min(old_W_gen.shape[1], self.num_hidden)
                new_W_gen = new_W_gen.at[:min_v, :min_h].set(old_W_gen[:min_v, :min_h])
            if old_b_rec is not None:
                min_h = min(old_b_rec.shape[0], self.num_hidden)
                new_b_rec = new_b_rec.at[:min_h].set(old_b_rec[:min_h])
            if old_b_gen is not None:
                min_v = min(old_b_gen.shape[0], self.num_visible)
                new_b_gen = new_b_gen.at[:min_v].set(old_b_gen[:min_v])
        
        # Update parameters
        if hasattr(self, "W_rec"):
            self.W_rec.value = new_W_rec
        else:
            self.W_rec = brainstate.ParamState(new_W_rec)
        
        if hasattr(self, "W_gen"):
            self.W_gen.value = new_W_gen
        else:
            self.W_gen = brainstate.ParamState(new_W_gen)
        
        if hasattr(self, "b_rec"):
            self.b_rec.value = new_b_rec
        else:
            self.b_rec = brainstate.ParamState(new_b_rec)
        
        if hasattr(self, "b_gen"):
            self.b_gen.value = new_b_gen
        else:
            self.b_gen = brainstate.ParamState(new_b_gen)
        
        # Update state variables
        if hasattr(self, "v"):
            self.v.value = jnp.zeros(self.num_visible, dtype=jnp.float32)
        else:
            self.v = brainstate.HiddenState(jnp.zeros(self.num_visible, dtype=jnp.float32))
        
        if hasattr(self, "h"):
            self.h.value = jnp.zeros(self.num_hidden, dtype=jnp.float32)
        else:
            self.h = brainstate.HiddenState(jnp.zeros(self.num_hidden, dtype=jnp.float32))
