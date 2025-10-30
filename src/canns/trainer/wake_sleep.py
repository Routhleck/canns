"""Wake-Sleep algorithm for Helmholtz Machines."""

from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp

from ..models.brain_inspired import BrainInspiredModel
from ._base import Trainer

__all__ = ["WakeSleepTrainer"]


class WakeSleepTrainer(Trainer):
    """
    Wake-Sleep algorithm trainer for Helmholtz Machines.

    The wake-sleep algorithm alternates between two phases:

    Wake Phase:
        - Present data to recognition network to infer hidden states
        - Train generative network to reconstruct the data from inferred states
        - Update rule: maximize P(visible | hidden) where hidden is from recognition

    Sleep Phase:
        - Sample hidden states from prior distribution
        - Generate visible states using generative network
        - Train recognition network to infer the hidden states that generated the data
        - Update rule: maximize P(hidden | visible) where visible is from generation

    This provides local learning signals for both networks without backpropagation
    through the entire architecture.

    Reference:
        Hinton, G. E., Dayan, P., Frey, B. J., & Neal, R. M. (1995).
        The "wake-sleep" algorithm for unsupervised neural networks.
        Science, 268(5214), 1158-1161.
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        learning_rate: float = 0.01,
        num_sleep_samples: int = 10,
        **kwargs,
    ):
        """
        Initialize Wake-Sleep trainer.

        Args:
            model: The Helmholtz Machine to train
            learning_rate: Learning rate for weight updates
            num_sleep_samples: Number of samples to generate in sleep phase
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate
        self.num_sleep_samples = num_sleep_samples

        # Track statistics
        self.wake_errors = []
        self.sleep_errors = []

    def train(self, train_data: Iterable):
        """
        Train Helmholtz Machine using wake-sleep algorithm.

        Args:
            train_data: Iterable of visible patterns
        """
        # Check required model methods
        required_methods = ["recognize", "generate", "sample_from_prior"]
        for method in required_methods:
            if not hasattr(self.model, method):
                raise AttributeError(f"Model must have '{method}' method for wake-sleep training")

        # Check required parameters
        required_params = ["W_rec", "W_gen", "b_rec", "b_gen"]
        for param in required_params:
            if not hasattr(self.model, param):
                raise AttributeError(f"Model must have '{param}' parameter")

        self.wake_errors = []
        self.sleep_errors = []

        # Wake phase: train on data
        self._wake_phase(train_data)

        # Sleep phase: train on generated samples
        self._sleep_phase()

    def _wake_phase(self, train_data: Iterable):
        """
        Wake phase: train generative network on data.

        Args:
            train_data: Real data samples
        """
        W_gen = self.model.W_gen.value
        b_gen = self.model.b_gen.value

        for pattern in train_data:
            v_data = jnp.asarray(pattern, dtype=jnp.float32)

            # Recognize: infer hidden state from data
            h_inferred = self.model.recognize(v_data, stochastic=True)

            # Train generative network to reconstruct data from inferred hidden
            # Update rule: ΔW_gen = η * (v_data - v_pred) * h_inferred^T
            v_pred = self.model.generate(h_inferred, stochastic=False)
            error = v_data - v_pred

            # Gradient update (sigmoid cross-entropy derivative)
            delta_W_gen = self.learning_rate * jnp.outer(error, h_inferred)
            delta_b_gen = self.learning_rate * error

            W_gen = W_gen + delta_W_gen
            b_gen = b_gen + delta_b_gen

            # Track error
            self.wake_errors.append(float(jnp.mean(error**2)))

        # Update generative parameters
        self.model.W_gen.value = W_gen
        self.model.b_gen.value = b_gen

    def _sleep_phase(self):
        """
        Sleep phase: train recognition network on generated samples.
        """
        W_rec = self.model.W_rec.value
        b_rec = self.model.b_rec.value

        for _ in range(self.num_sleep_samples):
            # Sample hidden state from prior
            h_sampled = self.model.sample_from_prior()

            # Generate visible state
            v_generated = self.model.generate(h_sampled, stochastic=True)

            # Train recognition network to infer the sampled hidden state
            # Update rule: ΔW_rec = η * (h_sampled - h_pred) * v_generated^T
            h_pred = self.model.recognize(v_generated, stochastic=False)
            error = h_sampled - h_pred

            # Gradient update
            delta_W_rec = self.learning_rate * jnp.outer(error, v_generated)
            delta_b_rec = self.learning_rate * error

            W_rec = W_rec + delta_W_rec
            b_rec = b_rec + delta_b_rec

            # Track error
            self.sleep_errors.append(float(jnp.mean(error**2)))

        # Update recognition parameters
        self.model.W_rec.value = W_rec
        self.model.b_rec.value = b_rec

    def predict(self, pattern, *args, **kwargs):
        """
        Predict (reconstruct) visible pattern through the model.

        Args:
            pattern: Input visible pattern

        Returns:
            Reconstructed pattern
        """
        if hasattr(self.model, "forward"):
            return self.model.forward(pattern)
        else:
            # Manual reconstruction
            v = jnp.asarray(pattern, dtype=jnp.float32)
            h = self.model.recognize(v, stochastic=False)
            v_recon = self.model.generate(h, stochastic=False)
            return v_recon

    def get_training_statistics(self) -> dict:
        """
        Get training statistics from last epoch.

        Returns:
            Dictionary with wake and sleep phase errors
        """
        stats = {}

        if len(self.wake_errors) > 0:
            stats["mean_wake_error"] = float(jnp.mean(jnp.array(self.wake_errors)))

        if len(self.sleep_errors) > 0:
            stats["mean_sleep_error"] = float(jnp.mean(jnp.array(self.sleep_errors)))

        return stats
