"""Hopfield Energy-based associative memory trainer."""

from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp
import numpy as np

from ..models.brain_inspired import BrainInspiredModel
from .hebbian import HebbianTrainer

__all__ = ["HopfieldEnergyTrainer"]


class HopfieldEnergyTrainer(HebbianTrainer):
    """
    Hopfield Energy-based associative memory trainer.

    Extends HebbianTrainer with specialized features for Hopfield networks:
    - Pattern storage capacity estimation
    - Pattern replay utilities
    - Energy diagnostics
    - Overlap metrics for pattern retrieval

    The Hopfield network stores patterns as attractors in an energy landscape.
    Energy function:
        E = -0.5 * s^T W s

    Training uses Hebbian learning to shape the energy landscape so that
    stored patterns correspond to local minima.

    Reference:
        Hopfield, J. J. (1982). Neural networks and physical systems with
        emergent collective computational abilities. PNAS, 79(8), 2554-2558.
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        **kwargs,
    ):
        """
        Initialize Hopfield Energy trainer.

        Args:
            model: The Hopfield network model to train
            **kwargs: Additional arguments passed to HebbianTrainer
        """
        super().__init__(model=model, **kwargs)

        # Storage for training patterns (for analysis)
        self.stored_patterns = []
        self.pattern_energies = []

    def train(self, train_data: Iterable):
        """
        Train Hopfield network and store patterns for analysis.

        Args:
            train_data: Iterable of patterns to store
        """
        # Store patterns for later analysis
        self.stored_patterns = [jnp.asarray(p, dtype=jnp.float32) for p in train_data]

        # Use parent HebbianTrainer for actual weight update
        super().train(self.stored_patterns)

        # Compute and store energy of each pattern
        self._compute_pattern_energies()

    def _compute_pattern_energies(self):
        """Compute energy for each stored pattern."""
        self.pattern_energies = []

        weight_attr = self.weight_attr or getattr(self.model, "weight_attr", "W")
        weight_param = getattr(self.model, weight_attr)
        W = weight_param.value

        for pattern in self.stored_patterns:
            # E = -0.5 * s^T W s
            energy = -0.5 * jnp.dot(pattern, jnp.dot(W, pattern))
            self.pattern_energies.append(float(energy))

    def compute_overlap(self, pattern1: jnp.ndarray, pattern2: jnp.ndarray) -> float:
        """
        Compute normalized overlap between two patterns.

        Args:
            pattern1: First pattern
            pattern2: Second pattern

        Returns:
            Overlap value between -1 and 1
        """
        p1 = jnp.asarray(pattern1, dtype=jnp.float32)
        p2 = jnp.asarray(pattern2, dtype=jnp.float32)
        return float(jnp.dot(p1, p2) / len(p1))

    def recall_pattern(
        self, noisy_pattern: jnp.ndarray, num_iter: int = 20, **kwargs
    ) -> tuple[jnp.ndarray, dict]:
        """
        Recall a stored pattern from a noisy/partial cue.

        Args:
            noisy_pattern: Input pattern (possibly corrupted)
            num_iter: Number of update iterations
            **kwargs: Additional arguments passed to predict

        Returns:
            Tuple of (recalled_pattern, diagnostics_dict)
        """
        # Use parent predict method
        recalled = self.predict(noisy_pattern, num_iter=num_iter, **kwargs)

        # Compute diagnostics
        diagnostics = self._compute_recall_diagnostics(noisy_pattern, recalled)

        return recalled, diagnostics

    def _compute_recall_diagnostics(
        self, input_pattern: jnp.ndarray, output_pattern: jnp.ndarray
    ) -> dict:
        """
        Compute diagnostic information for pattern recall.

        Args:
            input_pattern: Input (noisy) pattern
            output_pattern: Recalled pattern

        Returns:
            Dictionary with diagnostic metrics
        """
        diagnostics = {}

        # Find best matching stored pattern
        if len(self.stored_patterns) > 0:
            overlaps = [
                self.compute_overlap(output_pattern, stored) for stored in self.stored_patterns
            ]
            best_idx = int(np.argmax(overlaps))
            diagnostics["best_match_idx"] = best_idx
            diagnostics["best_match_overlap"] = overlaps[best_idx]

        # Input-output overlap
        diagnostics["input_output_overlap"] = self.compute_overlap(input_pattern, output_pattern)

        # Energy of recalled pattern
        weight_attr = self.weight_attr or getattr(self.model, "weight_attr", "W")
        weight_param = getattr(self.model, weight_attr)
        W = weight_param.value
        output_energy = -0.5 * jnp.dot(output_pattern, jnp.dot(W, output_pattern))
        diagnostics["output_energy"] = float(output_energy)

        return diagnostics

    def estimate_capacity(self) -> int:
        """
        Estimate theoretical storage capacity of the network.

        Returns:
            Estimated number of patterns that can be reliably stored
        """
        if hasattr(self.model, "storage_capacity"):
            return self.model.storage_capacity

        # Default estimate: N / (4 * ln(N))
        n = self.model.num_neurons if hasattr(self.model, "num_neurons") else 100
        return max(1, int(n / (4 * np.log(n))))

    def get_pattern_statistics(self) -> dict:
        """
        Get statistics about stored patterns.

        Returns:
            Dictionary with pattern statistics
        """
        stats = {
            "num_patterns": len(self.stored_patterns),
            "capacity_estimate": self.estimate_capacity(),
            "capacity_usage": len(self.stored_patterns) / max(1, self.estimate_capacity()),
        }

        if len(self.pattern_energies) > 0:
            stats["mean_pattern_energy"] = float(np.mean(self.pattern_energies))
            stats["std_pattern_energy"] = float(np.std(self.pattern_energies))
            stats["min_pattern_energy"] = float(np.min(self.pattern_energies))
            stats["max_pattern_energy"] = float(np.max(self.pattern_energies))

        return stats
