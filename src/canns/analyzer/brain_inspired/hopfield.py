"""Hopfield network analysis tools."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["HopfieldAnalyzer"]


class HopfieldAnalyzer:
    """
    Analyzer for Hopfield associative memory networks.

    Provides diagnostic and analysis tools for Hopfield networks including:
    - Pattern storage capacity estimation
    - Energy landscape analysis
    - Overlap metrics for pattern retrieval
    - Recall quality diagnostics

    The Hopfield network stores patterns as attractors in an energy landscape.
    Energy function:
        E = -0.5 * s^T W s

    Reference:
        Hopfield, J. J. (1982). Neural networks and physical systems with
        emergent collective computational abilities. PNAS, 79(8), 2554-2558.
    """

    def __init__(self, model, stored_patterns: list | None = None):
        """
        Initialize Hopfield analyzer.

        Args:
            model: The Hopfield network model to analyze
            stored_patterns: List of patterns stored in the network (optional)
        """
        self.model = model
        self.stored_patterns = stored_patterns if stored_patterns is not None else []
        self._pattern_energies = []

        # Compute energies if patterns provided
        if len(self.stored_patterns) > 0:
            self.compute_pattern_energies()

    def set_patterns(self, patterns: list):
        """
        Set the stored patterns and compute their energies.

        Args:
            patterns: List of patterns stored in the network
        """
        self.stored_patterns = [jnp.asarray(p, dtype=jnp.float32) for p in patterns]
        self.compute_pattern_energies()

    def compute_pattern_energies(self):
        """Compute energy for each stored pattern."""
        self._pattern_energies = []

        # Get weight matrix
        weight_attr = getattr(self.model, "weight_attr", "W")
        if callable(weight_attr):
            weight_attr = weight_attr()
        weight_param = getattr(self.model, weight_attr)
        W = weight_param.value

        for pattern in self.stored_patterns:
            # E = -0.5 * s^T W s
            energy = -0.5 * jnp.dot(pattern, jnp.dot(W, pattern))
            self._pattern_energies.append(float(energy))

    @property
    def pattern_energies(self) -> list[float]:
        """Get energies of stored patterns."""
        return self._pattern_energies

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

    def compute_energy(self, pattern: jnp.ndarray) -> float:
        """
        Compute energy of a given pattern.

        Args:
            pattern: Pattern to compute energy for

        Returns:
            Energy value E = -0.5 * s^T W s
        """
        # Get weight matrix
        weight_attr = getattr(self.model, "weight_attr", "W")
        if callable(weight_attr):
            weight_attr = weight_attr()
        weight_param = getattr(self.model, weight_attr)
        W = weight_param.value

        p = jnp.asarray(pattern, dtype=jnp.float32)
        return float(-0.5 * jnp.dot(p, jnp.dot(W, p)))

    def analyze_recall(self, input_pattern: jnp.ndarray, output_pattern: jnp.ndarray) -> dict:
        """
        Analyze pattern recall quality.

        Args:
            input_pattern: Input (noisy) pattern
            output_pattern: Recalled pattern

        Returns:
            Dictionary with diagnostic metrics:
                - best_match_idx: Index of best matching stored pattern
                - best_match_overlap: Overlap with best matching pattern
                - input_output_overlap: Overlap between input and output
                - output_energy: Energy of the recalled pattern
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
        diagnostics["output_energy"] = self.compute_energy(output_pattern)

        return diagnostics

    def estimate_capacity(self) -> int:
        """
        Estimate theoretical storage capacity of the network.

        Uses the rule of thumb: capacity ≈ N / (4 * ln(N))
        where N is the number of neurons.

        Returns:
            Estimated number of patterns that can be reliably stored
        """
        if hasattr(self.model, "storage_capacity"):
            return self.model.storage_capacity

        # Default estimate: N / (4 * ln(N))
        n = self.model.num_neurons if hasattr(self.model, "num_neurons") else 100
        return max(1, int(n / (4 * np.log(n))))

    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about stored patterns.

        Returns:
            Dictionary with pattern statistics:
                - num_patterns: Number of stored patterns
                - capacity_estimate: Theoretical capacity estimate
                - capacity_usage: Fraction of capacity used
                - mean_pattern_energy: Mean energy of stored patterns
                - std_pattern_energy: Standard deviation of energies
                - min_pattern_energy: Minimum energy
                - max_pattern_energy: Maximum energy
        """
        stats = {
            "num_patterns": len(self.stored_patterns),
            "capacity_estimate": self.estimate_capacity(),
            "capacity_usage": len(self.stored_patterns) / max(1, self.estimate_capacity()),
        }

        if len(self._pattern_energies) > 0:
            stats["mean_pattern_energy"] = float(np.mean(self._pattern_energies))
            stats["std_pattern_energy"] = float(np.std(self._pattern_energies))
            stats["min_pattern_energy"] = float(np.min(self._pattern_energies))
            stats["max_pattern_energy"] = float(np.max(self._pattern_energies))

        return stats

    def compute_weight_symmetry_error(self) -> float:
        """
        Compute the symmetry error of the weight matrix.

        Hopfield networks require symmetric weights (W_ij = W_ji).
        This metric quantifies how much the weight matrix deviates from symmetry.

        Returns:
            Symmetry error as ||W - W^T||_F / ||W||_F
        """
        weight_attr = getattr(self.model, "weight_attr", "W")
        if callable(weight_attr):
            weight_attr = weight_attr()
        weight_param = getattr(self.model, weight_attr)
        W = weight_param.value

        # Frobenius norm of asymmetry ("fro" is the standard numpy/jax parameter)
        asymmetry = W - W.T
        symmetry_error = float(jnp.linalg.norm(asymmetry, "fro") / jnp.linalg.norm(W, "fro"))
        return symmetry_error
