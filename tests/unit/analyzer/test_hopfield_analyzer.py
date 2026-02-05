"""Tests for Hopfield Analyzer."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from canns.analyzer.model_specific.hopfield import HopfieldAnalyzer
from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer


def test_hopfield_analyzer_initialization():
    """Test HopfieldAnalyzer initialization."""
    model = AmariHopfieldNetwork(num_neurons=10)

    analyzer = HopfieldAnalyzer(model)

    assert analyzer.model == model
    assert len(analyzer.stored_patterns) == 0
    assert len(analyzer.pattern_energies) == 0


def test_hopfield_analyzer_with_patterns():
    """Test analyzer with provided patterns."""
    model = AmariHopfieldNetwork(num_neurons=5)

    patterns = [
        jnp.array([1.0, -1.0, 1.0, -1.0, 1.0]),
        jnp.array([-1.0, 1.0, -1.0, 1.0, -1.0]),
    ]

    analyzer = HopfieldAnalyzer(model, stored_patterns=patterns)

    # Check stored patterns
    assert len(analyzer.stored_patterns) == 2
    assert jnp.allclose(analyzer.stored_patterns[0], patterns[0])
    assert jnp.allclose(analyzer.stored_patterns[1], patterns[1])

    # Check energies were computed
    assert len(analyzer.pattern_energies) == 2


def test_hopfield_analyzer_set_patterns():
    """Test setting patterns after initialization."""
    model = AmariHopfieldNetwork(num_neurons=4)

    analyzer = HopfieldAnalyzer(model)

    patterns = [
        jnp.array([1.0, 1.0, -1.0, -1.0]),
        jnp.array([1.0, -1.0, 1.0, -1.0]),
    ]

    analyzer.set_patterns(patterns)

    # Check that energies were computed
    assert len(analyzer.pattern_energies) == 2
    assert all(isinstance(e, float) for e in analyzer.pattern_energies)


def test_hopfield_overlap_computation():
    """Test overlap computation between patterns."""
    model = AmariHopfieldNetwork(num_neurons=4)

    analyzer = HopfieldAnalyzer(model)

    # Identical patterns
    p1 = jnp.array([1.0, -1.0, 1.0, -1.0])
    overlap = analyzer.compute_overlap(p1, p1)
    assert jnp.isclose(overlap, 1.0)

    # Opposite patterns
    p2 = jnp.array([-1.0, 1.0, -1.0, 1.0])
    overlap = analyzer.compute_overlap(p1, p2)
    assert jnp.isclose(overlap, -1.0)

    # Orthogonal patterns
    p3 = jnp.array([1.0, 1.0, -1.0, -1.0])
    overlap = analyzer.compute_overlap(p1, p3)
    assert jnp.isclose(overlap, 0.0)


def test_hopfield_pattern_recall_analysis():
    """Test pattern recall analysis with HebbianTrainer and HopfieldAnalyzer."""
    model = AmariHopfieldNetwork(num_neurons=5, asyn=False, activation="sign")

    # Train with HebbianTrainer
    trainer = HebbianTrainer(model)
    pattern = jnp.array([1.0, 1.0, -1.0, -1.0, 1.0])
    trainer.train([pattern])

    # Analyze with HopfieldAnalyzer
    analyzer = HopfieldAnalyzer(model, stored_patterns=[pattern])

    # Recall from noisy version
    noisy = jnp.array([1.0, -1.0, -1.0, -1.0, 1.0])  # One bit flipped
    recalled = trainer.predict(noisy, num_iter=10)

    # Analyze recall quality
    diagnostics = analyzer.analyze_recall(noisy, recalled)

    # Check diagnostics structure
    assert "best_match_idx" in diagnostics
    assert "best_match_overlap" in diagnostics
    assert "input_output_overlap" in diagnostics
    assert "output_energy" in diagnostics


def test_hopfield_capacity_estimation():
    """Test storage capacity estimation."""
    model = AmariHopfieldNetwork(num_neurons=100)

    analyzer = HopfieldAnalyzer(model)

    capacity = analyzer.estimate_capacity()

    # Should be approximately N / (4 * ln(N))
    expected = int(100 / (4 * jnp.log(100)))
    assert capacity > 0
    assert abs(capacity - expected) <= 1


def test_hopfield_pattern_statistics():
    """Test pattern statistics computation."""
    model = AmariHopfieldNetwork(num_neurons=8)

    # Train with patterns
    patterns = [
        jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        jnp.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
    ]

    trainer = HebbianTrainer(model)
    trainer.train(patterns)

    analyzer = HopfieldAnalyzer(model, stored_patterns=patterns)

    stats = analyzer.get_statistics()

    # Check statistics
    assert stats["num_patterns"] == 2
    assert "capacity_estimate" in stats
    assert "capacity_usage" in stats
    assert "mean_pattern_energy" in stats
    assert "min_pattern_energy" in stats
    assert "max_pattern_energy" in stats


def test_hopfield_weight_symmetry():
    """Test weight symmetry error computation."""
    model = AmariHopfieldNetwork(num_neurons=5)

    # Train with HebbianTrainer (should produce symmetric weights)
    trainer = HebbianTrainer(model)
    pattern = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0])
    trainer.train([pattern])

    analyzer = HopfieldAnalyzer(model)

    # Hebbian learning should produce symmetric weights
    symmetry_error = analyzer.compute_weight_symmetry_error()
    assert symmetry_error < 1e-5  # Should be very close to zero


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
