"""Tests for Hopfield Energy trainer."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from canns.models.brain_inspired import AmariHopfieldNetwork
from canns.trainer import HopfieldEnergyTrainer


def test_hopfield_energy_trainer_initialization():
    """Test HopfieldEnergyTrainer initialization."""
    model = AmariHopfieldNetwork(num_neurons=10)
    model.init_state()
    
    trainer = HopfieldEnergyTrainer(model)
    
    assert trainer.model == model
    assert len(trainer.stored_patterns) == 0
    assert len(trainer.pattern_energies) == 0


def test_hopfield_energy_trainer_stores_patterns():
    """Test that trainer stores patterns during training."""
    model = AmariHopfieldNetwork(num_neurons=5)
    model.init_state()
    
    trainer = HopfieldEnergyTrainer(model)
    
    # Train with patterns
    patterns = [
        jnp.array([1.0, -1.0, 1.0, -1.0, 1.0]),
        jnp.array([-1.0, 1.0, -1.0, 1.0, -1.0]),
    ]
    
    trainer.train(patterns)
    
    # Check stored patterns
    assert len(trainer.stored_patterns) == 2
    assert jnp.allclose(trainer.stored_patterns[0], patterns[0])
    assert jnp.allclose(trainer.stored_patterns[1], patterns[1])


def test_hopfield_energy_trainer_computes_pattern_energies():
    """Test that trainer computes energies for stored patterns."""
    model = AmariHopfieldNetwork(num_neurons=4)
    model.init_state()
    
    trainer = HopfieldEnergyTrainer(model)
    
    patterns = [
        jnp.array([1.0, 1.0, -1.0, -1.0]),
        jnp.array([1.0, -1.0, 1.0, -1.0]),
    ]
    
    trainer.train(patterns)
    
    # Check that energies were computed
    assert len(trainer.pattern_energies) == 2
    assert all(isinstance(e, float) for e in trainer.pattern_energies)


def test_hopfield_overlap_computation():
    """Test overlap computation between patterns."""
    model = AmariHopfieldNetwork(num_neurons=4)
    model.init_state()
    
    trainer = HopfieldEnergyTrainer(model)
    
    # Identical patterns
    p1 = jnp.array([1.0, -1.0, 1.0, -1.0])
    overlap = trainer.compute_overlap(p1, p1)
    assert jnp.isclose(overlap, 1.0)
    
    # Opposite patterns
    p2 = jnp.array([-1.0, 1.0, -1.0, 1.0])
    overlap = trainer.compute_overlap(p1, p2)
    assert jnp.isclose(overlap, -1.0)
    
    # Orthogonal patterns
    p3 = jnp.array([1.0, 1.0, -1.0, -1.0])
    overlap = trainer.compute_overlap(p1, p3)
    assert jnp.isclose(overlap, 0.0)


def test_hopfield_pattern_recall():
    """Test pattern recall with diagnostics."""
    model = AmariHopfieldNetwork(num_neurons=5, asyn=False, activation="sign")
    model.init_state()
    
    trainer = HopfieldEnergyTrainer(model)
    
    # Store a pattern
    pattern = jnp.array([1.0, 1.0, -1.0, -1.0, 1.0])
    trainer.train([pattern])
    
    # Recall from noisy version
    noisy = jnp.array([1.0, -1.0, -1.0, -1.0, 1.0])  # One bit flipped
    recalled, diagnostics = trainer.recall_pattern(noisy, num_iter=10)
    
    # Check diagnostics structure
    assert "best_match_idx" in diagnostics
    assert "best_match_overlap" in diagnostics
    assert "input_output_overlap" in diagnostics
    assert "output_energy" in diagnostics


def test_hopfield_capacity_estimation():
    """Test storage capacity estimation."""
    model = AmariHopfieldNetwork(num_neurons=100)
    model.init_state()
    
    trainer = HopfieldEnergyTrainer(model)
    
    capacity = trainer.estimate_capacity()
    
    # Should be approximately N / (4 * ln(N))
    expected = int(100 / (4 * jnp.log(100)))
    assert capacity > 0
    assert abs(capacity - expected) <= 1


def test_hopfield_pattern_statistics():
    """Test pattern statistics computation."""
    model = AmariHopfieldNetwork(num_neurons=8)
    model.init_state()
    
    trainer = HopfieldEnergyTrainer(model)
    
    # Train with patterns
    patterns = [
        jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        jnp.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
    ]
    
    trainer.train(patterns)
    
    stats = trainer.get_pattern_statistics()
    
    # Check statistics
    assert stats["num_patterns"] == 2
    assert "capacity_estimate" in stats
    assert "capacity_usage" in stats
    assert "mean_pattern_energy" in stats
    assert "min_pattern_energy" in stats
    assert "max_pattern_energy" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
