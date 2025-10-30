"""Tests for Oja's normalized Hebbian learning trainer."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from canns.models.brain_inspired import LinearHebbLayer
from canns.trainer import OjaTrainer


def test_oja_trainer_initialization():
    """Test OjaTrainer initialization."""
    model = LinearHebbLayer(input_size=10, output_size=5)
    model.init_state()
    
    trainer = OjaTrainer(model, learning_rate=0.01, normalize_weights=True)
    
    assert trainer.model == model
    assert trainer.learning_rate == 0.01
    assert trainer.normalize_weights is True
    assert trainer.weight_attr == "W"


def test_oja_trainer_basic_training():
    """Test basic Oja training on synthetic data."""
    # Create model
    model = LinearHebbLayer(input_size=4, output_size=2)
    model.init_state()
    
    # Create trainer
    trainer = OjaTrainer(model, learning_rate=0.1, normalize_weights=True)
    
    # Simple training data (4D vectors)
    train_data = [
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0, 0.0]),
        jnp.array([1.0, 1.0, 0.0, 0.0]),
    ]
    
    # Train
    trainer.train(train_data)
    
    # Check that weights have been updated (not all zeros)
    W = model.W.value
    assert not jnp.allclose(W, 0.0)
    
    # Check that weights are normalized (unit norm rows)
    norms = jnp.linalg.norm(W, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)


def test_oja_trainer_without_normalization():
    """Test Oja training without weight normalization."""
    model = LinearHebbLayer(input_size=3, output_size=2)
    model.init_state()
    
    trainer = OjaTrainer(model, learning_rate=0.05, normalize_weights=False)
    
    train_data = [
        jnp.array([1.0, 0.5, 0.2]),
        jnp.array([0.5, 1.0, 0.3]),
    ]
    
    trainer.train(train_data)
    
    # Weights should be updated but not necessarily normalized
    W = model.W.value
    assert not jnp.allclose(W, 0.0)


def test_oja_trainer_predict():
    """Test Oja trainer prediction."""
    model = LinearHebbLayer(input_size=3, output_size=2)
    model.init_state()
    
    # Set some non-zero weights
    model.W.value = jnp.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
    
    trainer = OjaTrainer(model)
    
    # Predict
    input_pattern = jnp.array([1.0, 1.0, 1.0])
    output = trainer.predict(input_pattern)
    
    # Check output shape
    assert output.shape == (2,)
    
    # Check output values (should be W @ x)
    expected = jnp.array([1.5, 1.5])
    assert jnp.allclose(output, expected)


def test_oja_trainer_convergence():
    """Test that Oja training extracts principal components."""
    # Create model
    model = LinearHebbLayer(input_size=3, output_size=1)
    model.init_state()
    
    trainer = OjaTrainer(model, learning_rate=0.01, normalize_weights=True)
    
    # Create data with clear principal component along [1, 1, 0]
    train_data = [
        jnp.array([1.0, 1.0, 0.1]),
        jnp.array([0.9, 1.1, 0.0]),
        jnp.array([1.1, 0.9, -0.1]),
        jnp.array([1.0, 1.0, 0.05]),
    ]
    
    # Train multiple epochs
    for _ in range(50):
        trainer.train(train_data)
    
    # Check that first weight vector aligns with PC direction
    W = model.W.value
    w1 = W[0]
    
    # First two components should be similar and larger than third
    assert abs(w1[0] - w1[1]) < 0.3  # Similar components
    assert abs(w1[2]) < abs(w1[0])  # Third component smaller


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
