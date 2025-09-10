"""Tests for the enhanced progress reporting architecture."""

import pytest
import numpy as np
from canns.models import AmariHopfieldNetwork
from canns.trainer import HebbianTrainer, create_progress_reporter


class TestProgressReporters:
    """Test progress reporter components."""
    
    def test_silent_reporter(self):
        """Test silent progress reporter."""
        reporter = create_progress_reporter("silent")
        task = reporter.start_task("test", total=10)
        reporter.update(task, 5)
        reporter.finish_task(task)
        # Silent reporter should not raise any errors
    
    def test_auto_reporter(self):
        """Test auto progress reporter."""
        reporter = create_progress_reporter("auto")
        with reporter.progress("test", total=5) as pbar:
            for i in range(5):
                reporter.update(pbar, 1)
        # Should work regardless of tqdm availability


class TestBackwardCompatibility:
    """Test backward compatibility with old trainer interface."""
    
    def test_old_trainer_interface(self):
        """Test that old code still works unchanged."""
        patterns = [
            np.array([1, -1, 1, -1]),
            np.array([-1, 1, -1, 1])
        ]
        
        # Old-style usage should work exactly as before
        model = AmariHopfieldNetwork(num_neurons=4, activation="sign")
        model.init_state()
        trainer = HebbianTrainer(model)
        trainer.train(patterns)
        
        result = trainer.predict(patterns[0])
        assert result.shape == patterns[0].shape
    
    def test_old_with_manual_loop(self):
        """Test that manual prediction loops still work."""
        patterns = [
            np.array([1, -1, 1, -1]),
            np.array([-1, 1, -1, 1])
        ]
        
        model = DiscreteHopfieldNetwork(num_neurons=4)
        model.init_state()
        trainer = HebbianTrainer(model, progress_mode="silent")
        trainer.train(patterns)
        
        # Manual loop like in old examples
        results = []
        for pattern in patterns:
            results.append(trainer.predict(pattern))
        
        assert len(results) == len(patterns)
        for result, pattern in zip(results, patterns):
            assert result.shape == pattern.shape


class TestNewFeatures:
    """Test new progress reporting features."""
    
    def setup_method(self):
        """Set up test data for each test."""
        self.patterns = [
            np.array([1, -1, 1, -1, 1, -1]),
            np.array([-1, 1, -1, 1, -1, 1])
        ]
        self.model = DiscreteHopfieldNetwork(num_neurons=6)
        self.model.init_state()
    
    def test_silent_trainer(self):
        """Test trainer with silent progress reporting."""
        trainer = HebbianTrainer(self.model, progress_mode="silent")
        trainer.train(self.patterns)
        results = trainer.predict_batch(self.patterns)
        
        assert len(results) == len(self.patterns)
        for result, pattern in zip(results, self.patterns):
            assert result.shape == pattern.shape
    
    def test_compiled_trainer(self):
        """Test trainer with compiled prediction."""
        trainer = HebbianTrainer(self.model, compiled_prediction=True)
        trainer.train(self.patterns)
        results = trainer.predict_batch(self.patterns)
        
        assert len(results) == len(self.patterns)
        for result, pattern in zip(results, self.patterns):
            assert result.shape == pattern.shape
    
    def test_batch_prediction(self):
        """Test batch prediction method."""
        trainer = HebbianTrainer(self.model, progress_mode="silent")
        trainer.train(self.patterns)
        
        # Test batch prediction
        results = trainer.predict_batch(
            self.patterns,
            show_sample_progress=False,
            show_iteration_progress=False
        )
        
        assert len(results) == len(self.patterns)
        
        # Results should be similar to individual predictions
        for i, pattern in enumerate(self.patterns):
            individual_result = trainer.predict(pattern)
            assert np.allclose(results[i], individual_result, rtol=1e-5)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        trainer = HebbianTrainer(self.model, progress_mode="silent")
        trainer.train(self.patterns)
        
        # Test progress callback
        callback_calls = []
        
        def test_callback(iteration, energy, converged, energy_change):
            callback_calls.append((iteration, energy, converged, energy_change))
        
        # Use model directly with callback
        result = self.model.predict(
            self.patterns[0],
            num_iter=5,
            compiled=False,
            progress_callback=test_callback
        )
        
        assert len(callback_calls) > 0
        assert result.shape == self.patterns[0].shape
        
        # Check callback data structure
        for call in callback_calls:
            iteration, energy, converged, energy_change = call
            assert isinstance(iteration, int)
            assert isinstance(energy, (int, float))
            assert isinstance(converged, bool)
            assert isinstance(energy_change, (int, float))
    
    def test_trainer_configuration(self):
        """Test dynamic trainer configuration."""
        trainer = HebbianTrainer(self.model, progress_mode="silent")
        trainer.train(self.patterns)
        
        # Test reconfiguration
        trainer.configure_progress(
            progress_mode="silent",
            compiled_prediction=False,
            show_iteration_progress=False
        )
        
        result = trainer.predict(self.patterns[0])
        assert result.shape == self.patterns[0].shape


class TestModelModes:
    """Test different model prediction modes."""
    
    def setup_method(self):
        """Set up test model."""
        self.model = DiscreteHopfieldNetwork(num_neurons=4)
        self.model.init_state()
        self.pattern = np.array([1, -1, 1, -1])
        self.model.apply_hebbian_learning([self.pattern])
    
    def test_compiled_vs_uncompiled(self):
        """Test compiled vs uncompiled prediction modes."""
        # Test compiled mode
        result1 = self.model.predict(self.pattern, compiled=True)
        
        # Test uncompiled mode
        result2 = self.model.predict(self.pattern, compiled=False)
        
        # Results should be very similar
        assert result1.shape == self.pattern.shape
        assert result2.shape == self.pattern.shape
        assert np.allclose(result1, result2, rtol=1e-5)
    
    def test_convergence_threshold(self):
        """Test convergence threshold parameter."""
        result = self.model.predict(
            self.pattern,
            compiled=False,
            convergence_threshold=1e-8
        )
        assert result.shape == self.pattern.shape
    
    def test_max_iterations(self):
        """Test maximum iterations parameter."""
        result = self.model.predict(
            self.pattern,
            num_iter=10,
            compiled=False
        )
        assert result.shape == self.pattern.shape


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_full_workflow(self):
        """Test complete workflow with different configurations."""
        # Create larger test data
        patterns = [
            np.random.choice([-1, 1], size=16),
            np.random.choice([-1, 1], size=16),
            np.random.choice([-1, 1], size=16)
        ]
        
        model = DiscreteHopfieldNetwork(num_neurons=16)
        model.init_state()
        
        # Test with different trainer configurations
        configs = [
            {"progress_mode": "silent", "compiled_prediction": True},
            {"progress_mode": "silent", "compiled_prediction": False},
            {"progress_mode": "auto", "compiled_prediction": True}
        ]
        
        for config in configs:
            trainer = HebbianTrainer(model, **config)
            trainer.train(patterns)
            
            # Test individual prediction
            result = trainer.predict(patterns[0])
            assert result.shape == patterns[0].shape
            
            # Test batch prediction
            results = trainer.predict_batch(
                patterns,
                show_sample_progress=False,
                show_iteration_progress=False
            )
            assert len(results) == len(patterns)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        model = DiscreteHopfieldNetwork(num_neurons=4)
        model.init_state()
        
        # Test invalid progress mode
        with pytest.raises(ValueError):
            create_progress_reporter("invalid_mode")
        
        # Test empty pattern list
        trainer = HebbianTrainer(model, progress_mode="silent")
        results = trainer.predict_batch([])
        assert len(results) == 0


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running basic functionality tests...")
    
    test_bc = TestBackwardCompatibility()
    test_bc.test_old_trainer_interface()
    print("✓ Backward compatibility test passed")
    
    test_nf = TestNewFeatures()
    test_nf.setup_method()
    test_nf.test_silent_trainer()
    print("✓ Silent trainer test passed")
    
    test_mm = TestModelModes()
    test_mm.setup_method()
    test_mm.test_compiled_vs_uncompiled()
    print("✓ Model modes test passed")
    
    print("All basic tests passed!")