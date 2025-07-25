"""
Comprehensive tests for the unified CANN implementation.

This module tests the new UnifiedCANN classes that eliminate code duplication
while maintaining the same functionality as the original implementations.
"""

import pytest
import numpy as np
import brainstate
import brainunit as u

from canns.models.basic.cann import (
    UnifiedCANN1D, UnifiedCANN2D, UnifiedCANN1D_SFA, UnifiedCANN2D_SFA,
    UnifiedBaseCANN, SFAMixin
)


class TestUnifiedBaseCANN:
    """Test the base unified CANN class."""
    
    def test_shape_validation(self):
        """Test that shape validation works correctly."""
        # Valid shapes
        assert UnifiedBaseCANN(10).shape == (10,)
        assert UnifiedBaseCANN((10,)).shape == (10,)
        assert UnifiedBaseCANN((10, 10)).shape == (10, 10)
        
        # Invalid shapes
        with pytest.raises(ValueError, match="Network size must be positive"):
            UnifiedBaseCANN(-5)
        
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            UnifiedBaseCANN((10, -5))
        
        with pytest.raises(ValueError, match="Only 1D and 2D networks supported"):
            UnifiedBaseCANN((10, 10, 10))
        
        with pytest.raises(TypeError, match="Shape must be int or tuple"):
            UnifiedBaseCANN("invalid")

    def test_parameter_validation(self):
        """Test that parameter validation works correctly."""
        # Valid parameters should work
        cann = UnifiedBaseCANN(10, tau=1.0, k=8.1, a=0.5, A=10.0, J0=4.0)
        assert cann.tau == 1.0
        assert cann.k == 8.1
        
        # Invalid parameters should raise errors
        with pytest.raises(ValueError, match="tau must be positive"):
            UnifiedBaseCANN(10, tau=-1.0)
        
        with pytest.raises(ValueError, match="k must be positive"):
            UnifiedBaseCANN(10, k=-1.0)
        
        with pytest.raises(ValueError, match="a must be positive"):
            UnifiedBaseCANN(10, a=-1.0)
        
        with pytest.raises(ValueError, match="A must be positive"):
            UnifiedBaseCANN(10, A=-1.0)
        
        with pytest.raises(ValueError, match="J0 must be positive"):
            UnifiedBaseCANN(10, J0=-1.0)
        
        with pytest.raises(ValueError, match="z_max.*must be greater than z_min"):
            UnifiedBaseCANN(10, z_min=1.0, z_max=-1.0)

    def test_1d_feature_space_setup(self):
        """Test 1D feature space initialization."""
        cann = UnifiedBaseCANN(50, z_min=-np.pi, z_max=np.pi)
        
        assert cann.ndim == 1
        assert cann.shape == (50,)
        assert len(cann.x) == 50
        assert cann.z_range == 2 * np.pi
        assert cann.rho == 50 / (2 * np.pi)
        assert cann.dx == 2 * np.pi / 50

    def test_2d_feature_space_setup(self):
        """Test 2D feature space initialization."""
        cann = UnifiedBaseCANN((20, 20), z_min=-np.pi, z_max=np.pi)
        
        assert cann.ndim == 2
        assert cann.shape == (20, 20)
        assert cann.length == 20
        assert len(cann.x) == 20
        assert cann.z_range == 2 * np.pi

    def test_position_validation_1d(self):
        """Test position validation for 1D networks."""
        cann = UnifiedBaseCANN(10)
        
        # Valid positions
        stimulus = cann.get_stimulus_by_pos(0.0)
        assert stimulus.shape == (10,)
        
        stimulus = cann.get_stimulus_by_pos([0.0])
        assert stimulus.shape == (10,)
        
        # Invalid positions
        with pytest.raises(ValueError, match="1D network expects scalar position"):
            cann.get_stimulus_by_pos([0.0, 1.0])

    def test_position_validation_2d(self):
        """Test position validation for 2D networks."""
        cann = UnifiedBaseCANN((10, 10))
        
        # Valid positions
        stimulus = cann.get_stimulus_by_pos([0.0, 0.0])
        assert stimulus.shape == (100,)  # Flattened 10x10
        
        stimulus = cann.get_stimulus_by_pos(np.array([0.0, 1.0]))
        assert stimulus.shape == (100,)
        
        # Invalid positions
        with pytest.raises(ValueError, match="2D network expects 2D position"):
            cann.get_stimulus_by_pos(0.0)
        
        with pytest.raises(ValueError, match="2D network expects 2D position"):
            cann.get_stimulus_by_pos([0.0, 1.0, 2.0])

    def test_sparse_vs_dense_connectivity(self):
        """Test that sparse and dense connectivity produce similar results."""
        # Create networks with same parameters but different connectivity
        cann_dense = UnifiedBaseCANN(30, use_sparse=False)
        cann_sparse = UnifiedBaseCANN(30, use_sparse=True, sparsity_threshold=1e-4)
        
        # Compare connectivity matrices
        dense_nnz = np.count_nonzero(np.abs(cann_dense.conn_mat) > 1e-6)
        sparse_nnz = np.count_nonzero(np.abs(cann_sparse.conn_mat) > 1e-6)
        
        # Sparse should have fewer connections
        assert sparse_nnz <= dense_nnz
        
        # But stimulus generation should be very similar
        stimulus_dense = cann_dense.get_stimulus_by_pos(0.0)
        stimulus_sparse = cann_sparse.get_stimulus_by_pos(0.0)
        
        # Should be identical (different thresholding shouldn't affect stimulus)
        np.testing.assert_array_equal(stimulus_dense, stimulus_sparse)


class TestUnifiedCANN1D:
    """Test the unified 1D CANN implementation."""
    
    def setup_method(self):
        """Set up test environment."""
        brainstate.environ.set(dt=0.1)

    def test_initialization(self):
        """Test 1D CANN initialization."""
        cann = UnifiedCANN1D(num=50, tau=1.0, k=8.1, a=0.5)
        
        assert cann.shape == (50,)
        assert cann.ndim == 1
        assert hasattr(cann, 'u')
        assert hasattr(cann, 'r')
        assert hasattr(cann, 'inp')
        
        # Check state shapes
        assert cann.u.value.shape == (50,)
        assert cann.r.value.shape == (50,)
        assert cann.inp.value.shape == (50,)

    def test_update_dynamics(self):
        """Test that the update method works correctly."""
        cann = UnifiedCANN1D(num=20, tau=1.0, k=8.1, a=0.5)
        
        # Set some input
        cann.inp.value = cann.get_stimulus_by_pos(0.0)
        
        # Initial states should be zero
        assert np.allclose(cann.u.value, 0.0)
        assert np.allclose(cann.r.value, 0.0)
        
        # After update, states should change
        cann.update()
        
        # u should have changed (received input)
        assert not np.allclose(cann.u.value, 0.0)
        
        # r should still be zero after first update (computed from initial u=0)  
        assert np.allclose(cann.r.value, 0.0)
        
        # After second update, r should reflect the updated u
        # Store the u value after first update to predict r after second update
        u_after_first = np.array(cann.u.value.copy())
        
        cann.update()
        
        # r should now reflect u_after_first (from beginning of second update)
        expected_r_raw = np.square(u_after_first)
        r_sum = np.sum(expected_r_raw)
        expected_r = expected_r_raw / (1.0 + cann.k * r_sum)
        
        np.testing.assert_allclose(cann.r.value, expected_r, rtol=1e-5)

    def test_multiple_updates(self):
        """Test that multiple updates maintain network stability."""
        cann = UnifiedCANN1D(num=30, tau=1.0, k=8.1, a=0.5)
        
        # Set input stimulus
        cann.inp.value = cann.get_stimulus_by_pos(0.5)
        
        states = []
        for _ in range(10):
            cann.update()
            states.append(np.array(cann.u.value.copy()))
        
        # Network should not explode
        for state in states:
            assert np.all(np.isfinite(state))
            assert np.max(np.abs(state)) < 100  # Reasonable bounds

    def test_stimulus_response(self):
        """Test that network responds correctly to stimulus."""
        cann = UnifiedCANN1D(num=40, tau=1.0, k=8.1, a=0.5)
        
        # Test stimulus at different positions
        positions = [-1.0, 0.0, 1.0]
        
        for pos in positions:
            stimulus = cann.get_stimulus_by_pos(pos)
            
            # Stimulus should be positive
            assert np.all(stimulus >= 0)
            
            # Should have a peak structure
            assert np.max(stimulus) > 0
            
            # Should be Gaussian-like (smooth)
            peak_idx = np.argmax(stimulus)
            if peak_idx > 0 and peak_idx < len(stimulus) - 1:
                # Check that it decreases away from peak
                assert stimulus[peak_idx] >= stimulus[peak_idx - 1]
                assert stimulus[peak_idx] >= stimulus[peak_idx + 1]


class TestUnifiedCANN2D:
    """Test the unified 2D CANN implementation."""
    
    def setup_method(self):
        """Set up test environment."""
        brainstate.environ.set(dt=0.1)

    def test_initialization(self):
        """Test 2D CANN initialization."""
        cann = UnifiedCANN2D(length=10, tau=1.0, k=8.1, a=0.5)
        
        assert cann.shape == (10, 10)
        assert cann.ndim == 2
        assert cann.length == 10
        
        # Check state shapes (flattened for computation)
        assert cann.u.value.shape == (100,)  # 10*10
        assert cann.r.value.shape == (100,)
        assert cann.inp.value.shape == (100,)

    def test_2d_update_dynamics(self):
        """Test 2D network update dynamics."""
        cann = UnifiedCANN2D(length=8, tau=1.0, k=8.1, a=0.5)
        
        # Set 2D input stimulus
        cann.inp.value = cann.get_stimulus_by_pos([0.0, 0.0])
        
        # Initial states
        initial_u = np.array(cann.u.value.copy())
        
        # Update - after first update, u changes but r reflects old u (=0)
        cann.update()
        
        # States should have changed
        assert not np.allclose(cann.u.value, initial_u)
        
        # r should still be zero after first update
        assert np.allclose(cann.r.value, 0.0)
        
        # Store u after first update for second update test
        u_after_first = np.array(cann.u.value.copy())
        
        # Second update - r should now reflect u from first update
        cann.update()
        
        # Check firing rate computation based on u_after_first
        expected_r_raw = np.square(u_after_first)
        r_sum = np.sum(expected_r_raw)
        expected_r = expected_r_raw / (1.0 + cann.k * r_sum)
        
        np.testing.assert_allclose(cann.r.value, expected_r, rtol=1e-5)

    def test_2d_stimulus_generation(self):
        """Test 2D stimulus generation."""
        cann = UnifiedCANN2D(length=6, tau=1.0, k=8.1, a=0.5)
        
        # Test stimulus at center
        stimulus = cann.get_stimulus_by_pos([0.0, 0.0])
        
        assert stimulus.shape == (36,)  # 6*6 flattened
        assert np.all(stimulus >= 0)
        assert np.max(stimulus) > 0
        
        # Test stimulus at different positions
        positions = [[-1.0, -1.0], [0.0, 1.0], [1.0, 0.0]]
        
        for pos in positions:
            stimulus = cann.get_stimulus_by_pos(pos)
            assert stimulus.shape == (36,)
            assert np.all(np.isfinite(stimulus))


class TestSFAMixin:
    """Test the spike-frequency adaptation mixin."""
    
    def test_sfa_parameter_validation(self):
        """Test SFA parameter validation."""
        # Valid parameters
        cann = UnifiedCANN1D_SFA(num=10, tau_v=50.0, m=0.3)
        assert cann.tau_v == 50.0
        assert cann.m == 0.3
        
        # Invalid parameters
        with pytest.raises(ValueError, match="tau_v must be positive"):
            UnifiedCANN1D_SFA(num=10, tau_v=-1.0)
        
        with pytest.raises(ValueError, match="m should typically be in"):
            UnifiedCANN1D_SFA(num=10, m=-0.1)

    def test_sfa_state_initialization(self):
        """Test that SFA states are properly initialized."""
        cann = UnifiedCANN1D_SFA(num=20, tau_v=30.0, m=0.5)
        
        # Should have adaptation variable
        assert hasattr(cann, 'v')
        assert cann.v.value.shape == (20,)
        assert np.allclose(cann.v.value, 0.0)  # Initially zero

    def test_sfa_update_dynamics(self):
        """Test SFA adaptation dynamics."""
        brainstate.environ.set(dt=0.1)
        cann = UnifiedCANN1D_SFA(num=15, tau_v=10.0, m=0.8)
        
        # Set some membrane potential
        cann.u.value = np.ones(15) * 2.0
        
        # Initial adaptation should be zero
        initial_v = np.array(cann.v.value.copy())
        assert np.allclose(initial_v, 0.0)
        
        # Update (this calls both CANN update and adaptation update)
        cann.update()
        
        # Adaptation variable should have changed toward membrane potential
        assert not np.allclose(cann.v.value, initial_v)
        
        # Should be moving toward m * u
        expected_change_direction = cann.m * cann.u.value
        assert np.all((cann.v.value - initial_v) * expected_change_direction >= 0)


class TestUnifiedCANN2D_SFA:
    """Test the unified 2D CANN with SFA."""
    
    def setup_method(self):
        """Set up test environment."""
        brainstate.environ.set(dt=0.1)

    def test_2d_sfa_initialization(self):
        """Test 2D CANN with SFA initialization."""
        cann = UnifiedCANN2D_SFA(length=5, tau_v=20.0, m=0.4)
        
        assert cann.shape == (5, 5)
        assert hasattr(cann, 'v')
        assert cann.v.value.shape == (25,)  # 5*5 flattened
        
        # SFA parameters
        assert cann.tau_v == 20.0
        assert cann.m == 0.4

    def test_2d_sfa_dynamics(self):
        """Test 2D CANN with SFA dynamics."""
        cann = UnifiedCANN2D_SFA(length=4, tau_v=15.0, m=0.6)
        
        # Set input and run dynamics
        cann.inp.value = cann.get_stimulus_by_pos([0.0, 0.0])
        
        initial_v = np.array(cann.v.value.copy())
        
        # Multiple updates
        for _ in range(3):
            cann.update()
        
        # Adaptation should have evolved
        assert not np.allclose(cann.v.value, initial_v)
        
        # Network should remain stable
        assert np.all(np.isfinite(cann.u.value))
        assert np.all(np.isfinite(cann.r.value))
        assert np.all(np.isfinite(cann.v.value))


class TestBackwardCompatibility:
    """Test that the unified implementation maintains backward compatibility."""
    
    def setup_method(self):
        """Set up test environment."""
        brainstate.environ.set(dt=0.1)

    def test_api_compatibility(self):
        """Test that the API is compatible with original implementation."""
        # Should be able to create with same parameters as original
        cann1d = UnifiedCANN1D(num=20, tau=1.0, k=8.1, a=0.5, A=10.0, J0=4.0)
        cann2d = UnifiedCANN2D(length=10, tau=1.0, k=8.1, a=0.5, A=10.0, J0=4.0)
        
        # Should have same interface
        assert hasattr(cann1d, 'update')
        assert hasattr(cann1d, 'get_stimulus_by_pos')
        assert hasattr(cann2d, 'update')
        assert hasattr(cann2d, 'get_stimulus_by_pos')
        
        # Should work with same usage pattern
        cann1d.inp.value = cann1d.get_stimulus_by_pos(0.0)
        cann2d.inp.value = cann2d.get_stimulus_by_pos([0.0, 0.0])
        
        cann1d.update()
        cann2d.update()
        
        # Should produce reasonable results
        assert np.all(np.isfinite(cann1d.u.value))
        assert np.all(np.isfinite(cann2d.u.value))


if __name__ == "__main__":
    pytest.main([__file__])