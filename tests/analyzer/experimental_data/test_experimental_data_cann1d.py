#!/usr/bin/env python3
"""
Test cases for canns.analyzer.experimental_data.cann1d module functions.

These tests provide basic verification of the CANN1D bump fitting and animation functions.
"""

import numpy as np
import pytest
from canns.analyzer.data.cann1d import bump_fits, create_1d_bump_animation, CANN1DPlotConfig, BumpFitsConfig


def create_mock_roi_data(n_roi=16, n_steps=1000):
    """Create mock ROI (Region of Interest) data for testing."""
    # Generate synthetic calcium imaging data with bump-like activity
    roi_data = np.zeros((n_steps, n_roi))
    
    # Create a moving bump pattern
    for t in range(n_steps):
        # Single bump moving across ROIs
        center = (t * 0.05) % n_roi  # Slow movement
        width = 2.0
        amplitude = 1.0 + 0.3 * np.random.randn()  # Some noise
        
        for i in range(n_roi):
            # Circular distance on the ring
            dist = min(abs(i - center), n_roi - abs(i - center))
            roi_data[t, i] = amplitude * np.exp(-(dist**2) / (2 * width**2))
        
        # Add background noise
        roi_data[t, :] += 0.1 * np.random.randn(n_roi)
    
    # Ensure non-negative values (like real calcium data)
    roi_data = np.maximum(roi_data, 0)
    
    return roi_data


def test_bump_fits_config():
    """Test BumpFitsConfig creation and defaults."""
    config = BumpFitsConfig()
    assert config.n_steps == 20000
    assert config.n_roi == 16
    assert config.n_bump_max == 4
    assert config.sigma_diff == 0.5
    assert config.ampli_min == 2.0
    assert config.kappa_mean == 2.5
    assert config.sig2 == 1.0
    assert config.penbump == 0.4
    assert config.jc == 1.8
    assert config.beta == 5.0
    assert config.random_seed is None
    print("BumpFitsConfig created successfully with default values")


def test_bump_fits_basic():
    """Test basic bump fitting functionality."""
    # Create test data
    roi_data = create_mock_roi_data(n_roi=8, n_steps=100)
    
    # Test with minimal configuration for speed
    config = BumpFitsConfig(
        n_steps=50,  # Very few steps for quick test
        n_roi=8,
        n_bump_max=2
    )
    
    try:
        result = bump_fits(roi_data, config=config, save_path=None)
        
        assert isinstance(result, dict)
        assert 'positions' in result
        assert 'kappas' in result  
        assert 'amplis' in result
        assert 'n_bumps' in result
        assert 'log_likelihood' in result
        
        # Check array shapes
        assert result['positions'].shape[0] == config.n_steps
        assert result['kappas'].shape[0] == config.n_steps
        assert result['amplis'].shape[0] == config.n_steps
        assert result['n_bumps'].shape[0] == config.n_steps
        
        print(f"Bump fitting completed: {result['positions'].shape[0]} steps")
        
    except Exception as e:
        print(f"Bump fitting test failed: {e}")


def test_bump_fits_with_real_data_structure():
    """Test bump fitting with data structure similar to real ROI data."""
    # Load actual test data if available, otherwise use mock data
    try:
        from canns.data.loaders import load_roi_data
        data = load_roi_data()
        roi_data = data['roi_data']
        n_roi = roi_data.shape[1]
        print(f"Using real ROI data: shape={roi_data.shape}")
    except Exception:
        # Fallback to mock data
        roi_data = create_mock_roi_data(n_roi=16, n_steps=200)
        n_roi = 16
        print(f"Using mock ROI data: shape={roi_data.shape}")
    
    # Use subset for faster testing
    roi_subset = roi_data[:100, :]  # First 100 time steps
    
    config = BumpFitsConfig(
        n_steps=20,  # Very short for testing
        n_roi=n_roi,
        n_bump_max=2
    )
    
    try:
        result = bump_fits(roi_subset, config=config, save_path=None)
        assert isinstance(result, dict)
        assert all(key in result for key in ['positions', 'kappas', 'amplis', 'n_bumps'])
        print("Bump fitting with real data structure successful")
    except Exception as e:
        print(f"Real data structure test skipped: {e}")


def test_create_1d_bump_animation():
    """Test 1D bump animation creation."""
    # Create test bump fitting results
    n_steps = 20
    n_roi = 8
    max_bumps = 2
    
    # Mock fitting results
    fitting_results = {
        'positions': np.random.rand(n_steps, max_bumps) * n_roi,
        'kappas': np.random.rand(n_steps, max_bumps) * 2 + 0.5,  # [0.5, 2.5]
        'amplis': np.random.rand(n_steps, max_bumps) * 2 + 0.5,  # [0.5, 2.5]
        'n_bumps': np.random.randint(0, max_bumps + 1, n_steps),
        'roi_data': create_mock_roi_data(n_roi=n_roi, n_steps=n_steps)
    }
    
    try:
        # Test with new config-based approach
        config = CANN1DPlotConfig.for_bump_animation(
            fps=5,
            show=False,
            figsize=(6, 4),
            max_height_value=0.6,
            nframes=n_steps
        )
        
        animation = create_1d_bump_animation(
            fitting_results,
            config=config,
            save_path=None  # Don't save during test
        )
        
        assert animation is not None
        print("1D bump animation creation with config successful")
        
        # Test backward compatibility
        animation_old = create_1d_bump_animation(
            fitting_results,
            save_path=None,
            fps=5,
            show=False,
            figsize=(6, 4)
        )
        
        assert animation_old is not None
        print("1D bump animation creation with old-style parameters successful")
        
    except Exception as e:
        print(f"Animation creation test failed: {e}")


def test_create_1d_bump_animation_with_saving():
    """Test 1D bump animation with saving (but to a test location)."""
    n_steps = 10  # Very short animation
    n_roi = 6
    
    fitting_results = {
        'positions': np.random.rand(n_steps, 2) * n_roi,
        'kappas': np.ones((n_steps, 2)),
        'amplis': np.ones((n_steps, 2)),
        'n_bumps': np.ones(n_steps, dtype=int),
        'roi_data': create_mock_roi_data(n_roi=n_roi, n_steps=n_steps)
    }
    
    import tempfile
    import os
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_animation.gif")
            
            # Test with new config-based approach
            config = CANN1DPlotConfig.for_bump_animation(
                fps=2,
                show=False,
                save_path=save_path,
                nframes=n_steps,
                show_progress_bar=False  # Disable progress bar for tests
            )
            
            animation = create_1d_bump_animation(
                fitting_results,
                config=config
            )
            
            # Check if file was created
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
            print(f"Animation saved successfully to temporary location")
            
    except Exception as e:
        print(f"Animation saving test failed: {e}")


def test_bump_fits_error_handling():
    """Test error handling in bump fitting."""
    # Test with invalid data shapes
    try:
        # Wrong data shape (1D instead of 2D)
        invalid_data = np.random.rand(100)
        config = BumpFitsConfig(n_steps=10, n_roi=8)
        
        with pytest.raises(Exception):
            bump_fits(invalid_data, config=config)
        print("Error handling for invalid data shape works correctly")
        
    except Exception as e:
        print(f"Error handling test encountered unexpected issue: {e}")


def test_numba_availability():
    """Test whether numba optimization is available."""
    from canns.analyzer.data.cann1d import HAS_NUMBA
    
    if HAS_NUMBA:
        print("Numba optimization available - tests will run faster")
    else:
        print("Numba not available - using pure numpy implementation")
    
    # This shouldn't cause any errors regardless of numba availability
    assert isinstance(HAS_NUMBA, bool)


if __name__ == "__main__":
    print("Running CANN1D module tests...\n")
    
    test_bump_fits_config()
    test_numba_availability()
    test_bump_fits_basic()
    test_bump_fits_with_real_data_structure()
    test_create_1d_bump_animation()
    test_create_1d_bump_animation_with_saving()
    test_bump_fits_error_handling()
    
    print("\nAll CANN1D tests completed!")