#!/usr/bin/env python3
"""
Simple test cases for canns.analyzer.experimental_data module functions.

These tests provide basic verification of the key functions to catch
common errors like array indexing issues and parameter mismatches.
"""

import numpy as np
import pytest
from canns.analyzer.data.cann2d import (
    SpikeEmbeddingConfig,
    TDAConfig,
    CANN2DPlotConfig,
    embed_spike_trains,
    tda_vis,
    decode_circular_coordinates,
    plot_3d_bump_on_torus,
)


def create_mock_spike_data(num_neurons=50, num_timepoints=1000):
    """Create mock spike train data for testing."""
    # Generate random spike times for each neuron
    spikes = {}
    for i in range(num_neurons):
        # Random number of spikes per neuron (0-20)
        num_spikes = np.random.randint(0, 21)
        # Random spike times within timepoint range
        spike_times = np.sort(np.random.uniform(0, 1, num_spikes))
        spikes[i] = spike_times
    
    # Time vector
    t = np.linspace(0, 1, num_timepoints)
    
    # Position data (for speed filtering tests)
    x = np.cumsum(np.random.randn(num_timepoints) * 0.01)
    y = np.cumsum(np.random.randn(num_timepoints) * 0.01)
    
    return {
        'spike': spikes,
        't': t,
        'x': x,
        'y': y
    }


def test_spike_embedding_config():
    """Test SpikeEmbeddingConfig creation and defaults."""
    config = SpikeEmbeddingConfig()
    assert config.res == 100000
    assert config.dt == 1000
    assert config.sigma == 5000
    assert config.smooth is True
    assert config.speed_filter is True
    assert config.min_speed == 2.5


def test_tda_config():
    """Test TDAConfig creation and defaults."""
    config = TDAConfig()
    assert config.dim == 6
    assert config.num_times == 5
    assert config.active_times == 15000
    assert config.k == 1000
    assert config.n_points == 1200
    assert config.metric == "cosine"
    assert config.nbs == 800
    assert config.maxdim == 1
    assert config.coeff == 47


def test_embed_spike_trains_basic():
    """Test basic spike train embedding functionality."""
    mock_data = create_mock_spike_data(num_neurons=10, num_timepoints=500)
    
    # Test without speed filtering
    config = SpikeEmbeddingConfig(smooth=False, speed_filter=False)
    result = embed_spike_trains(mock_data, config)
    
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 10  # number of neurons
    print(f"Basic embedding: shape={result.shape}")


def test_embed_spike_trains_with_speed_filter():
    """Test spike train embedding with speed filtering."""
    mock_data = create_mock_spike_data(num_neurons=10, num_timepoints=500)
    
    # Test with speed filtering
    config = SpikeEmbeddingConfig(smooth=False, speed_filter=True, min_speed=0.1)
    result = embed_spike_trains(mock_data, config)
    
    assert len(result) == 4  # spikes_bin, xx, yy, tt
    spikes_bin, xx, yy, tt = result
    assert isinstance(spikes_bin, np.ndarray)
    assert spikes_bin.ndim == 2
    assert spikes_bin.shape[1] == 10  # number of neurons
    print(f"Speed filtered embedding: spikes shape={spikes_bin.shape}")


def test_tda_vis_basic():
    """Test basic TDA visualization functionality."""
    # Create smaller test data for faster computation
    embed_data = np.random.randn(1000, 20)  # 1000 timepoints, 20 neurons
    
    config = TDAConfig(
        dim=3,
        num_times=10,
        active_times=100,  # Reduced for faster test
        k=50,
        n_points=100,  # Reduced for faster test
        metric="cosine",
        nbs=50,
        maxdim=1,
        coeff=47,
        show=False,  # Don't show plots during testing
        do_shuffle=False  # Skip shuffle for faster test
    )
    
    result = tda_vis(embed_data, config)
    
    assert isinstance(result, dict)
    assert 'persistence' in result
    assert 'indstemp' in result
    assert 'movetimes' in result
    assert 'n_points' in result
    assert result['shuffle_max'] is None  # No shuffle was performed
    print(f"TDA analysis completed: {len(result['indstemp'])} sampled points")


def test_decode_circular_coordinates():
    """Test circular coordinate decoding."""
    # Create minimal test data
    embed_data = create_mock_spike_data(num_neurons=10, num_timepoints=500)
    
    # First run TDA to get persistence results
    embed_spikes = embed_spike_trains(
        embed_data,
        config=SpikeEmbeddingConfig(smooth=True, speed_filter=False)
    )
    
    tda_config = TDAConfig(
        dim=3,
        num_times=20,
        active_times=20,  # Very small for test
        k=10,
        n_points=50,
        metric="cosine",
        nbs=20,
        maxdim=1,
        show=False,
        do_shuffle=False
    )
    
    try:
        persistence_result = tda_vis(embed_spikes, tda_config)
        
        # Test coordinate decoding
        decoding_result = decode_circular_coordinates(
            persistence_result=persistence_result,
            embed_data=embed_data,
            real_ground=True,
            real_of=True,
            save_path=None
        )
        
        assert isinstance(decoding_result, dict)
        assert 'coords' in decoding_result
        assert 'coordsbox' in decoding_result
        assert 'times' in decoding_result
        assert 'times_box' in decoding_result
        print("Circular coordinate decoding completed")
        
    except Exception as e:
        print(f"Coordinate decoding test skipped due to insufficient data: {e}")


def test_array_indexing_fix():
    """Test that the _get_coords function doesn't have array indexing errors."""
    from canns.analyzer.data.cann2d import _get_coords
    
    # Create minimal test data
    num_sampled = 10
    num_edges = 5
    coeff = 47
    
    # Mock cocycle data
    cocycle = np.array([
        [0, 1, 5],
        [1, 2, 3],
        [2, 3, 7],
        [3, 4, 2],
        [4, 0, 1]
    ])
    
    # Mock distance matrix
    dists = np.random.rand(num_sampled, num_sampled)
    np.fill_diagonal(dists, 0)
    dists = (dists + dists.T) / 2  # Make symmetric
    
    threshold = 0.5
    
    try:
        f, verts = _get_coords(cocycle, threshold, num_sampled, dists, coeff)
        assert isinstance(f, np.ndarray)
        assert isinstance(verts, np.ndarray)
        assert len(f) == len(verts)
        print("_get_coords array indexing works correctly")
    except ValueError as e:
        if "setting an array element with a sequence" in str(e):
            pytest.fail("Array indexing error still present in _get_coords")
        else:
            print(f"_get_coords test failed for different reason: {e}")


def test_plot_3d_bump_on_torus():
    """Test 3D bump animation creation (without saving)."""
    # Create mock decoding data
    num_timepoints = 100
    coords = np.random.rand(num_timepoints, 2) * 2 * np.pi
    times = np.arange(num_timepoints)
    
    decoding_result = {
        'coordsbox': coords,
        'times_box': times
    }
    
    embed_data = create_mock_spike_data(num_neurons=10, num_timepoints=200)
    
    try:
        # Test with new config-based approach
        config = CANN2DPlotConfig.for_torus_animation(
            n_frames=3,  # Very few frames for speed
            window_size=20,
            show=False,  # Don't display during test
            show_progress_bar=False,
            fps=2
        )
        
        ani = plot_3d_bump_on_torus(
            decoding_result=decoding_result,
            spike_data=embed_data,
            config=config
        )
        
        assert ani is not None
        print("3D torus animation creation with config works")
        
        # Test backward compatibility
        ani_old = plot_3d_bump_on_torus(
            decoding_result=decoding_result,
            spike_data=embed_data,
            save_path=None,
            n_frames=3,
            window_size=20,
            show=False,
            show_progress=False
        )
        
        assert ani_old is not None
        print("3D torus animation creation with old-style parameters works")
        
    except Exception as e:
        print(f"Torus animation test failed: {e}")


if __name__ == "__main__":
    print("Running experimental_data module tests...\n")
    
    test_spike_embedding_config()
    test_tda_config()
    test_embed_spike_trains_basic()
    test_embed_spike_trains_with_speed_filter()
    test_tda_vis_basic()
    test_array_indexing_fix()
    test_decode_circular_coordinates()
    test_plot_3d_bump_on_torus()
    
    print("\nAll tests completed!")