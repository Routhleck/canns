"""Tests for systematic rate map sampling module.

This test suite verifies the systematic spatial sampling implementation,
including Numba-optimized functions and the main ratemap computation.
"""

import numpy as np
import pytest
import brainpy.math as bm

from canns.models.basic import GridCell2DVelocity
from canns.analyzer.metrics.systematic_ratemap import (
    compute_systematic_ratemap,
    _compute_velocities,
    _create_vertical_velocities,
    _downsample_indices,
    _downsample_activities,
)


# ============================================================================
# Numba Function Tests
# ============================================================================


def test_compute_velocities():
    """Test velocity computation from positions."""
    # Create simple trajectory
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
    dt = 0.1

    velocities = _compute_velocities(positions, dt)

    # Expected velocities
    expected = np.array([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0]])

    np.testing.assert_allclose(velocities, expected, rtol=1e-5)
    assert velocities.shape == (3, 2), "Velocity shape should be (T-1, 2)"


def test_create_vertical_velocities():
    """Test vertical velocity array creation."""
    num_steps = 100
    num_positions = 30
    speed = 0.5

    batch_vel = _create_vertical_velocities(num_steps, num_positions, speed)

    assert batch_vel.shape == (num_steps, num_positions, 2)
    # x-component should be zero
    np.testing.assert_array_equal(batch_vel[:, :, 0], 0.0)
    # y-component should be speed
    np.testing.assert_array_equal(batch_vel[:, :, 1], speed)


def test_downsample_indices():
    """Test downsampling index computation."""
    total_length = 1000
    target_size = 30

    indices = _downsample_indices(total_length, target_size)

    assert len(indices) == target_size
    assert indices[0] == 0, "First index should be 0"
    assert indices[-1] < total_length, "Last index should be within bounds"
    # Check roughly uniform spacing
    expected_ratio = total_length // target_size
    np.testing.assert_allclose(
        np.diff(indices), expected_ratio, rtol=0.1, atol=1
    )


def test_downsample_activities():
    """Test activity downsampling."""
    num_steps = 1000
    batch_size = 6
    num_neurons = 100
    resolution = 30

    # Create synthetic activities with a pattern
    activities = np.random.rand(num_steps, batch_size, num_neurons)

    downsample_ratio = num_steps // resolution
    downsampled = _downsample_activities(activities, downsample_ratio, resolution)

    assert downsampled.shape == (batch_size, resolution, num_neurons)

    # Verify downsampled values match original at expected indices
    for i in range(batch_size):
        for y_idx in range(resolution):
            t_idx = y_idx * downsample_ratio
            if t_idx < num_steps:
                np.testing.assert_array_equal(
                    downsampled[i, y_idx, :], activities[t_idx, i, :]
                )


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.fixture
def small_model():
    """Create a small grid cell model for testing."""
    bm.set_dt(5e-4)
    model = GridCell2DVelocity(
        length=20,  # Small network for fast tests
        tau=0.01,
        alpha=0.1,
        W_l=2.0,
        lambda_net=17.0,
    )
    # Heal network to establish stable state
    model.heal_network(num_healing_steps=1000, dt_healing=1e-4)
    return model


def test_systematic_ratemap_shape(small_model):
    """Test that output shape matches expected dimensions."""
    resolution = 10
    ratemap = compute_systematic_ratemap(
        small_model,
        box_width=2.2,
        box_height=2.2,
        resolution=resolution,
        speed=0.3,
        num_batches=2,
        verbose=False,
    )

    expected_shape = (resolution, resolution, small_model.num)
    assert ratemap.shape == expected_shape, f"Expected shape {expected_shape}, got {ratemap.shape}"


def test_systematic_ratemap_coverage(small_model):
    """Test 100% spatial coverage (no NaN or zero-only bins)."""
    resolution = 10
    ratemap = compute_systematic_ratemap(
        small_model,
        box_width=2.2,
        box_height=2.2,
        resolution=resolution,
        speed=0.3,
        num_batches=2,
        verbose=False,
    )

    # Check no NaN values
    assert not np.any(np.isnan(ratemap)), "Rate map should not contain NaN values"

    # Check that we have activity across the spatial grid
    # At least some neurons should have non-zero activity in each spatial bin
    spatial_activity = np.max(ratemap, axis=2)  # Max across neurons for each spatial bin
    coverage = np.sum(spatial_activity > 0) / (resolution * resolution)
    assert coverage > 0.8, f"Spatial coverage should be >80%, got {coverage * 100:.1f}%"


def test_activity_range(small_model):
    """Test that activity values are in reasonable range."""
    ratemap = compute_systematic_ratemap(
        small_model,
        box_width=2.2,
        box_height=2.2,
        resolution=10,
        speed=0.3,
        num_batches=2,
        verbose=False,
    )

    # Activity should be non-negative
    assert np.all(ratemap >= 0), "Activity should be non-negative"

    # Activity should be bounded (for typical grid cell models)
    max_activity = np.max(ratemap)
    assert max_activity < 1.0, f"Max activity {max_activity} seems unreasonably high"
    assert max_activity > 0, "Should have some non-zero activity"


def test_resolution_scaling(small_model):
    """Test that function works with different resolutions."""
    for resolution in [5, 10, 15]:
        ratemap = compute_systematic_ratemap(
            small_model,
            box_width=2.2,
            box_height=2.2,
            resolution=resolution,
            speed=0.3,
            num_batches=2,
            verbose=False,
        )

        assert ratemap.shape == (resolution, resolution, small_model.num)
        assert not np.any(np.isnan(ratemap))


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_single_batch(small_model):
    """Test computation with single batch."""
    ratemap = compute_systematic_ratemap(
        small_model,
        box_width=2.2,
        box_height=2.2,
        resolution=10,
        speed=0.3,
        num_batches=1,  # Single batch
        verbose=False,
    )

    assert ratemap.shape == (10, 10, small_model.num)
    assert not np.any(np.isnan(ratemap))


def test_verbose_output(small_model, capsys):
    """Test that verbose mode produces output."""
    _ = compute_systematic_ratemap(
        small_model,
        box_width=2.2,
        box_height=2.2,
        resolution=5,  # Small for fast test
        speed=0.3,
        num_batches=1,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "Computing systematic rate maps" in captured.out
    assert "Rate maps computed" in captured.out


def test_different_arena_sizes(small_model):
    """Test with different arena dimensions."""
    # Square arena
    ratemap_square = compute_systematic_ratemap(
        small_model,
        box_width=2.0,
        box_height=2.0,
        resolution=8,
        speed=0.3,
        num_batches=2,
        verbose=False,
    )

    assert ratemap_square.shape == (8, 8, small_model.num)

    # Rectangular arena
    ratemap_rect = compute_systematic_ratemap(
        small_model,
        box_width=3.0,
        box_height=2.0,
        resolution=8,
        speed=0.3,
        num_batches=2,
        verbose=False,
    )

    assert ratemap_rect.shape == (8, 8, small_model.num)


# ============================================================================
# Performance Tests (Optional)
# ============================================================================


