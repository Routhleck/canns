"""
Tests for reproducibility of navigation tasks with random seeds.

This module tests that navigation tasks produce identical results when
using the same random seed, ensuring experimental reproducibility.
"""

import numpy as np
import pytest
import brainpy.math as bm

from canns.task.open_loop_navigation import OpenLoopNavigationTask

pytestmark = pytest.mark.integration


def test_open_loop_navigation_same_seed_reproducibility():
    """Test that OpenLoopNavigationTask produces identical results with same seed."""
    # Test parameters
    seed = 42
    duration = 5.0
    box_size = 2.2

    # Run 1: First execution with seed
    np.random.seed(seed)
    bm.random.seed(seed)
    bm.set_dt(0.1)

    task1 = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        start_pos=[box_size / 2, box_size / 2],
        speed_mean=0.3,
        speed_std=0.1,
        dt=bm.get_dt(),
        rng_seed=seed,
        progress_bar=False,
    )
    task1.get_data()

    pos1 = task1.data.position
    vel1 = task1.data.velocity
    speed1 = task1.data.speed

    # Run 2: Second execution with same seed
    np.random.seed(seed)
    bm.random.seed(seed)
    bm.set_dt(0.1)

    task2 = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        start_pos=[box_size / 2, box_size / 2],
        speed_mean=0.3,
        speed_std=0.1,
        dt=bm.get_dt(),
        rng_seed=seed,
        progress_bar=False,
    )
    task2.get_data()

    pos2 = task2.data.position
    vel2 = task2.data.velocity
    speed2 = task2.data.speed

    # Verify identical results
    assert np.allclose(pos1, pos2, rtol=1e-10, atol=1e-10), (
        "Positions should be identical with same seed"
    )
    assert np.allclose(vel1, vel2, rtol=1e-10, atol=1e-10), (
        "Velocities should be identical with same seed"
    )
    assert np.allclose(speed1, speed2, rtol=1e-10, atol=1e-10), (
        "Speeds should be identical with same seed"
    )

    # Additional check: exact equality for first few steps
    assert np.array_equal(pos1[:10], pos2[:10]), "First 10 positions should be exactly equal"


def test_open_loop_navigation_different_seeds():
    """Test that OpenLoopNavigationTask produces different results with different seeds."""
    # Test parameters
    duration = 5.0
    box_size = 2.2

    # Run 1: Seed 42
    np.random.seed(42)
    bm.random.seed(42)
    bm.set_dt(0.1)

    task1 = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        start_pos=[box_size / 2, box_size / 2],
        speed_mean=0.3,
        speed_std=0.1,
        dt=bm.get_dt(),
        rng_seed=42,
        progress_bar=False,
    )
    task1.get_data()

    pos1 = task1.data.position
    vel1 = task1.data.velocity

    # Run 2: Seed 123
    np.random.seed(123)
    bm.random.seed(123)
    bm.set_dt(0.1)

    task2 = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        start_pos=[box_size / 2, box_size / 2],
        speed_mean=0.3,
        speed_std=0.1,
        dt=bm.get_dt(),
        rng_seed=123,
        progress_bar=False,
    )
    task2.get_data()

    pos2 = task2.data.position
    vel2 = task2.data.velocity

    # Verify different results
    pos_diff = np.max(np.abs(pos1 - pos2))
    vel_diff = np.max(np.abs(vel1 - vel2))

    assert pos_diff > 1e-6, f"Positions should differ with different seeds (diff={pos_diff})"
    assert vel_diff > 1e-6, f"Velocities should differ with different seeds (diff={vel_diff})"


def test_open_loop_navigation_rng_seed_only_reproducibility():
    """Test that rng_seed alone ensures reproducibility without global seeds."""
    # Test parameters
    duration = 5.0
    box_size = 2.2
    rng_seed = 123

    # Important: Do NOT set global seeds to prove rng_seed alone works
    bm.set_dt(0.1)

    # Create two tasks with same rng_seed
    task1 = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        start_pos=[box_size / 2, box_size / 2],
        speed_mean=0.3,
        speed_std=0.1,
        dt=bm.get_dt(),
        rng_seed=rng_seed,
        progress_bar=False,
    )
    task1.get_data()

    task2 = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        start_pos=[box_size / 2, box_size / 2],
        speed_mean=0.3,
        speed_std=0.1,
        dt=bm.get_dt(),
        rng_seed=rng_seed,
        progress_bar=False,
    )
    task2.get_data()

    # Verify identical results
    assert np.allclose(task1.data.position, task2.data.position, rtol=1e-10, atol=1e-10), (
        "rng_seed alone should produce identical positions"
    )
    assert np.allclose(task1.data.velocity, task2.data.velocity, rtol=1e-10, atol=1e-10), (
        "rng_seed alone should produce identical velocities"
    )
    assert np.allclose(task1.data.speed, task2.data.speed, rtol=1e-10, atol=1e-10), (
        "rng_seed alone should produce identical speeds"
    )


def test_open_loop_navigation_reset_reproducibility():
    """Test that reset() maintains reproducibility with same seed."""
    seed = 42
    duration = 3.0
    box_size = 2.2

    # Create task with seed
    np.random.seed(seed)
    bm.random.seed(seed)
    bm.set_dt(0.1)

    task = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        start_pos=[box_size / 2, box_size / 2],
        speed_mean=0.3,
        speed_std=0.1,
        dt=bm.get_dt(),
        rng_seed=seed,
        progress_bar=False,
    )

    # Run 1
    task.get_data()
    pos1 = task.data.position.copy()
    vel1 = task.data.velocity.copy()

    # Reset and run again
    task.reset()
    task.get_data()
    pos2 = task.data.position.copy()
    vel2 = task.data.velocity.copy()

    # Verify identical results after reset
    assert np.allclose(pos1, pos2, rtol=1e-10, atol=1e-10), (
        "Reset should produce identical trajectories"
    )
    assert np.allclose(vel1, vel2, rtol=1e-10, atol=1e-10), (
        "Reset should produce identical velocities"
    )


def test_open_loop_navigation_no_seed_randomness():
    """Test that without seed, results are random (not identical)."""
    duration = 3.0
    box_size = 2.2

    # Run 1: No seed
    bm.set_dt(0.1)
    task1 = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        start_pos=[box_size / 2, box_size / 2],
        speed_mean=0.3,
        speed_std=0.1,
        dt=bm.get_dt(),
        rng_seed=None,  # No seed
        progress_bar=False,
    )
    task1.get_data()
    pos1 = task1.data.position

    # Run 2: No seed
    bm.set_dt(0.1)
    task2 = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        start_pos=[box_size / 2, box_size / 2],
        speed_mean=0.3,
        speed_std=0.1,
        dt=bm.get_dt(),
        rng_seed=None,  # No seed
        progress_bar=False,
    )
    task2.get_data()
    pos2 = task2.data.position

    # Verify results are different (with high probability)
    # Note: There's a tiny chance they could be identical by random chance,
    # but with 30+ timesteps and continuous random walk, this is extremely unlikely
    pos_diff = np.max(np.abs(pos1 - pos2))
    assert pos_diff > 1e-6, "Without seed, trajectories should be different (randomness working)"


def test_open_loop_navigation_seed_parameter_types():
    """Test that rng_seed accepts various input types correctly."""
    duration = 1.0
    box_size = 2.2
    bm.set_dt(0.1)

    # Test with integer seed
    task_int = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        rng_seed=42,
        progress_bar=False,
    )
    task_int.get_data()
    assert task_int.data.position is not None

    # Test with None (default random behavior)
    task_none = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        rng_seed=None,
        progress_bar=False,
    )
    task_none.get_data()
    assert task_none.data.position is not None

    # Test without specifying rng_seed (backward compatibility)
    task_default = OpenLoopNavigationTask(
        duration=duration,
        width=box_size,
        height=box_size,
        progress_bar=False,
    )
    task_default.get_data()
    assert task_default.data.position is not None
