"""Tests for cell classification metrics."""

import numpy as np
import pytest

from canns.analyzer.data.cell_classification import (
    GridnessAnalyzer,
    HeadDirectionAnalyzer,
    compute_2d_autocorrelation,
    compute_rate_map_from_binned,
)
from canns.analyzer.data.cell_classification import circ_r
from canns.analyzer.data.cell_classification import fit_ellipse

pytestmark = pytest.mark.integration


def test_circular_stats_mvl():
    """MVL should be high for concentrated angles and low for uniform angles."""
    rng = np.random.default_rng(0)
    concentrated = rng.normal(0.0, 0.1, 200)
    uniform = rng.uniform(-np.pi, np.pi, 2000)

    assert circ_r(concentrated) > 0.8
    assert circ_r(uniform) < 0.2


def test_grid_cell_classification_synthetic():
    """Synthetic grid pattern should yield a positive gridness score."""
    x = np.linspace(-2, 2, 100)
    xx, yy = np.meshgrid(x, x)

    theta1, theta2, theta3 = 0.0, np.pi / 3.0, 2.0 * np.pi / 3.0
    k = 2 * np.pi / 0.4

    grid_pattern = (
        np.cos(k * (xx * np.cos(theta1) + yy * np.sin(theta1)))
        + np.cos(k * (xx * np.cos(theta2) + yy * np.sin(theta2)))
        + np.cos(k * (xx * np.cos(theta3) + yy * np.sin(theta3)))
    ) / 3.0

    rate_map = (grid_pattern + 1.5) / 2.5 * 10.0

    autocorr = compute_2d_autocorrelation(rate_map)
    analyzer = GridnessAnalyzer()
    result = analyzer.compute_gridness_score(autocorr)

    assert np.isfinite(result.score)
    assert result.score > 0
    assert len(result.spacing) == 3
    assert len(result.orientation) == 3
    assert result.center_radius >= 0


def test_head_direction_classification_deterministic():
    """Deterministic directional spiking should be classified as HD."""
    time_stamps = np.linspace(0.0, 100.0, 10000)
    head_directions = (np.linspace(0.0, 8 * np.pi, len(time_stamps)) + np.pi) % (2 * np.pi) - np.pi

    preferred_dir = 0.5
    diff = np.arctan2(
        np.sin(head_directions - preferred_dir),
        np.cos(head_directions - preferred_dir),
    )
    spike_mask = np.abs(diff) < 0.25
    spike_times = time_stamps[spike_mask]

    analyzer = HeadDirectionAnalyzer(mvl_hd_threshold=0.3, strict_mode=False)
    result = analyzer.classify_hd_cell(spike_times, head_directions, time_stamps)

    assert result.is_hd
    assert result.mvl_hd > 0.3

    error = np.abs(
        np.arctan2(
            np.sin(result.preferred_direction - preferred_dir),
            np.cos(result.preferred_direction - preferred_dir),
        )
    )
    assert error < 0.3


def test_geometry_fit_ellipse():
    """Ellipse fitting should recover approximate center."""
    t = np.linspace(0.0, 2.0 * np.pi, 120)
    cx, cy = 5.0, 3.0
    rx, ry = 4.0, 2.0
    angle = np.pi / 6.0

    x = cx + rx * np.cos(t) * np.cos(angle) - ry * np.sin(t) * np.sin(angle)
    y = cy + rx * np.cos(t) * np.sin(angle) + ry * np.sin(t) * np.cos(angle)

    params = fit_ellipse(x, y)

    assert abs(params[0] - cx) < 0.5
    assert abs(params[1] - cy) < 0.5


def test_rate_map_from_binned():
    """Binned rate map should reflect occupancy-normalized spikes."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    spikes = np.zeros_like(x)
    spikes[::10] = 1.0

    rate_map, occupancy, _, _ = compute_rate_map_from_binned(x, y, spikes, bins=20)

    assert rate_map.shape == occupancy.shape
    assert np.all(rate_map[occupancy == 0] == 0)
