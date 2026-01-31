#!/usr/bin/env python3
"""
Smoke tests for the ASA experimental-data analysis API.
"""

import numpy as np
import pytest

from canns.analyzer import data
from canns.analyzer.visualization import PlotConfig
from canns.data.loaders import load_grid_data


def create_mock_spike_data(
    num_neurons: int = 30,
    num_timepoints: int = 400,
    density: float = 1.0,
    structured: bool = False,
    duration: float = 10.0,
    seed: int = 0,
):
    """Create mock spike train data for testing."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, num_timepoints, endpoint=False)
    dt = t[1] - t[0] if num_timepoints > 1 else duration

    if structured:
        theta1 = np.cumsum(rng.normal(scale=0.12, size=num_timepoints)) % (2 * np.pi)
        theta2 = np.cumsum(rng.normal(scale=0.10, size=num_timepoints)) % (2 * np.pi)
        x = (
            0.6 * np.cos(theta1)
            + 0.4 * np.cos(theta2)
            + rng.normal(scale=0.05, size=num_timepoints)
        )
        y = (
            0.6 * np.sin(theta1)
            + 0.4 * np.sin(theta2)
            + rng.normal(scale=0.05, size=num_timepoints)
        )

        base_rate = 0.2
        peak_rate = 1.5 * density
        kappa = 1.2

        spikes = {}
        pref1 = rng.uniform(0, 2 * np.pi, size=num_neurons)
        pref2 = rng.uniform(0, 2 * np.pi, size=num_neurons)
        for i in range(num_neurons):
            rate = base_rate + peak_rate * np.exp(
                kappa * (np.cos(theta1 - pref1[i]) + np.cos(theta2 - pref2[i]))
            )
            counts = rng.poisson(rate * dt)
            spike_times = []
            for ti, c in enumerate(counts):
                if c > 0:
                    spike_times.extend(t[ti] + rng.random(c) * dt)
            spikes[i] = np.sort(np.asarray(spike_times))
        return {"spike": spikes, "t": t, "x": x, "y": y}

    spikes = {}
    for i in range(num_neurons):
        low = max(5, int(25 * density))
        high = max(low + 1, int(60 * density))
        num_spikes = rng.integers(low, high)
        spike_times = np.sort(rng.uniform(0, duration, num_spikes))
        spikes[i] = spike_times

    x = np.cumsum(rng.standard_normal(num_timepoints) * 0.05)
    y = np.cumsum(rng.standard_normal(num_timepoints) * 0.05)
    return {"spike": spikes, "t": t, "x": x, "y": y}


def _subset_grid_data(
    grid_data: dict,
    num_neurons: int = 50,
    num_timepoints: int = 3000,
) -> dict:
    spike_data = grid_data["spike"]
    if hasattr(spike_data, "item") and callable(spike_data.item):
        spike_data = spike_data.item()

    t = grid_data["t"]
    end_idx = min(num_timepoints, len(t))
    if end_idx < 2:
        return grid_data

    t_window = t[:end_idx]
    t_min = t_window[0]
    t_max = t_window[-1]

    if isinstance(spike_data, dict):
        keys = sorted(spike_data.keys())[:num_neurons]
        spikes = {}
        for i, key in enumerate(keys):
            s = np.asarray(spike_data[key])
            spikes[i] = s[(s >= t_min) & (s <= t_max)]
    else:
        spikes = spike_data[:num_neurons]

    result = {"spike": spikes, "t": t_window}
    if "x" in grid_data:
        result["x"] = grid_data["x"][:end_idx]
    if "y" in grid_data:
        result["y"] = grid_data["y"][:end_idx]
    return result


def test_embed_spike_trains_basic():
    mock_data = create_mock_spike_data(num_neurons=12, num_timepoints=200)
    cfg = data.SpikeEmbeddingConfig(smooth=False, speed_filter=False)
    spikes, xx, yy, tt = data.embed_spike_trains(mock_data, config=cfg)

    assert isinstance(spikes, np.ndarray)
    assert spikes.ndim == 2
    assert spikes.shape[1] == 12
    assert xx is not None and yy is not None and tt is not None
    assert np.allclose(xx, mock_data["x"])
    assert np.allclose(yy, mock_data["y"])
    assert np.allclose(tt, mock_data["t"])


def test_embed_spike_trains_speed_filter():
    mock_data = create_mock_spike_data(num_neurons=8, num_timepoints=200)
    cfg = data.SpikeEmbeddingConfig(smooth=False, speed_filter=True, min_speed=0.01)
    spikes, xx, yy, tt = data.embed_spike_trains(mock_data, config=cfg)

    assert isinstance(spikes, np.ndarray)
    assert spikes.ndim == 2
    assert spikes.shape[1] == 8
    assert xx is not None and yy is not None and tt is not None


def test_tda_decode_and_cohomap():
    # grid_data = load_grid_data()
    # if grid_data is None:
    grid_data = create_mock_spike_data(
        num_neurons=50,
        num_timepoints=3000,
        density=2.0,
        structured=True,
        duration=30.0,
    )
    # else:
    #     grid_data = _subset_grid_data(grid_data, num_neurons=50, num_timepoints=3000)

    spike_cfg = data.SpikeEmbeddingConfig(smooth=True, speed_filter=False)
    spikes, *_ = data.embed_spike_trains(grid_data, config=spike_cfg)

    spike_data = dict(grid_data)
    spike_data["spike"] = spikes

    decoding = None
    for n_points in (120, 80, 60, 40):
        tda_cfg = data.TDAConfig(
            dim=3,
            num_times=8,
            active_times=max(3 * n_points, 60),
            k=20,
            n_points=n_points,
            metric="cosine",
            nbs=40,
            maxdim=1,
            coeff=47,
            show=False,
            do_shuffle=False,
            progress_bar=False,
        )

        persistence = data.tda_vis(spikes, config=tda_cfg)
        try:
            decoding = data.decode_circular_coordinates_multi(
                persistence_result=persistence,
                spike_data=spike_data,
                num_circ=1,
            )
            break
        except ValueError:
            continue

    if decoding is None:
        pytest.fail("decode_circular_coordinates_multi failed for all n_points settings")

    assert "coords" in decoding and "coordsbox" in decoding

    config = PlotConfig.for_static_plot(show=False)
    data.plot_cohomap_scatter_multi(
        decoding_result=decoding,
        position_data={"x": grid_data["x"], "y": grid_data["y"]},
        config=config,
    )


def test_cohospace_and_path_compare():
    mock_data = create_mock_spike_data(num_neurons=15, num_timepoints=250)
    spike_cfg = data.SpikeEmbeddingConfig(smooth=True, speed_filter=False)
    spikes, *_ = data.embed_spike_trains(mock_data, config=spike_cfg)

    rng = np.random.default_rng(1)
    coords = rng.random((spikes.shape[0], 2)) * 2 * np.pi

    data.plot_cohospace_scatter_trajectory_2d(coords, show=False)
    data.plot_cohospace_scatter_neuron_2d(
        coords=coords,
        activity=spikes,
        neuron_id=0,
        mode="fr",
        top_percent=5,
        show=False,
    )

    t_full = mock_data["t"]
    times_box = np.arange(0, len(t_full), 2)
    coords_box = rng.random((len(times_box), 2)) * 2 * np.pi

    t_aligned, x_aligned, y_aligned, coords_aligned, _ = data.align_coords_to_position_2d(
        t_full=mock_data["t"],
        x_full=mock_data["x"],
        y_full=mock_data["y"],
        coords2=coords_box,
        use_box=True,
        times_box=times_box,
        interp_to_full=True,
    )
    assert len(t_aligned) == len(mock_data["t"])
    coords_aligned = data.apply_angle_scale(coords_aligned, "rad")

    data.plot_path_compare_2d(
        x_aligned,
        y_aligned,
        coords_aligned,
        config=PlotConfig.for_static_plot(show=False),
    )


def test_fr_heatmap_and_frm(tmp_path):
    mock_data = create_mock_spike_data(num_neurons=10, num_timepoints=200)
    spike_cfg = data.SpikeEmbeddingConfig(smooth=False, speed_filter=True, min_speed=0.01)
    spikes, xx, yy, _ = data.embed_spike_trains(mock_data, config=spike_cfg)

    mat = data.compute_fr_heatmap_matrix(spikes, transpose=True, normalize=None)
    cfg = PlotConfig.for_static_plot(save_path=str(tmp_path / "fr_heatmap.png"), show=False)
    data.save_fr_heatmap_png(mat, config=cfg, dpi=120)

    frm_result = data.compute_frm(spikes, xx, yy, neuron_id=0, bins=20)
    frm_cfg = PlotConfig.for_static_plot(save_path=str(tmp_path / "frm.png"), show=False)
    data.plot_frm(frm_result.frm, config=frm_cfg, dpi=120)
