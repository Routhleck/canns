from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from canns.analyzer.data.asa.cohomap import cohomap, plot_cohomap
from canns.analyzer.data.asa.spatial_tda import (
    pca_reduce,
    permute_each_neuron_map,
    run_spatial_tda_from_asa,
)
from canns.analyzer.workflows import auto_grid_threshold
from canns.analyzer.workflows.phase_center_comparison import (
    _minimum_image_displacement_skew,
    compare_phase_centers_workflow,
)


def test_spatial_tda_pca_reduce_caps_components() -> None:
    points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    reduced, evr = pca_reduce(points, dim=7, standardize=False)

    assert reduced.shape == (3, 2)
    assert evr.shape == (2,)


def test_spatial_null_permutation_keeps_invalid_bins_invalid() -> None:
    fr_tensor = np.full((2, 2, 2), np.nan)
    valid_mask = np.array([[True, False], [True, False]])
    fr_tensor[valid_mask, 0] = [1.0, 2.0]
    fr_tensor[valid_mask, 1] = [3.0, 4.0]

    shuffled = permute_each_neuron_map(
        fr_tensor,
        np.random.default_rng(0),
        valid_mask=valid_mask,
    )

    assert np.isfinite(shuffled[valid_mask]).all()
    assert np.isnan(shuffled[~valid_mask]).all()


def test_spatial_tda_skip_embed_requires_dense_matrix(tmp_path) -> None:
    asa = {
        "spike": {0: np.array([0.1, 0.2]), 1: np.array([0.3])},
        "x": np.arange(4),
        "y": np.arange(4),
    }

    with pytest.raises(ValueError, match="dense numeric"):
        run_spatial_tda_from_asa(
            asa,
            compute_frm_fn=lambda *args, **kwargs: None,
            out_dir=tmp_path,
            skip_embed=True,
        )


def test_auto_grid_threshold_rejects_empty_explicit_topk(monkeypatch) -> None:
    asa = {
        "spike": np.ones((4, 3)),
        "x": np.linspace(0.0, 1.0, 4),
        "y": np.linspace(0.0, 1.0, 4),
        "t": np.arange(4),
    }
    monkeypatch.setattr(
        auto_grid_threshold,
        "_compute_grid_scores",
        lambda *args, **kwargs: np.array([0.9, 0.8, 0.7]),
    )

    with pytest.raises(ValueError, match="No top-k values"):
        auto_grid_threshold.analyze_auto_grid_threshold(
            asa,
            use_precomputed_spikes=True,
            topk_values=[99],
        )


def test_ecohomap_plots_all_decoded_dimensions() -> None:
    rng = np.random.default_rng(0)
    n_points = 80
    decoding = {
        "coordsbox": rng.uniform(0.0, 2 * np.pi, size=(n_points, 4)),
        "times_box": np.arange(n_points),
    }
    position = {
        "x": np.linspace(0.0, 1.0, n_points),
        "y": np.sin(np.linspace(0.0, 2 * np.pi, n_points)),
        "t": np.arange(n_points),
    }

    result = cohomap(decoding, position, bins=12, smooth_sigma=0.0, align_torus=False)
    fig = plot_cohomap(result, show=False)

    assert result["phase_maps"].shape[0] == 4
    assert np.allclose(result["phase_map1"], result["phase_maps"][0], equal_nan=True)
    assert np.allclose(result["phase_map2"], result["phase_maps"][1], equal_nan=True)
    assert len([ax for ax in fig.axes if ax.get_title().startswith("Phase Map")]) == 4
    plt.close(fig)


def test_ecohomap_supports_single_decoded_dimension() -> None:
    n_points = 80
    decoding = {
        "coordsbox": np.linspace(0.0, 2 * np.pi, n_points)[:, None],
        "times_box": np.arange(n_points),
    }
    position = {
        "x": np.linspace(0.0, 1.0, n_points),
        "y": np.cos(np.linspace(0.0, 2 * np.pi, n_points)),
        "t": np.arange(n_points),
    }

    result = cohomap(decoding, position, bins=12, smooth_sigma=0.0, align_torus=True)
    fig = plot_cohomap(result, show=False)

    assert result["phase_maps"].shape[0] == 1
    assert result["num_phase_maps"] == 1
    assert "phase_map2" not in result
    assert len([ax for ax in fig.axes if ax.get_title().startswith("Phase Map")]) == 1
    plt.close(fig)


def test_phase_center_minimum_image_uses_short_torus_displacement() -> None:
    start = np.array([0.05, 0.0])
    end = np.array([2 * np.pi - 0.05, 0.0])

    displacement = _minimum_image_displacement_skew(start, end)

    assert np.allclose(displacement, [-0.1, 0.0])
    assert np.linalg.norm(displacement) < 0.2


def test_phase_center_comparison_requires_both_cell_id_sides() -> None:
    decoding = {
        "coordsbox": np.array([[0.1, 0.2], [1.0, 1.1], [2.0, 2.1]]),
        "times_box": np.arange(3),
    }
    spikes = {"spike": np.ones((3, 2))}

    with pytest.raises(ValueError, match="both cell_ids_a and cell_ids_b"):
        compare_phase_centers_workflow(
            decoding,
            spikes,
            decoding,
            spikes,
            cell_ids_a=np.array([10, 11]),
            cell_ids_b=None,
        )
