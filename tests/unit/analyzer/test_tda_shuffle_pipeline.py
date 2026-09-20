from __future__ import annotations

import importlib
from dataclasses import replace
from threading import Event, Thread

import numpy as np
import pytest

from canns.analyzer.data.asa import tda
from canns.analyzer.data.asa.config import TDAConfig


@pytest.fixture
def activity():
    return np.arange(36, dtype=np.float64).reshape(12, 3)


@pytest.fixture
def config():
    return TDAConfig(
        dim=2,
        num_times=2,
        active_times=6,
        k=3,
        n_points=4,
        nbs=3,
        maxdim=2,
        show=False,
        progress_bar=False,
        num_shuffles=3,
    )


@pytest.fixture
def mocked_pipeline_concurrency(monkeypatch):
    # These scheduler unit tests replace the entire numerical ASA pipeline.
    monkeypatch.setattr(tda, "_check_shuffle_concurrency", lambda workers: None)


def persistence(activity, config):
    diagrams = [np.array([[0.0, 1.0], [0.0, np.inf]])]
    for dim in range(1, config.maxdim + 1):
        diagrams.append(np.array([[0.0, float(activity[0].sum()) + dim + 1]]))
    return dict(
        persistence={"dgms": diagrams}, movetimes=np.arange(6), indstemp=np.arange(4), n_points=4
    )


def test_real_and_null_use_same_full_pipeline_and_config(
    monkeypatch, activity, config, mocked_pipeline_concurrency
):
    seen = []
    shifts = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    config = replace(
        config,
        shuffle_shifts=shifts,
        shuffle_workers=2,
        standardize=False,
        dim=1,
        coeff=2,
        do_cocycles=False,
        ph_threshold=2.5,
        do_shuffle=True,
    )

    def compute(data, settings):
        seen.append((data.copy(), settings))
        return persistence(data, settings)

    def generic_shuffle(*args, **kwargs):
        pytest.fail("generic point-cloud shuffle must never analyze ASA activity")

    monkeypatch.setattr(tda, "_compute_real_persistence", compute)
    library = importlib.import_module("canns_lib.ripser")
    monkeypatch.setattr(library, "shuffle_null_model", generic_shuffle, raising=False)
    monkeypatch.delattr(library, "generate_offsets", raising=False)
    monkeypatch.setattr(tda, "_handle_visualization", lambda *args: None)
    result = tda.tda_vis(activity, config=config)
    assert len(seen) == 4
    np.testing.assert_array_equal(seen[0][0], activity)
    expected = [tda._shuffle_spike_trains(activity, row) for row in shifts]
    for actual, settings in seen[1:]:
        assert sum(np.array_equal(actual, shifted) for shifted in expected) == 1
        for key in (
            "dim",
            "num_times",
            "active_times",
            "k",
            "n_points",
            "metric",
            "nbs",
            "maxdim",
            "coeff",
            "standardize",
            "sampling_backend",
            "ph_threshold_policy",
            "ph_threshold",
            "do_cocycles",
        ):
            assert getattr(settings, key) == getattr(config, key)
    assert all(len(values) == 3 for values in result["shuffle_max"].values())


@pytest.mark.parametrize("workers", [1, 2])
def test_shuffle_matches_independent_complete_asa_rounds(monkeypatch, workers):
    """Compare the scheduler against explicit full ASA computations, not another backend label."""
    activity = np.random.default_rng(23).normal(size=(24, 4))
    shifts = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    config = TDAConfig(
        dim=2,
        num_times=2,
        active_times=12,
        k=5,
        n_points=8,
        nbs=5,
        maxdim=2,
        coeff=47,
        show=False,
        progress_bar=False,
        num_shuffles=2,
        shuffle_shifts=shifts,
        shuffle_workers=workers,
    )
    if workers > 1 and tda.HAS_NUMBA:
        from numba import config as numba_config
        from numba import get_num_threads, threading_layer

        if not numba_config.DISABLE_JIT:
            get_num_threads()
            if threading_layer() == "workqueue":
                monkeypatch.setattr(
                    tda, "_compute_real_persistence", lambda *args: pytest.fail("PH started")
                )
                with pytest.raises(ValueError, match="workqueue.*shuffle_workers=1"):
                    tda._run_shuffle_analysis(activity, config=config)
                return
    shifted = [
        np.column_stack([np.roll(activity[:, j], int(offset)) for j, offset in enumerate(row)])
        for row in shifts
    ]
    reference = [tda._compute_real_persistence(data, config) for data in shifted]
    compute = tda._compute_real_persistence
    captured = {}

    def record(data, settings):
        index = next(i for i, expected in enumerate(shifted) if np.array_equal(data, expected))
        result = compute(data, settings)
        captured[index] = result
        return result

    monkeypatch.setattr(tda, "_compute_real_persistence", record)
    maximums = tda._run_shuffle_analysis(activity, config=config)
    assert len(captured) == len(reference)
    for index, expected in enumerate(reference):
        actual = captured[index]
        for key in ("movetimes", "indstemp"):
            np.testing.assert_array_equal(actual[key], expected[key])
        for dim, diagram in enumerate(expected["persistence"]["dgms"]):
            np.testing.assert_array_equal(actual["persistence"]["dgms"][dim], diagram)
            finite = diagram[np.isfinite(diagram[:, 1])]
            expected_max = float(np.max(finite[:, 1] - finite[:, 0])) if len(finite) else 0.0
            assert maximums[dim][index] == expected_max
        for actual_dimension, expected_dimension in zip(
            actual["persistence"]["cocycles"], expected["persistence"]["cocycles"], strict=True
        ):
            assert len(actual_dimension) == len(expected_dimension)
            for a, b in zip(actual_dimension, expected_dimension, strict=True):
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("maxdim,coeff,cocycles", [(0, 2, False), (1, 47, True), (3, 2, True)])
def test_actual_pipeline_passes_scientific_parameters(
    monkeypatch, activity, config, maxdim, coeff, cocycles
):
    config = replace(
        config,
        standardize=False,
        maxdim=maxdim,
        coeff=coeff,
        do_cocycles=cocycles,
        ph_threshold_policy="max_finite_float32",
    )
    seen = {}

    def pca(data, times, dim, standardize):
        seen["pca"] = (times.copy(), dim, standardize)
        return np.ones((len(times), dim))

    def denoise(data, **kwargs):
        seen["denoise"] = kwargs
        return np.arange(kwargs["num_sample"]), None, None

    def graph(data, selected, **kwargs):
        seen["graph"] = kwargs
        matrix = np.ones((len(selected), len(selected))) * 1.23456789
        matrix[0, 1] = matrix[1, 0] = np.inf
        return matrix

    def ph(matrix, **kwargs):
        seen["ph"] = kwargs
        assert np.diag(matrix).tolist() == [0.0] * config.n_points
        return {"dgms": [np.empty((0, 2)) for _ in range(maxdim + 1)]}

    monkeypatch.setattr(tda, "_apply_pca_reduction", pca)
    monkeypatch.setattr(tda, "_sample_denoising", denoise)
    monkeypatch.setattr(tda, "_second_build", graph)
    monkeypatch.setattr(tda, "ripser", ph)
    tda._compute_real_persistence(activity, config)
    assert seen["pca"][1:] == (2, False)
    assert seen["denoise"] == dict(k=3, num_sample=4, omega=1, metric="cosine", backend="python")
    assert seen["graph"] == dict(metric="cosine", nbs=3)
    assert seen["ph"] == dict(
        maxdim=maxdim,
        coeff=coeff,
        do_cocycles=cocycles,
        distance_matrix=True,
        progress_bar=False,
        thresh=float(np.float32(1.23456789)),
    )


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("shuffle_backend", "python"),
        ("shuffle_backend", "canns_lib"),
        ("use_ffi_shuffle", True),
        ("use_ffi_shuffle", False),
        ("force_legacy", True),
    ],
)
def test_removed_backend_options_fail_instead_of_selecting_a_different_null(
    monkeypatch, activity, keyword, value
):
    monkeypatch.setattr(tda, "_compute_real_persistence", lambda *args: pytest.fail("PH started"))
    with pytest.raises(TypeError, match=keyword):
        tda.tda_vis(activity, **{keyword: value})
    with pytest.raises(TypeError, match=keyword):
        tda._run_shuffle_analysis(activity, **{keyword: value})


def test_unknown_legacy_keyword_is_not_silently_ignored(activity):
    with pytest.raises(TypeError, match="standardise"):
        tda.tda_vis(activity, standardise=False)


@pytest.mark.parametrize(
    "layer,workers,count,do_shuffle,reject",
    [
        ("workqueue", 1, 3, True, False),
        ("workqueue", 2, 1, True, False),
        ("workqueue", 2, 3, False, False),
        ("workqueue", 2, 3, True, True),
        ("tbb", 2, 3, True, False),
        ("omp", 2, 3, True, False),
    ],
)
def test_numba_concurrency_checked_before_real_analysis(
    monkeypatch, activity, config, layer, workers, count, do_shuffle, reject
):
    import numba

    initialized, seen = [], []
    monkeypatch.setattr(tda, "HAS_NUMBA", True)
    monkeypatch.setattr(numba.config, "DISABLE_JIT", False)
    monkeypatch.setattr(numba, "get_num_threads", lambda: initialized.append(True) or 1)

    def threading_layer():
        assert initialized  # Query only after the public initialization API.
        return layer

    def compute(data, settings):
        seen.append(data)
        return persistence(data, settings)

    monkeypatch.setattr(numba, "threading_layer", threading_layer)
    monkeypatch.setattr(tda, "_compute_real_persistence", compute)
    monkeypatch.setattr(tda, "_handle_visualization", lambda *args: None)
    settings = replace(config, shuffle_workers=workers, num_shuffles=count, do_shuffle=do_shuffle)
    if reject:
        with pytest.raises(ValueError, match="workqueue.*shuffle_workers=1"):
            tda.tda_vis(activity, config=settings)
        assert not seen
    else:
        tda.tda_vis(activity, config=settings)
        assert len(seen) == 1 + (count if do_shuffle else 0)
    if not do_shuffle or min(workers, count) == 1:
        assert not initialized


def test_disabled_jit_does_not_require_a_numba_thread_pool(monkeypatch):
    import numba

    monkeypatch.setattr(tda, "HAS_NUMBA", True)
    monkeypatch.setattr(numba.config, "DISABLE_JIT", True)
    monkeypatch.setattr(numba, "get_num_threads", lambda: pytest.fail("Numba pool initialized"))
    tda._check_shuffle_concurrency(2)


def test_asa_failure_preserves_index_offsets_and_original_error(monkeypatch, activity, config):
    shifts = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
    seen = []
    error = ArithmeticError("intentional failure")

    def compute(data, settings):
        seen.append(data.copy())
        if len(seen) == 2:
            raise error
        return persistence(data, settings)

    monkeypatch.setattr(tda, "_compute_real_persistence", compute)
    with pytest.raises(tda.ShuffleIterationError) as actual:
        tda._run_shuffle_analysis(activity, config=replace(config, shuffle_shifts=shifts))
    assert len(seen) == 2 and actual.value.index == 1
    assert actual.value.__cause__ is error and actual.value.original_exception is error
    np.testing.assert_array_equal(actual.value.offsets, shifts[1])
    assert not actual.value.offsets.flags.writeable


def test_parallel_asa_surfaces_failure_without_waiting_for_running_analysis(
    monkeypatch, activity, config, mocked_pipeline_concurrency
):
    shifts = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
    started, release, finished, returned = (Event() for _ in range(4))
    error = ArithmeticError("intentional concurrent failure")
    outcomes = []
    seen = []

    def compute(data, settings):
        seen.append(data.copy())
        if np.array_equal(data, activity):
            started.set()
            try:
                assert release.wait(10), "test did not release the running callback"
                return persistence(data, settings)
            finally:
                finished.set()
        assert started.wait(5), "first callback did not start"
        raise error

    def run():
        try:
            tda._run_shuffle_analysis(
                activity,
                config=replace(config, shuffle_shifts=shifts, shuffle_workers=2),
            )
        except BaseException as exc:
            outcomes.append(exc)
        finally:
            returned.set()

    monkeypatch.setattr(tda, "_compute_real_persistence", compute)
    supervisor = Thread(target=run)
    supervisor.start()
    try:
        assert started.wait(5)
        assert returned.wait(5), "failure waited for another running callback"
        assert not finished.is_set()
        assert len(outcomes) == 1 and isinstance(outcomes[0], tda.ShuffleIterationError)
        assert outcomes[0].index == 1 and outcomes[0].original_exception is error
        np.testing.assert_array_equal(outcomes[0].offsets, shifts[1])
        assert len(seen) == 2  # No third iteration is started after the failure.
    finally:
        release.set()
        supervisor.join(timeout=5)
        assert not supervisor.is_alive()
        assert finished.wait(5)


def test_successful_empty_diagram_is_zero_but_missing_dimension_fails(
    monkeypatch, activity, config
):
    monkeypatch.setattr(
        tda,
        "_compute_real_persistence",
        lambda data, cfg: {
            "persistence": {"dgms": [np.empty((0, 2)) for _ in range(cfg.maxdim + 1)]}
        },
    )
    out = tda._run_shuffle_analysis(activity, config=config)
    assert out == {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0]}
    monkeypatch.setattr(
        tda,
        "_compute_real_persistence",
        lambda data, cfg: {"persistence": {"dgms": [np.empty((0, 2))]}},
    )
    with pytest.raises(tda.ShuffleIterationError, match="missing or extra"):
        tda._run_shuffle_analysis(activity, config=config)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("seed", [21, np.int64(21)])
def test_seed_plan_matches_default_rng_and_keeps_input(
    monkeypatch, activity, config, mocked_pipeline_concurrency, workers, seed
):
    before = activity.copy()
    monkeypatch.setattr(tda, "_compute_real_persistence", persistence)
    config = replace(
        config,
        shuffle_seed=seed,
        shuffle_workers=workers,
        num_shuffles=7,
    )
    expected_shifts = np.random.default_rng(21).integers(0, 12, size=(7, 3), dtype=np.int64)
    expected = {d: [] for d in range(3)}
    for shifts in expected_shifts:
        data = tda._shuffle_spike_trains(activity, shifts)
        maxima, _ = tda._finite_lifetime_summary(persistence(data, config)["persistence"], 2)
        for dim in expected:
            expected[dim].append(maxima[dim])
    assert tda._run_shuffle_analysis(activity, config=config) == expected
    np.testing.assert_array_equal(activity, before)


@pytest.mark.parametrize("seed", [True, np.bool_(True), -1, 1.5, np.random.default_rng(9)])
def test_invalid_seed_rejected_before_real_without_mutating_generator(
    monkeypatch, activity, config, seed
):
    monkeypatch.setattr(tda, "_check_backend_availability", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tda, "_compute_real_persistence", lambda *args: pytest.fail("real analysis started")
    )
    generator = seed if isinstance(seed, np.random.Generator) else None
    before = repr(generator.bit_generator.state) if generator is not None else None
    with pytest.raises(ValueError, match="shuffle_seed.*nonnegative integer"):
        tda.tda_vis(
            activity,
            config=replace(config, do_shuffle=True, shuffle_seed=seed),
        )
    if generator is not None:
        assert repr(generator.bit_generator.state) == before


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"shuffle_workers": 0}, "positive"),
        ({"shuffle_shifts": np.zeros((2, 3), dtype=int)}, "integer"),
        ({"shuffle_shifts": np.zeros((3, 3))}, "integer"),
        ({"shuffle_shifts": np.full((3, 3), 12)}, "timepoints"),
        ({"shuffle_shifts": np.zeros((3, 3), dtype=int), "shuffle_seed": 1}, "exclusive"),
    ],
)
def test_bad_shuffle_plan_fails_before_pipeline(monkeypatch, activity, config, changes, match):
    monkeypatch.setattr(
        tda, "_compute_real_persistence", lambda *a: pytest.fail("pipeline started")
    )
    with pytest.raises(ValueError, match=match):
        tda._run_shuffle_analysis(activity, config=replace(config, **changes))


@pytest.mark.parametrize("changes", [{"k": 7}, {"n_points": 7}, {"nbs": 5}, {"dim": 4}])
def test_impossible_scientific_budgets_rejected(activity, config, changes):
    with pytest.raises(ValueError):
        tda._compute_real_persistence(activity, replace(config, **changes))


def test_dim_one_is_one_centered_principal_component():
    data = np.array([[1.0, 3.0, 2.0], [2.0, 6.0, 4.0], [3.0, 9.0, 6.0], [4.0, 12.0, 8.0]])
    scores, explained, _ = tda._pca(data, dim=1)
    assert scores.shape == (4, 1)
    np.testing.assert_allclose(scores.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(explained, [100.0], atol=1e-12)
    np.testing.assert_allclose(np.diff(scores[:, 0]) ** 2, 14.0, atol=1e-12)


def test_nonstandardized_pca_is_translation_invariant():
    data = np.array([[1.0, 4.0, 9.0], [3.0, 2.0, 8.0], [5.0, 1.0, 7.0], [7.0, 8.0, 2.0]])
    indices = np.arange(4)
    a = tda._apply_pca_reduction(data, indices, 2, False)
    b = tda._apply_pca_reduction(data + [100.0, -50.0, 20.0], indices, 2, False)
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(a.mean(axis=0), 0.0, atol=1e-12)


def test_threshold_policy_is_optional_and_respects_explicit_cutoff(config):
    matrix = np.array([[0.0, 2.123456789, np.inf], [2.123456789, 0.0, 1.0], [np.inf, 1.0, 0.0]])
    assert tda._ph_threshold_options(matrix, config) == {}
    assert tda._ph_threshold_options(matrix, replace(config, ph_threshold=1.5)) == {"thresh": 1.5}
    assert tda._ph_threshold_options(
        matrix, replace(config, ph_threshold_policy="max_finite_float32")
    ) == {"thresh": float(np.float32(2.123456789))}
    with pytest.raises(ValueError, match="cannot be combined"):
        tda._resolve_configuration(
            replace(config, ph_threshold=1.0, ph_threshold_policy="max_finite_float32")
        )


def test_rust_sampler_preserves_density_selection(monkeypatch):
    called = []

    def fuzzy(rows, cols, vals, n):
        assert rows.dtype == cols.dtype == np.dtype("int64")
        assert vals.dtype == np.dtype("float64")
        assert rows.flags.c_contiguous and cols.flags.c_contiguous and vals.flags.c_contiguous
        called.append(n)
        directed = np.zeros((n, n))
        directed[rows, cols] = vals
        return directed + directed.T - directed * directed.T

    monkeypatch.setattr(tda, "_rust_fuzzy_union_function", lambda: fuzzy)
    data = np.random.default_rng(2).normal(size=(18, 3))
    baseline = tda._sample_denoising_numba(data, 5, 8, 1.0, "euclidean")
    candidate = tda._sample_denoising(data, 5, 8, 1.0, "euclidean", backend="rust")
    assert called == [18]
    np.testing.assert_array_equal(candidate[0], baseline[0])
    np.testing.assert_allclose(candidate[1], baseline[1], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(candidate[2], baseline[2], rtol=1e-14, atol=1e-14)


def test_rust_sampling_unavailable_fails_before_real(monkeypatch, activity, config):
    monkeypatch.delattr(tda._ripser_core, "fuzzy_union", raising=False)
    monkeypatch.setattr(tda, "_compute_real_persistence", lambda *a: pytest.fail("real PH started"))
    with pytest.raises(ImportError, match="fuzzy_union"):
        tda.tda_vis(activity, config=replace(config, sampling_backend="rust"))


def test_higher_dimensions_remain_plotable():
    diagrams = {"dgms": [np.array([[0.0, 1.0]]) for _ in range(4)]}
    for fig in (tda._plot_barcode(diagrams), tda._plot_barcode_with_shuffle(diagrams, {})):
        assert len(fig.axes) == 4
        assert fig.axes[3].get_ylabel() == "$H_3$"
        tda.plt.close(fig)
