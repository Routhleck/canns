import numpy as np
import pytest

from canns.data import loaders


def test_validate_roi_data_valid():
    assert loaders.validate_roi_data(np.array([1.0, 2.0, 3.0]))
    assert loaders.validate_roi_data(np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0],
        np.zeros((1, 1, 1)),
        np.array([]),
        np.array([1.0, np.nan]),
    ],
)
def test_validate_roi_data_invalid(data):
    assert loaders.validate_roi_data(data) is False


def test_validate_grid_data_valid():
    data = {
        "spike": [np.array([0.1, 0.2, 0.3])],
        "t": np.array([0.1, 0.2, 0.3]),
    }
    assert loaders.validate_grid_data(data)


@pytest.mark.parametrize(
    "data",
    [
        "not-a-dict",
        {"t": np.array([0.1])},
        {"spike": [np.array([0.1])]},
        {"spike": "bad", "t": np.array([0.1])},
        {"spike": [], "t": np.array([0.1])},
        {"spike": [np.array([0.1])], "t": np.array([])},
        {"spike": [np.array([0.1])], "t": np.array([np.nan])},
    ],
)
def test_validate_grid_data_invalid(data):
    assert loaders.validate_grid_data(data) is False


def test_validate_grid_data_position_shape_mismatch():
    data = {
        "spike": [np.array([0.1])],
        "t": np.array([0.1, 0.2]),
        "x": np.array([0.1]),
    }
    assert loaders.validate_grid_data(data) is False


def test_preprocess_spike_data_filters():
    spikes = [np.array([0.0, 0.5, 1.0, 1.5]), np.array([0.2])]
    result = loaders.preprocess_spike_data(spikes, time_window=(0.0, 1.0), min_spike_count=2)
    assert result is not None
    assert len(result) == 1
    assert np.all(result[0] <= 1.0)


def test_preprocess_spike_data_no_valid_neurons():
    spikes = [np.array([0.1]), np.array([0.2])]
    assert loaders.preprocess_spike_data(spikes, min_spike_count=2) is None


def test_get_data_summary_roi():
    data = np.array([1.0, 2.0, 3.0])
    summary = loaders.get_data_summary(data)
    assert summary["type"] == "roi_data"
    assert summary["shape"] == (3,)
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["has_nan"] is False
    assert summary["has_inf"] is False


def test_get_data_summary_grid():
    data = {
        "spike": [np.array([0.1, 0.2]), np.array([0.3])],
        "t": np.array([0.0, 0.1, 0.2]),
        "x": np.array([0.0, 0.1, 0.2]),
        "y": np.array([1.0, 1.1, 1.2]),
    }
    summary = loaders.get_data_summary(data)
    assert summary["type"] == "grid_data"
    assert summary["n_neurons"] == 2
    assert summary["time_data"]["length"] == 3
    assert summary["x_data"]["range"] == 0.2
