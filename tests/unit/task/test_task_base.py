from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from canns.task._base import Task


@dataclass
class DummyData:
    x: np.ndarray
    y: np.ndarray


class DummyTask(Task):
    def get_data(self) -> None:
        self.data = DummyData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]))

    def show_data(self, show=True, save_path=None) -> None:  # pragma: no cover - no-op display
        return None


def test_save_data_requires_data(tmp_path):
    task = DummyTask()
    with pytest.raises(ValueError):
        task.save_data(str(tmp_path / "out.npz"))


def test_save_load_dataclass_roundtrip(tmp_path):
    task = DummyTask(data_class=DummyData)
    task.data = DummyData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]))
    path = tmp_path / "data" / "sample.npz"
    task.save_data(str(path))

    reloaded = DummyTask(data_class=DummyData)
    reloaded.load_data(str(path))

    assert isinstance(reloaded.data, DummyData)
    assert np.allclose(reloaded.data.x, task.data.x)
    assert np.allclose(reloaded.data.y, task.data.y)


def test_save_load_dict_roundtrip(tmp_path):
    task = DummyTask()
    task.data = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}
    path = tmp_path / "data.npz"
    task.save_data(str(path))

    reloaded = DummyTask()
    reloaded.load_data(str(path))

    assert isinstance(reloaded.data, dict)
    assert np.allclose(reloaded.data["a"], task.data["a"])
    assert np.allclose(reloaded.data["b"], task.data["b"])


def test_load_data_missing_file(tmp_path):
    task = DummyTask()
    with pytest.raises(ValueError):
        task.load_data(str(tmp_path / "missing.npz"))
