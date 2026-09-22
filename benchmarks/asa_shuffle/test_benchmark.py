"""Acceptance checks must reject dropped rounds and any altered full barcode."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "asa_benchmark", Path(__file__).with_name("benchmark.py")
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def fixture_pair(tmp_path):
    dirs = [tmp_path / name for name in ("old", "new")]
    for directory in dirs:
        directory.mkdir()
        np.savez(
            directory / "round_0.npz",
            H0=np.array([[0.0, np.inf]]),
            H1=np.array([[1.0, 2.0], [1.5, 1.6]]),
            C1_0=np.array([[0, 1, 3]]),
        )
        (directory / "round_0.json").write_text(json.dumps({"offsets": [1, 2], "checkpoints": {}}))
    return dirs


def test_full_array_comparison_accepts_essential_bar(tmp_path):
    old, new = fixture_pair(tmp_path)
    assert BENCH.compare_rounds(old, new, 1, 1) == 3


def test_missing_round_rejected(tmp_path):
    old, new = fixture_pair(tmp_path)
    with pytest.raises(FileNotFoundError):
        BENCH.compare_rounds(old, new, 2, 1)


def test_changed_short_bar_rejected_even_when_maximum_unchanged(tmp_path):
    old, new = fixture_pair(tmp_path)
    np.savez(
        new / "round_0.npz",
        H0=np.array([[0.0, np.inf]]),
        H1=np.array([[1.0, 2.0], [1.5, 1.7]]),
        C1_0=np.array([[0, 1, 3]]),
    )
    with pytest.raises(AssertionError, match="Full array differs"):
        BENCH.compare_rounds(old, new, 1, 1)


def test_missing_cocycle_rejected(tmp_path):
    old, new = fixture_pair(tmp_path)
    np.savez(
        new / "round_0.npz", H0=np.array([[0.0, np.inf]]), H1=np.array([[1.0, 2.0], [1.5, 1.6]])
    )
    with pytest.raises(AssertionError, match="Missing diagrams/cocycles"):
        BENCH.compare_rounds(old, new, 1, 1)
