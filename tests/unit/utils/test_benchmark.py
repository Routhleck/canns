from __future__ import annotations

import time

from canns.utils import benchmark


def test_benchmark_decorator_runs_and_returns(monkeypatch, capsys):
    calls = {"count": 0}

    def add_one(x: int) -> int:
        calls["count"] += 1
        return x + 1

    timestamps = iter([0.0, 0.1, 0.2, 0.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(timestamps))

    wrapped = benchmark(runs=2)(add_one)
    assert wrapped(1) == 2
    assert calls["count"] == 3

    output = capsys.readouterr().out
    assert "Start Benchmark" in output
    assert "Benchmark Results" in output
