#!/usr/bin/env python3
"""Isolated complete-ASA shuffle comparison against the original mp.Pool path.

No shortcut null model is timed. See README.md for baseline provenance,
instrumentation, cold-start semantics, and finite-threshold controls.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
import types
from pathlib import Path

STATE = {}
LOCAL = threading.local()
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(array):
    import numpy as np

    array = np.asarray(array)
    return dict(
        shape=list(array.shape),
        dtype=str(array.dtype),
        sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    )


def load_tda(source):
    # Reuse installed visualization dependencies without editing either source.
    importlib.import_module("canns.analyzer.visualization")
    parent = "canns.analyzer.shuffle_benchmark"
    package = parent + ".asa"
    for name in (parent, package):
        module = types.ModuleType(name)
        module.__path__ = [str(source)]
        sys.modules[name] = module
    for name in ("config", "tda"):
        full = package + "." + name
        spec = importlib.util.spec_from_file_location(full, Path(source) / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
    return sys.modules[package + ".tda"]


def legacy_worker(args):
    """Replay independent recorded shifts through the original worker body."""
    import numpy as np

    index, _, _ = args
    LOCAL.index = index
    LOCAL.stages = {}
    LOCAL.checkpoints = {}
    np.random.seed(STATE["seed"] + index)
    result = STATE["original_worker"](args)
    # The historical worker swallows failures; never accept a dropped round.
    if not (STATE["out"] / f"round_{index}.npz").is_file():
        raise RuntimeError(f"Legacy round {index} failed; retain its log")
    return result


def install_instrumentation(tda, out, finite, candidate, offsets):
    import numpy as np

    if candidate:
        original_shift = tda._shuffle_spike_trains
        lookup = {tuple(row): i for i, row in enumerate(offsets)}
        if len(lookup) != len(offsets):
            raise ValueError("Benchmark offsets must identify rounds uniquely")

        def shifted(activity, shifts=None):
            LOCAL.index = lookup[tuple(shifts)]
            LOCAL.stages = {}
            LOCAL.checkpoints = {}
            return original_shift(activity, shifts)

        tda._shuffle_spike_trains = shifted

    for name, key in (
        ("_pca", "pca"),
        ("_sample_denoising", "sampling"),
        ("_second_build", "graph"),
    ):
        original = getattr(tda, name)

        def instrument(*args, _fn=original, _key=key, **kwargs):
            start = time.perf_counter()
            result = _fn(*args, **kwargs)
            LOCAL.stages[_key] = time.perf_counter() - start
            value = result if _key == "graph" else result[0]
            LOCAL.checkpoints[_key] = array_digest(value)
            return result

        setattr(tda, name, instrument)

    original_ripser = tda.ripser

    def ripser(distance, **kwargs):
        # Only the explicitly named legacy-finite control injects a PH cutoff.
        if finite and not candidate:
            values = distance[np.isfinite(distance)].astype(np.float32)
            kwargs["thresh"] = float(values.max())
        start = time.perf_counter()
        result = original_ripser(distance, **kwargs)
        LOCAL.stages["ph"] = time.perf_counter() - start
        arrays = {f"H{i}": d for i, d in enumerate(result["dgms"])}
        for dim, cycles in enumerate(result["cocycles"]):
            for i, cycle in enumerate(cycles):
                arrays[f"C{dim}_{i}"] = cycle
        index = LOCAL.index
        np.savez(out / f"round_{index}.npz", **arrays)
        write(
            out / f"round_{index}.json",
            dict(
                index=index,
                offsets=offsets[index].tolist(),
                stages=LOCAL.stages,
                checkpoints=LOCAL.checkpoints,
                arrays={key: array_digest(a) for key, a in arrays.items()},
                threshold="inf"
                if not np.isfinite(kwargs.get("thresh", np.inf))
                else kwargs["thresh"],
            ),
        )
        return result

    tda.ripser = ripser


def worker(spec):
    import numpy as np

    out = Path(spec["out"])
    candidate = spec["variant"].startswith("pr-")
    finite = spec["variant"].endswith("finite")
    site = spec["candidate_site"] if candidate else spec["legacy_lib"]
    sys.path.insert(0, site)
    source = (Path(site) / "canns/analyzer/data/asa") if candidate else Path(spec["legacy_asa"])
    tda = load_tda(source)
    activity = np.load(spec["activity"])
    offsets = np.load(spec["offsets"])
    STATE.update(out=out, seed=spec["seed"])
    if not candidate:
        # Linux historical default; fork before any parallel Numba warm-up.
        mp.set_start_method("fork", force=True)
        STATE["original_worker"] = tda._process_single_shuffle
        tda._process_single_shuffle = legacy_worker
    install_instrumentation(tda, out, finite, candidate, offsets)
    import canns_lib

    identities = [
        source / "tda.py",
        source / "config.py",
        Path(canns_lib.__file__),
        *Path(canns_lib.__file__).parent.rglob("*.so"),
    ]
    write(
        out / "identity.json",
        dict(
            source_sha256={str(p): digest(p) for p in identities},
            versions={
                n: importlib.metadata.version(n)
                for n in ("numpy", "scipy", "numba", "scikit-learn", "canns-lib")
            },
            python=sys.version,
            platform=platform.platform(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            environment={k: os.environ.get(k) for k in (*THREAD_ENV, "NUMBA_THREADING_LAYER")},
        ),
    )
    parameters = dict(spec["parameters"], progress_bar=False)
    # Imports/loading excluded; JIT, pool creation, IPC, shifts and all rounds included.
    write(out / "ready.json", dict(pid=os.getpid()))
    while not (out / "go").exists():
        time.sleep(0.01)
    start = time.perf_counter()
    if candidate:
        config = tda.TDAConfig(
            **parameters,
            sampling_backend="rust",
            shuffle_shifts=offsets,
            ph_threshold_policy="max_finite_float32" if finite else "legacy",
            shuffle_workers=spec["workers"],
            num_shuffles=len(offsets),
            do_cocycles=True,
            standardize=True,
            show=False,
        )
        maxima = tda._run_shuffle_analysis(activity, config=config)
    else:
        maxima = tda._run_shuffle_analysis(
            activity, num_shuffles=len(offsets), num_cores=spec["workers"], **parameters
        )
    elapsed = time.perf_counter() - start
    for index in range(len(offsets)):
        if not (out / f"round_{index}.npz").is_file():
            raise RuntimeError(f"Missing round {index}")
    empty_counts = {}
    for dim in range(parameters["maxdim"] + 1):
        expected = []
        empty_counts[dim] = 0
        for index in range(len(offsets)):
            with np.load(out / f"round_{index}.npz") as z:
                bars = z[f"H{dim}"]
                bars = bars[np.isfinite(bars[:, 1])]
                if not len(bars):
                    empty_counts[dim] += 1
                    if candidate:
                        expected.append(0.0)
                else:
                    # Match the historical collector's native diagram dtype.
                    expected.append(float(np.max(bars[:, 1] - bars[:, 0])))
        observed = maxima.get(dim, [])
        if len(observed) != len(expected) or not np.allclose(
            observed, expected, rtol=2e-7, atol=5e-7
        ):
            raise RuntimeError(f"Returned maxima disagree with saved full H{dim} diagrams")
    write(
        out / "result.json",
        dict(
            status="complete",
            seconds=elapsed,
            rounds=len(offsets),
            empty_finite_counts=empty_counts,
            maxima={str(k): [float(v) for v in values] for k, values in maxima.items()},
        ),
    )


def compare_rounds(reference, candidate, count, maxdim):
    """Independently reopen all bars/cocycles; fail on absent rounds or fields."""
    import numpy as np

    comparisons = 0
    for i in range(count):
        with (
            np.load(Path(reference) / f"round_{i}.npz") as left,
            np.load(Path(candidate) / f"round_{i}.npz") as right,
        ):
            if set(left.files) != set(right.files) or not all(
                f"H{d}" in left for d in range(maxdim + 1)
            ):
                raise AssertionError(f"Missing diagrams/cocycles in round {i}")
            for key in left.files:
                if array_digest(left[key]) != array_digest(right[key]):
                    raise AssertionError(f"Full array differs: round {i}, {key}")
                comparisons += 1
        a = json.loads((Path(reference) / f"round_{i}.json").read_text())
        b = json.loads((Path(candidate) / f"round_{i}.json").read_text())
        for key in ("offsets", "checkpoints"):
            if a[key] != b[key]:
                raise AssertionError(f"Round {i}: {key} differs")
    return comparisons


def supervise(spec, timeout, max_tree_rss_gib=64):
    import psutil

    out = Path(spec["out"])
    out.mkdir()
    write(out / "spec.json", spec)
    env = dict(
        os.environ, **{k: "1" for k in THREAD_ENV}, NUMBA_THREADING_LAYER="omp", MPLBACKEND="Agg"
    )
    env.pop("PYTHONPATH", None)
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, __file__, "--worker", str(out / "spec.json")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        root = psutil.Process(process.pid)
        start = time.monotonic()
        peak_rss = peak_pss = 0
        last_pss = 0
        samples = 0
        while process.poll() is None:
            if time.monotonic() - start > timeout:
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                write(out / "failure.json", {"reason": "timeout", "seconds": timeout})
                raise RuntimeError(f"Timed out; evidence kept at {out}")
            if (out / "ready.json").exists() and not (out / "go").exists():
                (out / "go").touch()
            if (out / "go").exists() and not (out / "result.json").exists():
                rss = pss = 0
                collect_pss = time.monotonic() - last_pss >= 0.2
                for p in [root, *root.children(recursive=True)]:
                    try:
                        rss += p.memory_info().rss
                        if collect_pss:
                            pss += p.memory_full_info().pss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak_rss = max(peak_rss, rss)
                if rss > max_tree_rss_gib * 2**30:
                    import signal

                    # Stop only this benchmark's isolated process group.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    write(
                        out / "failure.json",
                        {
                            "reason": "process-tree RSS limit",
                            "rss_bytes": rss,
                            "limit_gib": max_tree_rss_gib,
                        },
                    )
                    raise RuntimeError(f"Memory limit exceeded; evidence kept at {out}")
                if collect_pss:
                    peak_pss = max(peak_pss, pss)
                    last_pss = time.monotonic()
                samples += 1
            time.sleep(0.02)
        if process.returncode:
            raise RuntimeError(f"Worker exited {process.returncode}; see {out / 'run.log'}")
    result = json.loads((out / "result.json").read_text())
    result.update(
        peak_tree_rss_bytes=peak_rss,
        peak_tree_pss_bytes=peak_pss,
        samples=samples,
        total_subprocess_seconds=time.monotonic() - start,
    )
    write(out / "result.json", result)
    return result


def suite(args):
    import numpy as np

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    plan.update(
        legacy_asa=str(args.legacy_asa.resolve()),
        legacy_lib=str(args.legacy_lib.resolve()),
        candidate_site=str(args.candidate_site.resolve()),
    )
    activity = (
        np.load(args.activity)
        if args.activity
        else np.random.RandomState(21).normal(size=(300, 20))
    )
    if activity.ndim != 2 or not np.isfinite(activity).all():
        raise ValueError("Activity must be a finite (time, neurons) matrix")
    np.save(out / "activity.npy", activity)
    offsets = np.array(
        [
            np.random.RandomState(plan["seed"] + i).randint(0, len(activity), activity.shape[1])
            for i in range(plan["rounds"])
        ],
        dtype=np.int64,
    )
    np.save(out / "offsets.npy", offsets)
    write(
        out / "manifest.json",
        dict(
            plan=plan,
            shape=list(activity.shape),
            activity_sha256=digest(out / "activity.npy"),
            offsets_sha256=digest(out / "offsets.npy"),
            harness_sha256=digest(__file__),
            host=platform.platform(),
            cpu=platform.processor(),
        ),
    )
    results = []
    references = {}
    for case in plan["cases"]:
        for workers in plan["workers"]:
            for repeat in range(plan["repeats"]):
                variants = list(plan["variants"])
                # Alternate order to reduce systematic order effects.
                if repeat % 2:
                    variants.reverse()
                for variant in variants:
                    name = f"{case['name']}_w{workers}_r{repeat}_{variant}"
                    spec = dict(
                        plan,
                        parameters=case["parameters"],
                        workers=workers,
                        variant=variant,
                        out=str(out / name),
                        activity=str(out / "activity.npy"),
                        offsets=str(out / "offsets.npy"),
                    )
                    print(name, flush=True)
                    result = supervise(spec, args.timeout, args.max_tree_rss_gib)
                    reference = references.setdefault(case["name"], spec["out"])
                    comparisons = compare_rounds(
                        reference, spec["out"], plan["rounds"], case["parameters"]["maxdim"]
                    )
                    row = dict(
                        case=case["name"],
                        workers=workers,
                        repeat=repeat,
                        variant=variant,
                        comparisons=comparisons,
                        **result,
                    )
                    results.append(row)
                    write(out / "results.json", results)
                    print(json.dumps(row), flush=True)
    write(out / "status.json", dict(status="complete", jobs=len(results)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--legacy-asa", type=Path)
    parser.add_argument("--legacy-lib", type=Path)
    parser.add_argument("--candidate-site", type=Path)
    parser.add_argument("--activity", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-tree-rss-gib", type=float, default=64)
    args = parser.parse_args()
    if args.worker:
        spec = json.loads(args.worker.read_text())
        try:
            worker(spec)
        except BaseException:
            write(Path(spec["out"]) / "failure.json", dict(traceback=traceback.format_exc()))
            raise
    else:
        suite(args)


if __name__ == "__main__":
    main()
