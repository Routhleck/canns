#!/usr/bin/env python3
"""Reopen benchmark evidence and report paired medians, without timing claims from failed jobs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from benchmark import array_digest, compare_rounds, digest, write


def check_job_coverage(plan, results):
    expected = {
        (case["name"], workers, repeat, variant)
        for case in plan["cases"]
        for workers in plan["workers"]
        for repeat in range(plan["repeats"])
        for variant in plan["variants"]
    }
    actual = [(r["case"], r["workers"], r["repeat"], r["variant"]) for r in results]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing, duplicated or unexpected benchmark jobs")
    if any(r["status"] != "complete" or r["rounds"] != plan["rounds"] for r in results):
        raise ValueError("Failed or incomplete round count")


def summarize(directories, output):
    import numpy as np

    rows = []
    reference = {}
    manifests = []
    arrays = 0
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifests.append(manifest)
        plan = manifest["plan"]
        result = json.loads((directory / "results.json").read_text())
        check_job_coverage(plan, result)
        if digest(directory / "offsets.npy") != manifest["offsets_sha256"]:
            raise ValueError("Saved shifts changed")
        offsets = np.load(directory / "offsets.npy")
        expected = (
            len(plan["cases"]) * len(plan["workers"]) * plan["repeats"] * len(plan["variants"])
        )
        if (
            len(result) != expected
            or json.loads((directory / "status.json").read_text())["status"] != "complete"
        ):
            raise ValueError(f"Incomplete suite: {directory}")
        if (
            manifest["activity_sha256"] != manifests[0]["activity_sha256"]
            or manifest["offsets_sha256"] != manifests[0]["offsets_sha256"]
        ):
            raise ValueError("Cannot combine different activities or shifts")
        for row in result:
            path = directory / f"{row['case']}_w{row['workers']}_r{row['repeat']}_{row['variant']}"
            measured = json.loads((path / "result.json").read_text())
            if any(row.get(k) != v for k, v in measured.items()):
                raise ValueError("Summary row differs from its measured job")
            case = next(c for c in plan["cases"] if c["name"] == row["case"])
            saved, settings = reference.setdefault(row["case"], (path, case["parameters"]))
            if settings != case["parameters"]:
                raise ValueError("Scientific parameters changed")
            checked = compare_rounds(saved, path, plan["rounds"], settings["maxdim"])
            if saved != path:
                arrays += checked
            stages = defaultdict(float)
            for i in range(plan["rounds"]):
                record = json.loads((path / f"round_{i}.json").read_text())
                if record["index"] != i or record["offsets"] != offsets[i].tolist():
                    raise ValueError("Round does not replay the recorded shifts")
                with np.load(path / f"round_{i}.npz") as payload:
                    actual = {key: array_digest(payload[key]) for key in payload.files}
                if actual != record["arrays"]:
                    raise ValueError("Payload differs from the recorded PH result")
                for k, v in record["stages"].items():
                    stages[k] += v
            rows.append(dict(row, **{f"summed_{k}_seconds": v for k, v in stages.items()}))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["workers"], row["variant"])].append(row)
    table = []
    for (case, workers, variant), values in sorted(groups.items()):
        table.append(
            dict(
                case=case,
                workers=workers,
                variant=variant,
                repeats=len(values),
                seconds_median=median(v["seconds"] for v in values),
                seconds_min=min(v["seconds"] for v in values),
                seconds_max=max(v["seconds"] for v in values),
                tree_pss_gib_median=median(v["peak_tree_pss_bytes"] / 2**30 for v in values),
                tree_rss_gib_median=median(v["peak_tree_rss_bytes"] / 2**30 for v in values),
            )
        )
    pairs = []
    for case, workers in sorted({(r["case"], r["workers"]) for r in rows}):
        for before, after, meaning in (
            ("legacy-default", "pr-finite", "combined opt-in workflow"),
            ("legacy-default", "pr-legacy", "same infinite threshold"),
            ("legacy-finite", "pr-finite", "same finite threshold"),
            ("pr-legacy", "pr-finite", "PH threshold within PR"),
        ):
            a = {r["repeat"]: r for r in groups.get((case, workers, before), [])}
            b = {r["repeat"]: r for r in groups.get((case, workers, after), [])}
            common = sorted(a.keys() & b.keys())
            if not common:
                continue
            ratios = [a[i]["seconds"] / b[i]["seconds"] for i in common]
            pairs.append(
                dict(
                    case=case,
                    workers=workers,
                    before=before,
                    after=after,
                    interpretation=meaning,
                    n=len(common),
                    median_speedup=median(ratios),
                    min_speedup=min(ratios),
                    max_speedup=max(ratios),
                    median_pss_reduction=median(
                        1 - b[i]["peak_tree_pss_bytes"] / a[i]["peak_tree_pss_bytes"]
                        for i in common
                    ),
                )
            )
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "summary.json",
        dict(
            jobs=len(rows),
            completed_round_executions=sum(r["rounds"] for r in rows),
            compared_arrays=arrays,
            full_array_equality=True,
            table=table,
            pairs=pairs,
            manifests=manifests,
            measurements=rows,
        ),
    )
    with (output / "measurements.csv").open("w") as stream:
        flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# Complete ASA shuffle benchmark",
        "",
        "All durations below are cold shuffle-batch wall times, including Numba JIT, pool creation,",
        "IPC, shifts, complete preprocessing and PH. Imports and input loading are excluded.",
        "Repetitions use the same recorded shifts; they are performance repetitions, not independent null draws.",
        "",
        "| Case | Workers | Variant | Median s [min, max] | Tree PSS GiB | Tree RSS GiB |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['case']} | {r['workers']} | {r['variant']} | {r['seconds_median']:.3f} [{r['seconds_min']:.3f}, {r['seconds_max']:.3f}] | {r['tree_pss_gib_median']:.3f} | {r['tree_rss_gib_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Case | Workers | Comparison | Paired median speedup [min, max] | Median PSS reduction |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for r in pairs:
        lines.append(
            f"| {r['case']} | {r['workers']} | {r['before']} → {r['after']} | {r['median_speedup']:.3f}× [{r['min_speedup']:.3f}, {r['max_speedup']:.3f}] | {r['median_pss_reduction']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Acceptance: {len(rows)} jobs; {sum(r['rounds'] for r in rows)} completed round executions; {arrays} full-array comparisons between distinct jobs passed (reference self-checks excluded).",
            "All H0/H1/H2 arrays applicable to each case, including short and essential bars and all returned cocycles, were reopened and compared exactly.",
            "Offsets, PCA output, selected indices and final distance-graph hashes also match.",
            "",
            "The combined comparison includes the opt-in finite PH threshold. It is not a Rust-only speedup.",
            "PSS is sampled every 0.2 s, RSS every approximately 0.02 s across the process tree; peaks are sampled lower bounds.",
            "Summed RSS counts shared fork pages more than once. PSS apportions shared pages and is the primary memory comparison.",
            "Stage times are sums across rounds and can exceed wall time when workers overlap. See measurements.csv.",
            "These are one-host measurements; repetition and fixed-shift counts are recorded in the plans. No universal speedup or significance claim is made.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.runs, args.output)
