#!/usr/bin/env python3
"""Plot the measured main comparison; control comparisons stay in RESULTS.md."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(summary, output):
    data = json.loads(summary.read_text())
    groups = sorted({(r["case"], r["workers"]) for r in data["table"]})
    selected = {(r["case"], r["workers"], r["variant"]): r for r in data["table"]}
    x = np.arange(len(groups))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for shift, variant, label, color in [
        (-width / 2, "legacy-default", "Original Python Pool / infinite PH", "#737B87"),
        (width / 2, "pr-finite", "PR / Rust sampling / finite PH", "#187DA8"),
    ]:
        rows = [selected[(*g, variant)] for g in groups]
        values = np.array([r["seconds_median"] for r in rows])
        errors = np.array(
            [
                [r["seconds_median"] - r["seconds_min"] for r in rows],
                [r["seconds_max"] - r["seconds_median"] for r in rows],
            ]
        )
        bars = axes[0].bar(
            x + shift, values, width, label=label, color=color, yerr=errors, capsize=3
        )
        axes[0].bar_label(bars, fmt="%.1f", padding=5, fontsize=9)
        bars = axes[1].bar(x + shift, [r["tree_pss_gib_median"] for r in rows], width, color=color)
        axes[1].bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    axes[0].set_ylabel("Batch wall time (seconds)")
    axes[1].set_ylabel("Peak process-tree PSS (GiB)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_xticks(
        x, [g[0].replace("grid1_", "").replace("_", " / ") + f"\n{g[1]} worker(s)" for g in groups]
    )
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.23)
    fig.suptitle("grid_1 complete ASA shuffle: time and memory", fontsize=15)
    fig.supxlabel(
        "Median of 3 cold batches; 4 fixed shifts/batch. Time error bars: min–max.\n"
        "Combined opt-in comparison; includes threshold changes. See matched-threshold controls.",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.output)
