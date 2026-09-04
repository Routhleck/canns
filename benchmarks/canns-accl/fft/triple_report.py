"""Three-platform comparison: Mac CPU vs Linux server CPU vs A100 GPU.

Reads the per-platform speed/accuracy CSVs and emits:
  - results/cann_fft_triple_summary.md  — per-(model, n) wall-time table
  - figures/fig_fft_triple_speed.png    — per-step time, three platforms
  - figures/fig_fft_triple_scan.png     — per-step inside T=200 scan
  - figures/fig_fft_triple_speedup.png  — speedup vs Mac CPU (3 platforms)

Hardware:
  - Mac CPU:   Apple M4, 10 cores, ARM
  - Server CPU: Intel Xeon Gold 6348 @ 2.6 GHz, 16 cores visible, AVX-512
  - GPU:       NVIDIA A100-SXM4-80GB (Ampere, TF32 by default)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_FIGS = _HERE / "figures"


def load_csv(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


# Backend display labels
BACKEND_LABELS = {
    "dense": "dense",
    "fft": "FFT (exact)",
    "svd_k1": "SVD k=1",
    "svd_k4": "SVD k=4",
    "svd_k16": "SVD k=16",
}

# Platform display labels
PLATFORM_LABELS = {
    "maccpu": "Mac M4 CPU",
    "servercpu": "Server Intel Xeon CPU",
    "gpu": "A100 GPU",
}

PLATFORM_COLORS = {
    "maccpu": "C0",
    "servercpu": "C1",
    "gpu": "C3",
}

PLATFORM_MARKERS = {
    "maccpu": "o",
    "servercpu": "s",
    "gpu": "^",
}


def build_summary(platforms: dict[str, list[dict]]) -> str:
    """Build a per-(model, n, backend) table of wall times across
    platforms, plus per-platform speedup over Mac CPU."""

    # Group by (model, n_param, backend)
    by_cell: dict[tuple[str, int, str], dict[str, dict]] = defaultdict(dict)
    for plat, rows in platforms.items():
        for r in rows:
            key = (r["model"], int(r["n_param"]), r["backend"])
            by_cell[key][plat] = r

    plat_names = list(platforms.keys())

    md = ["# FFT matvec — three-platform wall time\n"]
    md.append("**Hardware**\n")
    md.append("- **Mac CPU**: Apple M4, 10 cores, ARM64")
    md.append("- **Server CPU**: Intel Xeon Gold 6348 @ 2.6 GHz, 16 cores, AVX-512 (Linux)")
    md.append("- **GPU**: NVIDIA A100-SXM4-80GB (Ampere; cuBLAS uses TF32 by default)\n")

    md.append(
        "All numbers are median per-step wall time (ms) for a single "
        "recurrent matvec, after JIT warmup. Lower is better. "
        "`scan` is the per-step time inside a `lax.scan` of T=200 "
        "repeated matvecs (better reflects rollout cost).\n"
    )

    # Per (model) per (n) per (backend) — one table
    for model in ("cann1d", "cann2d"):
        md.append(f"## {model.upper()}\n")
        n_params = sorted({k[1] for k in by_cell if k[0] == model})
        backends = ("dense", "fft", "svd_k1", "svd_k4", "svd_k16")
        # Per-step table
        header = (
            ["n"]
            + [f"{PLATFORM_LABELS[p]} step" for p in plat_names]
            + [f"{PLATFORM_LABELS[p]} scan" for p in plat_names]
            + ["speedup over Mac (step)"]
            + ["speedup over Mac (scan)"]
        )
        md.append("### per-step (ms) and T=200 scan (ms)\n")
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for n_p in n_params:
            for backend in backends:
                key = (model, n_p, backend)
                if key not in by_cell:
                    continue
                row_data = by_cell[key]
                step_times = [
                    float(row_data[p]["per_step_ms"]) if p in row_data else float("nan")
                    for p in plat_names
                ]
                scan_times = [
                    float(row_data[p]["scan_per_step_ms"]) if p in row_data else float("nan")
                    for p in plat_names
                ]
                # Speedup over Mac CPU (maccpu)
                mac_step = (
                    step_times[plat_names.index("maccpu")]
                    if "maccpu" in plat_names
                    else float("nan")
                )
                mac_scan = (
                    scan_times[plat_names.index("maccpu")]
                    if "maccpu" in plat_names
                    else float("nan")
                )
                step_su = [mac_step / t if t > 0 else float("nan") for t in step_times]
                scan_su = [mac_scan / t if t > 0 else float("nan") for t in scan_times]
                row = [f"n={n_p} {BACKEND_LABELS[backend]}"]
                for t in step_times:
                    row.append(f"{t:.4f}")
                for t in scan_times:
                    row.append(f"{t:.4f}")
                for su in step_su:
                    row.append(f"{su:.2f}×")
                for su in scan_su:
                    row.append(f"{su:.2f}×")
                md.append("| " + " | ".join(row) + " |")
        md.append("")
    return "\n".join(md)


def fig_triple_speed(platforms: dict[str, list[dict]], out: Path):
    """Per-step time, all 3 platforms, dense vs FFT, 1D and 2D."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    titles = [("cann1d", "1D CANN"), ("cann2d", "2D CANN")]
    for ax, (m, title) in zip(axes, titles):
        for backend, ls, ms_size in [("dense", "-", 6), ("fft", "--", 5)]:
            for plat in platforms:
                rows = platforms[plat]
                by_n = {}
                for r in rows:
                    if r["model"] == m and r["backend"] == backend:
                        n_p = int(r["n_param"])
                        n_t = int(r["n_total"])
                        by_n[n_p] = (n_t, float(r["per_step_ms"]))
                sizes = sorted(by_n.keys())
                ns = [by_n[s][0] for s in sizes]
                ts = [by_n[s][1] for s in sizes]
                ax.loglog(
                    ns,
                    ts,
                    ls + PLATFORM_MARKERS[plat],
                    color=PLATFORM_COLORS[plat],
                    label=f"{PLATFORM_LABELS[plat]} {BACKEND_LABELS[backend]}",
                    lw=1.2,
                    ms=ms_size,
                    alpha=0.85,
                )
        ax.set_xlabel("n" if m == "cann1d" else "n (L²)")
        ax.set_ylabel("per-step time (ms)")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.3)
        ax.legend(loc="lower right", fontsize=6.5, ncol=1)
    fig.suptitle("Per-step wall time — Mac M4 CPU vs Intel Xeon CPU vs A100 GPU", y=1.04)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_triple_scan(platforms: dict[str, list[dict]], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    titles = [("cann1d", "1D CANN (T=200 scan)"), ("cann2d", "2D CANN (T=200 scan)")]
    for ax, (m, title) in zip(axes, titles):
        for backend, ls, ms_size in [("dense", "-", 6), ("fft", "--", 5)]:
            for plat in platforms:
                rows = platforms[plat]
                by_n = {}
                for r in rows:
                    if r["model"] == m and r["backend"] == backend:
                        n_p = int(r["n_param"])
                        n_t = int(r["n_total"])
                        by_n[n_p] = (n_t, float(r["scan_per_step_ms"]))
                sizes = sorted(by_n.keys())
                ns = [by_n[s][0] for s in sizes]
                ts = [by_n[s][1] for s in sizes]
                ax.loglog(
                    ns,
                    ts,
                    ls + PLATFORM_MARKERS[plat],
                    color=PLATFORM_COLORS[plat],
                    label=f"{PLATFORM_LABELS[plat]} {BACKEND_LABELS[backend]}",
                    lw=1.2,
                    ms=ms_size,
                    alpha=0.85,
                )
        ax.set_xlabel("n" if m == "cann1d" else "n (L²)")
        ax.set_ylabel("per-step time inside T=200 scan (ms)")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.3)
        ax.legend(loc="lower right", fontsize=6.5, ncol=1)
    fig.suptitle("Per-step inside T=200 lax.scan — three platforms", y=1.04)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_triple_speedup(platforms: dict[str, list[dict]], out: Path):
    """Speedup of each platform relative to Mac M4 CPU (the reference)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    titles = [("cann1d", "1D CANN"), ("cann2d", "2D CANN")]
    mac = "maccpu"
    for ax, (m, title) in zip(axes, titles):
        # Get Mac CPU per-(n, backend) for reference
        mac_data = {}
        for r in platforms[mac]:
            if r["model"] == m:
                mac_data[(int(r["n_param"]), r["backend"])] = float(r["per_step_ms"])
        for backend, ls, ms_size in [("dense", "-", 6), ("fft", "--", 5)]:
            for plat in platforms:
                if plat == mac:
                    continue
                rows = platforms[plat]
                by_n = {}
                for r in rows:
                    if r["model"] == m and r["backend"] == backend:
                        n_p = int(r["n_param"])
                        n_t = int(r["n_total"])
                        mac_t = mac_data.get((n_p, backend), float("nan"))
                        plat_t = float(r["per_step_ms"])
                        by_n[n_p] = (n_t, mac_t / plat_t if plat_t > 0 else float("nan"))
                sizes = sorted(by_n.keys())
                ns = [by_n[s][0] for s in sizes]
                ts = [by_n[s][1] for s in sizes]
                ax.loglog(
                    ns,
                    ts,
                    ls + PLATFORM_MARKERS[plat],
                    color=PLATFORM_COLORS[plat],
                    label=f"{PLATFORM_LABELS[plat]} {BACKEND_LABELS[backend]}",
                    lw=1.2,
                    ms=ms_size,
                    alpha=0.85,
                )
        ax.axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.5, label="Mac M4 baseline (1×)")
        ax.set_xlabel("n" if m == "cann1d" else "n (L²)")
        ax.set_ylabel("speedup vs Mac M4 CPU")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.3)
        ax.legend(loc="upper left", fontsize=6.5, ncol=1)
    fig.suptitle("Per-step speedup over Mac M4 CPU — higher is better", y=1.04)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    platforms = {
        "maccpu": load_csv(_RESULTS / "cann_fft_speed_maccpu.csv"),
        "servercpu": load_csv(_RESULTS / "cann_fft_speed_servercpu.csv"),
        "gpu": load_csv(_RESULTS / "cann_fft_speed_gpu.csv"),
    }
    for k, v in platforms.items():
        print(f"# {k}: {len(v)} rows")
    md = build_summary(platforms)
    out = _RESULTS / "cann_fft_triple_summary.md"
    out.write_text(md)
    print(f"# wrote {out}")

    fig_triple_speed(platforms, _FIGS / "fig_fft_triple_speed.png")
    fig_triple_scan(platforms, _FIGS / "fig_fft_triple_scan.png")
    fig_triple_speedup(platforms, _FIGS / "fig_fft_triple_speedup.png")
    print("# wrote figures/fig_fft_triple_{speed,scan,speedup}.png")


if __name__ == "__main__":
    main()
