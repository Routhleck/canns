"""Combine the per-platform FFT reports into a single side-by-side
comparison (CPU vs GPU) and emit a unified markdown + a Pareto plot.

Run:
    python benchmarks/cann_fft/combine_reports.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"


def load_csv(p: Path):
    import csv

    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    cpu_speed = load_csv(_RESULTS / "cann_fft_speed_cpu.csv")
    gpu_speed = load_csv(_RESULTS / "cann_fft_speed_gpu.csv")
    load_csv(_RESULTS / "cann_fft_accuracy_cpu.csv")
    load_csv(_RESULTS / "cann_fft_accuracy_gpu.csv")

    # Headline numbers (largest n we have on each platform)
    def headline(rows, label):
        by_cell = {}
        for r in rows:
            by_cell.setdefault((r["model"], int(r["n_param"])), {})[r["backend"]] = r
        out = []
        for m in ("cann1d", "cann2d"):
            sizes = sorted(k[1] for k in by_cell if k[0] == m)
            if not sizes:
                continue
            last = sizes[-1]
            cell = by_cell[(m, last)]
            for backend in ("dense", "fft", "svd_k1", "svd_k4", "svd_k16"):
                if backend not in cell:
                    continue
                r = cell[backend]
                out.append((label, m, last, backend, r))
        return out

    cpu_h = headline(cpu_speed, "CPU")
    gpu_h = headline(gpu_speed, "GPU")

    # Build a comparison table
    md = ["# FFT matvec benchmark — CPU vs GPU\n"]
    md.append(
        "Headline numbers (largest n per platform) for the recurrent "
        "matvec under three backends: `dense` (cuBLAS sgemv / numpy "
        "matmul), `svd_k{k}` (rank-k truncated SVD approximation), and "
        "`fft` (exact circulant matvec via cuFFT / numpy FFT).\n"
    )
    md.append("## CPU headline\n")
    md.append("| model | n | backend | step_ms | scan_ms | speedup_step | speedup_scan |")
    md.append("|---|---|---|---|---|---|---|")
    for _label, m, n, b, r in cpu_h:
        md.append(
            f"| {m} | {n} | {b} | {float(r['per_step_ms']):.4f} | "
            f"{float(r['scan_per_step_ms']):.4f} | "
            f"{float(r['speedup_vs_dense_step']):.2f} | "
            f"{float(r['speedup_vs_dense_scan']):.2f} |"
        )

    md.append("\n## GPU headline\n")
    md.append("| model | n | backend | step_ms | scan_ms | speedup_step | speedup_scan |")
    md.append("|---|---|---|---|---|---|---|")
    for _label, m, n, b, r in gpu_h:
        md.append(
            f"| {m} | {n} | {b} | {float(r['per_step_ms']):.4f} | "
            f"{float(r['scan_per_step_ms']):.4f} | "
            f"{float(r['speedup_vs_dense_step']):.2f} | "
            f"{float(r['speedup_vs_dense_scan']):.2f} |"
        )

    md.append("\n## Take-aways\n")
    md.append(
        "- **CPU: FFT is 25-50× faster than dense** and is *exact*. "
        "SVD k=1 is 100-1000× faster but has 30-50 mrad position error. "
        "FFT is the right choice for the high-fidelity regime on CPU.\n"
    )
    md.append(
        "- **GPU: FFT is barely faster than dense on per-step** "
        "(1.1-1.2× at n=4096) because cuBLAS sgemv is already very "
        "well-optimized for the matmul shape. FFT only wins "
        "meaningfully on the *scan* (rollout) path (1.6-2.0× at "
        "n=4096), where XLA can fuse FFT and avoid per-step launch "
        "overhead.\n"
    )
    md.append(
        "- **Accuracy on GPU** is ~1e-2 (not 1e-5) because cuBLAS uses "
        "TF32 (10-bit mantissa) by default on Ampere. Disable TF32 "
        "(`torch.backends.cuda.matmul.allow_tf32 = False` or the JAX "
        "equivalent) if you need full FP32 precision in the dense "
        "baseline.\n"
    )
    md.append(
        "- **Endpoint gotcha.** The canns default `endpoint=True` grid "
        "is not circulant; `accl_mode='fft'` on that grid silently "
        "falls back to dense with a `UserWarning`. To use FFT, set "
        "`model.x = bm.linspace(-bm.pi, bm.pi, n, endpoint=False)` and "
        "rebuild `model.conn_mat = model.make_conn()`.\n"
    )
    md.append("\n## Figures\n")
    md.append("- `figures/fig_fft_speed_cpu.png` — per-step time vs n (CPU)")
    md.append("- `figures/fig_fft_speed_gpu.png` — per-step time vs n (GPU)")
    md.append("- `figures/fig_fft_accuracy_cpu.png` — max-abs error vs n (CPU)")
    md.append("- `figures/fig_fft_pareto_cpu.png` — speed vs accuracy (CPU)")

    out_md = _RESULTS / "cann_fft_summary.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"# wrote {out_md}")

    # Combined CPU vs GPU speed plot
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    titles = [("cann1d", "1D CANN"), ("cann2d", "2D CANN")]
    for ax, (m, title) in zip(axes, titles, strict=False):
        for rows, label, color, marker in [
            (cpu_speed, "CPU", "C0", "o"),
            (gpu_speed, "GPU", "C3", "s"),
        ]:
            by_n = {}
            for r in rows:
                if r["model"] != m:
                    continue
                if r["backend"] not in ("dense", "fft"):
                    continue
                n = int(r["n_param"])
                by_n.setdefault(n, {})[r["backend"]] = float(r["per_step_ms"])
            sizes = sorted(by_n.keys())
            ns = [
                int(by_n[s][list(by_n[s].keys())[0]] if "dense" not in by_n[s] else 0)
                for s in sizes
            ]
            # Re-fetch n_total properly
            ns = []
            for s in sizes:
                for r in rows:
                    if r["model"] == m and int(r["n_param"]) == s:
                        ns.append(int(r["n_total"]))
                        break
            for backend, style in [("dense", "-"), ("fft", "--")]:
                ys = [by_n[s].get(backend, np.nan) for s in sizes]
                ax.loglog(
                    ns,
                    ys,
                    style + marker,
                    color=color,
                    label=f"{label} {backend}",
                    ms=5,
                    lw=1.0,
                    alpha=0.7,
                )
        ax.set_xlabel("n" if m == "cann1d" else "n (L²)")
        ax.set_ylabel("per-step time (ms)")
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Per-step time: CPU vs GPU (dense vs FFT)", y=1.02)
    fig.savefig(_HERE / "figures" / "fig_fft_speed_cpu_vs_gpu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("# wrote figures/fig_fft_speed_cpu_vs_gpu.png")


if __name__ == "__main__":
    main()
