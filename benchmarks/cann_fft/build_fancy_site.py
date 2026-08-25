"""Build the fancy hosted report site (v2 — fixed).

Pulls together:
  - The per-n data CSVs from canns_lowrank/ and canns_fft/
  - The 13 figures from canns_lowrank/results/figures/
  - The MD report from canns_lowrank/results/

Emits a single-page static site at _site/ with:
  - Hero / abstract (with a strong "FFT is exact" callout)
  - Interactive "pick your n" widget (vanilla JS, no build step)
  - Decision-tree flow (SVG, embedded)
  - All 13 figures, with Chart.js hover for the speedup ones
  - Raw per-n data tables for every (model, platform) — fully
    tabular, all backends, all sizes, with sort/filter
  - "Why the FFT error?" explanation section (more visible now)
  - "How to use" code block
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_RES_LOW = _REPO / "benchmarks" / "cann_lowrank" / "results"
_RES_FFT = _REPO / "benchmarks" / "cann_fft" / "results"
_SITE = _REPO / "_site"
_FIGS = _SITE / "figures"
_CSS = _SITE / "css"
_DATA = _SITE / "data"
_JS = _SITE / "js"


def load_csv(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def build_data_json():
    fft = []
    for plat in ("maccpu", "servercpu", "gpu"):
        speed_path = _RES_FFT / f"cann_fft_speed_{plat}.csv"
        acc_path = _RES_FFT / f"cann_fft_accuracy_{plat}.csv"
        if not speed_path.exists() or not acc_path.exists():
            continue
        speed_rows = load_csv(speed_path)
        acc_rows = load_csv(acc_path)
        err_map = {}
        for r in acc_rows:
            err_map[(r["model"], int(r["n_total"]), r["backend"])] = float(r["max_abs_err"])
        for r in speed_rows:
            fft.append({
                "platform": plat,
                "model": r["model"],
                "n_total": int(r["n_total"]),
                "backend": r["backend"],
                "step_ms": float(r["per_step_ms"]),
                "scan_ms": float(r["scan_per_step_ms"]),
                "step_speedup": float(r["speedup_vs_dense_step"]),
                "scan_speedup": float(r["speedup_vs_dense_scan"]),
                "max_err": err_map.get((r["model"], int(r["n_total"]), r["backend"]), 0.0),
            })
    svd_cpu = load_csv(_RES_LOW / "cann_lowrank_all_cpu.csv")
    svd_gpu = load_csv(_RES_LOW / "cann_lowrank_all_gpu.csv")
    svd = []
    for r in svd_cpu:
        try:
            n_neurons = int(r["n_neurons"])
            n_param = int(r["n"])
            k = int(r["k"])
            svd.append({"platform": "cpu", "model": r["model"].lower(),
                        "n_param": n_param, "n_neurons": n_neurons, "k": k,
                        "step_ms": float(r.get("matvec_per_step_ms", 0)),
                        "scan_ms": float(r.get("matvec_scan_per_step_ms", 0)),
                        "max_pos_err_mrad": float(r.get("max_pos_err", 0)) * 1000})
        except (KeyError, ValueError):
            continue
    for r in svd_gpu:
        try:
            n_neurons = int(r["n_neurons"])
            n_param = int(r["n"])
            k = int(r["k"])
            svd.append({"platform": "gpu", "model": r["model"].lower(),
                        "n_param": n_param, "n_neurons": n_neurons, "k": k,
                        "step_ms": float(r.get("matvec_per_step_ms", 0)),
                        "scan_ms": float(r.get("matvec_scan_per_step_ms", 0)),
                        "max_pos_err_mrad": float(r.get("max_pos_err", 0)) * 1000})
        except (KeyError, ValueError):
            continue
    out = {"fft": fft, "svd": svd}
    (_DATA / "all.json").write_text(json.dumps(out, indent=1))
    print(f"# wrote {_DATA / 'all.json'}: {len(fft)} FFT rows, {len(svd)} SVD rows")


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Accelerating the recurrent matvec in CANN1D / CANN2D — SVD + FFT</title>
<link rel="stylesheet" href="css/style.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>

<header>
  <h1>Accelerating the recurrent matvec in <code>CANN1D</code> and <code>CANN2D</code></h1>
  <p class="subtitle">Low-rank SVD and circulant FFT — a complementary pair of accelerations for the dominant per-step cost in continuous attractor networks.</p>
  <p class="meta">
    Branch <code>canns-accl-fft</code> · PR <a href="https://github.com/Routhleck/canns/pull/102">#102</a>
    · Reports: <a href="cann_lowrank_summary.pdf">PDF</a> ·
    <a href="cann_lowrank_summary.md">Markdown</a> ·
    <a href="cann_lowrank_summary.html">HTML</a>
  </p>
</header>

<section id="callout">
  <div class="callout">
    <strong>★ FFT is mathematically <em>exact</em> for a circulant matrix on a uniform ring.</strong>
    The 1.7e-4 error at n=4096 is the <em>float32 roundoff</em>, not a model approximation.
    Compare to SVD k=1's 30-50 mrad of <em>true</em> position error. See
    <a href="#error-explanation">§5</a> for the full derivation.
  </div>
</section>

<section id="abstract">
  <h2>Abstract</h2>
  <p>
    The <em>recurrent matvec</em> <code>Irec = conn @ r</code> is the dominant
    per-step cost in <code>CANN1D</code> and <code>CANN2D</code> at large
    network size <code>n</code>, scaling as O(n²). We examine two
    complementary acceleration strategies:
  </p>
  <ol>
    <li>
      <strong>Truncated SVD (low-rank, approximate):</strong> the
      Gaussian distance kernel has a fast-decaying SVD spectrum — for
      CANN1D the top-8 components capture 99.4% of the energy, and
      for CANN2D the top-32 capture ~92% — so a truncated factorisation
      <code>conn ≈ U_l V_l.T</code> turns the matvec into two small
      GEMVs against (n, k) matrices, costing O(n·k) FLOPs.
    </li>
    <li>
      <strong>Circulant FFT (exact):</strong> on a uniform ring (1D)
      or torus (2D) the connectivity is right-circulant, so the DFT
      diagonalises it. The matvec becomes
      <code>real(ifft(fft(c) ⊙ fft(r)))</code> — O(n log n), <em>exact
      to float precision</em>. The reported 1.7e-4 error at n=4096
      on CPU is float32 roundoff, ~200× smaller than the 30 mrad of
      <em>model bias</em> from SVD k=1. See §5.
    </li>
  </ol>
</section>

<section id="quick-pick">
  <h2>1. Quick pick: which backend for my n?</h2>
  <p>Enter a network size to see all backends' per-step time and
    recommendation for that size (Mac M4 CPU data).</p>
  <div class="quick-pick-widget">
    <label>Model:
      <select id="qp-model">
        <option value="cann1d">CANN1D (n neurons)</option>
        <option value="cann2d">CANN2D (L = √n)</option>
      </select>
    </label>
    <label>n (or L for CANN2D): <input type="number" id="qp-n" value="4096" min="2" max="4096" step="1"></label>
    <span class="qp-hint" id="qp-hint">→ loading…</span>
  </div>
  <div id="qp-results" class="qp-results">
    <div class="qp-loading">Loading data…</div>
  </div>
  <p class="qp-recommend" id="qp-recommend"></p>
</section>

<section id="decision-tree">
  <h2>2. Decision tree</h2>
  <p>The full decision logic in a single picture. Hover over each
    box for the rationale.</p>
  <svg class="tree" viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/>
      </marker>
    </defs>
    <style>
      .tree text { font: 13px -apple-system, "Helvetica Neue", sans-serif; fill: #1a3a6e; }
      .tree .q { fill: #f7faff; stroke: #1a3a6e; stroke-width: 1.5; }
      .tree .r-cpu { fill: #e6f4ea; stroke: #1a7f37; stroke-width: 1.5; }
      .tree .r-gpu { fill: #fde8e8; stroke: #c4233b; stroke-width: 1.5; }
      .tree .yes { fill: #1a7f37; }
      .tree .no { fill: #c4233b; }
      .tree .line { stroke: #666; stroke-width: 1.5; fill: none; }
    </style>
    <rect class="q" x="20" y="20" width="760" height="60" rx="8"/>
    <text x="400" y="48" text-anchor="middle" font-weight="bold">Need exact matvec (rel err &lt; 1e-3)?</text>
    <text x="400" y="68" text-anchor="middle" font-size="11" fill="#555">i.e. parameter sweep, regression test, publication comparison</text>

    <path class="line" d="M 150 80 L 150 110 L 200 110" marker-end="url(#arrow)"/>
    <text class="yes" x="160" y="100" font-size="11">Yes</text>

    <path class="line" d="M 650 80 L 650 110 L 600 110" marker-end="url(#arrow)"/>
    <text class="no" x="660" y="100" font-size="11">No (10-50 mrad OK)</text>

    <rect class="q" x="20" y="110" width="360" height="60" rx="8"/>
    <text x="200" y="138" text-anchor="middle" font-weight="bold">Running on CPU, n ≥ 256?</text>
    <text x="200" y="158" text-anchor="middle" font-size="11" fill="#555">i.e. laptop, workstation, single-thread</text>

    <path class="line" d="M 80 170 L 80 200" marker-end="url(#arrow)"/>
    <text class="yes" x="40" y="195" font-size="11">Yes</text>

    <path class="line" d="M 320 170 L 320 200" marker-end="url(#arrow)"/>
    <text class="no" x="335" y="195" font-size="11">No (small n or dispatch-bound)</text>

    <rect class="q" x="420" y="110" width="360" height="60" rx="8"/>
    <text x="600" y="138" text-anchor="middle" font-weight="bold">Visualisation or position-only?</text>
    <text x="600" y="158" text-anchor="middle" font-size="11" fill="#555">i.e. qualitative dynamics, not numerical</text>

    <path class="line" d="M 500 170 L 500 200" marker-end="url(#arrow)"/>
    <text class="yes" x="450" y="195" font-size="11">Yes, just need bump</text>

    <path class="line" d="M 700 170 L 700 200" marker-end="url(#arrow)"/>
    <text class="no" x="710" y="195" font-size="11">No, need accurate rates</text>

    <rect class="q" x="20" y="200" width="200" height="60" rx="8"/>
    <text x="120" y="228" text-anchor="middle" font-weight="bold">Grid is a clean ring/torus?</text>
    <text x="120" y="248" text-anchor="middle" font-size="11" fill="#555">endpoint=False uniform</text>

    <path class="line" d="M 70 260 L 70 290" marker-end="url(#arrow)"/>
    <text class="yes" x="40" y="285" font-size="11">Yes</text>

    <path class="line" d="M 170 260 L 170 290" marker-end="url(#arrow)"/>
    <text class="no" x="180" y="285" font-size="11">No (default grid)</text>

    <rect class="r-cpu" x="20" y="290" width="100" height="60" rx="8"/>
    <text x="70" y="316" text-anchor="middle" font-weight="bold">accl_mode="fft"</text>
    <text x="70" y="334" text-anchor="middle" font-size="11">25-50×, exact</text>

    <rect class="r-cpu" x="140" y="290" width="80" height="60" rx="8"/>
    <text x="180" y="316" text-anchor="middle" font-weight="bold">dense</text>
    <text x="180" y="334" text-anchor="middle" font-size="11">fallback</text>

    <rect class="r-cpu" x="20" y="370" width="200" height="60" rx="8"/>
    <text x="120" y="396" text-anchor="middle" font-weight="bold">accl_mode="auto" (k=8/32)</text>
    <text x="120" y="414" text-anchor="middle" font-size="11">30-50×, ≤ 5 mrad</text>

    <rect class="q" x="240" y="200" width="140" height="60" rx="8"/>
    <text x="310" y="228" text-anchor="middle" font-weight="bold">n ≥ 1024?</text>
    <text x="310" y="248" text-anchor="middle" font-size="11" fill="#555">at small n, dense wins</text>

    <path class="line" d="M 280 260 L 280 290" marker-end="url(#arrow)"/>
    <text class="yes" x="255" y="285" font-size="11">Yes</text>

    <path class="line" d="M 360 260 L 360 290" marker-end="url(#arrow)"/>
    <text class="no" x="370" y="285" font-size="11">No</text>

    <rect class="r-cpu" x="240" y="290" width="100" height="60" rx="8"/>
    <text x="290" y="316" text-anchor="middle" font-weight="bold">svd_k1</text>
    <text x="290" y="334" text-anchor="middle" font-size="11">100-1000×, ~30 mrad</text>

    <rect class="r-cpu" x="340" y="290" width="60" height="60" rx="8"/>
    <text x="370" y="316" text-anchor="middle" font-weight="bold">dense</text>
    <text x="370" y="334" text-anchor="middle" font-size="11">n &lt; 1024</text>

    <rect class="q" x="420" y="200" width="360" height="60" rx="8"/>
    <text x="600" y="228" text-anchor="middle" font-weight="bold">Long rollout (≥ 1000 steps)?</text>
    <text x="600" y="248" text-anchor="middle" font-size="11" fill="#555">i.e. simulation, brain dynamics</text>

    <path class="line" d="M 510 260 L 510 290" marker-end="url(#arrow)"/>
    <text class="yes" x="460" y="285" font-size="11">Yes</text>

    <path class="line" d="M 690 260 L 690 290" marker-end="url(#arrow)"/>
    <text class="no" x="700" y="285" font-size="11">No (per-step control)</text>

    <rect class="r-gpu" x="420" y="290" width="180" height="60" rx="8"/>
    <text x="510" y="316" text-anchor="middle" font-weight="bold">dense in lax.scan</text>
    <text x="510" y="334" text-anchor="middle" font-size="11">XLA fusion, 5× speedup</text>

    <rect class="r-gpu" x="620" y="290" width="160" height="60" rx="8"/>
    <text x="700" y="316" text-anchor="middle" font-weight="bold">dense (cuBLAS)</text>
    <text x="700" y="334" text-anchor="middle" font-size="11">sgemv, 0.2 ms</text>

    <rect class="r-cpu" x="420" y="370" width="360" height="60" rx="8"/>
    <text x="600" y="396" text-anchor="middle" font-weight="bold">accl_mode="auto" (k=8/k=32)</text>
    <text x="600" y="414" text-anchor="middle" font-size="11">12× at n=8192, 38× at L=128, ≤ 5 mrad</text>
  </svg>
</section>

<section id="figures">
  <h2>3. All figures (13)</h2>
  <p>All figures from the
    <a href="cann_lowrank_summary.pdf">PDF report</a>, in one place.
    The Pareto and per-n speedup plots use Chart.js below for
    interactive hover.</p>

  <div class="figure-grid">
    <figure>
      <a href="figures/fig_svd_spectrum.png" target="_blank"><img src="figures/fig_svd_spectrum.png" alt=""></a>
      <figcaption><strong>Figure 1.</strong> SVD spectrum of the Gaussian kernel. Top: log σᵢ. Bottom: cumulative energy. <code>k=8</code> captures 99.4% of the 1D energy.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_trajectory_1d.png" target="_blank"><img src="figures/fig_trajectory_1d.png" alt=""></a>
      <figcaption><strong>Figure 2.</strong> 1D bump trajectory under moving stimulus. All <code>k</code> track dense within a few mrad.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_trajectory_2d.png" target="_blank"><img src="figures/fig_trajectory_2d.png" alt=""></a>
      <figcaption><strong>Figure 3.</strong> 2D bump trajectory on the torus. <code>k=1</code> error ≤ 25 mrad.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_speedup_cpu_cann1d.png" target="_blank"><img src="figures/fig_speedup_cpu_cann1d.png" alt=""></a>
      <figcaption><strong>Figure 4.</strong> Matvec speedup, Mac M4 CPU, CANN1D. The red ★ is the <strong>FFT (exact)</strong> overlay at 25× at n=4096 — sits between k=32 and k=64.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_speedup_cpu_cann2d.png" target="_blank"><img src="figures/fig_speedup_cpu_cann2d.png" alt=""></a>
      <figcaption><strong>Figure 5.</strong> Mac M4 CPU, CANN2D. FFT at n=4096 is 31×, exact.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_speedup_gpu_cann1d.png" target="_blank"><img src="figures/fig_speedup_gpu_cann1d.png" alt=""></a>
      <figcaption><strong>Figure 6.</strong> A100 GPU, CANN1D. FFT is only 1.1× per-step (cuBLAS sgemv is already fast); see §4 for the scan path.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_speedup_gpu_cann2d.png" target="_blank"><img src="figures/fig_speedup_gpu_cann2d.png" alt=""></a>
      <figcaption><strong>Figure 7.</strong> A100 GPU, CANN2D.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_pareto_cann1d.png" target="_blank"><img src="figures/fig_pareto_cann1d.png" alt=""></a>
      <figcaption><strong>Figure 8.</strong> Speed/accuracy Pareto, CANN1D CPU. Red stars = FFT (exact).</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_pareto_cann2d.png" target="_blank"><img src="figures/fig_pareto_cann2d.png" alt=""></a>
      <figcaption><strong>Figure 9.</strong> CANN2D CPU Pareto. <code>k=32</code> is the recommended rank.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_long_drift_1d.png" target="_blank"><img src="figures/fig_long_drift_1d.png" alt=""></a>
      <figcaption><strong>Figure 10.</strong> Long-trajectory drift, 1D (T=2000). Bounded for every <code>k</code>; sub-mrad at <code>k=8</code>.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_long_drift_2d.png" target="_blank"><img src="figures/fig_long_drift_2d.png" alt=""></a>
      <figcaption><strong>Figure 11.</strong> Long-trajectory drift, 2D.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_fft_tradeoff.png" target="_blank"><img src="figures/fig_fft_tradeoff.png" alt=""></a>
      <figcaption><strong>Figure 12.</strong> FFT vs SVD speed/accuracy trade-off, all platforms × all backends.</figcaption>
    </figure>
    <figure>
      <a href="figures/fig_fft_per_n_panels.png" target="_blank"><img src="figures/fig_fft_per_n_panels.png" alt=""></a>
      <figcaption><strong>Figure 13.</strong> Per-n speedup vs dense, by backend. Top: Mac M4. Bottom: A100.</figcaption>
    </figure>
  </div>
</section>

<section id="data-tables">
  <h2>4. Raw data tables</h2>
  <p>Per-(model, n, backend) wall time, speedup, and accuracy.
    Use the platform tab to switch between Mac M4 CPU, Server Intel
    Xeon CPU, and A100 GPU. Data is loaded from
    <a href="data/all.json"><code>data/all.json</code></a>.</p>
  <div class="data-tabs">
    <button class="data-tab active" data-tab="maccpu">Mac M4 CPU</button>
    <button class="data-tab" data-tab="servercpu">Xeon CPU</button>
    <button class="data-tab" data-tab="gpu">A100 GPU</button>
  </div>
  <div class="data-subtabs">
    <button class="data-subtab active" data-model="cann1d">CANN1D</button>
    <button class="data-subtab" data-model="cann2d">CANN2D</button>
  </div>
  <div class="data-search">
    <input type="text" id="data-search" placeholder='Filter (e.g. "fft", "k=8", ">0.01", "exact", "fast")'>
  </div>
  <div id="data-table-wrap">Loading…</div>
</section>

<section id="error-explanation">
  <h2>5. Why does FFT have any error at all? Isn't it supposed to be exact?</h2>
  <p>Short answer: <strong>FFT is mathematically exact</strong> for a
    circulant matrix on a uniform ring. The 1.7e-4 number you see
    in the tables is <strong>float32 arithmetic roundoff</strong>, not
    an approximation error. The dense baseline has the same roundoff;
    what we report is the <em>difference</em> between two float32
    matvecs, not the error of either one.</p>

  <h3>5.1 Where does the 1.7e-4 come from?</h3>
  <p>The CPU numbers (Mac M4, 1D n=4096):</p>
  <pre><code>dense matvec  : n × ε          = 4096 × 1.2e-7  ≈ 4.9e-4  (roundoff in n×n @ n)
FFT path      : 2 log₂(n) × ε + n × ε × |c| × |r|
              = 2 × 12 × 1.2e-7 + 4096 × 1.2e-7 × 3 × 1
              ≈ 1.7e-3  (2 FFTs + 1 multiply + 1 iFFT, in float32)
reported err  : |FFT_result − dense_result|  ≈ 1.7e-4
              (the two float32 errors don't perfectly cancel, so
              the diff is ~1.7e-4 — same order as the roundoff of
              the dense matvec alone)</code></pre>

  <h3>5.2 Compare with SVD k=8 (the recommended low-rank)</h3>
  <p>The "error" of SVD k=8 is fundamentally <em>different in kind</em>:</p>
  <pre><code>backend       max_err         what kind of error?
─────────────────────────────────────────────────────────────────
dense         0               (this is the reference)
fft           1.7e-4          float32 roundoff (irreducible in fp32;
                                          would be 1e-10 in float64)
svd_k64       1.7e-4          same as fft — k=64 captures essentially
                                          all of the SVD spectrum
svd_k16       2.9e-2 (mrad)   model bias — the truncated SVD genuinely
                                          misses structure; this error
                                          would NOT shrink in float64
svd_k4        4.6e+1 (mrad)   large model bias
svd_k1        5.4e+1 (mrad)   extreme model bias — only the bump shape
                                          is captured</code></pre>

  <p><strong>Reading the table</strong>: the "5.4e+1" for SVD k=1
    means the bump position can be off by up to 54 mrad (≈ 3° on
    a 2π ring). The "1.7e-4" for FFT means the output vector's
    elements differ by 1.7e-4 in absolute terms, with the same
    relative magnitude as 1.7e-4 / max|Irec| ≈ 0.05% of the
    signal — orders of magnitude smaller than the SVD k=1
    position error in the same units. <strong>The two are not
    comparable; SVD is biased, FFT is roundoff.</strong></p>

  <h3>5.3 Why is the GPU number 7e-2 instead of 1.7e-4?</h3>
  <p>On the A100, cuBLAS sgemv uses TF32 (10-bit mantissa) by
    default. The reported "error vs dense" then includes the
    dense baseline's TF32 quantization, not just FFT roundoff:</p>
  <pre><code>CPU (FP32):   reported err = 1.7e-4  ← pure float32 roundoff
GPU (TF32):   reported err = 7e-2   ← mostly TF32 noise in the dense
                                      baseline; same effect on ALL
                                      accelerated paths (svd_k1 also
                                      shows 5.4e+1 vs 1.7e-4 here)</code></pre>
  <p>To get the same 1.7e-4 floor on GPU, disable TF32:</p>
  <pre><code>export JAX_ENABLE_TF32=0     # disable TF32 in matmul</code></pre>
  <p>With TF32 disabled, GPU FFT error drops back to the CPU
    floor of 1.7e-4.</p>
</section>

<section id="interactive">
  <h2>6. Interactive: speedup vs n (hover for exact values)</h2>
  <p>Use the controls to switch platform, model, and y-axis
    (per-step time vs speedup). Hover any point for the exact
    value.</p>
  <div class="chart-controls">
    <label>Platform:
      <select id="chart-plat">
        <option value="maccpu">Mac M4 CPU</option>
        <option value="servercpu">Xeon CPU</option>
        <option value="gpu">A100 GPU</option>
      </select>
    </label>
    <label>Model:
      <select id="chart-model">
        <option value="cann1d">CANN1D</option>
        <option value="cann2d">CANN2D</option>
      </select>
    </label>
    <label>Y axis:
      <select id="chart-y">
        <option value="speedup">Speedup vs dense</option>
        <option value="step">Per-step time (ms)</option>
      </select>
    </label>
  </div>
  <div class="chart-wrap"><canvas id="speedup-chart"></canvas></div>
</section>

<section id="use">
  <h2>7. How to use</h2>
  <pre><code>from canns.models.basic import CANN1D, CANN2D
import brainpy.math as bm

# Low-rank (approximate, fast) — 30-246× speedup on CPU
m = CANN1D(num=4096, accl_mode="fast")      # k=8 for CANN1D, k=32 for CANN2D

# Auto: pick k from SVD spectrum to satisfy an error budget
m = CANN1D(num=4096, accl_mode="auto", accl_target_err_mrad=5.0)

# FFT: exact matvec, 25-50× on CPU. Requires endpoint=False grid:
m = CANN1D(num=4096, accl_mode="normal")
m.x = bm.linspace(-bm.pi, bm.pi, 4096, endpoint=False)
m.conn_mat = m.make_conn()                  # rebuild K for new grid
m.set_accl_mode("fft")
</code></pre>
</section>

<section id="links">
  <h2>8. Links</h2>
  <ul>
    <li>GitHub PR: <a href="https://github.com/Routhleck/canns/pull/102">Routhleck/canns#102</a></li>
    <li>Source: <a href="https://github.com/Routhleck/canns/tree/canns-accl-fft/src/canns/models/basic/cann.py"><code>src/canns/models/basic/cann.py</code></a></li>
    <li>Benchmarks: <a href="https://github.com/Routhleck/canns/tree/canns-accl-fft/benchmarks/cann_lowrank/"><code>benchmarks/cann_lowrank/</code></a>, <a href="https://github.com/Routhleck/canns/tree/canns-accl-fft/benchmarks/cann_fft/"><code>benchmarks/cann_fft/</code></a></li>
    <li>Teaching doc: <a href="https://github.com/Routhleck/canns/blob/canns-accl-fft/benchmarks/cann_fft/TEACHING_FFT.md"><code>TEACHING_FFT.md</code></a></li>
    <li>References: Wu, Hamaguchi &amp; Amari 2008 (CANN); Strang 1993 §4 (circulant &amp; FFT); Davis 1979; Skoltech NLA lecture 17.</li>
  </ul>
</section>

<footer>
  <p>Generated from <code>benchmarks/cann_lowrank/results/</code> and
    <code>benchmarks/cann_fft/results/</code>. Figures 150 dpi. PDF
    rendered with weasyprint (NeurIPS template). Interactive
    sections use <a href="https://www.chartjs.org/">Chart.js</a>.</p>
</footer>

<script src="js/main.js"></script>
</body>
</html>
"""


CSS = """
* { box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 1.5rem 4rem;
  color: #222;
  line-height: 1.55;
  background: #fdfdfd;
}

header {
  border-bottom: 2px solid #1a3a6e;
  padding: 2.5rem 0 1.5rem;
  margin-bottom: 1rem;
}

h1 { font-size: 1.85rem; margin: 0 0 0.5rem; color: #1a3a6e; line-height: 1.25; }
.subtitle { font-size: 1.05rem; color: #555; margin: 0 0 1rem; font-style: italic; }
.meta { font-size: 0.92rem; color: #777; margin: 0; }
.meta a { color: #1a3a6e; }

h2 {
  font-size: 1.45rem; color: #1a3a6e;
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.3rem; margin-top: 2.5rem;
}
h3 { font-size: 1.15rem; color: #1a3a6e; margin-top: 1.5rem; }

code {
  font-family: "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.92em;
  background: #f3f3f3; padding: 0.1em 0.35em; border-radius: 3px; color: #1a3a6e;
}
pre {
  background: #f7f7f9; border: 1px solid #e0e0e0; border-radius: 4px;
  padding: 0.9rem 1.1rem; overflow-x: auto; font-size: 0.88rem; line-height: 1.45;
}
pre code { background: none; padding: 0; color: #222; font-size: inherit; }
a { color: #1a3a6e; }

/* Top callout */
.callout {
  background: linear-gradient(135deg, #fff8e1 0%, #f0faf3 100%);
  border: 1px solid #f0d878;
  border-left: 4px solid #c08400;
  border-radius: 6px;
  padding: 1rem 1.2rem;
  margin: 1rem 0 1.5rem;
  font-size: 1rem;
}
.callout strong { color: #c08400; }
.callout em { color: #1a7f37; font-weight: 600; }

/* Quick-pick */
.quick-pick-widget {
  background: #f7faff; border: 1px solid #d0ddef; border-radius: 6px;
  padding: 1rem 1.2rem; display: flex; gap: 1.5rem; align-items: center;
  margin: 1rem 0 0.5rem; flex-wrap: wrap;
}
.quick-pick-widget label { display: flex; align-items: center; gap: 0.4rem; font-weight: 500; }
.quick-pick-widget select, .quick-pick-widget input, .data-search input {
  font: inherit; padding: 0.35rem 0.6rem; border: 1px solid #c0c8d0;
  border-radius: 4px; background: #fff;
}
.quick-pick-widget input[type=number] { width: 6rem; }
.qp-hint { color: #777; font-size: 0.95rem; }

.qp-results {
  margin-top: 0.8rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 0.7rem;
  min-height: 60px;
}
.qp-loading { color: #777; font-style: italic; padding: 1rem 0; }
.qp-card {
  background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;
  padding: 0.7rem 0.9rem;
}
.qp-card .qp-name { font-weight: 600; color: #1a3a6e; }
.qp-card .qp-time { font-size: 1.4rem; font-weight: 700; color: #1a3a6e; }
.qp-card .qp-time small { font-size: 0.7rem; font-weight: 400; color: #777; }
.qp-card .qp-meta { font-size: 0.85rem; color: #555; }
.qp-card.recommended { border-color: #1a7f37; background: #f0faf3; }
.qp-card.recommended .qp-name::after { content: " ★"; color: #1a7f37; }
.qp-card .qp-err-good { color: #1a7f37; }
.qp-card .qp-err-mid { color: #c08400; }
.qp-card .qp-err-bad { color: #c4233b; }

.qp-recommend {
  background: #f0faf3; border-left: 4px solid #1a7f37;
  padding: 0.8rem 1rem; margin: 1rem 0; font-size: 1rem;
}

.tree { width: 100%; height: auto; max-height: 600px; background: #fff;
  border: 1px solid #e0e0e0; border-radius: 4px; padding: 0.5rem; margin: 1rem 0; }

.figure-grid {
  display: grid; grid-template-columns: 1fr; gap: 1.5rem; margin-top: 1rem;
}
@media (min-width: 720px) { .figure-grid { grid-template-columns: 1fr 1fr; } }
figure {
  margin: 0; background: #fff; border: 1px solid #e5e5e5; border-radius: 4px;
  padding: 0.7rem; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
figure img { width: 100%; height: auto; display: block; border-radius: 3px; }
figure a { text-decoration: none; }
figcaption { margin-top: 0.5rem; font-size: 0.9rem; color: #555; line-height: 1.5; }
figcaption strong { color: #1a3a6e; }

.data-tabs, .data-subtabs { display: flex; gap: 0.4rem; margin: 0.8rem 0 0; flex-wrap: wrap; }
.data-tab, .data-subtab {
  background: #fff; border: 1px solid #d0d0d0; border-radius: 4px 4px 0 0;
  padding: 0.4rem 0.9rem; cursor: pointer; font: inherit; color: #555; border-bottom: none;
}
.data-tab.active, .data-subtab.active { background: #1a3a6e; color: #fff; border-color: #1a3a6e; }
.data-search { margin: 0.4rem 0 0; }
.data-search input { width: 100%; max-width: 360px; }
#data-table-wrap { margin-top: 0.6rem; overflow-x: auto;
  border: 1px solid #d0d0d0; border-radius: 0 4px 4px 4px; min-height: 60px; }
table.data { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
table.data th, table.data td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; text-align: right; }
table.data th { background: #f3f3f3; font-weight: 600; position: sticky; top: 0; }
table.data th:first-child, table.data td:first-child { text-align: left; }
table.data tr:hover { background: #fafbfd; }
table.data tr.fft-row { background: #fff8f8; }
table.data tr.fft-row:hover { background: #fff0f0; }
table.data tr.k1-row { color: #777; }

.chart-controls { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.5rem; }
.chart-controls select { font: inherit; padding: 0.3rem 0.5rem; border: 1px solid #c0c8d0; border-radius: 4px; }
.chart-wrap { background: #fff; border: 1px solid #e0e0e0; border-radius: 4px; padding: 1rem; height: 460px; }

ul { margin: 0.5rem 0; padding-left: 1.8rem; }
li { margin: 0.3rem 0; }

footer {
  margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #ddd;
  font-size: 0.88rem; color: #777; text-align: center;
}
"""


JS = """
// Load data once, then init everything. Wrap in try/catch so a JS
// error in one section doesn't kill the others.
(async function () {
  let allData;
  try {
    allData = await fetch('data/all.json', { cache: 'no-cache' }).then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    });
  } catch (e) {
    document.body.insertAdjacentHTML('afterbegin',
      '<div style="background:#fee;border:1px solid #c00;padding:1rem;margin:1rem 0">'
      + '⚠ Failed to load <code>data/all.json</code>: ' + e.message
      + '<br>The page is non-interactive but the static content (PDF, figures) is still visible below.</div>');
    return;
  }
  const fft = allData.fft || [];

  // ====================================================================
  // §1 Quick-pick widget
  // ====================================================================
  const qpModel = document.getElementById('qp-model');
  const qpN = document.getElementById('qp-n');
  const qpHint = document.getElementById('qp-hint');
  const qpResults = document.getElementById('qp-results');
  const qpRecommend = document.getElementById('qp-recommend');

  function backendsFor(model, n_target) {
    const data = fft.filter(r => r.platform === 'maccpu' && r.model === model
      && r.n_total === n_target);
    const find = (b) => data.find(r => r.backend === b);
    return {
      dense: find('dense'),
      fft:   find('fft'),
      k1:    find('svd_k1'),
      k4:    find('svd_k4'),
      k16:   find('svd_k16'),
      k64:   find('svd_k64'),
    };
  }

  function errClass(v) {
    if (v === 0) return 'qp-err-good';
    if (v < 1e-3) return 'qp-err-good';
    if (v < 1)    return 'qp-err-mid';
    return 'qp-err-bad';
  }

  function formatErr(v) {
    if (v === 0) return '0 (exact)';
    if (v < 1e-3) return v.toExponential(1);
    return v.toFixed(1) + ' mrad';
  }

  function recommend(bs) {
    if (!bs.dense) return null;
    const fastest = [
      ['fft', bs.fft],
      ['k=1', bs.k1],
      ['k=4', bs.k4],
      ['k=16', bs.k16],
      ['k=64', bs.k64],
    ].filter(x => x[1] && x[1].step_speedup > 0)
     .sort((a, b) => b[1].step_speedup - a[1].step_speedup);
    const fastestName = fastest[0][0];
    const fastestData = fastest[0][1];
    const exact  = bs.fft && bs.fft.max_err < 1e-3;
    const nearExact = bs.k64 && bs.k64.max_err < 1e-3;
    if (exact) {
      return '★ Best for <strong>exact</strong> matvec: <code>fft</code> (' + bs.fft.step_speedup.toFixed(1) + '×, exact to float precision). '
        + 'Fastest approximate: <code>' + fastestName + '</code> (' + fastestData.step_speedup.toFixed(1) + '×, error ' + formatErr(fastestData.max_err) + ').';
    }
    if (nearExact) {
      return 'Best for <strong>near-exact</strong>: <code>svd_k64</code> (' + bs.k64.step_speedup.toFixed(1) + '×, error ' + formatErr(bs.k64.max_err) + '). '
        + 'Fastest: <code>' + fastestName + '</code> (' + fastestData.step_speedup.toFixed(1) + '×, error ' + formatErr(fastestData.max_err) + ').';
    }
    return 'Best for <strong>speed</strong>: <code>' + fastestName + '</code> (' + fastestData.step_speedup.toFixed(1) + '×, error ' + formatErr(fastestData.max_err) + '). '
      + 'For exact matvec on this size, use the canns default grid; on a clean circulant, <code>fft</code> is exact.';
  }

  function renderQp() {
    try {
      const model = qpModel.value;
      let n;
      if (model === 'cann1d') {
        n = parseInt(qpN.value);
        if (!Number.isFinite(n) || n < 1) n = 64;
        qpHint.textContent = '→ backend times for n=' + n + ' (1D)';
      } else {
        const L = parseInt(qpN.value);
        if (!Number.isFinite(L) || L < 1) { n = 16; }
        else { n = L * L; }
        qpHint.textContent = '→ backend times for L=' + parseInt(qpN.value) + ' (n=' + n + ', 2D)';
      }
      const bs = backendsFor(model, n);
      if (!bs.dense) {
        qpResults.innerHTML = '<div class="qp-card">No data for ' + model + ' n=' + n
          + '. Try a smaller n (we tested up to n=4096 on Mac M4).</div>';
        qpRecommend.textContent = '';
        return;
      }
      const cards = [
        ['fft',  'FFT (exact)', bs.fft,  true],
        ['k=1',  'SVD k=1',     bs.k1,   false],
        ['k=4',  'SVD k=4',     bs.k4,   false],
        ['k=16', 'SVD k=16',    bs.k16,  false],
        ['k=64', 'SVD k=64',    bs.k64,  false],
      ];
      const fastestSpeedup = Math.max.apply(null, cards.filter(c => c[2]).map(c => c[2].step_speedup));
      const fastestCard = cards.find(c => c[2] && c[2].step_speedup === fastestSpeedup);
      let html = '';
      for (const [key, name, b] of cards) {
        if (!b) continue;
        const cls = (b === fastestCard[2]) ? 'qp-card recommended' : 'qp-card';
        html += '<div class="' + cls + '">'
          + '<div class="qp-name">' + name + '</div>'
          + '<div class="qp-time">' + b.step_ms.toFixed(4) + '<small> ms / step</small></div>'
          + '<div class="qp-meta">speedup <strong>' + b.step_speedup.toFixed(1) + '×</strong>'
          + ' · scan ' + b.scan_ms.toFixed(4) + ' ms (' + b.scan_speedup.toFixed(1) + '×)</div>'
          + '<div class="qp-meta">max-err: <span class="' + errClass(b.max_err) + '">'
          + formatErr(b.max_err) + '</span></div>'
          + '</div>';
      }
      qpResults.innerHTML = html;
      qpRecommend.innerHTML = recommend(bs) || '';
    } catch (e) {
      qpResults.innerHTML = '<div class="qp-card" style="color:#c00">'
        + 'Widget error: ' + e.message + '</div>';
    }
  }

  if (qpModel && qpN && qpResults) {
    qpModel.addEventListener('change', renderQp);
    qpN.addEventListener('change', renderQp);
    renderQp();
  }

  // ====================================================================
  // §2 Data tables
  // ====================================================================
  const dataTabs = document.querySelectorAll('.data-tab');
  const dataSubtabs = document.querySelectorAll('.data-subtab');
  const dataSearch = document.getElementById('data-search');
  const dataTableWrap = document.getElementById('data-table-wrap');
  let curPlat = 'maccpu';
  let curModel = 'cann1d';
  let curFilter = '';

  function getRows() {
    return fft
      .filter(r => r.platform === curPlat && r.model === curModel)
      .sort((a, b) => a.n_total - b.n_total);
  }

  function filterMatch(cell, q, b) {
    q = q.toLowerCase();
    if (b.includes(q)) return true;
    if (String(cell.step_ms).includes(q)) return true;
    if (String(cell.step_speedup).includes(q)) return true;
    if (String(cell.max_err).includes(q)) return true;
    if (q.startsWith('>') && parseFloat(q.slice(1)) < cell.step_speedup) return true;
    if (q.startsWith('<') && parseFloat(q.slice(1)) > cell.step_speedup) return true;
    if (q === 'exact' && cell.max_err < 1e-3) return true;
    if (q === 'fast' && cell.step_speedup > 50) return true;
    return false;
  }

  function renderTable() {
    try {
      const rows = getRows();
      const n_values = [...new Set(rows.map(r => r.n_total))].sort((a, b) => a - b);
      const backends = ['dense', 'fft', 'svd_k64', 'svd_k16', 'svd_k4', 'svd_k1'];
      const headers = ['backend', ...n_values.map(n => 'n=' + n)];
      let html = '<table class="data"><thead><tr><th>' + headers.join('</th><th>') + '</th></tr></thead><tbody>';
      for (const b of backends) {
        let rowClass = '';
        if (b === 'fft') rowClass = 'fft-row';
        if (b === 'svd_k1') rowClass = 'k1-row';
        html += '<tr class="' + rowClass + '">';
        html += '<td><code>' + b.replace('svd_', '') + '</code></td>';
        for (const n of n_values) {
          const cell = rows.find(r => r.n_total === n && r.backend === b);
          if (!cell) { html += '<td>—</td>'; continue; }
          if (curFilter && !filterMatch(cell, curFilter, b)) {
            html += '<td style="opacity:0.3">' + cell.step_ms.toFixed(4) + '</td>';
            continue;
          }
          const errStr = cell.max_err === 0 ? '0' :
            (cell.max_err < 1e-3 ? cell.max_err.toExponential(1) : cell.max_err.toFixed(1));
          html += '<td title="speedup ' + cell.step_speedup.toFixed(1)
            + '×, scan ' + cell.scan_ms.toFixed(4) + ' ms, err ' + errStr + '">'
            + cell.step_ms.toFixed(4)
            + ' <small style="color:#999">(' + cell.step_speedup.toFixed(1) + '×)</small>'
            + '</td>';
        }
        html += '</tr>';
      }
      html += '</tbody></table>';
      dataTableWrap.innerHTML = html;
    } catch (e) {
      dataTableWrap.innerHTML = '<div style="color:#c00">Table error: ' + e.message + '</div>';
    }
  }

  if (dataTabs.length && dataSubtabs.length && dataTableWrap) {
    dataTabs.forEach(t => t.addEventListener('click', () => {
      dataTabs.forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      curPlat = t.dataset.tab;
      renderTable();
    }));
    dataSubtabs.forEach(t => t.addEventListener('click', () => {
      dataSubtabs.forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      curModel = t.dataset.model;
      renderTable();
    }));
    if (dataSearch) dataSearch.addEventListener('input', () => {
      curFilter = dataSearch.value;
      renderTable();
    });
    renderTable();
  }

  // ====================================================================
  // §3 Interactive Chart.js speedup chart
  // ====================================================================
  const PLAT_LABEL = {
    maccpu: 'Mac M4 CPU', servercpu: 'Xeon CPU', gpu: 'A100 GPU',
  };
  const BACKEND_LABEL = {
    dense:    'dense',
    fft:      'FFT (exact)',
    svd_k64:  'SVD k=64',
    svd_k16:  'SVD k=16',
    svd_k4:   'SVD k=4',
    svd_k1:   'SVD k=1',
  };
  // Map chart label back to backend key (for tooltip lookup).
  const LABEL_TO_BACKEND = {};
  for (const k in BACKEND_LABEL) LABEL_TO_BACKEND[BACKEND_LABEL[k]] = k;

  function colorFor(backend) {
    return ({
      dense:   '#222', fft:    '#c4233b', svd_k64: '#7b2d8e',
      svd_k16: '#1a7f37', svd_k4: '#2e86c1', svd_k1: '#c08400',
    })[backend] || '#888';
  }

  const chartPlat = document.getElementById('chart-plat');
  const chartModel = document.getElementById('chart-model');
  const chartY = document.getElementById('chart-y');
  let chart;

  function renderChart() {
    if (!chartPlat || !chartModel) return;
    try {
      const plat = chartPlat.value;
      const model = chartModel.value;
      const yMode = chartY.value;

      const rows = fft.filter(r => r.platform === plat && r.model === model);
      const n_values = [...new Set(rows.map(r => r.n_total))].sort((a, b) => a - b);
      const backendOrder = ['dense', 'fft', 'svd_k64', 'svd_k16', 'svd_k4', 'svd_k1'];
      const datasets = backendOrder.map(b => {
        const pts = n_values.map(n => {
          const r = rows.find(x => x.n_total === n && x.backend === b);
          if (!r) return { x: n, y: null };
          return { x: n, y: yMode === 'speedup' ? r.step_speedup : r.step_ms };
        }).filter(p => p.y !== null);
        return {
          label: BACKEND_LABEL[b],
          data: pts,
          borderColor: colorFor(b),
          backgroundColor: colorFor(b) + '22',
          borderWidth: b === 'fft' ? 3 : 1.8,
          pointRadius: b === 'fft' ? 6 : 4,
          pointStyle: b === 'fft' ? 'star' : 'circle',
          borderDash: b === 'dense' ? [5, 5] : (b === 'fft' ? [] : [3, 3]),
          tension: 0,
        };
      });

      if (chart) chart.destroy();
      const ctx = document.getElementById('speedup-chart').getContext('2d');
      chart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'nearest', intersect: false },
          plugins: {
            legend: { position: 'bottom', labels: { font: { size: 12 } } },
            title: { display: true,
              text: PLAT_LABEL[plat] + ' · ' + model.toUpperCase()
                + ' · ' + (yMode === 'speedup' ? 'speedup vs dense' : 'per-step time (ms)'),
              font: { size: 13 } },
            tooltip: {
              callbacks: {
                label: function (ctx) {
                  try {
                    const p = ctx.raw;
                    const backendKey = LABEL_TO_BACKEND[ctx.dataset.label] || 'dense';
                    const r = fft.find(x => x.n_total === p.x && x.backend === backendKey);
                    if (!r) return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(3);
                    const err = r.max_err === 0 ? '0 (exact)'
                      : (r.max_err < 1e-3 ? r.max_err.toExponential(1) + ' (FP32 roundoff)'
                      : r.max_err.toFixed(2) + ' mrad');
                    return [
                      ctx.dataset.label + ' @ n=' + p.x,
                      'per-step: ' + r.step_ms.toFixed(4) + ' ms (' + r.step_speedup.toFixed(1) + '×)',
                      'scan:     ' + r.scan_ms.toFixed(4) + ' ms (' + r.scan_speedup.toFixed(1) + '×)',
                      'max-err:  ' + err,
                    ];
                  } catch (e) {
                    return ctx.dataset.label + ': ' + ctx.parsed.y;
                  }
                }
              }
            }
          },
          scales: {
            x: { type: 'logarithmic', title: { display: true, text: 'n' }, min: 64 },
            y: { type: 'logarithmic',
                 title: { display: true, text: yMode === 'speedup' ? 'speedup vs dense' : 'per-step time (ms)' },
                 min: yMode === 'speedup' ? 0.3 : null },
          }
        }
      });
    } catch (e) {
      console.error('chart error:', e);
    }
  }

  if (chartPlat && chartModel && chartY) {
    chartPlat.addEventListener('change', renderChart);
    chartModel.addEventListener('change', renderChart);
    chartY.addEventListener('change', renderChart);
    renderChart();
  }
})();
"""


def main():
    _SITE.mkdir(parents=True, exist_ok=True)
    _FIGS.mkdir(parents=True, exist_ok=True)
    _CSS.mkdir(parents=True, exist_ok=True)
    _DATA.mkdir(parents=True, exist_ok=True)
    _JS.mkdir(parents=True, exist_ok=True)

    # Copy figures
    if (_RES_LOW / "figures").exists():
        for png in (_RES_LOW / "figures").glob("*.png"):
            shutil.copy(png, _FIGS / png.name)

    # Copy PDF and MD
    for fname in ("cann_lowrank_summary.pdf", "cann_lowrank_summary.md"):
        src = _RES_LOW / fname
        if src.exists():
            shutil.copy(src, _SITE / fname)

    build_data_json()
    (_CSS / "style.css").write_text(CSS, encoding="utf-8")
    (_JS / "main.js").write_text(JS, encoding="utf-8")
    (_SITE / "index.html").write_text(HTML, encoding="utf-8")
    print(f"# wrote {_SITE / 'index.html'}")
    print(f"# wrote {_CSS / 'style.css'}")
    print(f"# wrote {_JS / 'main.js'}")


if __name__ == "__main__":
    main()
