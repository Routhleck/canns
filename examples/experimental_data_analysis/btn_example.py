from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Silence known runtime warnings during clustering on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message="Found Intel OpenMP.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="threadpoolctl",
)
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak.*",
    category=UserWarning,
)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Ensure package root is on sys.path when running this file directly.
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))

from canns.utils.example_outputs import get_example_output_dir

OUTPUT_DIR = get_example_output_dir(__file__)

from canns.analyzer.data.cell_classification import (
    BTNAnalyzer,
    BTNConfig,
    plot_btn_autocorr_summary,
    plot_btn_distance_matrix,
)

# ==================== User settings ====================
npz_path = os.environ.get("BTN_NPZ_PATH", "28304_1_ASA_mec_full_cm.npz")
N_CLUSTERS = 3  # try 3 for BTN; try 4 if you want an extra small cluster
NBS = 40  # kNN neighbors; try 30-80
SMOOTH_SIGMA = 2.0  # try 2-4
MIN_SPIKES = 200  # set 0 to disable filtering
PLOT_DIAGRAM = False  # True to show Tomato persistence diagram

# Baseline clustering diagnostics (for data separability)
RUN_BASELINE = True
BASELINE_K = 3
BASELINE_GMM = True

# ==================== Load data ====================
npz = np.load(npz_path, allow_pickle=True)
spike_data = {k: npz[k] for k in npz.files}

# ==================== Optional spike-count filtering ====================
spike_raw = spike_data["spike"]
if hasattr(spike_raw, "item") and np.asarray(spike_raw).shape == ():
    spike_raw = spike_raw.item()

if isinstance(spike_raw, dict):
    spike_list = [np.asarray(spike_raw[k]) for k in spike_raw.keys()]
else:
    spike_list = [np.asarray(spike_raw[i]) for i in range(len(spike_raw))]

counts = np.array([len(s) for s in spike_list])
print("spike count stats:", {
    "min": int(counts.min()),
    "median": float(np.median(counts)),
    "max": int(counts.max()),
})

if MIN_SPIKES > 0:
    keep = counts >= MIN_SPIKES
    spike_list = [s for s, k in zip(spike_list, keep) if k]
    print(f"keeping {len(spike_list)}/{len(counts)} neurons with >= {MIN_SPIKES} spikes")

spike_data = dict(spike_data)
spike_data["spike"] = {i: s for i, s in enumerate(spike_list)}

# ==================== BTN clustering ====================
cfg = BTNConfig(
    n_clusters=N_CLUSTERS,
    nbs=NBS,
    maxt=0.2,
    res=1e-3,
    smooth_sigma=SMOOTH_SIGMA,
)
analyzer = BTNAnalyzer(cfg)
result = analyzer.classify_btn(
    spike_data,
    return_intermediates=True,
    plot_diagram=PLOT_DIAGRAM,
)

uniq, counts = np.unique(result.labels, return_counts=True)
print("cluster sizes (Tomato):", dict(zip(uniq, counts)))

# ==================== Baseline separability checks ====================
if RUN_BASELINE:
    X = result.intermediates["acorr_smooth"]
    if X.shape[0] >= BASELINE_K:
        km = KMeans(n_clusters=BASELINE_K, n_init=10, random_state=0)
        km_labels = km.fit_predict(X)
        try:
            sil = silhouette_score(X, km_labels, metric="cosine")
        except Exception:
            sil = float("nan")
        print("KMeans sizes:", dict(zip(*np.unique(km_labels, return_counts=True))))
        print("KMeans silhouette (cosine):", sil)

        if BASELINE_GMM:
            gmm = GaussianMixture(n_components=BASELINE_K, random_state=0)
            gmm_labels = gmm.fit_predict(X)
            try:
                sil_g = silhouette_score(X, gmm_labels, metric="cosine")
            except Exception:
                sil_g = float("nan")
            print("GMM sizes:", dict(zip(*np.unique(gmm_labels, return_counts=True))))
            print("GMM silhouette (cosine):", sil_g)
    else:
        print("Not enough neurons for baseline clustering.")

# ==================== Map clusters to BTN labels ====================
# Adjust this after inspecting cluster sizes and curves.
mapping = {0: "B", 1: "T", 2: "N"}

# ==================== Distance matrix heatmap ====================
plot_btn_distance_matrix(
    dist=result.intermediates["distance_matrix"],
    labels=result.labels,
    mapping=mapping,
    sort_by_label=True,
    save_path=str(OUTPUT_DIR / "btn_distance.png"),
    show=False,
)

# ==================== Mean autocorr curves ====================
plot_btn_autocorr_summary(
    acorr=result.intermediates["acorr"],
    labels=result.labels,
    bin_times=result.intermediates["bin_times"],
    mapping=mapping,
    normalize="probability",
    smooth_sigma=SMOOTH_SIGMA,
    save_path=str(OUTPUT_DIR / "btn_autocorr.png"),
    show=False,
)

print(f"Saved: {OUTPUT_DIR / 'btn_distance.png'}, {OUTPUT_DIR / 'btn_autocorr.png'}")
