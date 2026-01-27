from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _base_mask(shape: Tuple[int, int], center_bins: int = 2) -> np.ndarray:
    """Create a shared mask: center disk + outside circle (corners)."""
    h, w = shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # outer circle radius: half-min dimension
    outer_r = min(cy, cx)
    mask_outer = rr > outer_r

    mask_center = rr <= float(center_bins)
    return mask_outer | mask_center


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size or a.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    da = np.sqrt((a * a).mean())
    db = np.sqrt((b * b).mean())
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return float((a * b).mean() / (da * db))


def _vectorize_autocorrs(autocorrs: np.ndarray, center_bins: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorize autocorrs into point-cloud matrix X and return the shared mask."""
    if autocorrs.ndim != 3:
        raise ValueError(f"autocorrs must be (N,H,W), got {autocorrs.shape}")
    N, H, W = autocorrs.shape
    mask = _base_mask((H, W), center_bins=center_bins)
    # Use shared mask only; replace NaN with 0 so dimensions match across cells.
    ac = np.nan_to_num(autocorrs, nan=0.0, posinf=0.0, neginf=0.0)
    X = ac[:, ~mask]  # (N, n_features)
    return X.astype(np.float32, copy=False), mask


def _build_knn_graph(X: np.ndarray, k: int = 30, metric: str = "manhattan"):
    """Return an igraph graph built from kNN with edge weights."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise ImportError(f"scikit-learn is required for kNN graph: {e}")

    try:
        import igraph as ig
    except Exception as e:
        raise ImportError(f"python-igraph is required for Leiden clustering: {e}")

    N = X.shape[0]
    k_eff = min(max(int(k), 1), max(N - 1, 1))

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nbrs.fit(X)
    dist, ind = nbrs.kneighbors(X, return_distance=True)

    edges = []
    weights = []
    eps = 1e-6
    seen = set()
    for i in range(N):
        for jj in range(1, k_eff + 1):
            j = int(ind[i, jj])
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            d = float(dist[i, jj])
            w = 1.0 / (d + eps)
            edges.append((a, b))
            weights.append(w)

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights
    return g


def _leiden_membership(graph, resolution: float = 1.0) -> np.ndarray:
    """Run Leiden and return membership array."""
    try:
        import leidenalg
    except Exception as e:
        raise ImportError(f"leidenalg is required for Leiden clustering: {e}")

    part = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"] if "weight" in graph.es.attributes() else None,
        resolution_parameter=float(resolution),
    )
    return np.asarray(part.membership, dtype=int)


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def identify_grid_modules_and_stats(
    autocorrs: np.ndarray,
    *,
    gridness_analyzer,
    center_bins: int = 2,
    k: int = 30,
    resolution: float = 1.0,
    score_thr: float = 0.3,
    consistency_thr: float = 0.5,
    min_cells: int = 10,
    merge_corr_thr: float = 0.7,
    metric: str = "manhattan",
) -> Dict[str, Any]:
    """Identify grid modules with Leiden clustering on autocorrelogram point cloud.

    Parameters
    ----------
    autocorrs : np.ndarray
        Array of shape (N, H, W).
    gridness_analyzer :
        An instance that provides compute_gridness_score(autocorr)->GridnessResult.
    center_bins : int
        Radius (in bins) to mask around center peak.
    k : int
        Neighbors for kNN graph.
    resolution : float
        Leiden resolution parameter.
    score_thr, consistency_thr, min_cells, merge_corr_thr
        Module acceptance and merging thresholds.

    Returns
    -------
    dict with keys:
        module_id (N,), cluster_id (N,), modules (list of dict), params
    """
    if autocorrs.ndim != 3:
        raise ValueError(f"autocorrs must be (N,H,W). Got {autocorrs.shape}")
    N, H, W = autocorrs.shape

    X, mask = _vectorize_autocorrs(autocorrs, center_bins=int(center_bins))
    g = _build_knn_graph(X, k=int(k), metric=str(metric))
    cluster_id = _leiden_membership(g, resolution=float(resolution))

    # group members
    clusters: Dict[int, np.ndarray] = {}
    for cid in np.unique(cluster_id):
        clusters[int(cid)] = np.where(cluster_id == cid)[0]

    base_mask = mask  # shared
    ac = np.nan_to_num(autocorrs, nan=0.0, posinf=0.0, neginf=0.0)

    cluster_stats = {}
    candidate_cids = []

    # compute cluster metrics
    for cid, idxs in clusters.items():
        if idxs.size == 0:
            continue
        avg = ac[idxs].mean(axis=0)
        med = np.median(ac[idxs], axis=0)

        gr = gridness_analyzer.compute_gridness_score(med)
        grid_score = float(getattr(gr, "score", getattr(gr, "grid_score", np.nan)))


        flat_avg = avg[~base_mask]
        cors = []
        for i in idxs:
            c = _safe_corr(flat_avg, ac[i][~base_mask])
            cors.append(c)
        consistency = float(np.median(cors)) if len(cors) else 0.0

        cluster_stats[cid] = {
            "cid": cid,
            "size": int(idxs.size),
            "grid_score": grid_score,
            "consistency": consistency,
            "gridness_result": gr,
            "avg_autocorr": avg,
        }

        if (grid_score > float(score_thr)) and (consistency > float(consistency_thr)) and (idxs.size >= int(min_cells)):
            candidate_cids.append(cid)

    # Merge candidate clusters based on avg autocorr correlation
    cand = list(candidate_cids)
    dsu = _DSU(len(cand))
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            ci, cj = cand[i], cand[j]
            ai = cluster_stats[ci]["avg_autocorr"][~base_mask]
            aj = cluster_stats[cj]["avg_autocorr"][~base_mask]
            corr = _safe_corr(ai, aj)
            if corr > float(merge_corr_thr):
                dsu.union(i, j)

    # Build merged modules
    root_to_members: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cand):
        r = dsu.find(idx)
        root_to_members.setdefault(r, []).append(cid)

    modules = []
    module_id = np.full((N,), -1, dtype=int)

    # assign module ids in stable order
    roots = sorted(root_to_members.keys())
    for mid, r in enumerate(roots):
        member_cids = root_to_members[r]
        member_idxs = np.concatenate([clusters[c] for c in member_cids]) if member_cids else np.array([], dtype=int)

        # compute module-level stats (median over clusters)
        gs = [cluster_stats[c]["grid_score"] for c in member_cids]
        cs = [cluster_stats[c]["consistency"] for c in member_cids]
        grid_score = float(np.median(gs)) if gs else float("nan")
        consistency = float(np.median(cs)) if cs else float("nan")

        module_id[member_idxs] = mid

        modules.append({
            "module_id": mid,
            "clusters": member_cids,
            "indices": member_idxs.astype(int),
            "size": int(member_idxs.size),
            "grid_score": grid_score,
            "consistency": consistency,
        })

    n_grid_cells = int((module_id != -1).sum())
    out = {
        "cluster_id": cluster_id.astype(int),
        "module_id": module_id.astype(int),
        "n_units": int(N),
        "n_grid_cells": int(n_grid_cells),
        "n_modules": int(len(modules)),
        "modules": modules,
        "params": {
            "center_bins": int(center_bins),
            "k": int(k),
            "resolution": float(resolution),
            "score_thr": float(score_thr),
            "consistency_thr": float(consistency_thr),
            "min_cells": int(min_cells),
            "merge_corr_thr": float(merge_corr_thr),
            "metric": str(metric),
        },
    }
    return out
