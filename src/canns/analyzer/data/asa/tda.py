from __future__ import annotations

import logging
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from numbers import Integral
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from canns_lib import _ripser_core
from canns_lib.ripser import ripser
from matplotlib import gridspec
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing

from .config import ProcessingError, TDAConfig

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


def tda_vis(embed_data: np.ndarray, config: TDAConfig | None = None, **kwargs) -> dict[str, Any]:
    """
    Topological Data Analysis visualization with optional shuffle testing.

    Parameters
    ----------
    embed_data : np.ndarray
        Embedded spike train data of shape (T, N).
    config : TDAConfig, optional
        Configuration object with all TDA parameters. If None, legacy kwargs are used.
    **kwargs : Any
        Legacy keyword parameters (``dim``, ``num_times``, ``active_times``, ``k``,
        ``n_points``, ``metric``, ``nbs``, ``maxdim``, ``coeff``, ``show``,
        ``do_shuffle``, ``num_shuffles``, ``progress_bar``, ``standardize``).

    Returns
    -------
    dict
        Dictionary containing:
        - ``persistence``: persistence diagrams from real data.
        - ``indstemp``: indices of sampled points.
        - ``movetimes``: selected time points.
        - ``n_points``: number of sampled points.
        - ``shuffle_max``: shuffle analysis results (if ``do_shuffle=True``), else ``None``.

    Examples
    --------
    >>> from canns.analyzer.data import TDAConfig, tda_vis
    >>> cfg = TDAConfig(maxdim=1, do_shuffle=False, show=False)
    >>> result = tda_vis(embed_data, config=cfg)  # doctest: +SKIP
    >>> sorted(result.keys())
    ['indstemp', 'movetimes', 'n_points', 'persistence', 'shuffle_max']
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = TDAConfig(**kwargs)
    elif kwargs:
        raise TypeError("Pass either config or TDA keyword arguments, not both")
    config = _resolve_configuration(config)
    _check_backend_availability(config)
    if config.do_shuffle:
        _validate_shuffle_input(
            embed_data,
            config.num_shuffles,
            config.shuffle_seed,
            config.shuffle_shifts,
            config.shuffle_workers,
        )

    try:
        # Compute persistent homology for real data
        print("Computing persistent homology for real data...")
        real_persistence = _compute_real_persistence(embed_data, config)

        # Perform shuffle analysis if requested
        shuffle_max = None
        if config.do_shuffle:
            shuffle_max = _perform_shuffle_analysis(embed_data, config)

        # Visualization
        _handle_visualization(real_persistence["persistence"], shuffle_max, config)

        # Return results as dictionary
        return {
            "persistence": real_persistence["persistence"],
            "indstemp": real_persistence["indstemp"],
            "movetimes": real_persistence["movetimes"],
            "n_points": real_persistence["n_points"],
            "shuffle_max": shuffle_max,
        }

    except Exception as e:
        raise ProcessingError(f"TDA analysis failed: {e}") from e


def _compute_real_persistence(embed_data: np.ndarray, config: TDAConfig) -> dict[str, Any]:
    """Compute persistent homology for real data with progress tracking."""
    _validate_pipeline_parameters(embed_data, config)
    logging.info("Processing real data - Starting TDA analysis (5 steps)")

    # Step 1: Time point downsampling
    logging.info("Step 1/5: Time point downsampling")
    times_cube = _downsample_timepoints(embed_data, config.num_times)

    # Step 2: Select most active time points
    logging.info("Step 2/5: Selecting active time points")
    movetimes = _select_active_timepoints(embed_data, times_cube, config.active_times)

    # Step 3: PCA dimensionality reduction
    logging.info("Step 3/5: PCA dimensionality reduction")
    dimred = _apply_pca_reduction(embed_data, movetimes, config.dim, config.standardize)

    # Step 4: Point cloud sampling (denoising)
    logging.info("Step 4/5: Point cloud denoising")
    indstemp = _apply_denoising(dimred, config)

    # Step 5: Compute persistent homology
    logging.info("Step 5/5: Computing persistent homology")
    persistence = _compute_persistence_homology(dimred, indstemp, config)

    logging.info("TDA analysis completed successfully")

    # Return all necessary data in dictionary format
    return {
        "persistence": persistence,
        "indstemp": indstemp,
        "movetimes": movetimes,
        "n_points": config.n_points,
    }


def _downsample_timepoints(embed_data: np.ndarray, num_times: int) -> np.ndarray:
    """Downsample timepoints for computational efficiency."""
    return np.arange(0, embed_data.shape[0], num_times)


def _select_active_timepoints(
    embed_data: np.ndarray, times_cube: np.ndarray, active_times: int
) -> np.ndarray:
    """Select most active timepoints based on total activity."""
    activity_scores = np.sum(embed_data[times_cube, :], 1)
    # Match external TDAvis: sort indices first, then map to times_cube
    movetimes = np.sort(np.argsort(activity_scores)[-active_times:])
    return times_cube[movetimes]


def _apply_pca_reduction(
    embed_data: np.ndarray, movetimes: np.ndarray, dim: int, standardize: bool
) -> np.ndarray:
    """Apply PCA dimensionality reduction."""
    subset = embed_data[movetimes, :]
    if standardize:
        scaled_data = preprocessing.scale(subset)
    else:
        scaled_data = np.asarray(subset, dtype=np.float64)
        scaled_data = scaled_data - scaled_data.mean(axis=0)
    dimred, *_ = _pca(scaled_data, dim=dim)
    return dimred


def _apply_denoising(dimred: np.ndarray, config: TDAConfig) -> np.ndarray:
    """Apply point cloud denoising."""
    indstemp, *_ = _sample_denoising(
        dimred,
        k=config.k,
        num_sample=config.n_points,
        omega=1,  # Match external TDAvis: uses 1, not default 0.2
        metric=config.metric,
        backend=config.sampling_backend,
    )
    return indstemp


def _compute_persistence_homology(
    dimred: np.ndarray, indstemp: np.ndarray, config: TDAConfig
) -> dict[str, Any]:
    """Compute persistent homology using ripser."""
    d = _second_build(dimred, indstemp, metric=config.metric, nbs=config.nbs)
    np.fill_diagonal(d, 0)

    return ripser(
        d,
        maxdim=config.maxdim,
        coeff=config.coeff,
        do_cocycles=config.do_cocycles,
        distance_matrix=True,
        progress_bar=config.progress_bar,
        **_ph_threshold_options(d, config),
    )


def _perform_shuffle_analysis(embed_data: np.ndarray, config: TDAConfig) -> dict[int, Any]:
    """Perform shuffle analysis with progress tracking."""
    print(f"\nStarting shuffle analysis with {config.num_shuffles} iterations...")

    shuffle_max = _run_shuffle_analysis(
        embed_data,
        num_shuffles=config.num_shuffles,
        num_cores=config.shuffle_workers,
        progress_bar=config.progress_bar,
        config=config,
    )

    # Print shuffle analysis summary
    _print_shuffle_summary(shuffle_max)

    return shuffle_max


def _print_shuffle_summary(shuffle_max: dict[int, Any]) -> None:
    """Print summary of shuffle analysis results."""
    print("\nSummary of shuffle-based analysis:")
    for dim_idx in sorted(shuffle_max):
        if shuffle_max and dim_idx in shuffle_max and shuffle_max[dim_idx]:
            values = shuffle_max[dim_idx]
            print(
                f"H{dim_idx}: {len(values)} valid iterations | "
                f"Mean maximum persistence: {np.mean(values):.4f} | "
                f"99.9th percentile: {np.percentile(values, 99.9):.4f}"
            )


def _handle_visualization(
    real_persistence: dict[str, Any], shuffle_max: dict[int, Any] | None, config: TDAConfig
) -> None:
    """Handle visualization based on configuration."""
    if config.show:
        if config.do_shuffle and shuffle_max is not None:
            _plot_barcode_with_shuffle(real_persistence, shuffle_max)
        else:
            _plot_barcode(real_persistence)
        plt.show()
    else:
        plt.close()


def _compute_persistence(
    sspikes,
    dim=6,
    num_times=5,
    active_times=15000,
    k=1000,
    n_points=1200,
    metric="cosine",
    nbs=800,
    maxdim=1,
    coeff=47,
    progress_bar=True,
    standardize=True,
    sampling_backend="python",
    ph_threshold_policy="legacy",
    ph_threshold=None,
    do_cocycles=True,
):
    """Compatibility wrapper around the same complete real-data pipeline."""
    config = _resolve_configuration(
        TDAConfig(
            dim=dim,
            num_times=num_times,
            active_times=active_times,
            k=k,
            n_points=n_points,
            metric=metric,
            nbs=nbs,
            maxdim=maxdim,
            coeff=coeff,
            show=False,
            progress_bar=progress_bar,
            standardize=standardize,
            sampling_backend=sampling_backend,
            ph_threshold_policy=ph_threshold_policy,
            ph_threshold=ph_threshold,
            do_cocycles=do_cocycles,
        )
    )
    _check_backend_availability(config)
    return _compute_real_persistence(sspikes, config)["persistence"]


def _resolve_configuration(config: TDAConfig) -> TDAConfig:
    if config.sampling_backend not in {"python", "rust"}:
        raise ValueError("sampling_backend must be 'python' or 'rust'")
    if config.ph_threshold_policy not in {"legacy", "max_finite_float32"}:
        raise ValueError("ph_threshold_policy must be 'legacy' or 'max_finite_float32'")
    if config.ph_threshold is not None:
        if config.ph_threshold_policy != "legacy":
            raise ValueError("Explicit ph_threshold cannot be combined with max_finite_float32")
        if np.isnan(config.ph_threshold) or config.ph_threshold < 0:
            raise ValueError("ph_threshold must be nonnegative and not NaN")
    return config


def _validate_pipeline_parameters(activity, config):
    activity = np.asarray(activity)
    if activity.ndim != 2 or activity.dtype.kind not in "fiu" or 0 in activity.shape:
        raise ValueError("TDA activity must be a nonempty real numeric (time, neurons) matrix")
    if not np.isfinite(activity).all():
        raise ValueError("TDA activity must contain only finite values")
    for name in ("dim", "num_times", "active_times", "k", "n_points", "nbs"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(config.maxdim, bool)
        or not isinstance(config.maxdim, Integral)
        or config.maxdim < 0
    ):
        raise ValueError("maxdim must be a nonnegative integer")
    candidates = min(config.active_times, len(range(0, len(activity), config.num_times)))
    if candidates < 2:
        raise ValueError("PCA requires at least two active timepoints")
    if config.dim > min(candidates, activity.shape[1]):
        raise ValueError("dim exceeds the available PCA dimensions")
    if config.k > candidates or config.n_points > candidates:
        raise ValueError("k and n_points cannot exceed the available active timepoints")
    if config.nbs > config.n_points:
        raise ValueError("nbs cannot exceed n_points")


def _rust_fuzzy_union_function():
    function = getattr(_ripser_core, "fuzzy_union", None)
    if not callable(function):
        raise ImportError(
            "sampling_backend='rust' requires canns-lib with fuzzy_union; "
            "upgrade canns-lib or explicitly select sampling_backend='python'"
        )
    return function


def _check_backend_availability(config: TDAConfig):
    # Capability checks happen before real analysis; numerical failures never
    # select a different pipeline or restart with a different random shift.
    if config.sampling_backend == "rust":
        _rust_fuzzy_union_function()


def _ph_threshold_options(distance, config):
    if config.ph_threshold_policy == "legacy":
        return {} if config.ph_threshold is None else {"thresh": config.ph_threshold}
    if config.ph_threshold_policy != "max_finite_float32":
        raise ValueError("Unknown PH threshold policy")
    values = np.asarray(distance)
    if np.isnan(values).any() or np.isneginf(values).any():
        raise ValueError("PH graph contains NaN or negative infinity")
    with np.errstate(over="ignore", invalid="ignore"):
        finite = values[np.isfinite(values)].astype(np.float32)
    if not finite.size or not np.isfinite(finite).all():
        raise ValueError("PH graph finite edges must remain finite in float32")
    threshold = float(finite.max())
    if threshold >= np.finfo(np.float32).max:
        raise ValueError("Maximum finite threshold collides with Ripser's sentinel")
    return {"thresh": threshold}


def _pca(data, dim=2):
    """
    Perform PCA (Principal Component Analysis) for dimensionality reduction.

    Parameters:
        data (ndarray): Input data matrix of shape (N_samples, N_features).
        dim (int): Target dimension for PCA projection.

    Returns:
        components (ndarray): Projected data of shape (N_samples, dim).
        var_exp (list): Variance explained by each principal component.
        evals (ndarray): Eigenvalues corresponding to the selected components.
    """
    if dim == 1:
        centered = np.asarray(data) - np.mean(data, axis=0)
        covariance = np.atleast_2d(np.cov(centered, rowvar=False))
        values, vectors = np.linalg.eigh(covariance)
        index = int(np.argmax(values))
        total = float(values.sum())
        explained = [float(values[index]) / total * 100 if total else 0.0]
        return centered @ vectors[:, index : index + 1], explained, values[index : index + 1]
    if dim < 1:
        raise ValueError("dim must be positive")
    _ = data.shape
    # mean center the data
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = np.linalg.eig(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dim]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    tot = np.sum(evals)
    var_exp = [(i / tot) * 100 for i in sorted(evals[:dim], reverse=True)]
    components = np.dot(evecs.T, data.T).T
    return components, var_exp, evals[:dim]


def _sample_denoising(data, k=10, num_sample=500, omega=0.2, metric="euclidean", backend="python"):
    """
    Perform denoising and greedy sampling based on mutual k-NN graph.

    Parameters:
        data (ndarray): High-dimensional point cloud data.
        k (int): Number of neighbors for local density estimation.
        num_sample (int): Number of samples to retain.
        omega (float): Suppression factor during greedy sampling.
        metric (str): Distance metric used for kNN ('euclidean', 'cosine', etc).

    Returns:
        inds (ndarray): Indices of sampled points.
        d (ndarray): Pairwise similarity matrix of sampled points.
        Fs (ndarray): Sampling scores at each step.
    """
    if not 1 <= k <= len(data) or not 1 <= num_sample <= len(data):
        raise ValueError("k and num_sample must be in [1, available points]")
    if backend == "rust":
        return _sample_denoising_rust(data, k, num_sample, omega, metric)
    if backend != "python":
        raise ValueError("sampling backend must be 'python' or 'rust'")
    if HAS_NUMBA:
        return _sample_denoising_numba(data, k, num_sample, omega, metric)
    else:
        return _sample_denoising_numpy(data, k, num_sample, omega, metric)


def _sample_denoising_rust(data, k, num_sample, omega, metric):
    """Same density graph and greedy selection with bounded sorting temporaries."""
    fuzzy_union = _rust_fuzzy_union_function()
    n = data.shape[0]
    distances = squareform(pdist(data, metric))
    width = k
    indices = np.empty((n, width), dtype=np.int64)
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        indices[start:stop] = np.argsort(distances[start:stop], axis=1)[:, :width]
    knn_distances = np.take_along_axis(distances, indices, axis=1)
    del distances
    sigmas, rhos = _smooth_knn_dist(knn_distances, k, local_connectivity=0)
    rows, cols, vals = _compute_membership_strengths(indices, knn_distances, sigmas, rhos)
    del indices, knn_distances, sigmas, rhos
    adjacency = fuzzy_union(
        np.ascontiguousarray(rows, dtype=np.int64),
        np.ascontiguousarray(cols, dtype=np.int64),
        np.ascontiguousarray(vals, dtype=np.float64),
        n,
    )
    del rows, cols, vals
    selected, scores = _greedy_sampling_numba(adjacency, num_sample, omega)
    sampled = _build_distance_matrix_numba(adjacency, selected)
    return selected, sampled, scores


def _sample_denoising_numpy(data, k=10, num_sample=500, omega=0.2, metric="euclidean"):
    """Original numpy implementation for fallback."""
    n = data.shape[0]
    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = _smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = result + transpose - prod_matrix
    result.eliminate_zeros()
    X = result.toarray()
    F = np.sum(X, 1)
    Fs = np.zeros(num_sample)
    Fs[0] = np.max(F)
    i = np.argmax(F)
    inds_all = np.arange(n)
    inds_left = inds_all > -1
    inds_left[i] = False
    inds = np.zeros(num_sample, dtype=int)
    inds[0] = i
    for j in np.arange(1, num_sample):
        F -= omega * X[i, :]
        Fmax = np.argmax(F[inds_left])
        # Exactly match external TDAvis implementation (including the indexing logic)
        Fs[j] = F[Fmax]
        i = inds_all[inds_left][Fmax]

        inds_left[i] = False
        inds[j] = i
    d = np.zeros((num_sample, num_sample))

    for j, i in enumerate(inds):
        d[j, :] = X[i, inds]
    return inds, d, Fs


def _sample_denoising_numba(data, k=10, num_sample=500, omega=0.2, metric="euclidean"):
    """Optimized numba implementation."""
    n = data.shape[0]
    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    sigmas, rhos = _smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals = _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)

    # Build symmetric adjacency matrix using optimized function
    X_adj = _build_adjacency_matrix_numba(rows, cols, vals, n)

    # Greedy sampling using optimized function
    inds, Fs = _greedy_sampling_numba(X_adj, num_sample, omega)

    # Build final distance matrix
    d = _build_distance_matrix_numba(X_adj, inds)

    return inds, d, Fs


@njit(fastmath=True)
def _build_adjacency_matrix_numba(rows, cols, vals, n):
    """Build symmetric adjacency matrix efficiently with numba.

    This matches the scipy sparse matrix operations:
    result = result + transpose - prod_matrix
    where prod_matrix = result.multiply(transpose)
    """
    # Initialize matrices
    X = np.zeros((n, n), dtype=np.float64)
    X_T = np.zeros((n, n), dtype=np.float64)

    # Build adjacency matrix and its transpose simultaneously
    for i in range(len(rows)):
        X[rows[i], cols[i]] = vals[i]
        X_T[cols[i], rows[i]] = vals[i]  # Transpose

    # Apply the symmetrization formula: A = A + A^T - A ⊙ A^T (vectorized)
    # This matches scipy's: result + transpose - prod_matrix
    X[:, :] = X + X_T - X * X_T

    return X


@njit(fastmath=True)
def _greedy_sampling_numba(X, num_sample, omega):
    """Optimized greedy sampling with numba."""
    n = X.shape[0]
    F = np.sum(X, axis=1)
    Fs = np.zeros(num_sample)
    inds = np.zeros(num_sample, dtype=np.int64)
    inds_left = np.ones(n, dtype=np.bool_)

    # Initialize with maximum F
    i = np.argmax(F)
    Fs[0] = F[i]
    inds[0] = i
    inds_left[i] = False

    # Greedy sampling loop
    for j in range(1, num_sample):
        # Update F values
        for k in range(n):
            F[k] -= omega * X[i, k]

        # Find maximum among remaining points (matching numpy logic exactly)
        max_val = -np.inf
        max_idx = -1
        for k in range(n):
            if inds_left[k] and F[k] > max_val:
                max_val = F[k]
                max_idx = k

        # Record the F value using the selected index (matching external TDAvis)
        i = max_idx
        Fs[j] = F[i]
        inds[j] = i
        inds_left[i] = False

    return inds, Fs


@njit(fastmath=True)
def _build_distance_matrix_numba(X, inds):
    """Build final distance matrix efficiently with numba."""
    num_sample = len(inds)
    d = np.zeros((num_sample, num_sample))

    for j in range(num_sample):
        for k in range(num_sample):
            d[j, k] = X[inds[j], inds[k]]

    return d


@njit(fastmath=True)
def _smooth_knn_dist(distances, k, n_iter=64, local_connectivity=0.0, bandwidth=1.0):
    """
    Compute smoothed local distances for kNN graph with entropy balancing.

    Parameters:
        distances (ndarray): kNN distance matrix.
        k (int): Number of neighbors.
        n_iter (int): Number of binary search iterations.
        local_connectivity (float): Minimum local connectivity.
        bandwidth (float): Bandwidth parameter.

    Returns:
        sigmas (ndarray): Smoothed sigma values for each point.
        rhos (ndarray): Minimum distances (connectivity cutoff) for each point.
    """
    target = np.log2(k) * bandwidth
    #    target = np.log(k) * bandwidth
    #    target = k

    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = np.inf
        mid = 1.0

        # Vectorized computation of non-zero distances
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > 1e-5:
                    rho[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1])
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        # Vectorized binary search loop - compute all at once instead of loop
        for _ in range(n_iter):
            # Vectorized computation: compute all distances at once
            d_array = distances[i, 1:] - rho[i]
            # Vectorized conditional: use np.where for conditional computation
            psum = np.sum(np.where(d_array > 0, np.exp(-(d_array / mid)), 1.0))

            if np.fabs(psum - target) < 1e-5:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        # Optimized mean computation - reuse ith_distances
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < 1e-3 * mean_ith_distances:
                result[i] = 1e-3 * mean_ith_distances
        else:
            if result[i] < 1e-3 * mean_distances:
                result[i] = 1e-3 * mean_distances

    return result, rho


@njit(parallel=True, fastmath=True)
def _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):
    """
    Compute membership strength matrix from smoothed kNN graph.

    Parameters:
        knn_indices (ndarray): Indices of k-nearest neighbors.
        knn_dists (ndarray): Corresponding distances.
        sigmas (ndarray): Local bandwidths.
        rhos (ndarray): Minimum distance thresholds.

    Returns:
        rows (ndarray): Row indices for sparse matrix.
        cols (ndarray): Column indices for sparse matrix.
        vals (ndarray): Weight values for sparse matrix.
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    cols = np.zeros((n_samples * n_neighbors), dtype=np.int64)
    vals = np.zeros((n_samples * n_neighbors), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                # val = ((knn_dists[i, j] - rhos[i]) / (sigmas[i]))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def _second_build(data, indstemp, nbs=800, metric="cosine"):
    """
    Reconstruct distance matrix after denoising for persistent homology.

    Parameters:
        data (ndarray): PCA-reduced data matrix.
        indstemp (ndarray): Indices of sampled points.
        nbs (int): Number of neighbors in reconstructed graph.
        metric (str): Distance metric ('cosine', 'euclidean', etc).

    Returns:
        d (ndarray): Symmetric distance matrix used for persistent homology.
    """
    # Filter the data using the sampled point indices
    data = data[indstemp, :]

    # Compute the pairwise distance matrix
    X = squareform(pdist(data, metric))
    knn_indices = np.argsort(X)[:, :nbs]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()

    # Compute smoothed kernel widths
    sigmas, rhos = _smooth_knn_dist(knn_dists, nbs, local_connectivity=0)
    rows, cols, vals = _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)

    # Construct a sparse graph
    result = coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = result + transpose - prod_matrix
    result.eliminate_zeros()

    # Build the final distance matrix
    d = result.toarray()
    # Match external TDAvis: direct negative log without epsilon handling
    # Temporarily suppress divide by zero warning to match external behavior
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -np.log(d)
    np.fill_diagonal(d, 0)

    return d


def _fast_pca_transform(data, components):
    """Fast PCA transformation using numba."""
    return np.dot(data, components.T)


def _run_shuffle_analysis(
    sspikes, num_shuffles=None, num_cores=None, progress_bar=None, *, config=None, **kwargs
):
    """Run the complete real-data pipeline after every independent neuron shift.

    CANNs owns the analysis and bounded scheduler. Every real and shuffled
    analysis calls the same Rust PH backend through canns_lib.ripser.ripser.
    Failures retain the iteration and offsets; there is no backend fallback.
    """
    if config is None:
        config = TDAConfig(**kwargs)
    elif kwargs:
        raise TypeError("Pass either config or shuffle TDA keyword arguments, not both")
    overrides = {}
    if num_shuffles is not None:
        overrides["num_shuffles"] = num_shuffles
    if num_cores is not None:
        overrides["shuffle_workers"] = num_cores
    if progress_bar is not None:
        overrides["progress_bar"] = progress_bar
    config = replace(config, **overrides)
    config = _resolve_configuration(config)
    _check_backend_availability(config)
    activity, offsets = _validate_shuffle_input(
        sspikes,
        config.num_shuffles,
        config.shuffle_seed,
        config.shuffle_shifts,
        config.shuffle_workers,
    )
    if offsets is None:
        offsets = np.random.default_rng(config.shuffle_seed).integers(
            0, activity.shape[0], size=(config.num_shuffles, activity.shape[1]), dtype=np.int64
        )
    offsets.flags.writeable = False
    # Keep the input and random plan stable while workers run. Scientific
    # settings are shared with real analysis; only presentation controls differ.
    activity = np.array(activity, copy=True)
    activity.flags.writeable = False
    analysis_config = replace(config, do_shuffle=False, show=False, progress_bar=False)
    return _run_asa_shuffle(activity, offsets, config, analysis_config)


def _shuffle_persistence_pipeline(activity, *, config):
    """Private ASA round, never a callback passed to another library."""
    persistence = _compute_real_persistence(activity, config)["persistence"]
    if len(persistence["dgms"]) != config.maxdim + 1:
        raise ValueError("Pipeline returned missing or extra homology dimensions")
    return persistence


def _validate_shuffle_input(activity, count, seed, shifts, workers):
    for name, value in (("num_shuffles", count), ("shuffle_workers", workers)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    _check_shuffle_concurrency(min(count, workers))
    activity = np.asarray(activity)
    if activity.ndim != 2 or 0 in activity.shape or activity.dtype.kind not in "fiu":
        raise ValueError("Shuffle activity must be a nonempty real numeric (time, neurons) matrix")
    if not np.isfinite(activity).all():
        raise ValueError("Shuffle activity must contain only finite values")
    if seed is not None and shifts is not None:
        raise ValueError("shuffle_seed and shuffle_shifts are mutually exclusive")
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0
    ):
        raise ValueError("shuffle_seed must be None or a nonnegative integer")
    offsets = None
    if shifts is not None:
        offsets = np.asarray(shifts)
        if offsets.shape != (count, activity.shape[1]) or offsets.dtype.kind not in "iu":
            raise ValueError("shuffle_shifts must be an integer (num_shuffles, neurons) matrix")
        if (offsets < 0).any() or (offsets >= activity.shape[0]).any():
            raise ValueError("shuffle_shifts must be in [0, timepoints)")
        offsets = np.array(offsets, dtype=np.int64, copy=True)
    return activity, offsets


def _check_shuffle_concurrency(workers):
    """Reject Numba workqueue before concurrent ASA analyses can abort Python."""
    if HAS_NUMBA and workers > 1:
        from numba import config, get_num_threads, threading_layer

        if config.DISABLE_JIT:
            return
        # This public API initializes the selected layer before querying it.
        get_num_threads()
        if threading_layer() == "workqueue":
            raise ValueError(
                "Concurrent ASA shuffles cannot use Numba's non-thread-safe workqueue layer. "
                "Set shuffle_workers=1, or configure an installed thread-safe Numba "
                "threading backend before starting Python."
            )


class ShuffleIterationError(ProcessingError):
    """A failed ASA shuffle is evidence, never a missing/zero null draw."""

    def __init__(self, index, offsets, original_exception):
        self.index = int(index)
        self.offsets = np.array(offsets, dtype=np.int64, copy=True)
        self.offsets.flags.writeable = False
        self.original_exception = original_exception
        super().__init__(
            f"Shuffle {index} failed with offsets {self.offsets.tolist()}: {original_exception}"
        )


def _finite_lifetime_summary(persistence, maxdim):
    diagrams = persistence["dgms"]
    if len(diagrams) != maxdim + 1:
        raise ValueError("Pipeline returned missing or extra homology dimensions")
    maxima, essential = {}, {}
    for dim, diagram in enumerate(diagrams):
        a = np.asarray(diagram)
        if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in "fiu":
            raise ValueError(f"H{dim} diagram must be a real numeric (n, 2) array")
        if not np.isfinite(a[:, 0]).all() or np.isnan(a[:, 1]).any():
            raise ValueError(f"H{dim} has invalid birth/death values")
        if np.isneginf(a[:, 1]).any() or (a[:, 1] < a[:, 0]).any():
            raise ValueError(f"H{dim} has invalid death values")
        finite = a[np.isfinite(a[:, 1])]
        essential[dim] = int(np.isposinf(a[:, 1]).sum())
        if not len(finite):
            maxima[dim] = 0.0
        elif a.dtype.kind in "iu":
            maxima[dim] = float(max(int(death) - int(birth) for birth, death in finite))
        else:
            cast = finite.astype(np.result_type(a.dtype, np.float64), copy=False)
            maxima[dim] = float(np.max(cast[:, 1] - cast[:, 0]))
        if not np.isfinite(maxima[dim]):
            raise ValueError(f"H{dim} finite lifetime overflowed")
    return maxima, essential


def _run_asa_shuffle(activity, offsets, config, analysis_config):
    """Execute complete ASA analyses with bounded concurrency and ordered results."""
    results = [None] * config.num_shuffles

    def work(index):
        try:
            shifted = _shuffle_spike_trains(activity, offsets[index])
            persistence = _shuffle_persistence_pipeline(shifted, config=analysis_config)
            return _finite_lifetime_summary(persistence, config.maxdim)
        except Exception as exc:
            raise ShuffleIterationError(index, offsets[index], exc) from exc

    workers = min(config.shuffle_workers, config.num_shuffles)
    if workers == 1:
        for index in range(config.num_shuffles):
            results[index] = work(index)
    else:
        # Never submit the entire batch: at most workers complete analyses are
        # queued/running, with one shifted activity matrix per active worker.
        executor = ThreadPoolExecutor(max_workers=workers)
        pending = {}
        next_index = 0
        try:
            while next_index < config.num_shuffles or pending:
                while next_index < config.num_shuffles and len(pending) < workers:
                    pending[executor.submit(work, next_index)] = next_index
                    next_index += 1
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=pending.__getitem__):
                    index = pending.pop(future)
                    results[index] = future.result()
        except BaseException:
            for future in pending:
                future.cancel()
            # Running analyses cannot be forcibly stopped; surface the failed
            # iteration immediately and cancel any work that has not started.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    if any(any(count for dim, count in essential.items() if dim > 0) for _, essential in results):
        warnings.warn(
            "Shuffle diagrams contain essential higher-dimensional intervals; "
            "shuffle_max summarizes finite lifetimes only.",
            RuntimeWarning,
            stacklevel=3,
        )
    return {dim: [result[0][dim] for result in results] for dim in range(config.maxdim + 1)}


def _run_shuffle_analysis_multiprocessing(
    sspikes, num_shuffles=1000, num_cores=4, progress_bar=True, **kwargs
):
    """Compatibility entry point for the bounded complete-pipeline reference."""
    return _run_shuffle_analysis(sspikes, num_shuffles, num_cores, progress_bar, **kwargs)


def _process_single_shuffle(args):
    """Compatibility worker; numerical errors propagate instead of returning {}."""
    index, activity, kwargs = args
    config = _resolve_configuration(TDAConfig(**kwargs))
    offsets = np.random.default_rng().integers(0, len(activity), size=activity.shape[1])
    try:
        persistence = _shuffle_persistence_pipeline(
            _shuffle_spike_trains(activity, offsets), config=config
        )
        return _finite_lifetime_summary(persistence, config.maxdim)[0]
    except Exception as exc:
        raise ShuffleIterationError(index, offsets, exc) from exc


def _shuffle_spike_trains(sspikes, offsets=None):
    """Positive independent circular shifts on each neuron's full time series."""
    shuffled = np.empty_like(sspikes)
    if offsets is None:
        offsets = np.random.randint(0, len(sspikes), size=sspikes.shape[1])
    for neuron, shift in enumerate(offsets):
        shuffled[:, neuron] = np.roll(sspikes[:, neuron], int(shift))
    return shuffled


def _plot_barcode(persistence):
    """
    Plot barcode diagram from persistent homology result.

    Parameters:
        persistence (dict): Persistent homology result with 'dgms' key.
    """
    alpha = 1
    inf_delta = 0.1
    dgms = persistence["dgms"]
    maxdim = len(dgms) - 1
    colormap = np.tile([0, 0.55, 0.2], (maxdim + 1, 1))
    dims = np.arange(maxdim + 1)
    labels = [f"$H_{dim}$" for dim in dims]

    # Determine axis range
    min_birth, max_death = 0, 0
    for dim in dims:
        persistence_dim = dgms[dim][~np.isinf(dgms[dim][:, 1]), :]
        if persistence_dim.size > 0:
            min_birth = min(min_birth, np.min(persistence_dim))
            max_death = max(max_death, np.max(persistence_dim))

    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta
    axis_start = min_birth - delta

    # Create plot
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(len(dims), 1)

    for dim in dims:
        axes = plt.subplot(gs[dim])
        axes.axis("on")
        axes.set_yticks([])
        axes.set_ylabel(labels[dim], rotation=0, labelpad=20, fontsize=12)

        d = np.copy(dgms[dim])
        d[np.isinf(d[:, 1]), 1] = infinity
        dlife = d[:, 1] - d[:, 0]

        # Select top 30 bars by lifetime
        dinds = np.argsort(dlife)[-30:]
        if dim > 0:
            dinds = dinds[np.flip(np.argsort(d[dinds, 0]))]

        axes.barh(
            0.5 + np.arange(len(dinds)),
            dlife[dinds],
            height=0.8,
            left=d[dinds, 0],
            alpha=alpha,
            color=colormap[dim],
            linewidth=0,
        )

        axes.plot([0, 0], [0, len(dinds)], c="k", linestyle="-", lw=1)
        axes.plot([0, len(dinds)], [0, 0], c="k", linestyle="-", lw=1)
        axes.set_xlim([axis_start, infinity])

    plt.tight_layout()
    return fig


def _plot_barcode_with_shuffle(persistence, shuffle_max):
    """
    Plot barcode with shuffle region markers.
    """
    # Handle case where shuffle_max is None
    if shuffle_max is None:
        shuffle_max = {}

    alpha = 1
    inf_delta = 0.1
    maxdim = len(persistence["dgms"]) - 1
    colormap = np.tile([0, 0.55, 0.2], (maxdim + 1, 1))
    dims = np.arange(maxdim + 1)

    min_birth, max_death = 0, 0
    for dim in dims:
        # Filter out infinite values
        valid_bars = [bar for bar in persistence["dgms"][dim] if not np.isinf(bar[1])]
        if valid_bars:
            min_birth = min(min_birth, np.min(valid_bars))
            max_death = max(max_death, np.max(valid_bars))

    # Handle case with no valid bars
    if max_death == 0 and min_birth == 0:
        min_birth = 0
        max_death = 1

    delta = (max_death - min_birth) * inf_delta
    infinity = max_death + delta

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(len(dims), 1)

    # Get shuffle thresholds (99.9th percentile for each dimension)
    thresholds = {}
    for dim in dims:
        if dim in shuffle_max and shuffle_max[dim]:
            thresholds[dim] = np.percentile(shuffle_max[dim], 99.9)
        else:
            thresholds[dim] = 0

    labels = [f"$H_{dim}$" for dim in dims]

    for _, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes.axis("on")
        axes.set_yticks([])
        if dim < len(labels):
            axes.set_ylabel(labels[dim], rotation=0, labelpad=20, fontsize=12)

        # Do not pre-filter out infinite bars; copy the full diagram instead
        d = np.copy(persistence["dgms"][dim])
        if d.size == 0:
            d = np.zeros((0, 2))

        # Map infinite death values to a finite upper bound for visualization
        d[np.isinf(d[:, 1]), 1] = infinity
        dlife = d[:, 1] - d[:, 0]

        # Select top 30 longest-lived bars
        if len(dlife) > 0:
            dinds = np.argsort(dlife)[-30:]
            if dim > 0:
                dinds = dinds[np.flip(np.argsort(d[dinds, 0]))]

            threshold = thresholds.get(dim, 0)

            # Draw paper-style shuffle shadow: the longest shuffle lifetime
            # aligned to the lower value (birth) of each original bar.
            for i, idx in enumerate(dinds):
                if threshold > 0:
                    axes.barh(
                        0.5 + i,
                        threshold,
                        height=0.9,
                        left=d[idx, 0],
                        alpha=0.35,
                        color="gray",
                        linewidth=0,
                        zorder=1,
                    )
                axes.barh(
                    0.5 + i,
                    dlife[idx],
                    height=0.62,
                    left=d[idx, 0],
                    alpha=alpha,
                    color=colormap[dim],
                    linewidth=0,
                    zorder=2,
                )

            indsall = len(dinds)
        else:
            indsall = 0

        axes.plot([0, 0], [0, indsall], c="k", linestyle="-", lw=1)
        axes.plot([0, indsall], [0, 0], c="k", linestyle="-", lw=1)
        axes.set_xlim([min_birth - delta, infinity])

    plt.tight_layout()
    return fig
