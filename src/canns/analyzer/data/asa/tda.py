from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from canns_lib.ripser import ripser
from matplotlib import gridspec
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from tqdm import tqdm

from .config import Constants, ProcessingError, TDAConfig

try:
    from numba import jit, njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(x):
        return range(x)


def tda_vis(embed_data: np.ndarray, config: TDAConfig | None = None, **kwargs) -> dict[str, Any]:
    """
    Topological Data Analysis visualization with optional shuffle testing.

    Parameters:
        embed_data : ndarray
            Embedded spike train data.
        config : TDAConfig, optional
            Configuration object with all TDA parameters
        **kwargs : backward compatibility parameters

    Returns:
        dict : Dictionary containing:
            - persistence: persistence diagrams from real data
            - indstemp: indices of sampled points
            - movetimes: selected time points
            - n_points: number of sampled points
            - shuffle_max: shuffle analysis results (if do_shuffle=True, otherwise None)
    """
    # Handle backward compatibility and configuration
    if config is None:
        config = TDAConfig(
            dim=kwargs.get("dim", 6),
            num_times=kwargs.get("num_times", 5),
            active_times=kwargs.get("active_times", 15000),
            k=kwargs.get("k", 1000),
            n_points=kwargs.get("n_points", 1200),
            metric=kwargs.get("metric", "cosine"),
            nbs=kwargs.get("nbs", 800),
            maxdim=kwargs.get("maxdim", 1),
            coeff=kwargs.get("coeff", 47),
            show=kwargs.get("show", True),
            do_shuffle=kwargs.get("do_shuffle", False),
            num_shuffles=kwargs.get("num_shuffles", 1000),
            progress_bar=kwargs.get("progress_bar", True),
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

    logging.info("Processing real data - Starting TDA analysis (5 steps)")

    # Step 1: Time point downsampling
    logging.info("Step 1/5: Time point downsampling")
    times_cube = _downsample_timepoints(embed_data, config.num_times)

    # Step 2: Select most active time points
    logging.info("Step 2/5: Selecting active time points")
    movetimes = _select_active_timepoints(embed_data, times_cube, config.active_times)

    # Step 3: PCA dimensionality reduction
    logging.info("Step 3/5: PCA dimensionality reduction")
    dimred = _apply_pca_reduction(embed_data, movetimes, config.dim)

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

def _apply_pca_reduction(embed_data: np.ndarray, movetimes: np.ndarray, dim: int) -> np.ndarray:
    """Apply PCA dimensionality reduction."""
    scaled_data = preprocessing.scale(embed_data[movetimes, :])
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
        do_cocycles=True,
        distance_matrix=True,
        progress_bar=config.progress_bar,
    )

def _perform_shuffle_analysis(embed_data: np.ndarray, config: TDAConfig) -> dict[int, Any]:
    """Perform shuffle analysis with progress tracking."""
    print(f"\nStarting shuffle analysis with {config.num_shuffles} iterations...")

    # Create parameters dict for shuffle analysis
    shuffle_params = {
        "dim": config.dim,
        "num_times": config.num_times,
        "active_times": config.active_times,
        "k": config.k,
        "n_points": config.n_points,
        "metric": config.metric,
        "nbs": config.nbs,
        "maxdim": config.maxdim,
        "coeff": config.coeff,
    }

    shuffle_max = _run_shuffle_analysis(
        embed_data,
        num_shuffles=config.num_shuffles,
        num_cores=Constants.MULTIPROCESSING_CORES,
        progress_bar=config.progress_bar,
        **shuffle_params,
    )

    # Print shuffle analysis summary
    _print_shuffle_summary(shuffle_max)

    return shuffle_max

def _print_shuffle_summary(shuffle_max: dict[int, Any]) -> None:
    """Print summary of shuffle analysis results."""
    print("\nSummary of shuffle-based analysis:")
    for dim_idx in [0, 1, 2]:
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
):
    # Time point downsampling
    times_cube = np.arange(0, sspikes.shape[0], num_times)

    # Select most active time points
    movetimes = np.sort(np.argsort(np.sum(sspikes[times_cube, :], 1))[-active_times:])
    movetimes = times_cube[movetimes]

    # PCA dimensionality reduction
    scaled_data = preprocessing.scale(sspikes[movetimes, :])
    dimred, *_ = _pca(scaled_data, dim=dim)

    # Point cloud sampling (denoising)
    indstemp, *_ = _sample_denoising(dimred, k, n_points, 1, metric)

    # Build distance matrix
    d = _second_build(dimred, indstemp, metric=metric, nbs=nbs)
    np.fill_diagonal(d, 0)

    # Compute persistent homology
    persistence = ripser(
        d,
        maxdim=maxdim,
        coeff=coeff,
        do_cocycles=True,
        distance_matrix=True,
        progress_bar=progress_bar,
    )

    return persistence

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
    if dim < 2:
        return data, [0]
    m, n = data.shape
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

def _sample_denoising(data, k=10, num_sample=500, omega=0.2, metric="euclidean"):
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
    if HAS_NUMBA:
        return _sample_denoising_numba(data, k, num_sample, omega, metric)
    else:
        return _sample_denoising_numpy(data, k, num_sample, omega, metric)

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



def _run_shuffle_analysis(sspikes, num_shuffles=1000, num_cores=4, progress_bar=True, **kwargs):
    """Perform shuffle analysis with optimized computation."""
    return _run_shuffle_analysis_multiprocessing(
        sspikes, num_shuffles, num_cores, progress_bar, **kwargs
    )

def _run_shuffle_analysis_multiprocessing(
    sspikes, num_shuffles=1000, num_cores=4, progress_bar=True, **kwargs
):
    """Original multiprocessing implementation for fallback."""
    # Use numpy arrays with NaN for failed results (more efficient than None filtering)
    max_lifetimes = {
        0: np.full(num_shuffles, np.nan),
        1: np.full(num_shuffles, np.nan),
        2: np.full(num_shuffles, np.nan),
    }

    # Estimate runtime with a test iteration
    logging.info("Running test iteration to estimate runtime...")

    _ = _process_single_shuffle((0, sspikes, kwargs))

    # Prepare task list
    tasks = [(i, sspikes, kwargs) for i in range(num_shuffles)]
    logging.info(
        f"Starting shuffle analysis with {num_shuffles} iterations using {num_cores} cores..."
    )

    # Use multiprocessing pool for parallel processing
    with mp.Pool(processes=num_cores) as pool:
        results = list(pool.imap(_process_single_shuffle, tasks))
        logging.info("Shuffle analysis completed")

    # Collect results - use indexing instead of append for better performance
    for idx, res in enumerate(results):
        for dim, lifetime in res.items():
            max_lifetimes[dim][idx] = lifetime

    # Filter out NaN values (failed results) - convert to list for consistency
    for dim in max_lifetimes:
        max_lifetimes[dim] = max_lifetimes[dim][~np.isnan(max_lifetimes[dim])].tolist()

    return max_lifetimes


@njit(fastmath=True)

def _process_single_shuffle(args):
    """Process a single shuffle task."""
    i, sspikes, kwargs = args
    try:
        shuffled_data = _shuffle_spike_trains(sspikes)
        persistence = _compute_persistence(shuffled_data, **kwargs)

        dim_max_lifetimes = {}
        for dim in [0, 1, 2]:
            if dim < len(persistence["dgms"]):
                # Filter out infinite values
                valid_bars = [bar for bar in persistence["dgms"][dim] if not np.isinf(bar[1])]
                if valid_bars:
                    lifetimes = [bar[1] - bar[0] for bar in valid_bars]
                    if lifetimes:
                        dim_max_lifetimes[dim] = max(lifetimes)
        return dim_max_lifetimes
    except Exception as e:
        print(f"Shuffle {i} failed: {str(e)}")
        return {}

def _shuffle_spike_trains(sspikes):
    """Perform random circular shift on spike trains."""
    shuffled = sspikes.copy()
    num_neurons = shuffled.shape[1]

    # Independent shift for each neuron
    for n in range(num_neurons):
        shift = np.random.randint(0, int(shuffled.shape[0] * 0.1))
        shuffled[:, n] = np.roll(shuffled[:, n], shift)

    return shuffled



def _plot_barcode(persistence):
    """
    Plot barcode diagram from persistent homology result.

    Parameters:
        persistence (dict): Persistent homology result with 'dgms' key.
    """
    cs = np.repeat([[0, 0.55, 0.2]], 3).reshape(3, 3).T  # RGB color for each dimension
    alpha = 1
    inf_delta = 0.1
    colormap = cs
    dgms = persistence["dgms"]
    maxdim = len(dgms) - 1
    dims = np.arange(maxdim + 1)
    labels = ["$H_0$", "$H_1$", "$H_2$"]

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

    cs = np.repeat([[0, 0.55, 0.2]], 3).reshape(3, 3).T
    alpha = 1
    inf_delta = 0.1
    colormap = cs
    maxdim = len(persistence["dgms"]) - 1
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

    for _, dim in enumerate(dims):
        axes = plt.subplot(gs[dim])
        axes.axis("off")

        # Add gray background to represent shuffle region
        if dim in thresholds:
            axes.axvspan(0, thresholds[dim], alpha=0.2, color="gray", zorder=-3)
            axes.axvline(x=thresholds[dim], color="gray", linestyle="--", alpha=0.7)

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

            # Mark significant bars
            significant_bars = []
            for idx in dinds:
                if dlife[idx] > thresholds.get(dim, 0):
                    significant_bars.append(idx)

            # Draw bars
            for i, idx in enumerate(dinds):
                color = "red" if idx in significant_bars else colormap[dim]
                axes.barh(
                    0.5 + i,
                    dlife[idx],
                    height=0.8,
                    left=d[idx, 0],
                    alpha=alpha,
                    color=color,
                    linewidth=0,
                )

            indsall = len(dinds)
        else:
            indsall = 0

        axes.plot([0, 0], [0, indsall], c="k", linestyle="-", lw=1)
        axes.plot([0, indsall], [0, 0], c="k", linestyle="-", lw=1)
        axes.set_xlim([0, infinity])
        axes.set_title(f"$H_{dim}$", loc="left")

    plt.tight_layout()
    return fig
