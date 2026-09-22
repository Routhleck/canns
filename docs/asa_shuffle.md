# Complete-pipeline ASA shuffle

ASA now uses the same analysis function and configuration for real activity and
each shuffled activity matrix. Each neuron is independently circular-shifted
over the full supplied time axis. Every round then repeats time downsampling,
activity-based timepoint selection, standardization, PCA, density sampling,
distance-graph construction and persistent homology.

The former native shuffle instead measured Euclidean distances between neurons'
time series. Its null distribution was a different statistical object from the
real ASA time-state analysis. That shortcut is no longer used.

## Configure the real and null analyses together

```python
from canns.analyzer.data import TDAConfig, tda_vis

# activity is your finite (timepoints, neurons) matrix after spike embedding.
# These sampling budgets require at least 15000 candidate timepoints.
config = TDAConfig(
    dim=6,
    num_times=5,
    active_times=15000,
    k=1000,
    n_points=1200,
    nbs=800,
    metric="cosine",
    maxdim=2,
    coeff=47,
    standardize=True,
    do_cocycles=True,
    do_shuffle=True,
    num_shuffles=100,
    shuffle_seed=17,
    sampling_backend="rust",
    ph_threshold_policy="max_finite_float32",
    shuffle_workers=1,
    show=False,
    progress_bar=False,
)
result = tda_vis(activity, config=config)
```

This example requires the companion canns-lib release with the `fuzzy_union`
kernel for Rust density sampling. Select `sampling_backend="python"` to retain
the existing sampling implementation. In both cases, PH is computed by the same
Rust Ripser backend through `canns_lib.ripser.ripser`. Missing sampling
capabilities are reported before real analysis starts. Numerical failures never
trigger an automatic backend change.

Use either a `TDAConfig` or equivalent keyword arguments to `tda_vis`, not both.
Unrecognized parameters and conflicting options raise errors. Budgets are
validated: `k` and `n_points` must fit the available selected timepoints, and
`nbs` must not exceed `n_points`. `active_times` remains an upper bound; fewer
available candidates are allowed when the other budgets still fit.

## Division of responsibilities

`TDAConfig` explicitly defines the ASA analysis, including activity selection,
PCA, density sampling, graph construction and PH parameters. CANNs applies the
same configuration to the real data and every shifted activity matrix; no
analysis callback or arbitrary callback-argument dictionary is passed across
the library boundary.

CANNs generates circular-shift offsets locally and uses Rust numerical kernels
from canns-lib. The library's separate `shuffle_null_model` function accepts
point-cloud distance and PH
parameters explicitly. It does not perform ASA activity selection, PCA or graph
construction, so CANNs does not use it for ASA null distributions.

## Independent engineering options

| Option | Default | Effect |
|---|---|---|
| `sampling_backend` | `"python"` | Optional `"rust"` uses bounded row-sorting temporaries and a dense fuzzy-union kernel for density sampling. The final graph-building method is unchanged. |
| `ph_threshold_policy` | `"legacy"` | Optional `"max_finite_float32"` uses each graph's greatest finite float32 edge value, retaining every finite edge. |
| `ph_threshold` | `None` | Optional explicit filtration cutoff, applied to real and null. It cannot be combined with `"max_finite_float32"`. |
| `shuffle_workers` | `1` | Maximum simultaneous complete analyses. More workers multiply memory pressure; there is no promised linear speedup. |
| `shuffle_seed` | `None` | Local NumPy `default_rng` seed for offsets, without changing global random state. |
| `shuffle_shifts` | `None` | Explicit integer offsets for exact replay; mutually exclusive with `shuffle_seed`. |
| `do_cocycles` | `True` | Whether PH returns cocycles. Forwarded consistently for real and null. |

The Rust option still uses dense quadratic storage. This change does not make
arbitrarily large sample budgets inexpensive. Library PH calls without progress
callbacks release the Python GIL, but NumPy/BLAS and PH may also have their own
threads. Choose concurrency using the complete workflow's measured memory use.

Concurrent rounds also require a thread-safe Numba threading layer. If Numba
selects `workqueue`, CANNs rejects more than one simultaneous shuffle before
real PH starts, because concurrent calls can abort Python. Use
`shuffle_workers=1`, or configure an installed thread-safe Numba backend such as
TBB or OpenMP before starting Python (see Numba's
[`NUMBA_THREADING_LAYER` setting](https://numba.readthedocs.io/en/stable/user/threading-layer.html)).
CANNs never changes that environment setting automatically. A batch
with only one round does not require concurrent support.

The finite-threshold policy is an opt-in optimization, not a fixed percentile
cutoff. Missing edges stay infinite; higher-dimensional essential intervals are
retained and must not be interpreted as finite lifetimes. An explicitly chosen
smaller `ph_threshold` changes the filtration and its scientific interpretation.

## Replay and failures

Offsets have shape `(num_shuffles, neurons)`, with integer values in `[0, T)`.
Positive offsets mean `np.roll(activity[:, neuron], offset)`. Zero is allowed.
The scheduler generates all offsets before starting work, so the seed and
result order are independent of `shuffle_workers`. To reuse an old batch's exact
shifts, pass its saved offsets, rather than assuming a different RNG reproduces
them from the same integer seed.

```python
config.shuffle_seed = None
config.shuffle_shifts = recorded_offsets
config.num_shuffles = len(recorded_offsets)
result = tda_vis(activity, config=config)
```

On failure, `tda_vis` raises `ProcessingError` with the underlying exception
chained. A shuffle exception exposes `index`, `offsets` and
`original_exception`. No failed round is discarded, replaced by another seed,
or assigned zero. No partial null distribution is returned. Already-running
thread analyses may need to finish before resources can be reclaimed.

## Migration from earlier shuffle interfaces

Remove `shuffle_backend`, `use_ffi_shuffle` and `force_legacy` from calls and
saved configuration dictionaries. These obsolete selectors now raise
`TypeError` rather than silently choosing a different null model. There is one
ASA shuffle path: local offset generation and a bounded scheduler, followed by
the complete ASA pipeline and Rust PH for every round. There is no separate
NumPy PH implementation to select.

Keep scientific parameters in `TDAConfig`. `sampling_backend` controls only the
density-sampling implementation, while `shuffle_workers` controls concurrency.
Neither option changes the PH backend or skips scientific preprocessing.

## What the result means

For a reproducible comparison with the original Python multiprocessing path,
see the [complete ASA shuffle benchmark](../../benchmarks/asa_shuffle/README.md).
It records complete diagrams and cocycles, measures the whole process tree,
and separates finite-threshold gains from comparisons at the same PH threshold.
The historical shortcut's speedup numbers do not measure this complete pipeline.

`result["shuffle_max"]` retains one maximum **finite** lifetime per round and
dimension, including every successful round in index order. Empty finite
diagrams have maximum zero only after a successful computation. Essential H1 or
higher classes emit a warning because these finite maxima do not test their
significance. This ASA entry point returns per-round finite maxima; the generic
canns-lib shuffle API's detailed diagrams describe its own point-cloud null and
are not a substitute for ASA per-round diagrams.

This change corrects pipeline and parameter consistency. It does not by itself
make every ASA configuration an exact reproduction of Gardner et al. (2022).
Spike embedding, cell selection, shift distribution, analysis parameters and
the final significance rule remain scientific choices. For example, the
example above uses `k=1000` and 100 shuffles; the paper describes `k=1500` and
1000 shuffles. Existing ASA plotting still uses the 99.9th empirical percentile
of per-round finite maxima, whereas the paper used the overall maximum across
its shuffles. Those thresholds are different. With 100 shuffles, the smallest
corrected Monte Carlo p-value is `1 / 101`, not `p < 0.001`.

See the [paper's Methods](https://www.nature.com/articles/s41586-021-04268-7)
and the companion canns-lib shuffle guide for its point-cloud null model,
offset replay and migration details.
