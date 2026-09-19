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
    shuffle_backend="canns_lib",
    sampling_backend="rust",
    ph_threshold_policy="max_finite_float32",
    shuffle_workers=1,
    show=False,
    progress_bar=False,
)
result = tda_vis(activity, config=config)
```

This example requires the companion canns-lib release with its public
`canns_lib.ripser.shuffle_null_model(..., pipeline=...)` API and `fuzzy_union`
kernel. During migration, explicitly select `shuffle_backend="python"` and
`sampling_backend="python"` to use the local full-pipeline reference with an
older canns-lib PH backend. Missing capabilities are reported before the real
analysis starts. A numerical failure never triggers an automatic backend change.

Use either a `TDAConfig` or equivalent keyword arguments to `tda_vis`, not both.
Unrecognized parameters and conflicting options raise errors. Budgets are
validated: `k` and `n_points` must fit the available selected timepoints, and
`nbs` must not exceed `n_points`. `active_times` remains an upper bound; fewer
available candidates are allowed when the other budgets still fit.

## Independent engineering options

| Option | Default | Effect |
|---|---|---|
| `shuffle_backend` | `"canns_lib"` | Selects the new library scheduler or the local `"python"` scheduler. Both call the complete real-data pipeline. |
| `sampling_backend` | `"python"` | Optional `"rust"` uses bounded row-sorting temporaries and a dense fuzzy-union kernel for density sampling. The final graph-building method is unchanged. |
| `ph_threshold_policy` | `"legacy"` | Optional `"max_finite_float32"` uses each graph's greatest finite float32 edge value, retaining every finite edge. |
| `ph_threshold` | `None` | Optional explicit filtration cutoff, applied to real and null. It cannot be combined with `"max_finite_float32"`. |
| `shuffle_workers` | `1` | Maximum simultaneous complete analyses. More workers multiply memory pressure; there is no promised linear speedup. |
| `shuffle_seed` | `None` | Local NumPy `default_rng` seed for offsets, without changing global random state. |
| `shuffle_shifts` | `None` | Explicit integer offsets for exact replay; mutually exclusive with `shuffle_seed`. |
| `do_cocycles` | `True` | Whether PH returns cocycles. Forwarded consistently for real and null. |

The Rust option still uses dense quadratic storage. This change does not make
arbitrarily large sample budgets inexpensive. The library's no-callback PH
route releases the Python GIL, but NumPy/BLAS and PH may also have their own
threads. Choose concurrency using the complete workflow's measured memory use.

The finite-threshold policy is an opt-in optimization, not a fixed percentile
cutoff. Missing edges stay infinite; higher-dimensional essential intervals are
retained and must not be interpreted as finite lifetimes. An explicitly chosen
smaller `ph_threshold` changes the filtration and its scientific interpretation.

## Replay and failures

Offsets have shape `(num_shuffles, neurons)`, with integer values in `[0, T)`.
Positive offsets mean `np.roll(activity[:, neuron], offset)`. Zero is allowed.
Both schedulers generate all offsets before starting work, so the seed and
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
thread callbacks may need to finish before resources can be reclaimed.

The deprecated `use_ffi_shuffle=True/False` maps to the library/Python scheduler
respectively, with a warning. It never selects the removed neuron-distance
algorithm. An inconsistent explicit `shuffle_backend` is rejected.

## What the result means

`result["shuffle_max"]` retains one maximum **finite** lifetime per round and
dimension, including every successful round in index order. Empty finite
diagrams have maximum zero only after a successful computation. Essential H1 or
higher classes emit a warning because these finite maxima do not test their
significance. For full per-round diagrams and essential counts, use the public
canns-lib API's `return_details=True` with an appropriate complete callback.

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
and the companion canns-lib shuffle guide for the callback contract, full
outputs and migration details.
