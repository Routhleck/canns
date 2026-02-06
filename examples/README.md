# Examples

This folder contains runnable example scripts grouped by topic. If a script
writes files (plots, animations, data dumps), outputs go into a single
location: `examples/outputs/`.

## Layout

- `brain_inspired/`: classical learning rules and brain-inspired algorithms
  (Hopfield, Oja, BCM, STDP, etc.)
- `cann/`: continuous attractor neural network (CANN) models and navigation
  demos
- `cell_classification/`: grid / cell classification examples
- `experimental_data_analysis/`: ASA pipeline utilities (TDA, decoding,
  CohoMap/CohoSpace, path compare, firing-rate maps)
- `slow_points_analysis/`: dynamical system fixed/slow point analysis
- `outputs/`: generated artifacts from example runs (ignored by git)

## Outputs

By default, each script writes to:

```
examples/outputs/<category>/<script_name>/
```

You can override the base output directory with:

```
CANNS_EXAMPLES_OUTPUT_DIR=/path/to/outputs
```

Some scripts also expose `--out-dir` to override the directory per run.

## Running

Run examples from the repository root. For instance:

```
uv run python examples/cann/grid_cell_velocity_path_integration.py
```

```
uv run python examples/experimental_data_analysis/cohomap_example.py --data /path/to/asa.npz
```
