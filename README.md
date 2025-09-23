# CANNs: Continuous Attractor Neural Networks Toolkit

[![Contributors][contributors-shield]][contributors-url]
[![Stars][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

> 中文说明请见 [README_zh.md](README_zh.md)

CANNs is a Python library built on top of the Brain Simulation Ecosystem (`brainstate`, `brainunit`) that streamlines experimentation with continuous attractor neural networks and related brain-inspired models. It delivers ready-to-use models, task generators, analysis tools, and pipelines so neuroscience and AI researchers can move from ideas to reproducible simulations quickly.

## Highlights

- **Model families** – `canns.models.basic` ships 1D/2D CANNs (including SFA variants and hierarchical networks), while `canns.models.brain_inspired` adds Hopfield-style systems.
- **Task-first API** – `canns.task.tracking` and `canns.task.spatial_navigation` generate smooth tracking inputs, population coding stimuli, or import experimental trajectories.
- **Rich analysis suite** – `canns.analyzer` covers energy landscapes, tuning curves, spike embeddings, UMAP/TDA helpers, and theta-sweep animations.
- **Unified training** – `canns.trainer.HebbianTrainer` implements generic Hebbian learning and prediction, layered on the abstract `Trainer` base.
- **Pipelines out of the box** – `canns.pipeline.ThetaSweepPipeline` orchestrates navigation tasks, direction/grid-cell networks, and visualisation in a single call.
- **Extensible foundations** – base classes (`BasicModel`, `Task`, `Trainer`, `Pipeline`) keep custom components consistent with the built-in ecosystem.

## Installation

```bash
# CPU-only installation
pip install canns

# Optional accelerators (Linux only)
pip install canns[cuda12]
pip install canns[tpu]
```

## Quick Start

```python
import brainstate
from canns.models.basic import CANN1D
from canns.task.tracking import SmoothTracking1D

brainstate.environ.set(dt=0.1)

cann = CANN1D(num=512)
cann.init_state()

task = SmoothTracking1D(
    cann_instance=cann,
    Iext=(0.0, 0.5, 1.0, 1.5),
    duration=(5.0, 5.0, 5.0, 5.0),
    time_step=brainstate.environ.get_dt(),
)
task.get_data()

def step(t, stimulus):
    cann(stimulus)
    return cann.u.value, cann.inp.value

us, inputs = brainstate.compile.for_loop(
    step,
    task.run_steps,
    task.data,
    pbar=brainstate.compile.ProgressBar(10),
)
```

For an end-to-end theta sweep workflow, see `examples/pipeline/theta_sweep_from_external_data.py` or the `ThetaSweepPipeline` notebook in the docs.

## Documentation & Notebooks

- [01_quick_start.ipynb](docs/en/notebooks/01_quick_start.ipynb) – condensed tour of the library layout.
- [00_design_philosophy.ipynb](docs/en/notebooks/00_design_philosophy.ipynb) – detailed design rationale for each module.
- Interactive launchers: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/routhleck/canns/HEAD?filepath=docs%2Fen%2Fnotebooks) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/routhleck/canns/blob/master/docs/en/notebooks/)

## Development Workflow

```bash
# Create the dev environment (uv-based)
make install

# Format and lint (ruff, codespell, etc.)
make lint

# Run the test suite (pytest)
make test
```

Additional scripts live under `devtools/` and `scripts/`.

## Repository Layout

```
src/canns/            Core library modules (models, tasks, analyzers, trainer, pipeline)
docs/                 Sphinx documentation and notebooks
examples/             Ready-to-run scripts for models, analysis, and pipelines
tests/                Pytest coverage for key behaviours
```

## Contributing

Contributions are welcome! Please open an issue or discussion if you plan significant changes. Pull requests should follow the existing lint/test workflow (`make lint && make test`).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

[contributors-shield]: https://img.shields.io/github/contributors/routhleck/canns.svg?style=for-the-badge
[contributors-url]: https://github.com/routhleck/canns/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/routhleck/canns.svg?style=for-the-badge
[stars-url]: https://github.com/routhleck/canns/stargazers
[issues-shield]: https://img.shields.io/github/issues/routhleck/canns.svg?style=for-the-badge
[issues-url]: https://github.com/routhleck/canns/issues
[license-shield]: https://img.shields.io/github/license/routhleck/canns.svg?style=for-the-badge
[license-url]: https://github.com/routhleck/canns/blob/master/LICENSE
