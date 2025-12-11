# CANNs: Continuous Attractor Neural Networks Toolkit

<div align="center">
  <img src="images/logo.svg" alt="CANNs Logo" width="350">
</div>

[<img src="https://badges.ws/badge/status-beta-yellow" />](https://github.com/routhleck/canns)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/canns)
[<img src="https://badges.ws/maintenance/yes/2025" />](https://github.com/routhleck/canns)
<picture><img src="https://badges.ws/github/release/routhleck/canns" /></picture>
<picture><img src="https://badges.ws/github/license/routhleck/canns" /></picture>
[![DOI](https://zenodo.org/badge/1001781809.svg)](https://doi.org/10.5281/zenodo.17412545)


<picture><img src="https://badges.ws/github/stars/routhleck/canns?logo=github" /></picture>
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/canns?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/canns)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Routhleck/canns)
[<img src="https://badges.ws/badge/Buy_Me_a_Coffee-ff813f?icon=buymeacoffee" />](https://buymeacoffee.com/forrestcai6)

> 中文说明请见 [README_zh.md](README_zh.md)

CANNs is a Python library built on top of brainpy with performance‑critical modules accelerated by a dedicated Rust backend (`canns-lib`). It streamlines experimentation with continuous attractor neural networks and related brain‑inspired models, providing ready‑to‑use models, task generators, analysis tools, and pipelines so neuroscience and AI researchers can move from ideas to reproducible simulations quickly.

## Highlights

- **Model families** – `canns.models.basic` ships 1D/2D CANNs (including SFA variants and hierarchical networks), while `canns.models.brain_inspired` adds Hopfield-style systems.
- **Task-first API** – `canns.task.tracking` and `canns.task.open_loop_navigation` generate smooth tracking inputs, population coding stimuli, or import experimental trajectories.
- **Rich analysis suite** – `canns.analyzer` covers energy landscapes, tuning curves, spike embeddings, UMAP/TDA helpers, and theta-sweep animations.
- **Unified training** – `canns.trainer.HebbianTrainer` implements generic Hebbian learning and prediction, layered on the abstract `Trainer` base.
- **Pipelines out of the box** – `canns.pipeline.ThetaSweepPipeline` orchestrates navigation tasks, direction/grid-cell networks, and visualisation in a single call.
- **Extensible foundations** – base classes (`BasicModel`, `Task`, `Trainer`, `Pipeline`) keep custom components consistent with the built-in ecosystem.

## Visual Gallery

<div align="center">
<table>
<tr>
<td align="center" width="50%" valign="top">
<h4>1D CANN Smooth Tracking</h4>
<img src="docs/_static/smooth_tracking_1d.gif" alt="1D CANN Smooth Tracking" width="320">
<br><em>Real-time dynamics during smooth tracking</em>
</td>
<td align="center" width="50%" valign="top">
<h4>2D CANN Population Encoding</h4>
<img src="docs/_static/CANN2D_encoding.gif" alt="2D CANN Encoding" width="320">
<br><em>Spatial information encoding patterns</em>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<h4>Theta Sweep Analysis</h4>
<img src="docs/_static/theta_sweep_animation.gif" alt="Theta Sweep Animation" width="600">
<br><em>Grid cell and head direction networks with theta rhythm modulation</em>
</td>
</tr>
<tr>
<td align="center" width="50%" valign="top">
<h4>Bump Analysis</h4>
<img src="docs/_static/bump_analysis_demo.gif" alt="Bump Analysis Demo" width="320">
<br><em>1D bump fitting and analysis</em>
</td>
<td align="center" width="50%" valign="top">
<h4>Torus Topology Analysis</h4>
<img src="docs/_static/torus_bump.gif" alt="Torus Bump Analysis" width="320">
<br><em>3D torus visualization and decoding</em>
</td>
</tr>
</table>
</div>

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
import brainpy as bp
import brainpy.math as bm
from canns.models.basic import CANN1D
from canns.task.tracking import SmoothTracking1D

bm.set_dt(0.1)

cann = CANN1D(num=512)

task = SmoothTracking1D(
    cann_instance=cann,
    Iext=(0.0, 0.5, 1.0, 1.5),
    duration=(5.0, 5.0, 5.0, 5.0),
    time_step=bm.get_dt(),
)
task.get_data()

def step(t, stimulus):
    cann(stimulus)
    return cann.u.value, cann.inp.value

us, inputs = bm.for_loop(
    step,
    (
	task.run_steps,
	task.data,
    )
)
```

For an end-to-end theta sweep workflow, see `examples/pipeline/theta_sweep_from_external_data.py` or the `ThetaSweepPipeline` notebook in the docs.

## Documentation & Notebooks

- [Quick Start Guide](https://routhleck.com/canns/en/notebooks/01_quick_start.html) – condensed tour of the library layout.
- [Design Philosophy](https://routhleck.com/canns/en/notebooks/00_design_philosophy.html) – detailed design rationale for each module.
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

## Citation

If you use CANNs in your research, please cite it using the information from our [CITATION.cff](CITATION.cff) file or use the following:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17412545.svg)](https://doi.org/10.5281/zenodo.17412545)

```bibtex
@software{he_2025_canns,
  author       = {He, Sichao},
  title        = {CANNs: Continuous Attractor Neural Networks Toolkit},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v0.9.0},
  doi          = {10.5281/zenodo.17412545},
  url          = {https://github.com/Routhleck/canns}
}
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
