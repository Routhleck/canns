# CANNs：连续吸引子神经网络工具包

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


> 英文版说明请参见 [README.md](README.md)

CANNs 是一个构建于 brainpy 之上的 Python 库，并通过专门的 Rust 后端（`canns-lib`）加速部分性能敏感模块，旨在提升连续吸引子神经网络（Continuous Attractor Neural Networks, CANNs）及相关类脑模型的实验效率。它提供可直接使用的模型、任务生成器、分析工具与流水线（pipelines），让神经科学与人工智能研究者从想法到可复现仿真更加高效。

## 亮点概览

- **模型家族** – `canns.models.basic` 提供 1D/2D CANN（含 SFA 变体与分层网络），`canns.models.brain_inspired` 进一步加入类 Hopfield 系统。
- **任务优先 API** – `canns.task.tracking` 与 `canns.task.open_loop_navigation` 可生成平滑跟踪输入、群体编码刺激，或导入实验轨迹。
- **丰富分析套件** – `canns.analyzer` 覆盖能量景观、调谐曲线、脉冲嵌入、UMAP/TDA 辅助工具，以及 theta 扫描动画。
- **统一训练框架** – `canns.trainer.HebbianTrainer` 实现通用的 Hebb 学习与预测，基于抽象 `Trainer` 基类。
- **即取即用的流水线** – `canns.pipeline.ThetaSweepPipeline` 一次性编排导航任务、方向/网格细胞网络与可视化。
- **可扩展基础** – 基类（`BasicModel`, `Task`, `Trainer`, `Pipeline`）让自定义组件与内置生态保持一致。

## 可视化展示

<div align="center">
<table>
<tr>
<td align="center" width="50%" valign="top">
<h4>1D CANN 平滑跟踪</h4>
<img src="docs/_static/smooth_tracking_1d.gif" alt="1D CANN 平滑跟踪" width="320">
<br><em>平滑跟踪过程中的实时动态</em>
</td>
<td align="center" width="50%" valign="top">
<h4>2D CANN 群体编码</h4>
<img src="docs/_static/CANN2D_encoding.gif" alt="2D CANN 编码" width="320">
<br><em>空间信息编码活动模式</em>
</td>
</tr>
<tr>
<td colspan="2" align="center">
<h4>Theta 扫描分析</h4>
<img src="docs/_static/theta_sweep_animation.gif" alt="Theta 扫描动画" width="600">
<br><em>网格细胞和头朝向网络的 theta 节律调制</em>
</td>
</tr>
<tr>
<td align="center" width="50%" valign="top">
<h4>凸包分析</h4>
<img src="docs/_static/bump_analysis_demo.gif" alt="凸包分析演示" width="320">
<br><em>1D 凸包拟合与分析</em>
</td>
<td align="center" width="50%" valign="top">
<h4>环面拓扑分析</h4>
<img src="docs/_static/torus_bump.gif" alt="环面凸包分析" width="320">
<br><em>3D 环面可视化与解码</em>
</td>
</tr>
</table>
</div>

## 安装方式

```bash
# 仅 CPU 安装
pip install canns

# 可选加速（仅 Linux）
pip install canns[cuda12]
pip install canns[tpu]
```

## 快速开始

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
    task.run_steps,
    task.data,
)
```

若需端到端的 theta 扫描流程，请参见 `examples/pipeline/theta_sweep_from_external_data.py` 或文档中的 `ThetaSweepPipeline` 笔记本。

## 文档与示例笔记本

- [快速开始指南](https://routhleck.com/canns/zh/notebooks/01_quick_start.html) – 库结构速览。
- [设计理念](https://routhleck.com/canns/zh/notebooks/00_design_philosophy.html) – 各模块设计理念详解。
- 交互式运行： [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/routhleck/canns/HEAD?filepath=docs%2Fzh%2Fnotebooks) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/routhleck/canns/blob/master/docs/zh/notebooks/)

## 开发流程

```bash
# 创建开发环境（基于 uv）
make install

# 格式化与静态检查（ruff, codespell 等）
make lint

# 运行测试（pytest）
make test
```

更多脚本位于 `devtools/` 与 `scripts/` 目录。

## 仓库结构

```
src/canns/            核心库模块（模型、任务、分析器、训练器、流水线）
docs/                 Sphinx 文档与笔记本
examples/             可直接运行的模型、分析与流水线脚本
tests/                关键行为的 Pytest 覆盖
```

## 引用本项目

如果您在研究中使用了 CANNs，请通过我们的 [CITATION.cff](CITATION.cff) 文件或以下信息引用本项目：

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

## 参与贡献

欢迎贡献！如计划重大修改，请先发起 issue 或 discussion。Pull Request 请遵循现有工作流（`make lint && make test`）。

## 许可证

采用 Apache License 2.0。详见 [LICENSE](LICENSE)。
（注：本中文翻译仅供参考，如与英文版存在差异，以英文 LICENSE 为准。）

[contributors-shield]: https://img.shields.io/github/contributors/routhleck/canns.svg?style=for-the-badge
[contributors-url]: https://github.com/routhleck/canns/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/routhleck/canns.svg?style=for-the-badge
[stars-url]: https://github.com/routhleck/canns/stargazers
[issues-shield]: https://img.shields.io/github/issues/routhleck/canns.svg?style=for-the-badge
[issues-url]: https://github.com/routhleck/canns/issues
[license-shield]: https://img.shields.io/github/license/routhleck/canns.svg?style=for-the-badge
[license-url]: https://github.com/routhleck/canns/blob/master/LICENSE
