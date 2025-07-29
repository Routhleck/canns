[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]

<!-- 语言切换 -->
**语言**: [English](README.md) | **中文**

<!-- PROJECT LOGO -->
<br />

<div align="center">
  <a href="https://github.com/routhleck/canns">
    <img src="images/logo.svg" alt="Logo" height="100">
  </a>

<h3 align="center">连续吸引子神经网络 (CANNs) Python 库</h3>

  <p align="center">
    专为连续吸引子神经网络 (CANNs) 和其他脑启发计算模型设计的 Python 库。提供统一的高级 API，让研究者和开发者能够轻松加载、分析和训练最先进的 CANN 架构，快速实验和部署脑启发解决方案。
    <br />
    <a href="docs/zh/"><strong>📖 中文文档 »</strong></a>
    <br />
    <br />
    <a href="#快速开始">🚀 快速开始</a>
    &middot;
    <a href="#示例">💡 示例</a>
    &middot;
    <a href="https://github.com/routhleck/canns/issues/new?labels=bug&template=bug-report---.md">🐛 报告问题</a>
    &middot;
    <a href="https://github.com/routhleck/canns/issues/new?labels=enhancement&template=feature-request---.md">✨ 功能请求</a>
  </p>
</div>

---

> ⚠️ **开发状态**: 本项目正在积极开发中。功能和接口可能在未来版本中进一步完善和调整。

## 📋 目录

- [关于项目](#关于项目)
- [核心特性](#核心特性) 
- [快速开始](#快速开始)
- [安装](#安装)
- [使用示例](#使用示例)
- [交互式文档](#交互式文档)
- [项目结构](#项目结构)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [联系方式](#联系方式)

## 🎯 关于项目

CANNs (Continuous Attractor Neural Networks) 是一个专门为连续吸引子神经网络研究设计的 Python 库。该库基于现代科学计算框架构建，为神经科学研究、计算建模和脑启发算法开发提供了强大而易用的工具集。

### 🔬 什么是连续吸引子神经网络？

连续吸引子神经网络是一类特殊的神经网络模型，能够在连续的状态空间中维持稳定的活动模式。这类网络在以下方面表现突出：

- **空间表征**: 通过神经元群体活动编码连续的空间位置
- **工作记忆**: 维持和更新动态信息
- **路径积分**: 基于运动信息推算位置变化
- **平滑跟踪**: 对连续变化目标的跟踪

## ✨ 核心特性

### 🏗️ 丰富的模型库
- **CANN1D/2D**: 一维和二维连续吸引子网络
- **SFA 模型**: 集成慢特征分析的高级模型
- **分层网络**: 支持多层级信息处理的复杂架构

### 🎮 任务导向设计  
- **路径积分**: 空间导航和位置估计任务
- **目标跟踪**: 平滑连续的动态目标跟踪
- **可扩展框架**: 轻松添加自定义任务类型

### 📊 强大的分析工具
- **实时可视化**: 能量景观、神经活动动画
- **统计分析**: 发放率、调谐曲线、群体动力学
- **数据处理**: z-score 归一化、时间序列分析

### ⚡ 高性能计算
- **JAX 加速**: 基于 JAX 的高效数值计算
- **GPU 支持**: CUDA 和 TPU 硬件加速
- **并行处理**: 大规模网络仿真优化

## 🚀 快速开始

### 安装

```bash
# 基础安装 (CPU)
pip install canns

# GPU 支持 (Linux)
pip install canns[cuda12]

# TPU 支持 (Linux)
pip install canns[tpu]
```

### 基础示例

```python
import brainstate
from canns.models.basic import CANN1D
from canns.task.tracking import SmoothTracking1D
from canns.analyzer.visualize import energy_landscape_1d_animation

# 设置计算环境
brainstate.environ.set(dt=0.1)

# 创建一维CANN网络
cann = CANN1D(num=512)
cann.init_state()

# 定义平滑跟踪任务
task = SmoothTracking1D(
    cann_instance=cann,
    Iext=(1., 0.75, 2., 1.75, 3.),  # 外部输入序列
    duration=(10., 10., 10., 10.),   # 每个阶段持续时间
    time_step=brainstate.environ.get_dt(),
)

# 获取任务数据
task.get_data()

# 定义仿真步骤
def run_step(t, inputs):
    cann(inputs)
    return cann.u.value, cann.inp.value

# 运行仿真
us, inps = brainstate.compile.for_loop(
    run_step,
    task.run_steps,
    task.data,
    pbar=brainstate.compile.ProgressBar(10)
)

# 生成能量景观动画
energy_landscape_1d_animation(
    {'u': (cann.x, us), 'Iext': (cann.x, inps)},
    time_steps_per_second=100,
    fps=20,
    title='平滑跟踪任务',
    save_path='tracking_demo.gif'
)
```

## 📖 交互式文档

我们提供了完整的交互式文档，您可以直接在浏览器中运行代码示例：

### 🌐 在线运行
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/routhleck/canns/HEAD?filepath=docs%2Fzh%2Fnotebooks) **MyBinder** - 免费在线 Jupyter 环境
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/routhleck/canns/blob/master/docs/zh/notebooks/) **Google Colab** - 需要 Google 账户

### 📚 文档结构
- **[入门指南](docs/zh/notebooks/01_introduction.ipynb)** - CANNs 基础概念和使用介绍
- **[快速开始](docs/zh/notebooks/02_quickstart.ipynb)** - 常用场景和快速上手
- **[核心概念](docs/zh/notebooks/03_core_concepts.ipynb)** - 数学原理和理论基础

## 📁 项目结构

```
canns/
├── src/canns/
│   ├── models/          # 神经网络模型
│   │   ├── basic/       # 基础CANN模型
│   │   ├── brain_inspired/  # 脑启发模型
│   │   └── hybrid/      # 混合模型
│   ├── task/            # 任务定义
│   │   ├── tracking.py      # 跟踪任务
│   │   └── path_integration.py  # 路径积分
│   ├── analyzer/        # 分析工具
│   │   ├── utils.py         # 分析工具函数
│   │   └── visualize.py     # 可视化工具
│   ├── trainer/         # 训练框架
│   └── pipeline/        # 数据流水线
├── examples/            # 使用示例
├── docs/               # 文档
│   ├── en/             # 英文文档
│   └── zh/             # 中文文档
├── tests/              # 单元测试
└── binder/             # Binder 配置
```

## 💡 使用示例

### 二维空间跟踪

```python
from canns.models.basic import CANN2D

# 创建二维网络
cann2d = CANN2D(shape=(64, 64))
cann2d.init_state()

# 二维跟踪任务...
```

### 分层路径积分

```python
from canns.models.basic import HierarchicalNetwork
from canns.task.path_integration import PathIntegration

# 创建分层网络
hierarchical = HierarchicalNetwork(
    layers=[512, 256, 128],
    connectivity='feedforward'
)

# 路径积分任务
path_task = PathIntegration(
    network=hierarchical,
    trajectory_data='path_data.npz'
)
```

### 自定义可视化

```python
from canns.analyzer.visualize import (
    raster_plot,
    tuning_curve_plot,
    firing_rate_analysis
)

# 光栅图
raster_plot(spike_data, save_path='raster.png')

# 调谐曲线
tuning_curve_plot(responses, stimuli, save_path='tuning.png')

# 发放率分析
firing_rate_analysis(activity_data, save_path='firing_rate.png')
```

## 🛠️ 开发环境

### 依赖项

- **Python**: >= 3.11
- **BrainX**: 核心计算框架
- **JAX**: 高性能数值计算
- **ratinabox**: 空间认知建模
- **matplotlib**: 数据可视化
- **tqdm**: 进度显示

### 开发工具

- **pytest**: 单元测试
- **ruff**: 代码格式化和检查
- **basedpyright**: 类型检查
- **codespell**: 拼写检查

## 🤝 贡献指南

我们欢迎社区贡献！请查看以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)  
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范

- 遵循 PEP 8 代码风格
- 添加必要的类型注解
- 编写单元测试
- 更新相关文档

## 📄 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

**项目维护者**: Sichao He  
**邮箱**: sichaohe@outlook.com  
**项目链接**: [https://github.com/routhleck/canns](https://github.com/routhleck/canns)

---

<div align="center">

如果这个项目对您有帮助，请考虑给我们一个 ⭐️

</div>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/routhleck/canns.svg?style=for-the-badge
[contributors-url]: https://github.com/routhleck/canns/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/routhleck/canns.svg?style=for-the-badge
[forks-url]: https://github.com/routhleck/canns/network/members
[stars-shield]: https://img.shields.io/github/stars/routhleck/canns.svg?style=for-the-badge
[stars-url]: https://github.com/routhleck/canns/stargazers
[issues-shield]: https://img.shields.io/github/issues/routhleck/canns.svg?style=for-the-badge
[issues-url]: https://github.com/routhleck/canns/issues
[license-shield]: https://img.shields.io/github/license/routhleck/canns.svg?style=for-the-badge
[license-url]: https://github.com/routhleck/canns/blob/master/LICENSE.txt