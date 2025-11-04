快速开始
========

本笔记概括 CANNs 设计哲学的要点，帮助你迅速熟悉库的结构。

在深入阅读完整的设计哲学笔记或源码之前，可先将它当作高层检查清单。

模块概览
--------

- ``model`` 内置模型包。

  - ``basic`` 核心 CANN 模型及其变体。
  - ``brain_inspired`` 各类脑启发模型。
  - ``hybrid`` 将 CANN 与其他架构（如 ANN）混合的模型。

- ``task`` 负责生成、持久化、导入与可视化刺激的任务工具。
- ``analyzer`` 专注模型与数据可视化的分析工具。

  - ``model analyzer`` 能量景观、发放率、调谐曲线等模型分析函数。
  - ``data analyzer`` 面向实验数据或虚拟 RNN 动力学的分析流程。

- ``trainer`` 统一的训练与预测接口。
- ``pipeline`` 将上述模块串联的端到端流水线。

模型模块速览
------------

``models``
~~~~~~~~~~

总览
^^^^

模型模块实现不同维度的基础
CANN、脑启发变体以及混合模型，是整个库的核心，并与其他模块协同覆盖多种场景。

实现按照类型分组：

- Basic Models (:mod:`~src.canns.models.basic`) 标准 CANN 结构及其扩展。
- Brain-Inspired Models (:mod:`~src.canns.models.brain_inspired`)
  脑启发网络实现。
- Hybrid Models (:mod:`~src.canns.models.hybrid`) 将 CANN 与 ANN
  等机制结合的混合模型。

这些模型依赖 `Brain Simulation
Ecosystem <https://brainmodeling.readthedocs.io/index.html>`__\ ，尤其是
`brainstate <https://brainstate.readthedocs.io>`__\ 。\ ``brainstate``
基于 JAX/BrainUnit，提供 ``brainstate.nn.Dynamics``
抽象、\ ``State``/``HiddenState``/``ParamState``
容器、\ ``brainstate.environ`` 的统一时间步控制，以及
``brainstate.compile.for_loop``\ 、\ ``brainstate.random``
等工具。借助这些组件，模型只需描述状态变量与更新规则，时间推进、并行与随机数管理由
brainstate 负责，从而降低实现成本。

任务模块速览
------------

:mod:`~src.canns.task`
~~~~~~~~~~~~~~~~~

总览
^^^^

任务模块（:mod:`~src.canns.task`）负责生成、保存、读取、导入并展示 CANN
相关刺激。它提供多种预设任务，同时允许根据特定需求扩展。主要类型包括
:class:`~src.canns.task.tracking.SmoothTracking1D`、:class:`~src.canns.task.tracking.SmoothTracking2D` 和
:class:`~src.canns.task.open_loop_navigation.OpenLoopNavigationTask`。

分析器模块速览
--------------

:mod:`~src.canns.analyzer`
~~~~~~~~~~~~~~~~~~~~~

总览
^^^^

分析器模块（:mod:`~src.canns.analyzer`）为 CANN
模型和实验数据提供可视化与统计分析工具，涵盖模型分析与数据分析两大类。关键组件包括
:class:`~src.canns.analyzer.plotting.PlotConfigs` 配置系统和可视化函数如
:func:`~src.canns.analyzer.plotting.energy_landscape_1d_animation` 和
:func:`~src.canns.analyzer.plotting.energy_landscape_2d_animation`。

训练器要点
----------

:mod:`~src.canns.trainer`
~~~~~~~~~~~~~~~~~~~~

总览
^^^^

训练器模块（:mod:`~src.canns.trainer`）提供统一的训练与评估接口。目前以 Hebbian
学习为核心，后续可扩展其他策略。核心类型包括 :class:`~src.canns.trainer.HebbianTrainer`、
:class:`~src.canns.trainer.OjaTrainer`、:class:`~src.canns.trainer.SangerTrainer`、
:class:`~src.canns.trainer.BCMTrainer` 和 :class:`~src.canns.trainer.STDPTrainer` 等。

流水线概览
----------

:mod:`~src.canns.pipeline`
~~~~~~~~~~~~~~~~~~~~~

总览
^^^^

流水线模块（:mod:`~src.canns.pipeline`）将模型、任务、分析器与训练器串联成端到端流程，便于用最少代码完成常见需求。
核心管道包括 :class:`~src.canns.pipeline.ThetaSweepPipeline` 用于空间导航和 Theta 扫描分析。

下一步
------

- 阅读
  `设计哲学 <https://routhleck.com/canns/zh/notebooks/00_design_philosophy.html>`__
  获取完整设计理念。
- 浏览 `examples/  <https://github.com/Routhleck/canns/tree/master/examples>`__ 目录了解各模块的实际用法。
- 按照各节提供的扩展指引定制自己的组件。
