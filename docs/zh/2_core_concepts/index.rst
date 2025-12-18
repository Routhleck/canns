========
核心概念
========

深入探讨库的设计、架构和理论基础。

本节解释 CANNs 库的设计原则、模块组织和概念基础。这些文档侧重于"为什么"和"何时"而非"如何"——帮助您理解库的架构并就使用组件做出明智决策。

.. toctree::
   :maxdepth: 1
   :caption: 主题:

   01_design_philosophy
   02_model_collections
   03_task_generators
   04_analysis_methods
   05_brain_inspired_training

概述
--------

:doc:`01_design_philosophy`
   理解库的架构、核心设计原则和四个应用场景。了解关注点分离、可扩展性、BrainPy 集成和性能策略。

:doc:`02_model_collections`
   探索三个模型类别：基础 CANN 模型、具有学习机制的脑启发模型，以及结合 CANNs 与 ANNs 的混合模型。理解 BrainPy 基础及如何实现自定义模型。

:doc:`03_task_generators`
   任务生成理念和可用范式。了解追踪任务（群体编码、模板匹配、平滑追踪）和导航任务（闭环、开环）。理解模型-任务耦合和设计考虑。

:doc:`04_analysis_methods`
   综合分析工具：用于仿真的模型分析器、用于实验记录的数据分析器、用于不动点的 RNN 动力学分析，以及用于几何结构的拓扑数据分析。

:doc:`05_brain_inspired_training`
   脑启发学习机制和 Trainer 框架。了解活动依赖可塑性、学习规则（Hebbian、STDP、BCM），以及如何实现生物学合理的自定义训练器。
