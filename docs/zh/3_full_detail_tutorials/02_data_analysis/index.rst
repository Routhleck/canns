场景2: 数据分析与神经解码
==========================

关于实验神经数据分析、拓扑解码与 RNN 动力学的综合教程。

教程列表
--------

.. toctree::
   :maxdepth: 1
   :caption: 实验数据分析

   01_asa_pipeline
   02_cann1d_bump_fit
   04_cell_classification

.. toctree::
   :maxdepth: 1
   :caption: RNN动力学分析

   03_flipflop_tutorial

教程概述
--------

**ASA pipeline 教程**

覆盖从 ``spike/x/y/t`` 输入到 TDA、解码、CohoMap/CohoSpace/PathCompare
与 FR/FRM 的完整流程，并对应到仓库内示例脚本。

**1D CANN ROI bump 拟合教程**

展示如何使用 ``roi_bump_fits`` 提取 bump 参数，并生成动画用于
分析环形吸引子动态。

**Cell Classification 教程**

展示基于 GridScore 与自相关特征的细胞分类流程，
包括单细胞评分与网格模块划分示例。

**RNN不动点分析教程（FlipFlop任务）**

本教程提供了使用 ``FixedPointFinder`` 工具分析循环神经网络（RNN）动力学特性的详细指南：

- **理论基础**: 理解动力系统中不动点的概念
- **FlipFlop任务**: 训练RNN完成多通道记忆任务
- **不动点查找**: 使用优化方法识别稳定和不稳定的不动点
- **可视化分析**: 通过PCA降维在状态空间中显示不动点分布
- **多配置比较**: 比较2位、3位和4位任务的不动点结构

**关键发现**: 对于N位FlipFlop任务，成功训练的RNN学会创建2^N个稳定不动点——每个不动点对应一个唯一的记忆状态组合。

示例代码
--------

- ``examples/experimental_data_analysis``：ASA pipeline 相关脚本（TDA/解码/CohoMap/CohoSpace/PathCompare/FR 等）
- ``examples/cell_classification``：细胞分类与相关分析示例

相关资源
--------

您可能会发现这些资源有帮助：

- :doc:`../01_cann_modeling/index`——理解CANN模型
- :doc:`../04_pipeline/index`——端到端研究工作流
- 核心概念文档——详细的分析方法
