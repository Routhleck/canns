教程
====

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。

欢迎来到 CANNs 教程！本教程采用 **场景驱动** 的方式，帮助你根据实际需求快速找到相关内容。

如何使用本教程
--------------

不同于传统的"模块介绍"方式，我们的教程围绕 **你想要完成的任务** 来组织：

- **想分析 CANN 动力学？** → :doc:`cann_dynamics/index`
- **想建模空间导航？** → :doc:`spatial_navigation/index`
- **想训练记忆网络？** → :doc:`memory_networks/index`
- **想学习无监督算法？** → :doc:`unsupervised_learning/index`

每个场景都包含：

1. **场景描述** - 你将解决什么问题
2. **完整示例** - 可直接运行的代码
3. **逐步解析** - 代码如何工作
4. **结果分析** - 如何解读输出
5. **扩展方向** - 下一步学什么

教程场景
--------

.. toctree::
   :maxdepth: 1
   :caption: 选择你的场景

   cann_dynamics/index
   spatial_navigation/index
   memory_networks/index
   unsupervised_learning/index
   receptive_fields/index
   temporal_learning/index
   experimental_analysis/index
   advanced_workflows/index

场景概览
--------

1. CANN 动力学分析
~~~~~~~~~~~~~~~~~~

**场景**：理解 CANN 如何响应不同输入，分析 bump 动力学。

**关键词**：追踪、调谐曲线、振荡、可视化

**适合**：初学者、需要理解基础模型的研究者

→ :doc:`cann_dynamics/index`

2. 空间导航建模
~~~~~~~~~~~~~~~

**场景**：模拟大脑的空间导航系统（网格细胞、位置细胞、Theta 节律）。

**关键词**：路径积分、分层网络、Theta sweep、海马体

**适合**：神经科学研究者、空间认知研究者

→ :doc:`spatial_navigation/index`

3. 记忆网络训练
~~~~~~~~~~~~~~~

**场景**：使用 Hopfield 网络实现联想记忆和模式存储。

**关键词**：Hebbian、模式完成、能量函数、容量分析

**适合**：学习神经网络基础的学生、记忆研究者

→ :doc:`memory_networks/index`

4. 无监督学习
~~~~~~~~~~~~~

**场景**：通过 Oja 和 Sanger 规则提取主成分。

**关键词**：PCA、Hebbian、权重归一化、降维

**适合**：对生物启发学习感兴趣的研究者

→ :doc:`unsupervised_learning/index`

5. 感受野发展
~~~~~~~~~~~~~

**场景**：使用 BCM 规则训练方向选择性神经元。

**关键词**：BCM、滑动阈值、感受野、方向选择性

**适合**：视觉神经科学研究者、发展可塑性研究者

→ :doc:`receptive_fields/index`

6. 时序模式学习
~~~~~~~~~~~~~~~

**场景**：使用 STDP 训练脉冲神经网络学习时序模式。

**关键词**：STDP、脉冲时序、因果关系、LTP/LTD

**适合**：脉冲神经网络研究者、时序编码研究者

→ :doc:`temporal_learning/index`

7. 实验数据分析
~~~~~~~~~~~~~~~

**场景**：分析真实的神经记录数据，拟合 bump 活动。

**关键词**：bump 拟合、ROI 数据、钙成像、预处理

**适合**：实验神经科学家、数据分析人员

→ :doc:`experimental_analysis/index`

8. 高级工作流
~~~~~~~~~~~~~

**场景**：构建端到端 Pipeline，实现复杂的自动化工作流。

**关键词**：Pipeline、批量处理、参数扫描、自动化

**适合**：需要高效工作流的研究者和工程师

→ :doc:`advanced_workflows/index`

获取帮助
--------

- **示例代码**：:doc:`../examples/README` - 所有可运行的示例
- **API 文档**：:doc:`../../autoapi/index` - 详细的 API 参考
- **社区支持**：`GitHub Issues <https://github.com/your-org/canns/issues>`_

