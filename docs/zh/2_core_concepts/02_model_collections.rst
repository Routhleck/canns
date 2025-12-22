==================
模型集合
==================

本文档介绍 CANNs 库中不同类别的模型及其扩展方法。

概述
========

模型模块（``canns.models``）实现了各种 CANN 架构及其变体。模型分为三个类别：

**基础模型** (``canns.models.basic``)
   标准 CANN 实现和变体

**类脑模型** (``canns.models.brain_inspired``)
   具有生物学习机制的模型

**混合模型** (``canns.models.hybrid``)
   CANN 与人工神经网络的组合

所有模型都基于 BrainPy 的动力学框架构建，该框架提供状态管理、时间步进和 JIT 编译功能。

基础模型
============

基础模型实现了核心的 **Wu-Amari-Wong 连续吸引子神经网络（CANNs）** 动力学 :cite:p:`amari1977dynamics,wu2008dynamics,fung2010moving,wu2016continuous`，如理论神经科学文献所述。它们使用预定义的连接模式（通常是高斯核）和固定参数。

可用的基础模型
----------------------

模型按 ``canns.models.basic`` 中的模块文件组织：

原始 CANN (cann.py)
~~~~~~~~~~~~~~~~~~~~~

核心连续吸引子神经网络实现。

``CANN1D``
   一维连续吸引子网络。默认使用 512 个神经元排列在环上，采用高斯循环连接。适用于头部方向编码 :cite:p:`taube1990head` 和角度变量。

``CANN1D_SFA``
   带有尖峰频率适应的 CANN1D。它增加了活动依赖的负反馈，并能实现自持续波传播。用于建模内在动力学。

``CANN2D``
   二维连续吸引子网络，神经元排列在环面上。适用于位置场编码 :cite:p:`o1971hippocampus` 和空间变量。

``CANN2D_SFA``
   带有尖峰频率适应的 CANN2D。支持二维行波。

层次化路径积分模型 (hierarchical_model.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

层次化模型 :cite:p:`chu2025localized` 结合多种细胞类型用于空间认知。

``GaussRecUnits``
   具有高斯连接的循环单元。

``NonRecUnits``
   用于比较的非循环单元。

``BandCell``
   用于一维路径积分的带状细胞。

``GridCell``
   具有多个尺度的单个网格细胞 :cite:p:`hafting2005microstructure` 模块。

``HierarchicalPathIntegrationModel``
   完整的路径积分 :cite:p:`mcnaughton2006path` 系统，包含网格细胞和位置细胞。

``HierarchicalNetwork``
   结合多种细胞类型用于空间认知。

Theta扫描模型 (theta_sweep_model.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

为 theta :cite:p:`chu2024firing,ji2025systems` 节律分析和空间导航研究 :cite:p:`mi2014spike,li2025dynamics,ji2025systems` 设计的模型。

``DirectionCellNetwork`` :cite:p:`ji2025phase`
   头部方向细胞网络。

``GridCellNetwork`` :cite:p:`ji2025systems`
   网格细胞模块网络。

``PlaceCellNetwork`` :cite:p:`chu2024firing`
   基于网格细胞输入的位置细胞网络。

实现基础模型
--------------------------

每个基础模型都继承自 ``canns.models.basic.BasicModel`` 或 ``canns.models.basic.BasicModelGroup``。

构造函数设置
~~~~~~~~~~~~~~~~~

使用总神经元数调用父构造函数::

   super().__init__(math.prod(shape), **kwargs)

在 ``self.shape`` 和 ``self.varshape`` 中存储形状信息，以正确处理维度。

必需的方法
~~~~~~~~~~~~~~~~

**连接矩阵** (``make_conn()``)
   生成循环连接矩阵。典型实现使用高斯核：

   - 计算神经元之间的成对距离
   - 应用指定宽度的高斯函数
   - 将结果存储在 ``self.conn_mat`` 中

   参见 ``src/canns/models/basic/cann.py`` 的参考实现。

**刺激生成** (``get_stimulus_by_pos(pos)``)
   将特征空间位置转换为外部输入模式。由任务模块调用以生成神经输入：

   - 以位置坐标作为输入
   - 返回与网络大小匹配的刺激向量
   - 使用高斯波包或类似的局部化模式

**更新动力学** (``update(inputs)``)
   定义单步状态演化：

   - 读取当前状态
   - 基于 CANN 方程计算导数
   - 应用时间步：``new_state = old_state + derivative * bm.get_dt()``
   - 写入更新后的状态

**诊断属性**
   提供用于分析的有用信息：

   - ``self.x``: 特征空间坐标
   - ``self.rho``: 神经元密度
   - 用于波包追踪的峰值检测方法

类脑模型
=====================

类脑模型具有生物学上合理的学习机制。与具有固定权重的基础模型不同，这些网络通过局部的、活动依赖的可塑性修改其连接。

关键特征
-------------------

**局部学习规则**
   权重更新仅依赖于突触前和突触后活动

**无误差反向传播**
   学习在没有显式误差信号的情况下发生

**基于能量的动力学**
   网络状态演化以最小化能量函数

**吸引子形成**
   存储的模式成为动力学的不动点

可用的类脑模型
--------------------------------

``AmariHopfieldNetwork``
   经典联想记忆模型 :cite:p:`amari1977neural,hopfield1982neural`，具有二进制模式存储。Hebbian 学习 :cite:p:`hebb2005organization` 用于权重形成。内容可寻址记忆。

``LinearLayer``
   具有可学习权重的线性层，用于比较和测试。支持各种无监督学习规则，包括用于主成分提取的 Oja 规则 :cite:p:`oja1982simplified` 和用于多个主成分的 Sanger 规则 :cite:p:`sanger1989optimal`。

``SpikingLayer``
   具有生物学真实尖峰动力学的脉冲神经网络层。

实现类脑模型
-----------------------------------

继承自 ``canns.models.brain_inspired.BrainInspiredModel`` 或 ``canns.models.brain_inspired.BrainInspiredModelGroup``。

状态和权重变量
~~~~~~~~~~~~~~~~~~~~~~~~~~~

定义状态变量和可训练权重：

- ``self.s``: 状态向量（``bm.Variable``）
- ``self.W``: 连接权重（``bm.Variable``）

所有状态和权重变量在 BrainPy 中使用 ``bm.Variable``。

权重属性
~~~~~~~~~~~~~~~~

如果权重存储在不同名称下，覆盖 ``weight_attr`` 属性::

   @property
   def weight_attr(self):
       return 'W'  # 或自定义属性名称

更新动力学
~~~~~~~~~~~~~~~

在 ``update(...)`` 中定义当前权重下的状态演化。通常涉及矩阵-向量乘法和激活函数。

能量函数
~~~~~~~~~~~~~~~

返回当前状态的标量能量值。训练器使用它来监控收敛::

   @property
   def energy(self):
       return -0.5 * state @ weights @ state

Hebbian 学习
~~~~~~~~~~~~~~~~

在 ``apply_hebbian_learning(patterns)`` 中可选的权重更新自定义实现。如果未提供，训练器使用默认的外积规则::

   W += learning_rate * patterns.T @ patterns

动态调整大小
~~~~~~~~~~~~~~~~

可选支持在保留学习结构的同时更改网络大小：``resize(num_neurons, preserve_submatrix)``

参见 ``src/canns/models/brain_inspired/hopfield.py`` 的参考实现。

混合模型
=============

.. note::

   混合模型将 CANN 动力学与其他神经网络架构相结合（开发中）。愿景包括：

   - 嵌入在更大人工神经网络中的 CANN 模块
   - 用于端到端训练的可微分 CANN 层
   - 吸引子动力学与前馈处理的集成
   - 将生物学合理性与深度学习能力相结合

   当前状态：在 ``canns.models.hybrid`` 中存在用于未来实现的占位符模块结构。

BrainPy 基础
==================

所有模型利用 BrainPy 的 :cite:p:`wang2023brainpy` 基础设施：

动力学抽象
--------------------

``bp.DynamicalSystem`` 提供：

- 自动状态跟踪
- JIT 编译支持
- 可组合的子模块

状态容器
----------------

``bm.Variable``
   所有状态变量（可变、内部或可学习参数）的通用容器

这些容器支持透明的 JAX :cite:p:`jax2018github` 变换，同时保持直观的面向对象语法。

时间管理
---------------

``brainpy.math`` 提供时间步管理：

- ``bm.set_dt(0.1)``: 设置模拟时间步
- ``bm.get_dt()``: 检索当前时间步

这确保了模型、任务和训练器之间的一致性。

编译仿真
-------------------

``bm.for_loop`` 实现高效仿真：

- 用于 GPU/TPU 加速的 JIT 编译
- 自动微分支持
- 进度跟踪集成

总结
=======

CANNs 模型集合提供：

1. **基础模型** - 可立即使用的标准 CANN 实现
2. **类脑模型** - 具有局部学习能力的网络
3. **混合模型** - 与深度学习的未来集成（开发中）

每个类别通过基类继承遵循一致的模式，使库既强大又可扩展。BrainPy 基础处理复杂性，允许用户专注于定义神经动力学而不是实现细节。
