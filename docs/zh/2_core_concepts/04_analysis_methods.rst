================
分析方法
================

本文档解释了 CANNs 库中的分析和可视化工具。

概览
========

分析器模块（``canns.analyzer``）提供了用于可视化和解释仿真输出和实验数据的工具。它根据数据源和分析类型组织为不同的组件：

模块结构
--------

.. admonition:: 新组织结构（v2.0+）
   :class: note

   分析器模块按功能组织：

   * **metrics/** - 计算分析（无 matplotlib 依赖）

     * ``spatial_metrics`` - 空间指标计算
     * ``utils`` - Spike train 转换工具
     * ``experimental/`` - CANN1D/2D 实验数据分析

   * **visualization/** - 绘图和动画（基于 matplotlib）

     * ``config`` - PlotConfig 统一配置系统
     * ``spatial_plots`` - 空间可视化
     * ``energy_plots`` - 能量景观可视化
     * ``spike_plots`` - Raster 图和发放率图
     * ``tuning_plots`` - 调谐曲线可视化
     * ``experimental/`` - 实验数据可视化

   * **slow_points/** - 不动点分析
   * **model_specific/** - 专用模型分析器

.. grid:: 2 2 2 4
   :gutter: 3

   .. grid-item-card:: 📊 模型分析器
      :class-header: bg-primary text-white text-center

      分析 CANN 仿真输出

   .. grid-item-card:: 📈 数据分析器
      :class-header: bg-success text-white text-center

      分析实验神经记录

   .. grid-item-card:: 🔬 RNN 动力学分析
      :class-header: bg-info text-white text-center

      研究不动点和慢流形

   .. grid-item-card:: 🌐 拓扑数据分析
      :class-header: bg-warning text-dark text-center

      检测神经活动中的几何结构

模型分析器
==============

模型分析器可视化 CANN 仿真的输出，专注于网络活动模式及其随时间的演化。

核心功能
-----------------

.. tab-set::

   .. tab-item:: 📹 活动可视化

      .. list-table::
         :widths: 40 60

         * - ``animate_dynamics()``
           - 动画显示放电率随时间的演化
         * - ``plot_network_state()``
           - 当前活动模式的快照
         * - ``plot_bump_trajectory()``
           - 跟踪活动波包中心位置

   .. tab-item:: ⚡ 能量景观

      .. list-table::
         :widths: 40 60

         * - ``energy_landscape_1d()``
           - 可视化吸引子盆地结构
         * - ``energy_landscape_2d()``
           - 二维能量表面
         * - **用途**
           - 展示不同状态如何与吸引子最小值相关

   .. tab-item:: 🔗 连接性

      .. list-table::
         :widths: 40 60

         * - ``plot_weight_matrix()``
           - 可视化循环连接
         * - ``plot_connection_profile()``
           - 单个神经元的连接模式
         * - **用途**
           - 揭示墨西哥帽或其他核结构

设计哲学
-----------------

.. important::

   模型分析器函数接收仿真结果作为数组而不是模型对象。这种独立性意味着：

   * 相同的可视化适用于不同的模型类型
   * 结果可以保存并稍后分析
   * 分析期间不依赖模型内部结构

   **函数接受标准化格式：**

   * 放电率作为 ``(time, neurons)`` 数组
   * 膜电位作为 ``(time, neurons)`` 数组
   * 用于活动波包定位的空间坐标

PlotConfig 系统
-----------------

.. admonition:: 配置模式
   :class: tip

   库使用 ``PlotConfig`` 数据类进行可视化配置：

   **好处：**

   * ✅ **可重用性**：相同的配置应用于多个图
   * ✅ **类型安全**：在构造时验证参数
   * ✅ **共享**：在函数之间传递配置对象

   **常见配置包括：**

   * ``figsize``：图形尺寸
   * ``interval``：动画速度
   * ``colormap``：颜色方案选择
   * ``show_colorbar``：切换颜色图例

   虽然 PlotConfig 提供便利，但为了向后兼容，仍然支持直接参数传递。

数据分析器
=============

数据分析器处理实验神经记录，通常是脉冲序列或放电率估计。

与模型分析器的关键区别
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 方面
     - 模型分析器
     - 数据分析器
   * - **输入数据**
     - 干净的仿真输出
     - 脉冲序列——稀疏、离散事件——和放电率估计
   * - **关注点**
     - 可视化 CANN 动力学
     - 解码神经活动、拟合参数模型
   * - **噪声**
     - 最小（仿真）
     - 可能有噪声或不完整的记录

功能
------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📊 群体活动分析
      :class-header: bg-light text-center

      * 从神经群体估计活动波包位置
      * 拟合高斯曲线到活动模式
      * 跟踪解码位置随时间的变化

   .. grid-item-card:: 🔬 虚拟数据生成
      :class-header: bg-light text-center

      * 创建用于算法测试的合成脉冲序列
      * 生成真值场景
      * 验证分析流水线

   .. grid-item-card:: 📈 统计工具
      :class-header: bg-light text-center

      * 调谐曲线估计
      * 角度变量的圆形统计
      * 误差量化指标

   .. grid-item-card:: 🎯 用例
      :class-header: bg-light text-center

      * 针对实验记录验证 CANN 模型
      * 为神经数据开发解码算法
      * 用模拟实验测试理论预测

RNN 动力学分析
=====================

该组件将递归神经网络作为动力系统进行分析 :cite:p:`sussillo2013opening`，找到不动点 :cite:p:`golub2018fixedpointfinder` 并表征相空间结构。

目的
-------

.. note::

   CANN 模型是连续时间动力系统。理解它们的行为需要：

   * 识别稳定的不动点（吸引子）
   * 找到不稳定的不动点（鞍点、排斥子）
   * 映射动力学集中的慢流形

方法
-------

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: 📍 不动点查找
      :class-header: bg-primary text-white

      定位动力学消失的状态（du/dt = 0）：

      * 数值求根
      * 多个初始条件以进行彻底搜索
      * 通过稳定性分类（特征值分析）

   .. grid-item-card:: 📊 稳定性分析
      :class-header: bg-success text-white

      表征不动点附近的动力学：

      * Jacobian 计算
      * 特征值分解
      * 吸引子 vs. 鞍点 vs. 排斥子分类

   .. grid-item-card:: 🌀 慢流形识别
      :class-header: bg-info text-white

      在状态空间中找到低维结构：

      * 降维
      * 识别慢动力学的方向
      * 可视化状态空间组织

当前范围
-------------

.. admonition:: 实现状态
   :class: note

   目前专注于分析 RNN 模型（包括 CANN 作为特例）。为以下提供工具：

   * 理解内在网络动力学
   * 表征吸引子景观
   * 研究参数变化下的分岔

拓扑数据分析（TDA）
================================

TDA 工具 :cite:p:`carlsson2009topology` 使用持续同调 :cite:p:`edelsbrunner2010computational` 检测高维神经活动数据中的几何和拓扑结构。

为什么 CANN 需要 TDA
-----------------

.. important::

   CANN 活动模式通常存在于低维流形上：

   * **环吸引子**：圆上的活动（1D 环面）
   * **环面吸引子**：2D 环面上的活动（网格细胞）
   * **球面吸引子**：球面上的活动

   传统方法可能会遗漏这些结构。TDA 提供数学上严格的检测。

可用工具
---------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔬 持续同调
      :class-header: bg-light text-center

      * 加速的 ripser 实现
      * 检测拓扑特征（环、空洞）
      * 持续图量化特征显著性

   .. grid-item-card:: 📉 降维
      :class-header: bg-light text-center

      * 用户应用外部工具（UMAP、PCA 等）
      * 库提供预处理实用程序
      * 降维表示的可视化

用例
---------

.. tab-set::

   .. tab-item:: 🧭 网格细胞分析

      网格细胞在环面上编码位置。TDA 可以：

      * ✅ 验证神经记录中的环面结构 :cite:p:`carlsson2009topology,edelsbrunner2010computational`
      * ✅ 量化活动与理论预测的匹配程度
      * ✅ 检测与理想拓扑的偏差

   .. tab-item:: 🔍 吸引子结构发现

      对于未知网络：

      * ✅ 从活动模式推断吸引子几何
      * ✅ 测试关于编码流形的假设
      * ✅ 将实验数据与模型预测进行比较

实现说明
---------------------

.. admonition:: 技术细节
   :class: tip

   * **Ripser 集成**，用于快速持续同调 :cite:p:`carlsson2009topology,edelsbrunner2010computational`
   * **外部依赖**，用于某些高级方法
   * **专注于工具**，与吸引子网络研究相关

   有关性能详细信息，请参阅 ``canns-lib`` Ripser 模块（1.13 倍平均加速，最高 1.82 倍）。

总结
=======

分析模块提供全面的工具：

.. grid:: 2 2 2 4
   :gutter: 2

   .. grid-item-card::
      :class-header: bg-primary text-white text-center

      1️⃣
      ^^^
      **模型分析器**：使用标准化函数可视化 CANN 仿真输出

   .. grid-item-card::
      :class-header: bg-success text-white text-center

      2️⃣
      ^^^
      **数据分析器**：处理实验记录和合成神经数据

   .. grid-item-card::
      :class-header: bg-info text-white text-center

      3️⃣
      ^^^
      **RNN 动力学**：研究不动点和相空间结构

   .. grid-item-card::
      :class-header: bg-warning text-dark text-center

      4️⃣
      ^^^
      **TDA**：检测神经表征的拓扑属性

这些工具既支持正向建模（仿真分析）又支持逆向工程（实验数据解释）——支持从理论到验证的完整研究周期。
