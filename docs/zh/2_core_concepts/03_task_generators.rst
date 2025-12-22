================
任务生成器
================

本文档说明 CANNs 库中的任务生成理念和可用范式。

概述
========

任务模块（``canns.task``）为 CANN 仿真生成实验数据，并支持保存、加载、导入和可视化功能。它提供了标准化的范式，抽象了常见的实验场景，确保可重复性和便利性。

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 📊 生成输入序列
      :class-header: bg-primary text-white text-center

      创建驱动网络动态的时变外部输入

   .. grid-item-card:: 🎯 提供真实值
      :class-header: bg-success text-white text-center

      为分析和比较提供轨迹信息

任务类别
===============

任务根据其建模的认知功能分为两大类。

.. tab-set::

   .. tab-item:: 📍 追踪任务

      **追踪任务模拟网络跟随外部信号的场景。** CANN 中的活动波包追踪移动的刺激位置。

      .. grid:: 1
         :gutter: 2

         .. grid-item-card:: 🔵 群体编码
            :class-header: bg-light

            网络在固定位置接收静态输入。测试基本吸引子稳定性和群体表征准确性。

         .. grid-item-card:: 🎯 模板匹配
            :class-header: bg-light

            网络接收短暂的、可能含噪声的输入。测试模式补全和识别能力。

         .. grid-item-card:: 🔄 平滑追踪
            :class-header: bg-light

            **最常用的范式**

            网络接收连续移动的输入信号。测试具有不同速度和方向的动态追踪能力。

            **可用实现：**

            * ``SmoothTracking1D``：用于环形网络的一维追踪
            * ``SmoothTracking2D``：用于环面网络的二维追踪（开发中）

   .. tab-item:: 🧭 导航任务

      **导航任务模拟空间移动场景** :cite:p:`mcnaughton2006path`，其中网络通过路径积分 :cite:p:`etienne2004path,samsonovich1997path` 接收速度或朝向信息，而不是直接的位置输入。

      .. grid:: 1
         :gutter: 2

         .. grid-item-card:: 🔁 闭环导航
            :class-header: bg-light

            网络基于自运动信号更新其内部表征。来自环境的反馈可以纠正误差。

         .. grid-item-card:: ➡️ 开环导航
            :class-header: bg-light

            网络在没有外部反馈的情况下积分速度输入。测试路径积分能力和随时间累积的误差。

      .. note::

         导航任务不需要直接的模型耦合，因为它们提供更丰富的数据（速度、角度等），用户可根据具体应用进行解释。

模型-任务耦合
===================

为何存在耦合
-------------------

追踪任务需要在构造时传入 CANN 模型实例::

   task = SmoothTracking1D(cann_instance=cann, ...)

.. important::

   这种耦合的存在是因为追踪任务需要访问 ``cann.get_stimulus_by_pos()``。该方法将抽象位置坐标转换为与网络编码方案匹配的具体神经输入模式。

   **耦合为用户提供了便利：**

   * 自动生成与网络拓扑匹配的刺激
   * 任务和模型之间的编码一致性
   * 减少常见用例的样板代码

何时需要耦合
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 任务类型
     - 需要 ``cann_instance``
     - 提供的数据
   * - **追踪任务**
     - ✅ 是
     - 神经空间中的输入模式
   * - | 群体编码
       | 模板匹配
       | 平滑追踪
     - ✅ 是
     - 内部使用 ``model.get_stimulus_by_pos()``
   * - **导航任务**
     - ❌ 否
     - 速度、朝向、位置数据
   * - | 闭环导航
       | 开环导航
     - ❌ 否
     - 用户决定如何转换为神经输入

.. admonition:: 设计原理
   :class: note

   这种区别反映了这些范式的不同性质。追踪涉及对网络的直接感觉输入，而导航涉及基于自运动的内部状态更新。

任务组件
===============

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⚙️ 任务配置
      :class-header: bg-light text-center

      任务通过构造函数参数进行配置：

      * **目标位置**：刺激出现或移动到的位置
      * **持续时间**：每个片段持续多长时间
      * **时间步长**：时间分辨率（来自 ``bm.get_dt()``）
      * **附加参数**：速度曲线、噪声水平、初始条件

   .. grid-item-card:: 📊 数据生成
      :class-header: bg-light text-center

      ``get_data()`` 方法返回：

      * **输入序列**：随时间变化的神经输入数组（用于追踪任务）
      * **轨迹信息**：位置、速度、时间戳
      * **元数据**：用于文档记录的任务参数

   .. grid-item-card:: 💾 数据持久化
      :class-header: bg-light text-center

      任务支持保存和加载：

      * ``save(filename)``：存储任务数据以实现可重复性
      * ``load(filename)``：重新加载先前生成的任务
      * 标准格式确保兼容性

   .. grid-item-card:: 📥 轨迹导入
      :class-header: bg-light text-center

      **开发中的功能**

      该库支持从实验记录导入外部轨迹。这使得：

      * 重放真实动物的运动路径
      * 针对实验数据进行验证
      * 将模型预测与神经记录进行比较

任务使用模式
====================

标准工作流
-----------------

.. admonition:: 典型使用步骤
   :class: tip

   1. **创建模型实例**
   2. **配置任务**，设置位置和持续时间
   3. **生成数据**，使用 ``get_data()``
   4. **运行仿真**，将任务输入馈送到模型
   5. **分析结果**，将模型输出与任务轨迹进行比较

多次试验生成
--------------------------

任务支持生成多次试验：

* 相同范式，不同随机种子
* 系统性参数变化
* 批处理能力

参数扫描
----------------

将任务与分析流程结合以：

* 测试模型在不同条件下的鲁棒性
* 寻找最优参数范围
* 表征吸引子属性

设计考虑
=====================

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: ⏱️ 时间步长一致性
      :class-header: bg-warning text-dark text-center

      任务使用 ``bm.get_dt()`` 来确保时间分辨率与仿真环境匹配。

      **始终在创建任务之前设置全局时间步长：**

      .. code-block:: python

         bm.set_dt(0.1)
         task = SmoothTracking1D(...)

   .. grid-item-card:: 🎯 位置编码
      :class-header: bg-info text-white text-center

      任务在抽象特征空间（角度、坐标）中操作。神经活动模式的转换由以下处理：

      * 对于直接耦合，使用 ``model.get_stimulus_by_pos()``
      * 对于解耦场景，使用用户定义的编码

   .. grid-item-card:: 🔧 可扩展性
      :class-header: bg-success text-white text-center
      :columns: 12

      可以通过以下方式创建自定义任务：

      * 从基础任务类继承
      * 实现所需的数据生成方法
      * 遵循输出格式的约定

总结
=======

任务模块提供：

.. grid:: 2 2 2 4
   :gutter: 2

   .. grid-item-card::
      :class-header: bg-light text-center

      1️⃣
      ^^^
      **追踪任务**：直接刺激跟随（群体编码、模板匹配、平滑追踪）

   .. grid-item-card::
      :class-header: bg-light text-center

      2️⃣
      ^^^
      **导航任务**：自运动积分（闭环、开环导航）

   .. grid-item-card::
      :class-header: bg-light text-center

      3️⃣
      ^^^
      **模型耦合**：追踪任务的自动刺激生成

   .. grid-item-card::
      :class-header: bg-light text-center

      4️⃣
      ^^^
      **灵活性**：导航任务允许用户自定义输入解释

任务将实验范式抽象为可重用的组件——使得能够在标准化条件下系统地研究 CANN 动态。耦合设计在常见情况的便利性和专门应用的灵活性之间取得平衡。
