教程3：模型画廊 TUI 教程
=================

本教程介绍 Model Gallery TUI 的使用方法。Gallery 以交互式终端界面运行三类模型
（CANN1D / CANN2D / GridCell）的完整分析流程，**覆盖**
``canns-experiments/figure2`` 中所有子图对应的可视化结果，并以 PNG 输出。

全结果合成图
--------------------

参见 :ref:`全结果合成图 <fig-gallery-full>`，后续将替换为真实输出。

.. _fig-gallery-full:

.. figure:: /_static/figure2_full.png
   :alt: figure2 全结果图
   :width: 90%

   全结果合成图

教程目标
--------

- 在 TUI 中选择模型与分析模式并生成图像结果
- 了解每个模型的 5 类标准可视化（对齐 :ref:`全结果合成图 <fig-gallery-full>` 的子图）
- 掌握结果目录结构、文件命名与日志路径

适用人群
--------

- 希望快速复现实验脚本可视化的研究者
- 需要在终端内完成模型分析与结果浏览的用户

前置准备
--------

- 已安装 CANNs（建议 ``pip install -e .`` 或 ``pip install canns``）
- 终端宽度建议不低于 120 列，至少 100 列

启动 Model Gallery TUI
----------------------

推荐方式：使用统一入口 ``canns-tui``，启动后选择 **Model Gallery**。

.. code-block:: bash

   canns-tui
   # 启动后在选择页选择 Model Gallery

也可以直接启动 Gallery：

.. code-block:: bash

   python -m canns.pipeline.gallery
   # 或
   canns-gallery

启动后若出现尺寸警告，请调大终端窗口或缩小字体。

界面概览
--------

.. figure:: /_static/gallery_tui_overview.png
   :alt: Model Gallery TUI 界面总览
   :width: 90%

   Model Gallery TUI 界面总览

界面结构
--------

- 左侧：模型选择、分析模式、运行按钮、进度与状态
- 中部：参数面板（可滚动）
- 右侧：结果预览与日志

界面元素清单
------------

左侧操作区
^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Model**
     - 选择 CANN1D / CANN2D / GridCell。
   * - **Analysis**
     - 选择对应模型的分析模式（见下方“模型与分析一览”）。
   * - **Run**
     - 启动当前分析任务。
   * - **进度条**
     - 显示任务执行进度。
   * - **Status**
     - 显示运行状态（Idle / Running / Success / Error）。

中部参数区
^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - 当前模型的参数面板；内容可滚动。
   * - **模型参数**
     - 包含网络规模、时间步长、连接参数等。
   * - **分析参数**
     - 不同分析模式具有不同参数（如 duration / resolution / warmup）。

右侧结果区
^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Image Preview**
     - 结果图片的终端预览（字符图）。可用 **Open** 外部查看。
   * - **Log Viewer**
     - 任务运行日志（含错误与路径提示）。

快捷键
------

.. list-table::
   :widths: 30 70

   * - **Ctrl-W**
     - 切换工作目录（Workdir）。
   * - **Ctrl-R**
     - 运行当前分析。
   * - **F5**
     - 刷新预览（重新加载结果图片）。
   * - **Esc**
     - 退出应用。

模型与分析一览
--------------

Gallery 的分析项与 ``canns-experiments/figure2`` 对齐（见 :ref:`全结果合成图 <fig-gallery-full>`）：

**CANN1D** （A.1–A.5 / 对应 :ref:`全结果合成图 <fig-gallery-full>` 第一行）：

- Connectivity Matrix
- Energy Landscape
- Tuning Curves
- Template Matching
- Neural Manifold

**CANN2D** （B.1–B.5 / 对应 :ref:`全结果合成图 <fig-gallery-full>` 第二行）：

- Connectivity Matrix
- Energy Landscape
- Firing Field
- Trajectory Comparison
- Neural Manifold

**GridCell** （C.1–C.5 / 对应 :ref:`全结果合成图 <fig-gallery-full>` 第三行）：

- Connectivity Matrix
- Energy Landscape
- Firing Field（systematic rate map）
- Path Integration
- Neural Manifold

参数说明（重点参数）
-------------------

**CANN1D**:

- ``num``: 神经元数量（默认 256）。
- ``dt``: 时间步长（默认 0.1）。
- ``tuning_neurons``: 以逗号分隔的神经元索引（如 ``64,128,192``）。
- ``manifold_warmup``: 去除前期动态的 warmup 时长。

**CANN2D**:

- ``length``: 网络边长（默认 64）。
- ``field_resolution``: firing field 网格分辨率（默认 80）。
- ``traj_warmup``: 轨迹对比 warmup。
- ``manifold_downsample``: manifold 采样稀疏化。

**GridCell**:

- ``dt``: 网格细胞模型时间步长（默认 ``5e-4``）。
- ``field_resolution``: systematic rate map 分辨率（默认 100）。
- ``energy_heal_steps`` / ``path_heal_steps``: 初始化稳定化步数。

操作流程
--------

1. 启动 ``canns-tui``，在选择页进入 **Model Gallery**。\n
2. 在左侧选择 **Model** 与 **Analysis**。\n
3. 在中部参数区调整模型与分析参数（可滚动）。\n
4. 点击 **Run**，等待执行完成。\n
5. 在右侧预览结果图片或点击 **Open** 使用外部查看器。\n

结果输出
--------

所有结果均以 **PNG** 保存到工作目录下：

``<workdir>/Results/gallery/<model>/``

文件命名示例：

``cann1d_energy_seed42.png``

日志与预览
----------

运行日志会实时显示在右侧 **Log Viewer**。如需保存或复制日志，
可在终端中直接选择复制，或在需要时再补充日志文件输出功能。

界面截图占位
------------

.. figure:: /_static/gallery_tui_results.png
   :alt: Gallery 结果界面
   :width: 90%

   结果界面截图

常见问题
--------

- **滚动条不显示**：请扩大终端高度或使用滚轮滚动参数区。
- **运行时间较长**：部分分析（如 CANN2D/ GridCell）默认参数较重，可适当降低
  ``duration`` / ``resolution`` / ``num_batches``。

下一步
------

- 等待替换真实界面截图与综合图
- 根据你的实际参数需求继续扩展 Gallery 分析项
