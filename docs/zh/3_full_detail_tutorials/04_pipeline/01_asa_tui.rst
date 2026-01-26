教程1：ASA TUI 端到端分析教程
======================

本教程介绍 ASA TUI（Attractor Structure Analyzer）如何将 ASA pipeline 的数据预处理、
TDA、解码与可视化整合为交互式终端工作流。你将学会用界面完成分析并在工作目录中管理结果。

教程目标
--------

- 使用 ASA TUI 完成一次端到端分析
- 理解输入数据结构、参数含义与输出目录
- 掌握常见分析模式与依赖关系

适用人群
--------

- 需要快速分析实验神经记录的研究者
- 希望通过可视化界面完成 ASA pipeline 的用户
- 需要复用 TDA/解码/可视化流程的实验组

前置准备
--------

- 已安装 CANNs（建议 ``pip install -e .`` 或 ``pip install canns``）
- 终端宽度建议不低于 120 列，至少 100 列
- 准备好 ASA 或 Neuron + Trajectory 数据

启动 ASA TUI
------------

在项目环境中执行：

.. code-block:: bash

   python -m canns.pipeline.asa
   # or
   canns-tui

启动后若出现尺寸警告，请调大终端窗口或缩小字体。

界面概览
--------

界面分为三列：

- 左侧：工作目录、运行按钮、进度条、状态信息
- 中间：参数面板 + 工作目录文件树
- 右侧：结果预览与日志输出

.. figure:: /_static/asa_tui_overview.png
   :alt: ASA TUI 界面总览
   :width: 90%

   ASA TUI 主界面总览

快捷键
------

.. list-table::
   :widths: 30 70

   * - 快捷键
     - 功能
   * - ``Ctrl-W``
     - 选择工作目录
   * - ``Ctrl-R``
     - 运行分析
   * - ``F5``
     - 刷新预览
   * - ``?``
     - 打开帮助
   * - ``Esc``
     - 退出应用
   * - ``Tab``
     - 面板间切换焦点

工作流程概览
------------

1. 选择工作目录（包含数据文件）
2. 选择输入模式与数据文件
3. 配置预处理参数
4. 进入分析页并选择分析模式
5. 运行分析并查看 Results 输出

步骤 1：选择工作目录
----------------------

点击左侧 **Change Workdir** 按钮或按 ``Ctrl-W``，选择包含数据的目录。
文件树会同步刷新，便于选取输入数据和查看输出结果。

.. figure:: /_static/asa_tui_workdir.png
   :alt: 选择工作目录
   :width: 85%

   选择工作目录弹窗

步骤 2：选择输入模式与文件
--------------------------

输入模式位于 **Input Mode** 下拉框：

- **ASA File**：单个 ``.npz`` 文件，包含 ``spike`` 与 ``t`` （推荐包含 ``x``/``y``）。
- **Neuron + Traj**：神经元文件与轨迹文件分离。

ASA 文件格式
^^^^^^^^^^^^

至少包含 ``spike`` 与 ``t``：

- ``spike``: ``T x N`` 的稠密矩阵，或可嵌入的 spike 数据结构
- ``t``: 时间序列（与 ``spike`` 同步）
- 可选：``x`` / ``y`` 用于轨迹相关分析

.. code-block:: python

   import numpy as np
   np.savez(
       "session_asa.npz",
       spike=spike,
       t=t,
       x=x,
       y=y,
   )

在文件树中选中 ``.npz`` 后，日志区会提示已选择 ASA 文件。

Neuron + Traj 模式
^^^^^^^^^^^^^^^^^^

需要两类 ``.npz``：

- **Neuron**: 包含 ``spike``（或整个文件本身为 spike 数组）
- **Traj**: 包含 ``x`` / ``y`` / ``t``

.. code-block:: python

   np.savez("neuron.npz", spike=spike)
   np.savez("traj.npz", x=x, y=y, t=t)

.. note::
   当前版本 TUI 的文件树仅支持直接选取 ASA 文件。若需使用 Neuron + Traj，
   建议先在脚本中合并为 ASA ``.npz``，或后续按需扩展选择逻辑。

步骤 3：预处理设置
------------------

**Method** 支持两种选项：

- ``None``：假设输入已是稠密 ``T x N`` 矩阵
- ``Embed Spike Trains``：将 spike times 嵌入为稠密矩阵（推荐）

关键参数说明：

.. list-table::
   :widths: 25 75

   * - 参数
     - 含义
   * - ``res``
     - 嵌入分辨率/采样长度
   * - ``dt``
     - 采样时间步长
   * - ``sigma``
     - 平滑核尺度
   * - ``smooth``
     - 是否进行平滑
   * - ``speed_filter``
     - 是否对速度进行过滤
   * - ``min_speed``
     - 最小速度阈值

设置完成后点击 **Continue →** 进入分析页。

.. figure:: /_static/asa_tui_preprocess.png
   :alt: 预处理参数设置
   :width: 90%

   预处理参数区域

步骤 4：选择分析模式
--------------------

分析页支持以下模式：

- **TDA**：持续同调分析，生成 barcode
- **CohoMap**：基于 TDA 的相位解码与 cohomap
- **PathCompare**：真实轨迹 vs 解码轨迹
- **CohoSpace**：相位流形投影与轨迹可视化
- **FR**：群体放电率热图
- **FRM**：单神经元放电率地图
- **GridScore**：网格细胞指标分析

依赖关系：

- CohoMap 需要先完成 TDA
- PathCompare / CohoSpace 依赖 CohoMap

步骤 5：运行与结果查看
----------------------

点击 **Run Analysis** 或按 ``Ctrl-R`` 开始分析。运行进度与日志会实时显示。
结果默认保存在：

``<workdir>/Results/<dataset_id>/``

其中 ``dataset_id`` 由输入文件名与哈希前缀组成。
常见输出示例：

- ``TDA/``: ``barcode.png``、``persistence.npz``
- ``CohoMap/``: ``cohomap.png``、``decoding.npz``
- ``PathCompare/``: ``path_compare.png``
- ``CohoSpace/``: ``cohospace_trajectory.png``、（可选）``cohospace_neuron_*.png``
- ``FR/``: ``fr_heatmap.png``
- ``FRM/``: ``frm_neuron_<id>.png``
- ``GridScore/``: ``gridscore_distribution.png``、``gridscore.npz``

日志文件默认写入：

``<workdir>/Results/<dataset_id>/asa_tui.log``

你也可以在文件树中选中图片文件进行预览。

.. figure:: /_static/asa_tui_results.png
   :alt: 结果预览
   :width: 90%

   分析结果与日志区域
