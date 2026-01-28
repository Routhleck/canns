教程3：ASA GUI 端到端分析教程
===========================

本教程介绍 ASA GUI（Attractor Structure Analyzer）如何将 ASA pipeline 的数据预处理、
TDA、解码与可视化整合为图形化工作流。你将学会用窗口界面完成分析并在结果目录中管理输出。

.. note::
   若想深入了解分析原理与参数选择，请参阅
   :doc:`../02_data_analysis/01_asa_pipeline`。

教程目标
--------

- 使用 ASA GUI 完成一次端到端分析
- 理解输入数据结构、参数含义与输出目录
- 掌握常见分析模式与依赖关系

适用人群
--------

- 需要图形界面完成 ASA pipeline 的用户
- 想快速浏览分析结果与图表的实验研究者
- 希望复用 TDA/解码/可视化流程的实验组

前置准备
--------

- 已安装 CANNs（建议 ``pip install -e .`` 或 ``pip install canns``）
- 安装 GUI 依赖：``pip install canns[gui]``（包含 PySide6）
- 可选：``pip install qtawesome``（用于导航栏图标）
- 准备好 ASA ``.npz`` 数据

.. note::
   ASA GUI 当前仅支持 **ASA .npz** 输入。
   “Neuron + Trajectory” 相关控件仍处于预留状态。

启动 ASA GUI
------------

在项目环境中执行：

.. code-block:: bash

   canns-gui
   # 或
   python -m canns.pipeline.asa_gui

启动后会显示主窗口，默认尺寸约 1200×800，可根据屏幕调整大小。

界面概览
--------

主窗口顶部包含：

- **Preprocess** 与 **Analysis** 导航按钮
- **Light / Dark** 主题切换
- **Help** 帮助按钮（打开快速使用说明）
- **中文 / English** 双语切换

Preprocess 页：输入配置、预处理参数与日志。
Analysis 页：分析参数、结果预览与文件输出。

.. figure:: /_static/asa_gui_overview.png
   :alt: ASA GUI 界面总览
   :width: 90%

   ASA GUI 主界面总览（占位图）

界面元素清单
------------

Preprocess 页面
^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Input / Mode**
     - 固定为 ``ASA (.npz)``；当前 GUI 仅支持 ASA 输入。
   * - **Preset**
     - 预设模板：``grid`` / ``hd`` / ``none``（影响分析页默认值）。
   * - **ASA file**
     - 拖拽或点击 **Browse** 选择 ``.npz`` 文件。
   * - **Preprocess / Method**
     - 选择 ``None`` 或 ``Embed spike trains``。
   * - **Embedding 参数**
     - ``res`` / ``dt`` / ``sigma`` / ``smooth`` / ``speed_filter`` / ``min_speed``。
   * - **Pre-classification**
     - 预留选项（``none`` / ``grid`` / ``hd``），当前仅记录，不影响分析。
   * - **Run / Stop / Progress**
     - 启动或终止预处理，并显示进度条。
   * - **Logs**
     - 显示预处理过程日志。

.. figure:: /_static/asa_gui_preprocess.png
   :alt: ASA GUI 预处理界面
   :width: 85%

   ASA GUI 预处理页（占位图）

Analysis 页面
^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Analysis Parameters**
     - 选择分析模块并设置参数。
   * - **Analysis module**
     - 支持 ``TDA`` / ``CohoMap`` / ``PathCompare`` / ``CohoSpace`` / ``FR`` / ``FRM`` / ``GridScore``。
   * - **Preprocess (Standardization)**
     - ``StandardScaler`` 标准化选项。
   * - **Help**
     - 打开快速帮助与常见提示。
   * - **Language**
     - 在中文 / English 之间切换界面语言。
   * - **Run Analysis / Stop / Progress**
     - 运行分析并显示进度。
   * - **结果标签页**
     - ``Barcode`` / ``CohoMap`` / ``Path Compare`` / ``CohoSpace`` / ``FR`` / ``FRM`` / ``GridScore`` / ``Files``。
   * - **Files**
     - 列出输出文件并可 **Open Folder** 打开结果目录。

.. figure:: /_static/asa_gui_tda.png
   :alt: ASA GUI TDA 模式
   :width: 85%

   TDA 模式示例（占位图）

.. figure:: /_static/asa_gui_cohomap.png
   :alt: ASA GUI CohoMap 模式
   :width: 85%

   CohoMap 模式示例（占位图）

.. figure:: /_static/asa_gui_path_compare.png
   :alt: ASA GUI Path Compare 模式
   :width: 85%

   Path Compare 模式示例（占位图）

.. figure:: /_static/asa_gui_cohospace.png
   :alt: ASA GUI CohoSpace 模式
   :width: 85%

   CohoSpace 模式示例（占位图）

.. figure:: /_static/asa_gui_fr.png
   :alt: ASA GUI FR 模式
   :width: 85%

   FR 模式示例（占位图）

.. figure:: /_static/asa_gui_frm.png
   :alt: ASA GUI FRM 模式
   :width: 85%

   FRM 模式示例（占位图）

.. figure:: /_static/asa_gui_gridscore.png
   :alt: ASA GUI GridScore 模式
   :width: 85%

   GridScore 模式示例（占位图）

.. figure:: /_static/asa_gui_help_preprocess.png
   :alt: ASA GUI 预处理帮助
   :width: 60%

   Preprocess 帮助面板（占位图）

.. figure:: /_static/asa_gui_help_tda.png
   :alt: ASA GUI TDA 帮助
   :width: 60%

   TDA 帮助面板（占位图）

工作流程概览
------------

1. 启动 ASA GUI
2. 选择 ASA ``.npz`` 输入文件
3. 配置预处理参数并运行 Preprocess
4. 进入 Analysis 页选择分析模块
5. 运行分析并在标签页查看结果

步骤 1：选择 ASA 文件
----------------------

在 **ASA file** 区域拖拽或点击 **Browse** 选择 ``.npz`` 文件。
界面会提示期望的字段：``spike`` / ``x`` / ``y`` / ``t``。

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

步骤 2：设置预处理
------------------

在 **Preprocess** 区域选择预处理方法：

- **None**：直接使用原始 spike 结构
- **Embed spike trains**：生成稠密矩阵供 TDA/FR/FRM 使用

如需嵌入，设置 ``res`` / ``dt`` / ``sigma`` / ``smooth`` /
``speed_filter`` / ``min_speed`` 等参数，然后点击 **Run Preprocess**。

步骤 3：选择分析模块
--------------------

切换到 **Analysis** 页面，选择分析模块并配置参数：

- **TDA**：持久同调分析与条形码
- **CohoMap / CohoSpace**：解码并绘制空间结构
- **Path Compare**：轨迹比较（含动画输出）
- **FR / FRM**：放电率热图与神经元放电率图
- **GridScore**：网格评分与神经元浏览器

点击 **Run Analysis** 开始运行。

步骤 4：查看结果
----------------

运行完成后，在右侧标签页查看结果：

- **图像标签页**：支持内嵌预览与 **Open Image** 外部打开
- **GridScore**：包含分布图与神经元 inspector
- **Files**：列出输出文件并可打开输出目录

结果输出结构
------------

输出目录默认位于当前工作目录下：

``Results/<dataset>_<hash>/``

其中 ``<dataset>`` 由输入文件名生成，``<hash>`` 为输入哈希前缀。
目录中会包含分析结果与缓存（``.asa_cache``），以加速重复运行。

注意事项
--------

- GUI 的工作目录为 **启动 GUI 时的当前目录**，可在命令行中切换目录后再启动。
- 如果需要 ``Neuron + Trajectory`` 输入，请使用 ASA TUI 或脚本流程。
