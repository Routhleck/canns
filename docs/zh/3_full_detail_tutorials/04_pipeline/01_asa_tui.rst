教程2：ASA TUI 端到端分析教程（旧版）
==============================

本教程介绍 ASA TUI（Attractor Structure Analyzer）如何将 ASA pipeline 的数据预处理、
TDA、解码与可视化整合为交互式终端工作流。你将学会用界面完成分析并在工作目录中管理结果。

.. note::
   ASA TUI 为旧版界面，仅作为过渡使用，推荐优先使用
   :doc:`ASA GUI 教程 <03_asa_gui>`。

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
   # 或
   canns-tui

.. note::
   ``canns-tui`` 现在是统一入口，启动后先选择 **ASA** 或 **Model Gallery**。
   若只想直接进入 ASA，可使用 ``python -m canns.pipeline.asa``。

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

界面元素清单
------------

左侧操作区
^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Workdir 标签**
     - 显示当前工作目录路径。
   * - **Change Workdir**
     - 打开工作目录选择窗口（等同 ``Ctrl-W``）。
   * - **Page 指示器**
     - 显示当前页面：``Preprocess`` 或 ``Analysis``。
   * - **Continue →**
     - 触发预处理并切换到分析页（仅在预处理页显示）。
   * - **← Back**
     - 返回预处理页（仅在分析页显示）。
   * - **Run Analysis**
     - 运行当前分析模式（仅在分析页显示）。
   * - **Stop**
     - 请求终止当前运行的任务（分析页可用）。
   * - **进度条**
     - 显示预处理/分析进度。
   * - **Status**
     - 显示运行状态（Idle / Running / Success / Error）。

中部参数区
^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - 参数控制区域，按预处理页与分析页切换。
   * - **Input Mode**
     - 选择 ``ASA File`` 或 ``Neuron + Traj``。
   * - **Preset**
     - 预设模板：``Grid`` / ``HD`` / ``None``（会刷新 TDA/GridScore 默认值）。
   * - **Method**
     - 预处理方法：``None`` / ``Embed Spike Trains``。
   * - **Files in Workdir**
     - 工作目录文件树，选中图片可预览，选中 ``.npz`` 可作为输入。

右侧结果区
^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Setup 标签页**
     - 显示快速操作提示。
   * - **Results 标签页**
     - 结果预览与提示信息；完成分析后自动切换。
   * - **Log Viewer**
     - 实时显示运行日志；同时写入日志文件。

.. note::
   ASA TUI 的图片预览以 **终端字符图** 形式展示（基于 ``climage``）。
   若系统具备图形界面和图像查看器，可点击 **Open** 在外部查看真实图片。

快捷键
------

.. list-table::
   :widths: 30 70

   * - 快捷键
     - 功能
   * - ``Ctrl-W``
     - 选择工作目录
   * - ``Ctrl-R``
     - 运行当前页面动作（预处理页=Continue，分析页=Run Analysis）
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

工作目录相关说明
^^^^^^^^^^^^^^^^

- 工作目录决定 **文件树** 与 **结果输出目录** 的根位置。
- 切换工作目录会重置当前输入状态，需要重新选择数据文件。
- 按 ``F5`` 可刷新文件树与当前路径显示。

步骤 2：选择输入模式与文件
--------------------------

输入模式位于 **Input Mode** 下拉框：

- **ASA File**：单个 ``.npz`` 文件，包含 ``spike`` 与 ``t``（推荐包含 ``x``/``y``）。
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

文件树与预览
^^^^^^^^^^^^

- 选中 ``.npz`` 且输入模式为 **ASA File** 时，会设置 ASA 输入文件。
- 选中 ``.png/.jpg/.jpeg/.gif/.bmp`` 时，会在右侧预览该图片。
- 结果生成后，文件树会自动刷新，便于浏览输出目录。

步骤 3：预处理设置
------------------

**Method** 支持两种选项：

- ``None``：假设输入已是稠密 ``T x N`` 矩阵
- ``Embed Spike Trains``：将 spike times 嵌入为稠密矩阵（推荐）

Preset 说明
^^^^^^^^^^^

- ``Grid`` / ``HD`` 会将 TDA 与 GridScore 参数恢复为相应默认值。
- ``None`` 不做改动，保留当前参数输入。
- 若你已手动调整参数，切换 preset 可能覆盖这些值。

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

.. note::
   只有选择 ``Embed Spike Trains`` 时，上述参数才可编辑；选择 ``None`` 会禁用它们。

设置完成后点击 **Continue →** 进入分析页。

.. figure:: /_static/asa_tui_preprocess.png
   :alt: 预处理参数设置
   :width: 90%

   预处理参数区域

预处理页按钮行为
^^^^^^^^^^^^^^^^

- **Continue →**：触发预处理任务；完成后自动切换到分析页。
- 预处理过程中按钮会被锁定，避免重复提交。
- 若输入数据缺失必需字段，会弹出错误窗口提示。

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

选择分析模式后，参数面板会自动切换对应参数组；其中 **Decode / CohoMap** 参数在
CohoMap / PathCompare / CohoSpace 三种模式之间共享。

依赖关系：

- CohoMap 需要先完成 TDA
- PathCompare / CohoSpace 依赖 CohoMap

分析参数详解
^^^^^^^^^^^^

TDA 参数
~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``dim``
     - 嵌入维度
   * - ``num_times``
     - 采样次数
   * - ``active_times``
     - 活跃时间长度
   * - ``k``
     - 近邻数量
   * - ``n_points``
     - 采样点数量
   * - ``metric``
     - 距离度量（cosine / euclidean / correlation）
   * - ``nbs``
     - 邻域采样数
   * - ``maxdim``
     - 最大同调维度
   * - ``coeff``
     - 同调系数
   * - ``do_shuffle``
     - 是否随机打乱
   * - ``num_shuffles``
     - 打乱次数（仅在 ``do_shuffle`` 开启时可编辑）
   * - ``standardize``
     - 是否使用 StandardScaler 标准化

Decode / CohoMap 参数
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``decode_version``
     - 解码版本：``v2`` （multi）或 ``v0`` （legacy）
   * - ``num_circ``
     - 期望环的数量
   * - ``cohomap_subsample``
     - cohomap 轨迹下采样
   * - ``real_ground`` / ``real_of``
     - 仅 ``v0`` 模式启用

.. note::
   当 ``decode_version`` 为 ``v2`` 时，``real_ground`` 与 ``real_of`` 会被禁用。

PathCompare 参数
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``use_box``
     - 使用 ``coordsbox`` / ``times_box`` 裁剪
   * - ``interp_to_full``
     - 仅在 ``use_box`` 开启时可用
   * - ``dim_mode``
     - ``1d`` 或 ``2d``
   * - ``dim`` / ``dim1`` / ``dim2``
     - 维度选择（随 ``dim_mode`` 启用）
   * - ``coords_key`` / ``times_box_key``
     - 可选的字段名覆盖
   * - ``slice_mode``
     - ``time`` 或 ``index``
   * - ``tmin`` / ``tmax``
     - 时间切片（``-1`` 表示自动）
   * - ``imin`` / ``imax``
     - 索引切片（``-1`` 表示自动）
   * - ``stride``
     - 采样步长
   * - ``theta_scale``
     - 角度单位（rad / deg / unit / auto）

.. note::
   ``dim_mode`` 为 ``1d`` 时仅启用 ``dim``；为 ``2d`` 时启用 ``dim1``/``dim2``。
   ``slice_mode`` 为 ``time`` 时启用 ``tmin``/``tmax``，为 ``index`` 时启用 ``imin``/``imax``。
   ``use_box`` 关闭时会禁用 ``interp_to_full``。

CohoSpace 参数
~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``dim_mode``
     - ``1d`` 或 ``2d``
   * - ``dim`` / ``dim1`` / ``dim2``
     - 维度选择（随 ``dim_mode`` 启用）
   * - ``mode``
     - 使用 ``fr`` 或 ``spike`` 信号
   * - ``top_percent``
     - 用于选择高活跃神经元的百分比
   * - ``view``
     - ``both`` / ``single`` / ``population``
   * - ``neuron_id``
     - 单神经元视图时可指定
   * - ``subsample``
     - 轨迹下采样倍率
   * - ``unfold``
     - ``square`` 或 ``skew`` 展开
   * - ``skew_show_grid`` / ``skew_tiles``
     - 仅在 ``unfold=skew`` 时启用

.. note::
   ``dim_mode`` 为 ``1d`` 时仅启用 ``dim``；为 ``2d`` 时启用 ``dim1``/``dim2``。
   ``unfold`` 为 ``skew`` 时启用 ``skew_show_grid`` 与 ``skew_tiles``。

FR 参数
~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``neuron_start`` / ``neuron_end``
     - 神经元范围（留空表示全部）
   * - ``time_start`` / ``time_end``
     - 时间范围（留空表示全部）
   * - ``mode``
     - ``fr`` 或 ``spike``
   * - ``normalize``
     - ``zscore_per_neuron`` / ``minmax_per_neuron`` / ``none``

.. note::
   ``neuron_start`` / ``neuron_end`` 与 ``time_start`` / ``time_end`` 留空时视为全范围。

FRM 参数
~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``neuron_id``
     - 需要绘制的神经元编号
   * - ``bins``
     - 空间网格数量
   * - ``min_occupancy``
     - 最小占据阈值
   * - ``smoothing``
     - 是否平滑
   * - ``smooth_sigma``
     - 平滑尺度
   * - ``mode``
     - ``fr`` 或 ``spike``

GridScore 参数
~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``annulus_inner`` / ``annulus_outer``
     - 环形区域内外半径
   * - ``bin_size``
     - 空间分箱尺寸
   * - ``smooth_sigma``
     - 平滑尺度

步骤 5：运行与结果查看
----------------------

点击 **Run Analysis** 或按 ``Ctrl-R`` 开始分析。运行进度与日志会实时显示。
最终完成后的可视化结果会在图像预览控件预览

.. figure:: /_static/asa_tui_results.png
   :alt: 结果预览
   :width: 90%

   分析结果与日志区域

图像预览控件
^^^^^^^^^^^^

在 **Results** 标签页的 Image Preview 区域可直接操作：

- **Load**：读取输入框中的路径并在 TUI 内预览；支持相对 workdir 的相对路径。
- **Open**：调用系统图像查看器打开原图（需要操作系统有图形界面）。
- **Zoom + / Zoom -**：放大或缩小终端字符图预览。
- **Fit**：重置缩放级别并适配当前预览区域。
- **↑/↓/←/→**：在预览区域内平移（仅影响视图，不改变图像文件）。

.. note::
   如果未安装 ``climage``，预览区域会显示文件名提示；建议安装后获得字符图预览效果。

最终结果默认保存在：

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



按钮与状态的运行逻辑
^^^^^^^^^^^^^^^^^^^^

- **Run Analysis** 会读取当前页面参数并启动分析任务。
- 运行时 **Stop** 可请求取消任务；日志会提示取消状态。
- 预处理与分析完成后会自动刷新文件树并切换到结果页。
- 结果预览优先显示 ``barcode`` / ``cohomap`` / ``path_compare`` / ``cohospace_trajectory`` /
  ``fr_heatmap`` / ``frm`` / ``distribution`` 等产物。

弹窗与帮助
----------

- 终端尺寸不足时，会弹出警告窗口提示推荐大小。
- 参数或输入错误时，会弹出错误窗口并显示原因。
- 按 ``?`` 打开帮助屏，列出快捷键与流程说明。
