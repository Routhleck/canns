================
分析方法
================

本文档整理 ``canns.analyzer`` 当前实现的分析与可视化工具，覆盖数据分析（ASA pipeline）、
模型可视化、空间指标、RNN 动力学以及模型专用分析。

.. note::
   如需深入了解拓扑数据分析与神经群体流形研究，可参阅
   :cite:p:`carlsson2009topology,edelsbrunner2010computational,Vaupel2023Duality,Gardner2022Toroidal`。

概览
========

分析器模块按功能组织，结构如下：

- **data/** - 实验数据分析与 ASA pipeline

  - ``asa/`` - embed/TDA/解码/CohoMap/CohoSpace/PathCompare/FR/FRM/Spatial TDA
  - ``asa/fly_roi.py`` - ROI 1D bump 拟合与动画（Fly ROI）
  - ``legacy/`` - 早期 CANN2D 数据分析器（兼容旧工作流）
  - ``workflows/`` - 由基础 ASA 模块拼接得到的组合工作流

- **metrics/** - 计算分析（无 matplotlib 依赖）

  - ``spatial_metrics`` - 空间指标计算
  - ``systematic_ratemap`` - 系统化放电场统计
  - ``utils`` - Spike train 转换工具

- **visualization/** - 绘图和动画（基于 matplotlib）

  - ``core`` - PlotConfig 统一配置系统
  - ``spatial_plots`` - 空间可视化
  - ``energy_plots`` - 能量景观可视化
  - ``spike_plots`` - Raster 图和发放率图
  - ``tuning_plots`` - 调谐曲线可视化
  - ``theta_sweep_plots`` - Theta 扫描可视化

- **slow_points/** - 不动点分析（Fixed Points）
- **model_specific/** - 专用模型分析器（如 Hopfield）

数据分析器（ASA pipeline）
===========================

ASA pipeline 面向实验神经记录，核心输入为 ``spike/x/y/t``。典型流程：

1. ``embed_spike_trains`` —— 将 spike times 嵌入为 ``(T, N)`` 连续矩阵
2. ``tda_vis`` —— 持续同调 + barcode
3. ``decode_circular_coordinates_multi`` —— 解码相位轨迹
4. ``cohomap`` / ``plot_cohomap`` —— 将解码相位按真实 ``x/y`` 空间分箱，生成 EcohoMap
5. ``cohospace`` / ``plot_cohospace`` / ``compute_cohoscore_1d`` / ``compute_cohoscore_2d`` —— 神经元相位选择性与 EcohoSpace
6. ``plot_path_compare_1d`` / ``plot_path_compare_2d`` —— 真实轨迹 vs 解码轨迹
7. ``compute_fr_heatmap_matrix`` / ``compute_frm`` —— FR Heatmap / FRM

.. note::
   ASA 的交互式 TUI 由 ``canns.pipeline.asa`` 提供，复用同一套 Python API。

输入格式
----------------

- ``spike`` 支持 dict / list-of-arrays / ``(T, N)`` 矩阵
- ``x/y/t`` 用于轨迹对齐与速度过滤
- ``embed_spike_trains`` 返回 ``(spikes_bin, xx, yy, tt)``，当 ``speed_filter=True`` 时
  会对齐 ``x/y/t`` 并进行速度过滤

核心函数
----------------

.. list-table::
   :widths: 35 65

   * - **预处理**
     - ``SpikeEmbeddingConfig`` / ``embed_spike_trains``
   * - **TDA**
     - ``TDAConfig`` / ``tda_vis``
   * - **解码**
     - ``decode_circular_coordinates`` / ``decode_circular_coordinates_multi``
   * - **相位与路径**
     - ``cohomap`` / ``plot_cohomap`` / ``plot_cohomap_scatter_multi`` /
       ``plot_path_compare_1d`` / ``plot_path_compare_2d``
   * - **流形与投影**
     - ``plot_projection`` / ``plot_2d_bump_on_manifold`` / ``plot_3d_bump_on_torus`` /
       ``CANN2DPlotConfig``
   * - **CohoSpace**
     - ``cohospace`` / ``plot_cohospace`` /
       ``plot_cohospace_trajectory_1d`` / ``plot_cohospace_trajectory_2d`` /
       ``plot_cohospace_neuron_1d`` / ``plot_cohospace_neuron_2d`` /
       ``plot_cohospace_population_1d`` / ``plot_cohospace_population_2d`` /
       ``compute_cohoscore_1d`` / ``compute_cohoscore_2d``
   * - **FR/FRM**
     - ``compute_fr_heatmap_matrix`` / ``save_fr_heatmap_png`` / ``compute_frm`` / ``plot_frm``
   * - **路径对齐**
     - ``align_coords_to_position_1d`` / ``align_coords_to_position_2d`` / ``apply_angle_scale``
   * - **Spatial TDA**
     - ``spatial_embedding.py`` / ``spatial_tda.py``
   * - **组合工作流**
     - ``workflows.auto_grid_threshold`` / ``workflows.phase_center_comparison``
   * - **ROI 1D bump**
     - ``roi_bump_fits`` / ``create_1d_bump_animation`` / ``CANN1DPlotConfig``

Time TDA 与 Spatial TDA
-----------------------

ASA 中现在区分两类点云构造方式：

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 方法
     - 输入点云
     - 主要用途
   * - **Time TDA**
     - 群体活动的时间采样点 ``r(t)``
     - 动力学轨迹、解码、CohoMap/CohoSpace 与 PathCompare。
   * - **Spatial TDA**
     - 按空间位置分箱后的 firing-rate 向量 ``r(x)``
     - 空间表征流形与 Fig.4C-style barcode 分析。

Spatial TDA 由 ``canns.analyzer.data.asa`` 中的 ``spatial_embedding.py`` 与
``spatial_tda.py`` 实现。示例脚本
``examples/experimental_data_analysis/spatial_tda_from_asa.py`` 提供 ASA ``.npz``
文件的命令行入口。

.. figure:: /_static/asa_spatial_tda_barcode_example.png
   :width: 70%
   :align: center

   Spatial TDA 示例输出。该流程先把时间序列活动重写为空间位置上的
   firing-rate 向量 ``r(x)``，再对空间 bin 构成的点云计算 persistent homology。

组合工作流
----------

``canns.analyzer.workflows`` 存放由基础 ASA 模块拼接得到的高层工作流：

- ``auto_grid_threshold.py``：按 grid score 排序 sweep 神经元子集并报告候选阈值。
- ``phase_center_comparison.py``：比较不同 session / condition 的 phase center，并使用 torus-aware displacement。

这些 workflow 不引入新的底层分析定义，而是把已有 ASA 模块按常用科研问题进行组合：

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Workflow
     - 解决的问题
     - 典型输出
   * - ``auto_grid_threshold.py``
     - 自动尝试不同 grid-score cutoff / top-k 神经元集合，帮助选择更适合 TDA 或 coho 分析的 grid-cell 子集。
     - 阈值 sweep 表、best subset、barcode / coho 质量指标。
   * - ``phase_center_comparison.py``
     - 对齐两个 session 或 condition 中的同一批神经元，比较它们在 coho torus 上的 phase center 位移。
     - 黑红点 paired phase-center 图、torus-aware displacement summary。

.. figure:: /_static/asa_auto_grid_threshold_example.png
   :width: 80%
   :align: center

   自动 grid-score 阈值筛选示例。横向 sweep 不同子集大小或阈值，用于快速检查
   哪一组神经元更容易得到稳定的拓扑与 coho 结果。

.. figure:: /_static/asa_phase_center_comparison_example.png
   :width: 70%
   :align: center

   Phase-center comparison 示例。黑点与红点表示两个 session / condition 中同一神经元的
   phase center，连线使用 torus 周期边界下的最短位移。

可视化与模型输出分析
===========================

模型仿真结果的主要可视化集中在 ``canns.analyzer.visualization``：

.. list-table::
   :widths: 30 70

   * - **能量景观**
     - ``energy_landscape_1d_static`` / ``energy_landscape_1d_animation`` /
       ``energy_landscape_2d_static`` / ``energy_landscape_2d_animation``
   * - **脉冲图**
     - ``raster_plot`` / ``population_activity_heatmap`` / ``average_firing_rate_plot``
   * - **空间可视化**
     - ``plot_firing_field_heatmap`` / ``plot_autocorrelation`` /
       ``plot_grid_score`` / ``plot_grid_spacing_analysis`` /
       ``create_grid_cell_tracking_animation``
   * - **调谐曲线**
     - ``tuning_curve``
   * - **Theta 扫描**
     - ``create_theta_sweep_grid_cell_animation`` /
       ``create_theta_sweep_place_cell_animation`` /
       ``plot_grid_cell_manifold`` / ``plot_population_activity_with_theta``

PlotConfig 系统
----------------

``PlotConfig`` 与 ``PlotConfigs`` 提供统一的绘图配置入口：

- 共享 ``figsize``、``title``、``xlabel`` 等配置
- 支持动画参数（``fps``、``interval``）与输出路径（``save_path``）
- 兼容旧的关键字参数方式

空间指标与统计（metrics）
===========================

``canns.analyzer.metrics`` 提供无绘图库依赖的计算函数：

- ``spatial_metrics``

  - ``compute_firing_field``：根据 ``(T, N)`` 与 ``(T, 2)`` 计算空间放电场
  - ``gaussian_smooth_heatmaps``：热图平滑
  - ``compute_spatial_autocorrelation``：空间自相关
  - ``compute_grid_score`` / ``find_grid_spacing``：网格细胞指标

- ``systematic_ratemap``

  - ``compute_systematic_ratemap``：系统化空间采样（Burak & Fiete 风格）

RNN 动力学分析（slow_points）
================================

``canns.analyzer.slow_points`` 提供不动点分析工具：

- ``FixedPointFinder``：在 RNN/CANN 动力系统中寻找不动点
- ``FixedPoints``：存储与分析不动点集合
- ``plot_fixed_points_2d`` / ``plot_fixed_points_3d``：可视化
- ``save_checkpoint`` / ``load_checkpoint``：保存/恢复搜索状态

模型专用分析
================

当前提供 Hopfield 相关分析：

- ``HopfieldAnalyzer``
   - ``compute_overlap`` / ``compute_energy``
   - ``analyze_recall`` （输入/输出一致性）
   - ``estimate_capacity`` / ``get_statistics``

拓扑数据分析（TDA）
=======================

TDA 在 ASA pipeline 中实现，依赖 ``canns.analyzer.data.asa.tda``：

- ``TDAConfig`` 定义降维、采样与持久同调参数
- ``tda_vis`` 负责计算与可视化 barcode
- 内部使用 ``canns_lib.ripser`` 加速持续同调计算

TDA 适用于检测 **环/环面** 等低维拓扑结构，可用于评估实验记录与模型预测的一致性。

总结
=======

``canns.analyzer`` 提供从 **实验数据分析** 到 **模型可视化** 再到 **动力学与专用模型诊断** 的全链路工具。
推荐工作流：

- 实验记录 → ASA pipeline（``canns.analyzer.data``）
- 仿真输出 → visualization + metrics
- 动力学研究 → slow_points
- 专用模型 → model_specific
