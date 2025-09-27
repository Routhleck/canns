CANN 网络示例
=============

本章聚焦 ``examples/cann/`` 内直接驱动 CANN 网络的脚本，涵盖 1D/2D bump 追踪、调谐曲线、
以及 theta 调制驱动的可视化。每个示例都明确指出使用到的模型、任务和分析函数。

cann1d_oscillatory_tracking.py
------------------------------

- **路径**：`examples/cann/cann1d_oscillatory_tracking.py <https://github.com/Routhleck/canns/blob/master/examples/cann/cann1d_oscillatory_tracking.py>`_
- **目的**：演示 1D CANN 在 ``SmoothTracking1D`` 输入下的平滑跟踪，并输出能量景观动图。
- **核心 API**：

  - :class:`CANN1D <src.canns.models.basic.CANN1D>`
  - :class:`SmoothTracking1D <src.canns.task.tracking.SmoothTracking1D>`
  - :func:`energy_landscape_1d_animation() <src.canns.analyzer.plotting.energy.energy_landscape_1d_animation>`
- **运行流程**：
  1. 设置 ``brainstate.environ.set(dt=0.1)`` 并初始化 512 节点的 1D CANN。
  2. 构造目标位置序列 (``Iext``) 与时间 (``duration``)，调用 ``task.get_data()`` 生成刺激。
  3. 使用 brainstate.compile.for_loop 执行网络更新，收集网络状态与输入轨迹。
  4. 调用 :func:`canns.analyzer.plotting.energy_landscape_1d_animation` 生成 GIF（可配合 ``PlotConfigs.energy_landscape_1d_animation`` 预设参数）。
- **输出**：生成 test_smooth_tracking_1d.gif（或自定义路径）。
- **延伸**：

  - 可替换为 :class:`CANN1D_SFA <src.canns.models.basic.CANN1D_SFA>` 对比带自适应电流的响应。
  - 修改 ``Iext`` 为更复杂的轨迹，或搭配 ``tuning_curve`` 检测稳态激活。

cann1d_tuning_curve.py
----------------------

- **路径**：`examples/cann/cann1d_tuning_curve.py <https://github.com/Routhleck/canns/blob/master/examples/cann/cann1d_tuning_curve.py>`_
- **目的**：在 1D CANN 中采集选定神经元的平均放电率，绘制调谐曲线。
- **核心 API**：
  - :class:`CANN1D <src.canns.models.basic.CANN1D>`
  - :class:`SmoothTracking1D <src.canns.task.tracking.SmoothTracking1D>`
  - :func:`tuning_curve() <src.canns.analyzer.plotting.tuning.tuning_curve>`
- **运行流程**：
  1. 构建覆盖 [-π, π] 的 1D CANN 和追踪任务（包含多个位置切换）。
  2. 编译运行网络，记录发放率与外部输入。
  3. 利用 PlotConfigs.tuning_curve 指定神经元索引、直方图 bins 与绘图样式。
  4. 调用 :func:`canns.analyzer.plotting.tuning_curve` 绘制曲线（默认直接展示）。
- **输出**：交互图或保存的 PNG（可通过 PlotConfigs.tuning_curve 配置 ``save_path``）。
- **延伸**：

  - 将 ``neuron_indices_to_plot`` 扩展为更多索引，或传入 ``pref_stim=cann.x`` 对齐首选值。
  - 与 ``HebbianTrainer.predict()`` 结合，验证学习前后的调谐变化。

cann2d_tracking.py
------------------

- **路径**：`examples/cann/cann2d_tracking.py <https://github.com/Routhleck/canns/blob/master/examples/cann/cann2d_tracking.py>`_
- **目的**：演示 2D CANN 在多目标输入下的活动迁移，并生成 2D 能量景观动图。
- **核心 API**：

  - :class:`CANN2D <src.canns.models.basic.CANN2D>`
  - :class:`SmoothTracking2D <src.canns.task.tracking.SmoothTracking2D>`
  - :func:`energy_landscape_2d_animation() <src.canns.analyzer.plotting.energy.energy_landscape_2d_animation>`
- **运行流程**：
  1. 创建 ``length=100`` 的 2D CANN 并初始化状态。
  2. 指定一组二维 ``Iext`` 位置和 ``duration``，以 ``SmoothTracking2D`` 产生输入轨迹。
  3. 调用 brainstate.compile.for_loop 更新网络，记录 u/r/inp。
  4. 采用 PlotConfigs.energy_landscape_2d_animation 渲染活动热图动画。
- **输出**：生成 CANN2D_encoding.gif。
- **延伸**：

  - 调整 ``length`` 控制分辨率；若担心性能，可减少 ``time_steps_per_second``。
  - 搭配 ``SmoothTracking1D`` 运行对比实验，观察维度扩展的影响。

theta_sweep_grid_cell_network.py
--------------------------------

- **路径**：`examples/cann/theta_sweep_grid_cell_network.py <https://github.com/Routhleck/canns/blob/master/examples/cann/theta_sweep_grid_cell_network.py>`_
- **目的**：使用方向细胞与网格细胞模型运行短时 theta sweep，并生成多种图像。
- **核心 API**：

  - :class:`DirectionCellNetwork <src.canns.models.basic.theta_sweep_model.DirectionCellNetwork>`
  - :class:`GridCellNetwork <src.canns.models.basic.theta_sweep_model.GridCellNetwork>`
  - :mod:`canns.analyzer.theta_sweep <src.canns.analyzer.theta_sweep>`
- **运行流程**：
  1. 建立 ``SpatialNavigationTask``，生成场地内的短程轨迹及 theta 辅助数据。
  2. 初始化方向细胞/网格细胞网络，循环调用 ``calculate_theta_modulation`` 更新活动。
  3. 调用 :func:`plot_population_activity_with_theta() <src.canns.analyzer.theta_sweep.plot_population_activity_with_theta>`、:func:`plot_grid_cell_manifold() <src.canns.analyzer.theta_sweep.plot_grid_cell_manifold>` 与 :func:`create_theta_sweep_animation() <src.canns.analyzer.theta_sweep.create_theta_sweep_animation>` 输出图像与动图。
- **输出**：生成 theta_sweep_animation.gif 及若干 PNG。
- **延伸**：

  - 调整 ``mapping_ratio``、``theta_strength_*``，对比不同调制深度。
  - 将生成的数据输入 :class:`ThetaSweepPipeline <src.canns.pipeline.theta_sweep.ThetaSweepPipeline>` 复用可视化流程。

更多脚本
--------

- :doc:`tasks` 中的 ``hierarchical_path_integration.py``、``import_external_trajectory.py``
  展示了与 ``SpatialNavigationTask`` 联动的场景，适用于空间导航或外部数据导入。
- 若需要纯视觉化对照，可直接查看仓库根目录下生成的 GIF/PNG（文件名以 ``test_`` 开头）。
