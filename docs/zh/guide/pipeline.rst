Theta Sweep 流水线示例
==========================

:class:`ThetaSweepPipeline <src.canns.pipeline.theta_sweep.ThetaSweepPipeline>` 将导航任务、方向/网格细胞模型和可视化串联成
端到端流程。本章列出两个等级的示例，展示如何从外部轨迹数据启动分析，以及如何全面定制参数。

theta_sweep_from_external_data.py
---------------------------------

- **路径**：`examples/pipeline/theta_sweep_from_external_data.py <https://github.com/Routhleck/canns/blob/master/examples/pipeline/theta_sweep_from_external_data.py>`_
- **场景**：使用 Catmull–Rom 样条生成的外部轨迹，直接调用流水线完成 theta sweep 分析。
- **流程概述**：

  1. 生成平滑闭合轨迹（或替换为实验轨迹），构建 ``times`` 与 ``positions``。
  2. 初始化 ThetaSweepPipeline（使用默认网络与 theta 参数）。
  3. 调用 ``pipeline.run(output_dir="theta_sweep_results")``，自动导出动画与关键图像。
- **输出**：

  - ``theta_sweep_results/`` 目录，包含动画 GIF/MP4、人口活动热图、轨迹分析图。
  - 终端打印分析摘要（时间长度、保存位置）。
- **延伸**：

  - 结合 :doc:`tasks` 的 ``import_external_trajectory.py``，以真实数据替换样例轨迹。
  - 通过 ThetaSweepPipeline(..., env_size=?, dt=?) 与 run(..., animation_fps=?) 控制分辨率与图像质量。

advanced_theta_sweep_pipeline.py
--------------------------------

- **路径**：`examples/pipeline/advanced_theta_sweep_pipeline.py <https://github.com/Routhleck/canns/blob/master/examples/pipeline/advanced_theta_sweep_pipeline.py>`_
- **场景**：面向高级用户，展示如何覆写流水线的所有配置，包括网络尺寸、theta 周期、输出质量等。
- **流程概述**：

  1. 构造包含 L 形轨迹的样条路径，并添加有控制的扰动。
  2. 在构造函数中同时传入 ``direction_cell_params``、``grid_cell_params``、``theta_params``
     与 ``spatial_nav_params``，实现全量定制。
  3. 调用 ``run(..., save_animation=True, save_plots=True, verbose=True)`` 获取完整结果，并查看返回的 ``results`` 字典。
  4. 脚本还示范如何从 ``results["data"]`` 中提取 ``gc_activity``、``theta_phase`` 等数组做二次分析。
- **输出**：``advanced_theta_sweep_results/`` 内的动画、分析图、模拟数据缓存。
- **延伸**：

  - 改变 ``theta_strength_hd/gc`` 或 ``theta_cycle_len``，对比不同节律设置的影响。
  - 将 ``grid_activity``、``dc_activity`` 导出为 ``.npz``，与实验数据直接比较。

使用建议
--------

- 两个示例都依赖 :mod:`canns.task.open_loop_navigation <src.canns.task.open_loop_navigation>` 与 :mod:`canns.analyzer.theta_sweep <src.canns.analyzer.theta_sweep>`。
  若需进一步了解其内部实现，可回顾 :doc:`tasks` / :doc:`models` 章节。
- 渲染动画时可能需要数分钟，请关注终端进度条；若在无显示环境运行，可保持 ``show=False``。
