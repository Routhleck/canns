任务与导航示例
==============

``examples/cann/`` 与 ``examples/pipeline/`` 中有若干脚本展示如何使用
:mod:`canns.task <src.canns.task>` 生成轨迹、导入外部数据，并驱动更复杂的网络结构。本章选取两类典型示例。

import_external_trajectory.py
-----------------------------

- **路径**：`examples/cann/import_external_trajectory.py <https://github.com/Routhleck/canns/blob/master/examples/cann/import_external_trajectory.py>`_
- **目标**：演示如何替换 ``OpenLoopNavigationTask`` 默认的随机轨迹，改为加载外部位置数据。
- **关键流程**：

  1. 手动生成带噪声的随机行走轨迹（可替换为从文件读取）。
  2. 初始化 :class:`OpenLoopNavigationTask <src.canns.task.open_loop_navigation.OpenLoopNavigationTask>` 并调用
     ``import_data(position_data=..., times=...)``。
  3. 执行 ``calculate_theta_sweep_data()`` 以获得线速度/角速度增益，为后续 theta sweep 做准备。
  4. 调用 ``show_trajectory_analysis`` 输出 PNG，同时自绘 Matplotlib 图对比导入数据。
- **输出**：``import_external_trajectory.png``、``our_data_comparison.png`` 及一系列统计信息。
- **延伸**：

  - 将 ``positions`` 替换为实验数据；若包含朝向，可在 ``import_data`` 中传入 ``head_direction``。
  - 保存 ``snt.save_data(...)``，让后续示例直接复用相同轨迹。

hierarchical_path_integration.py
--------------------------------

- **路径**：`examples/cann/hierarchical_path_integration.py <https://github.com/Routhleck/canns/blob/master/examples/cann/hierarchical_path_integration.py>`_
- **目标**：演示层次化路径积分网络如何与 ``OpenLoopNavigationTask`` 协同运作。
- **关键流程**：

  1. 创建长时间 (``duration=1000``) 的空间导航任务，并保存轨迹到 ``trajectory_test.npz``。
  2. 构建 :class:`HierarchicalNetwork <src.canns.models.basic.hierarchical_model.HierarchicalNetwork>`，包含带状细胞、网格细胞、位置细胞模块。
  3. 通过 ``brainstate.compile.for_loop`` 先执行初始化阶段（``loc_input_stre`` 充当校准），再运行完整轨迹。
  4. 使用 :func:`benchmark() <src.canns.misc.benchmark.benchmark>` 比较编译循环性能。
- **输出**：生成 trajectory_graph.png、band_grid_place_activity.npz（可选）。
- **延伸**：

  - 将脚本与 :doc:`models` 的 CANN 示例结合，分析不同连接参数对路径积分的影响。
  - 使用 ``OpenLoopNavigationTask.import_data`` 替换随机轨迹，再运行层次网络以匹配实验数据。

提示
----

- ``OpenLoopNavigationTask`` 默认依赖 ``Ratinabox``，在首次运行时会生成内置环境；可通过传入
  ``walls``、``objects`` 等参数自定义布局。
- 若需要批量生成轨迹，可在脚本外部循环调用 ``task.get_data()`` 并保存，以便流水线示例直接消费。


闭环导航工具
--------------

- **路径**：``src/canns/task/closed_loop_navigation.py``
- **目标**：在 ``Ratinabox`` 闭环代理基础上，提供环境感知的移动规划工具，包括移动代价网格和测地线
  可视化。
- **关键流程**：

  1. 初始化 :class:`ClosedLoopNavigationTask <src.canns.task.closed_loop_navigation.ClosedLoopNavigationTask>`
     或便捷的 :class:`TMazeClosedLoopNavigationTask <src.canns.task.closed_loop_navigation.TMazeClosedLoopNavigationTask>`。
  2. 调用 :meth:`build_movement_cost_grid()
     <src.canns.task.closed_loop_navigation.ClosedLoopNavigationTask.build_movement_cost_grid>` 并指定 ``dx``、``dy``
     将环境离散化，阻塞单元格会被标记为 ``INT32_MAX`` 权重。
  3. 通过 ``show_data(overlay_movement_cost=True, cost_grid=...)`` 叠加网格观察墙体/洞口，或使用
     :meth:`show_geodesic_distance_matrix()
     <src.canns.task.closed_loop_navigation.ClosedLoopNavigationTask.show_geodesic_distance_matrix>` 计算可通行单元格之间的最短路径矩阵。
- **输出**：带有阻塞/可通行着色以及单元格权重文本的 matplotlib 图像，另有密集测地线距离矩阵可供后续规划使用。
- **延伸**：

  - 将返回的 :class:`MovementCostGrid
    <src.canns.task.closed_loop_navigation.MovementCostGrid>` 传递给自定义规划器或导出为调试文件。
  - 可参考 ``tests/task/closed_loop_navigation`` 中的 pytest 用例，按需拓展迷宫结构的回归测试。
  - 运行 ``uv run python examples/cann/closed_loop_complex_environment.py`` 快速生成包含墙体与洞口的
    可视化示例，结果会保存到 ``figures/closed_loop_complex``。
