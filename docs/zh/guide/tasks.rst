任务与导航示例
==============

``examples/cann/`` 与 ``examples/pipeline/`` 中有若干脚本展示如何使用
:mod:`canns.task <src.canns.task>` 生成轨迹、导入外部数据，并驱动更复杂的网络结构。本章选取两类典型示例。

import_external_trajectory.py
-----------------------------

- **路径**：`examples/cann/import_external_trajectory.py <https://github.com/Routhleck/canns/blob/master/examples/cann/import_external_trajectory.py>`_
- **目标**：演示如何替换 ``SpatialNavigationTask`` 默认的随机轨迹，改为加载外部位置数据。
- **关键流程**：

  1. 手动生成带噪声的随机行走轨迹（可替换为从文件读取）。
  2. 初始化 :class:`SpatialNavigationTask <src.canns.task.spatial_navigation.SpatialNavigationTask>` 并调用
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
- **目标**：演示层次化路径积分网络如何与 ``SpatialNavigationTask`` 协同运作。
- **关键流程**：

  1. 创建长时间 (``duration=1000``) 的空间导航任务，并保存轨迹到 ``trajectory_test.npz``。
  2. 构建 :class:`HierarchicalNetwork <src.canns.models.basic.hierarchical_model.HierarchicalNetwork>`，包含带状细胞、网格细胞、位置细胞模块。
  3. 通过 ``brainstate.compile.for_loop`` 先执行初始化阶段（``loc_input_stre`` 充当校准），再运行完整轨迹。
  4. 使用 :func:`benchmark() <src.canns.misc.benchmark.benchmark>` 比较编译循环性能。
- **输出**：生成 trajectory_graph.png、band_grid_place_activity.npz（可选）。
- **延伸**：

  - 将脚本与 :doc:`models` 的 CANN 示例结合，分析不同连接参数对路径积分的影响。
  - 使用 ``SpatialNavigationTask.import_data`` 替换随机轨迹，再运行层次网络以匹配实验数据。

提示
----

- ``SpatialNavigationTask`` 默认依赖 ``Ratinabox``，在首次运行时会生成内置环境；可通过传入
  ``walls``、``objects`` 等参数自定义布局。
- 若需要批量生成轨迹，可在脚本外部循环调用 ``task.get_data()`` 并保存，以便流水线示例直接消费。
