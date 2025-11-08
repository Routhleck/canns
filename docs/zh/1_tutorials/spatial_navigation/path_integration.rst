路径积分和位置编码
==================

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。



场景描述
--------

你想要理解动物如何通过整合运动信息（速度和方向）来维持对自身位置的连续表示。这是空间导航的基础机制，被称为"路径积分"（path integration）或"航位推算"（dead reckoning）。

你将学到
--------

- 路径积分的数学原理
- 使用CANN实现路径积分的机制
- 格点细胞和位置细胞的角色
- 运动输入如何驱动位置表示的更新
- 误差累积和校正机制

完整示例
--------

.. literalinclude:: ../../../../examples/cann/hierarchical_path_integration.py
   :language: python
   :linenos:

逐步解析
--------

1. **创建开环导航任务**

   .. code-block:: python

      from canns.task.open_loop_navigation import OpenLoopNavigationTask

      task = OpenLoopNavigationTask(
          width=5,
          height=5,
          speed_mean=0.04,      # 平均速度
          speed_std=0.016,      # 速度变异性
          duration=50000.0,     # 持续时间（毫秒）
          dt=0.05,              # 时间步长
          start_pos=(2.5, 2.5), # 起始位置
          progress_bar=True
      )

   **说明**：
   - 模拟在5×5环境中的随机游走
   - 速度随时间波动（生物现实性）
   - 返回速度和位置轨迹

2. **初始化分层网络**

   .. code-block:: python

      from canns.models.basic import HierarchicalNetwork

      network = HierarchicalNetwork(
          num_module=5,      # 5个格点模块
          num_place=30,      # 30个位置细胞
      )
      network.init_state()

   **说明**：
   - 包含多个尺度的格点细胞模块
   - 位置细胞通过整合格点输入形成
   - 模拟哺乳动物的真实导航系统

3. **运行路径积分**

   .. code-block:: python

      def run_step(t, velocity, position):
          network(
              velocity=velocity,     # 速度输入
              loc=position,          # 当前位置（用于学习）
              loc_input_stre=0.      # 定位输入强度
          )
          return (
              network.band_x_fr.value,
              network.band_y_fr.value,
              network.grid_fr.value,
              network.place_fr.value
          )

      # 编译运行
      results = brainstate.compile.for_loop(
          run_step,
          time_indices,
          velocity_data,
          position_data
      )

   **说明**：
   - Band细胞编码运动方向（X和Y）
   - Grid细胞形成周期性的格点模式
   - Place细胞编码特定位置

4. **分析位置编码**

   .. code-block:: python

      from canns.analyzer.spatial import compute_firing_field

      # 计算每个细胞的位置激活热图
      grid_heatmaps = compute_firing_field(
          grid_activity,
          animal_trajectory,
          width=5,
          height=5
      )

   **说明**：
   - 热图显示细胞在环境中的活动模式
   - 格点细胞应显示规则的六边形模式
   - 位置细胞应显示特定的"位置场"

运行结果
--------

运行此脚本会生成：

1. **轨迹图** (`trajectory_graph.png`)

   - 显示动物在环境中的运动轨迹
   - 帮助理解输入数据的复杂性

2. **神经活动热图**

   - **格点细胞热图**：规则的六边形模式
   - **Band细胞热图**：方向和运动的表示
   - **位置细胞热图**：单个或多个位置场

3. **性能指标**

   - 仿真时间：~10-30秒
   - 内存使用：~500 MB
   - 生成的文件：~50个热图

关键概念
--------

**路径积分的数学模型**

位置更新遵循简单的积分方程：

.. math::

   \vec{p}(t+\Delta t) = \vec{p}(t) + \vec{v}(t) \cdot \Delta t

其中：
- p(t)：当前位置
- v(t)：速度
- Δt：时间步长

**格点细胞的六边形编码**

格点细胞形成规则的六边形网格：

.. code-block:: text

   俯视图：
   ○───○───○───○
    ╱ ╲ ╱ ╲ ╱ ╲ ╱ ╲
   ○───○───○───○
    ╲ ╱ ╲ ╱ ╲ ╱ ╲ ╱
   ○───○───○───○

   - 每个格点细胞对应六边形中心
   - 多个尺度的模块编码不同分辨率
   - Mouton & Moser的发现（2005），诺奖）

**多模块整合**

分层结构的优势：

.. code-block:: text

   大尺度格点模块 (>1m)
        ↓
   中尺度格点模块
        ↓
   小尺度格点模块 (<10cm)
        ↓
   位置细胞（place cells）

   - 每个模块独立进行路径积分
   - 不同模块的尺度比例为2-3倍
   - 组合产生精确的位置表示

实验变化
--------

**1. 改变环境大小**

.. code-block:: python

   # 大环境（更多位置细胞活动）
   task = OpenLoopNavigationTask(width=10, height=10)

   # 小环境（更简单的表示）
   task = OpenLoopNavigationTask(width=2, height=2)

**2. 分析误差累积**

.. code-block:: python

   # 比较网络预测位置与实际位置
   predicted_positions = decode_from_place_cells(place_activity)
   actual_positions = task.data.position

   position_error = np.linalg.norm(
       predicted_positions - actual_positions,
       axis=1
   )

   print(f"平均位置误差: {np.mean(position_error):.3f}m")
   print(f"最大位置误差: {np.max(position_error):.3f}m")

**3. 改变速度统计**

.. code-block:: python

   # 快速运动（难以追踪）
   task = OpenLoopNavigationTask(
       speed_mean=0.08,
       speed_std=0.032
   )

   # 缓慢运动（容易精确追踪）
   task = OpenLoopNavigationTask(
       speed_mean=0.01,
       speed_std=0.004
   )

**4. 分析格点细胞的格点间距**

.. code-block:: python

   import numpy as np
   from scipy.fft import fft2

   # 计算热图的频谱
   for module_idx, heatmap in enumerate(grid_heatmaps):
       fft_result = np.abs(fft2(heatmap))
       # 从频谱提取格点间距
       spacing = analyze_grid_spacing(fft_result)
       print(f"模块{module_idx}格点间距: {spacing:.2f}cm")

相关API
-------

- :class:`~src.canns.models.basic.HierarchicalNetwork` - 分层导航网络
- :class:`~src.canns.task.open_loop_navigation.OpenLoopNavigationTask` - 开环导航任务
- :func:`~src.canns.analyzer.spatial.compute_firing_field` - 位置激活热图计算

生物学背景
----------

**Entorhinal Cortex (内嗅皮层)**

外部脑室中脑（MEC）包含：

1. **Grid细胞**：在大约60,000个MEC细胞中占5-10%
   - 规则的六边形网格模式
   - 多个模块的不同间距
   - 为路径积分提供度量衡

2. **Head Direction细胞**：编码动物头部方向
   - 调谐曲线是钟形的
   - 全球极化，与环境对齐

**Hippocampus (海马体)**

- **Place细胞**：在特定位置有强响应
- 通过整合格点和方向信息形成
- 支持空间记忆和计划

**生物的路径积分能力**

- 沙漠蚂蚁可以在没有视觉线索下返回巢穴
- 蜜蜂在蜂房中进行摇晃舞蹈时进行路径积分
- 大鼠即使在黑暗中也能建立位置图

更多资源
--------

- :doc:`hierarchical_network` - 理解分层结构
- :doc:`theta_modulation` - Theta节律在导航中的角色
- :doc:`grid_place_cells` - 格点细胞和位置细胞的关系
常见问题
--------

**Q: 为什么使用格点细胞？**

A: 格点细胞提供高效的空间编码：
   - 六边形编码比直角坐标更高效
   - 多模块结构允许任意精度
   - 符合生物观察数据

**Q: 位置误差会无限增长吗？**

A: 是的，在没有修正的开环路径积分中会累积。但大脑有机制来修正：
   - 视觉输入（地标）重置位置
   - 嗅觉线索（odor)
   - 触觉反馈
   - 本体感觉信息

**Q: 如何从细胞活动解码位置？**

A: 可以使用最大似然估计：

   .. code-block:: python

      def decode_position(place_cell_activity):
          # 假设每个位置细胞是高斯调谐的
          likelihood = np.ones(num_positions)
          for cell_idx, activity in enumerate(place_cell_activity):
              likelihood *= likelihood_given_activity(
                  activity,
                  place_field_of_cell[cell_idx]
              )
          return positions[np.argmax(likelihood)]

下一步
------

完成本教程后，推荐：

1. 分析不同格点模块的角色
2. 测试在有视觉线索时的学习
3. 比较不同动物的导航能力
4. 阅读 :doc:`grid_place_cells` 了解细胞间的相互作用
