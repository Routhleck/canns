格点细胞和位置细胞的相互作用
===========================

场景描述
--------

你想要理解格点细胞（Grid Cells）和位置细胞（Place Cells）是如何协作的，以及它们如何形成了大脑中的空间表示。这两种细胞类型代表了不同层次的空间编码。

你将学到
--------

- 格点细胞和位置细胞的特性对比
- 它们的神经解剖学位置
- 相互连接和信息流
- 如何从格点输入产生位置场
- 多尺度表示的整合

完整示例
--------

基于分层网络模型的格点-位置细胞分析：

.. code-block:: python

   import numpy as np
   from canns.models.basic import HierarchicalNetwork
   from canns.task.open_loop_navigation import OpenLoopNavigationTask
   from canns.analyzer.spatial import compute_firing_field

   # 创建导航环境和网络
   task = OpenLoopNavigationTask(width=5, height=5, duration=50000)
   task.get_data()

   network = HierarchicalNetwork(num_module=5, num_place=30)
   network.init_state()

   # 运行网络
   def run_step(t, velocity, position):
       network(velocity=velocity, loc=position, loc_input_stre=0)
       return (network.grid_fr.value, network.place_fr.value)

   grid_activity, place_activity = brainstate.compile.for_loop(
       run_step,
       time_indices,
       task.data.velocity,
       task.data.position
   )

   # 分析单个细胞的位置场
   for grid_cell_idx in range(grid_activity.shape[1]):
       grid_heatmap = compute_firing_field(
           grid_activity[:, grid_cell_idx].reshape(-1, 1),
           task.data.position
       )
       # 应该显示规则的六边形模式

   for place_cell_idx in range(place_activity.shape[1]):
       place_heatmap = compute_firing_field(
           place_activity[:, place_cell_idx].reshape(-1, 1),
           task.data.position
       )
       # 应该显示单个高斯位置场

逐步解析
--------

1. **格点细胞的特性**

   .. code-block:: python

      # 格点细胞的特征
      格点细胞特性 = {
          '位置': '内嗅皮层MEC',
          '放电模式': '规则六边形格点',
          '调谐宽度': '30-60 cm',
          '位置场数': '多个周期性位置场',
          '功能': '提供度量衡'
      }

   **格点模式**：

   .. code-block:: text

      一个格点细胞的激活热图：

         █ █ █ █ █
        █ █ █ █ █ █
       █ █ █ █ █ █ █
        █ █ █ █ █ █
         █ █ █ █ █

      - 多个分散的位置场
      - 规则的六边形对称性
      - 占据整个环境

2. **位置细胞的特性**

   .. code-block:: python

      # 位置细胞的特征
      位置细胞特性 = {
          '位置': '海马体CA1',
          '放电模式': '单个位置场',
          '调谐宽度': '20-40 cm',
          '位置场数': '通常1个（可多个）',
          '功能': '编码动物位置'
      }

   **位置场形状**：

   .. code-block:: text

      一个位置细胞的激活热图：

         ░░░░░░░░░░
         ░░░░░░░░░░
         ░░░████░░░
         ░░░████░░░
         ░░░░░░░░░░

      - 单一的、局限的活动区域
      - 高斯形的轮廓
      - 对应环境的特定位置

3. **格点-位置细胞的整合机制**

   .. code-block:: python

      # 位置细胞如何从格点细胞形成？
      # 通过线性组合和非线性整合

      position_cell_activity = 0
      for grid_cell_idx in range(num_grid_cells):
          weight = synaptic_strength[grid_cell_idx]
          position_cell_activity += weight * grid_cell_activity[grid_cell_idx]

      # 非线性：只有输入超过阈值才激活
      position_cell_output = relu(position_cell_activity - threshold)

   **关键原理**：
   - 位置细胞是多个格点细胞的线性组合
   - 组合系数通过学习确定
   - 实现从多格点到单位置的映射

4. **"汇聚"与"散射"**

   .. code-block:: text

      格点细胞 → 位置细胞：汇聚
      ┌─────────────────────┐
      │  位置细胞（单一位置场）│
      └─────────────────────┘
             ▲  ▲  ▲  ▲  ▲
             │  │  │  │  │
      ┌──────┴──┴──┴──┴──┴──┐
      │ 格点细胞 1 2 3 4 5... │
      └───────────────────────┘

      位置细胞 → 格点细胞：散射
      ┌─────────────────────┐
      │ 位置细胞（单一位置）  │
      └─────────────────────┘
             ▼  ▼  ▼  ▼  ▼
             │  │  │  │  │
      ┌──────┴──┴──┴──┴──┴──┐
      │ 格点细胞 1 2 3 4 5... │
      └───────────────────────┘

关键概念
--------

**中国剩余定理在脑中的实现**

多格点模块组合产生唯一位置：

.. code-block:: python

   # 5个模块，每个有不同的周期
   periods = [6, 10, 15, 25, 40]  # 厘米

   # 位置300cm：
   位置 % 6 = 0    → 格点模块1的相位A
   位置 % 10 = 0   → 格点模块2的相位B
   位置 % 15 = 0   → 格点模块3的相位C
   位置 % 25 = 0   → 格点模块4的相位D
   位置 % 40 = 20  → 格点模块5的相位E

   组合(A,B,C,D,E) → 唯一位置

**符号距离（Metric Distance）**

格点细胞编码的是距离，不仅是位置：

.. code-block:: text

   两个位置之间的距离可从格点活动推断：
   - 相同相位的格点细胞 → 距离是周期的倍数
   - 位置A和位置B的格点模式相似 → 距离较近
   - 位置A和位置C的格点模式差异大 → 距离较远

实验变化
--------

**1. 分析格点-位置的编码权重**

.. code-block:: python

   # 计算位置细胞对每个格点细胞的依赖
   from sklearn.linear_model import LinearRegression

   X = grid_activity  # 格点细胞活动
   y = place_activity  # 位置细胞活动

   model = LinearRegression().fit(X, y)
   weights = model.coef_  # [num_place_cells, num_grid_cells]

   # 绘制权重矩阵
   plt.imshow(weights, aspect='auto', cmap='RdBu_r')
   plt.xlabel('格点细胞')
   plt.ylabel('位置细胞')
   plt.title('格点到位置的连接权重')

**2. 测试解码精度**

.. code-block:: python

   # 从格点活动解码位置（不使用位置细胞）
   grid_decoder = LinearRegression().fit(grid_activity, position)
   grid_predicted_pos = grid_decoder.predict(grid_activity_test)

   # 从位置细胞活动解码位置
   place_decoder = LinearRegression().fit(place_activity, position)
   place_predicted_pos = place_decoder.predict(place_activity_test)

   # 比较精度
   grid_error = np.mean(np.linalg.norm(grid_predicted_pos - true_position, axis=1))
   place_error = np.mean(np.linalg.norm(place_predicted_pos - true_position, axis=1))

**3. 破坏一个格点模块观察影响**

.. code-block:: python

   # 模拟一个格点模块的损伤
   damaged_grid_activity = grid_activity.copy()
   damaged_grid_activity[:, :, 0] = 0  # 破坏模块0

   # 重新训练位置细胞解码器
   damaged_decoder = LinearRegression().fit(
       damaged_grid_activity, position
   )
   damaged_error = np.mean(...)

   # 与完整系统比较
   print(f"完整系统误差: {normal_error:.2f} cm")
   print(f"损伤后误差: {damaged_error:.2f} cm")
   print(f"误差增加: {(damaged_error/normal_error - 1)*100:.1f}%")

相关API
-------

- :class:`~src.canns.models.basic.HierarchicalNetwork` - 完整的格点-位置网络
- :func:`~src.canns.analyzer.spatial.compute_firing_field` - 激活热图
- :class:`~src.canns.models.basic.CANN2D` - 单个格点模块

生物学应用
----------

**导航的多层次表示**

1. **格点细胞**（MEC）：
   - 提供稳定的、不变的坐标系
   - 在不同环境中保持相同的间距
   - 支持路径积分

2. **位置细胞**（海马体）：
   - 环境特异性
   - 支持情境记忆
   - 参与认知地图的形成

3. **头方向细胞**（MEC）：
   - 编码朝向
   - 全局极化
   - 与位置和格点细胞协同工作

**临床意义**

- 老年性痴呆和格点细胞异常
- 创伤性脑损伤和空间定向障碍
- 航海能力丧失可能预示神经退行性疾病

更多资源
--------

- :doc:`path_integration` - 理解路径积分
- :doc:`hierarchical_network` - 分层结构
常见问题
--------

**Q: 为什么位置细胞只有一个位置场，而格点细胞有多个？**

A: 因为位置细胞是多个格点模块的非线性组合。通过选择适当的输入权重和阈值，多个周期性的格点模式可以组合出单一的、局限的位置场。

**Q: 大脑中的格点间距是固定的吗？**

A: 基本上是固定的，但在学习过程中可能会调整。更重要的是，不同动物物种有不同的基准间距（与体型相关）。

**Q: 位置细胞的学习如何发生？**

A: 通过监督学习或强化学习。外部信号（视觉地标、奖励）强化某些格点-位置的关联，逐渐塑造位置细胞的调谐曲线。

下一步
------

1. 分析在新环境中位置场的快速形成
2. 研究格点和位置之间的信息流
3. 比较不同物种的编码策略
4. 阅读 :doc:`complex_environments` 了解更复杂的场景
