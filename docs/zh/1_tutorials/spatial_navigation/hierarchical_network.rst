分层导航网络架构
================

场景描述
--------

你想要理解如何将多个尺度的神经编码（格点细胞、Band细胞、位置细胞）组织成统一的导航系统。分层网络结构是大脑内嗅皮层和海马体的真实组织原理。

你将学到
--------

- 分层网络的架构设计
- 不同神经元类型的功能角色
- 尺度之间的整合机制
- 网络初始化和状态管理
- 多模块编码的优势

完整示例
--------

.. literalinclude:: ../../../../examples/cann/hierarchical_path_integration.py
   :language: python
   :linenos:

逐步解析
--------

1. **分层网络的组件**

   .. code-block:: python

      from canns.models.basic import HierarchicalNetwork

      network = HierarchicalNetwork(
          num_module=5,           # 5个格点模块
          num_place=30,          # 30个位置细胞
      )
      network.init_state()

   **网络结构**：

   .. code-block:: text

      运动输入（速度、位置）
              │
              ↓
         Band细胞层
        (X方向和Y方向)
              │
              ↓
       ┌──────┴──────┬────────┬────────┐
       ↓             ↓        ↓        ↓
     Module 1    Module 2  Module 3  Module 5
    (粗尺度)     (中尺度)  (细尺度)  (超细)
     格点细胞     格点细胞   格点细胞   格点细胞
       │            │        │        │
       └────────────┴────────┴────────┘
              │
              ↓
         位置细胞层
              │
              ↓
          位置表示

2. **Band细胞层（运动编码）**

   .. code-block:: python

      # Band细胞编码X和Y方向的运动
      band_x_response = network.band_x_fr.value  # X方向编码
      band_y_response = network.band_y_fr.value  # Y方向编码

      # 运动输入通过高斯编码
      # v_x = 0.02 m/s → X方向Band细胞激活
      # v_y = 0.01 m/s → Y方向Band细胞激活

   **说明**：
   - Band细胞与运动方向选择性相同
   - 提供了连续运动信息
   - 驱动所有格点模块的更新

3. **多尺度格点模块**

   .. code-block:: python

      # 每个模块代表不同的空间尺度
      module_activities = [
          network.grid_fr.value  # 聚合的格点活动
      ]

      # 模块间的尺度比例
      scale_ratios = [1.0, 2.4, 5.76, 13.8, 33.1]  # 2.4倍缩放

      # 每个模块的格点间距
      grid_spacings = {
          'module_1': 50,   # cm
          'module_2': 120,  # cm
          'module_3': 290,  # cm
          'module_4': 700,  # cm
          'module_5': 1700  # cm
      }

   **说明**：
   - 相邻模块间距约2.4倍关系
   - 小模块提供精细定位
   - 大模块提供广泛导航
   - 多尺度编码避免歧义

4. **位置细胞层（整合层）**

   .. code-block:: python

      # 位置细胞通过整合所有格点模块
      place_activity = network.place_fr.value

      # 单个位置细胞的活动形成"位置场"
      # 通过中国剩余定理原理：
      #   不同模块的周期组合产生唯一位置表示

   **数学原理** （中国剩余定理）：

   .. code-block:: text

      假设有3个模块，周期分别为 3, 5, 7

      模块1 (周期3): 0 1 2 0 1 2 0 1 2 ...
      模块2 (周期5): 0 1 2 3 4 0 1 2 3 ...
      模块3 (周期7): 0 1 2 3 4 5 6 0 1 ...

      组合: (0,0,0) → 位置0（唯一）
           (1,1,1) → 位置1（唯一）
           (2,2,2) → 位置2（唯一）
           ...
           (1,3,5) → 位置36（唯一）

      通过5个模块可覆盖巨大的表示空间！

5. **网络的学习机制**

   .. code-block:: python

      # 定位输入强度控制学习
      network(
          velocity=velocity_input,
          loc=actual_position,           # 教师信号
          loc_input_stre=input_strength  # 0 = 无学习，100 = 强学习
      )

      # loc_input_stre=100 时：使用位置信号校准网络
      # loc_input_stre=0 时：纯路径积分（开环）

运行结果
--------

运行分层网络会生成：

1. **轨迹和活动数据**

   - 动物轨迹：动物在环境中的运动路径
   - Band细胞活动：运动方向编码
   - 格点活动：所有5个模块的格点图案
   - 位置细胞活动：位置编码

2. **激活热图**

   .. code-block:: text

      Grid Module 1（粗尺度）：
      ○   ○   ○   ○
       ○   ○   ○   ○    大型六边形

      Grid Module 5（细尺度）：
      ○○○○○○○○○○
      ○○○○○○○○○○    小型六边形，高分辨率
      ○○○○○○○○○○

      Place Cells：
      █     █       █    每个位置细胞代表特定位置

3. **性能指标**

   - 仿真时间：~20秒
   - 网络大小：~5000个神经元
   - 内存使用：~500 MB
   - 轨迹长度：~500,000个时间步

关键概念
--------

**尺度不变性和符号距离问题**

为什么需要多个尺度？

.. code-block:: text

   单一格点模块的问题：
   ○───○───○───○───○───○
   位置在（0,0）时：模式A
   位置在（6,0）时：也是模式A（因为周期为6）
   → 无法区分！

   多模块的解决方案：
   Module 1 (周期 6): 位置0 → A,  位置6 → A (歧义)
   Module 2 (周期 7): 位置0 → B,  位置6 → C (区分！)
   → 组合 (A, B) 唯一对应位置0
   → 组合 (A, C) 唯一对应位置6

**模块独立性和冗余性**

- 每个模块独立进行路径积分
- 一个模块的错误不影响其他模块
- 提供鲁棒性和错误校正能力

**缩放不变的导航**

多模块结构使导航在不同环境中缩放：

.. code-block:: python

   # 在小鼠环境中
   grid_spacings = [50, 120, 290, 700, 1700]  # cm

   # 在大象环境中
   grid_spacings = [5, 12, 29, 70, 170]  # m（缩放后）

实验变化
--------

**1. 改变模块数量**

.. code-block:: python

   # 更少模块（低精度）
   network = HierarchicalNetwork(num_module=2, num_place=10)

   # 更多模块（高精度）
   network = HierarchicalNetwork(num_module=8, num_place=50)

**2. 分析单个模块的格点间距**

.. code-block:: python

   from canns.analyzer.spatial import compute_firing_field

   # 为每个模块计算热图
   for module_idx in range(5):
       heatmap = compute_firing_field(
           module_activities[module_idx],
           animal_trajectory
       )
       # 从热图提取格点间距

**3. 测试位置解码精度**

.. code-block:: python

   def decode_position_from_place_cells(place_activity):
       """从位置细胞活动解码位置"""
       # 使用最大活动的细胞作为位置估计
       active_cell = np.argmax(place_activity)
       estimated_position = cell_to_position_map[active_cell]
       return estimated_position

   # 评估解码精度
   decoded_positions = [decode_position_from_place_cells(p)
                       for p in place_activity]
   position_errors = np.linalg.norm(
       np.array(decoded_positions) - animal_trajectory,
       axis=1
   )

**4. 研究模块间的相位关系**

.. code-block:: python

   # 格点模块的活动周期
   for module_idx in range(5):
       # 计算活动的自相关
       autocorr = compute_autocorrelation(module_activities[module_idx])
       period = find_peak_distance(autocorr)
       print(f"Module {module_idx} 周期: {period:.1f}cm")

相关API
-------

- :class:`~src.canns.models.basic.HierarchicalNetwork` - 分层导航网络
- :class:`~src.canns.models.basic.CANN2D` - 基础2D CANN（用于单个模块）
- :func:`~src.canns.analyzer.spatial.compute_firing_field` - 热图计算

生物学应用
----------

**多尺度导航的优势**

1. **精确定位**：细尺度模块提供厘米级精度
2. **广泛覆盖**：粗尺度模块覆盖数公里范围
3. **错误校正**：大尺度模块检测小尺度漂移
4. **能量效率**：稀疏编码，减少神经活动

**进化和发展**

- 更大的动物有更大的格点间距
- 不同物种的模块数不同
- 年轻动物的格点细胞可能尺度不同

更多资源
--------

- :doc:`path_integration` - 理解基本的路径积分
- :doc:`grid_place_cells` - 格点和位置细胞的相互作用
- :doc:`complex_environments` - 在复杂环境中的导航
常见问题
--------

**Q: 为什么位置细胞会在多个地方活跃？**

A: 因为它是多个格点模块的组合。每个格点模块有周期性模式，组合产生多个位置场。通常通过学习，网络会强化一个主要位置场。

**Q: 多少个模块"足够"？**

A: 取决于环境大小和所需精度。数学上，5个模块可以用2.4倍的缩放比例覆盖任意大的空间。大多数啮齿动物使用7-8个尺度。

**Q: 模块如何同步？**

A: 通过位置输入。定期的外部定位输入（视觉地标等）将所有模块重置到一致状态，防止漂移累积。

下一步
------

完成本教程后，推荐：

1. 改变模块数和观察精度变化
2. 分析在视觉约束下的学习过程
3. 研究不同环境大小的适应
4. 阅读 :doc:`grid_place_cells` 了解细胞间的相互作用
