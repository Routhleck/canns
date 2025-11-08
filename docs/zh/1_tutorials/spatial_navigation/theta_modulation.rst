Theta节律和时序编码
===================

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。



场景描述
--------

你想要理解大脑中Theta节律（~8Hz）如何与空间编码相互作用，以及"Theta Phase Precession"机制如何实现高精度的时间和空间编码。

你将学到
--------

- Theta节律的性质和来源
- Theta Phase Precession的机制
- 时序代码和位置代码的结合
- Theta节律在路径积分中的角色
- 振荡和CANN相互作用

完整示例
--------

基于带有Theta节律的导航任务：

.. code-block:: python

   import numpy as np
   import brainstate
   from canns.models.basic import CANN1D
   from canns.task.tracking import SmoothTracking1D

   # 设置Theta节律（~8 Hz）
   theta_frequency = 8.0  # Hz
   theta_period = 1.0 / theta_frequency  # 秒

   # 创建Theta调制的输入
   def create_theta_modulated_input(position, time, theta_freq=8.0):
       """创建被Theta节律调制的位置输入"""
       theta_phase = np.sin(2 * np.pi * theta_freq * time)
       # 输入强度按Theta振荡调制
       return position * (1.0 + 0.5 * theta_phase)

   # 运行带Theta调制的CANN追踪
   cann = CANN1D(num=512)
   cann.init_state()

   firing_rates = []
   phases = []

   for t in np.linspace(0, 10, 1000):  # 10秒
       position = np.sin(t)  # 正弦变化的位置
       theta_input = create_theta_modulated_input(position, t)

       # 创建输入刺激
       stimulus = np.zeros(512)
       stimulus[int(256 + 256 * position)] = theta_input

       cann(stimulus)
       firing_rates.append(cann.r.value.copy())
       phases.append(np.angle(np.exp(1j * 2 * np.pi * 8 * t)))

   firing_rates = np.array(firing_rates)
   phases = np.array(phases)

逐步解析
--------

1. **Theta节律的基础**

   .. code-block:: python

      # Theta节律是一个周期振荡
      theta_frequency = 8.0  # Hz（5-12Hz范围内）
      theta_period = 1.0 / theta_frequency  # ~125 ms

      # 标准Theta信号
      theta_signal = np.sin(2 * np.pi * theta_frequency * t)

   **特征**：
   - 频率：5-12 Hz（取决于行为和物种）
   - 幅度：0.5-2 mV
   - 来源：中隔核（medial septum）
   - 功能：协调多脑区的活动

2. **Phase Precession机制**

   .. code-block:: python

      # Phase Precession：位置细胞在穿过其位置场时
      # 其放电与Theta节律的相位逐渐提前

      # 当动物进入位置场：
      位置 = 0.5, Theta相位 = 0° → 细胞开始放电
      位置 = 0.6, Theta相位 = 30° → 细胞更强放电
      位置 = 0.7, Theta相位 = 60° → 更强
      位置 = 0.8, Theta相位 = 90° → 最强
      位置 = 0.9, Theta相位 = 120° → 开始减弱
      位置 = 1.0, Theta相位 = 180° → 离开位置场

   **结果**：
   - 放电时间相对于位置提前
   - 一个周期内可编码多个位置
   - 时间乘以速度 = 位置信息

3. **在CANN中实现Phase Precession**

   .. code-block:: python

      def analyze_phase_precession(firing_rates, phases, position_trajectory):
          """分析Phase Precession"""
          precession_data = []

          for neuron_idx in range(firing_rates.shape[1]):
              neuron_activity = firing_rates[:, neuron_idx]

              # 找到该神经元的峰值活动
              peak_times = np.where(neuron_activity > np.max(neuron_activity) * 0.5)[0]

              if len(peak_times) > 1:
                  # 计算放电时间和Theta相位的关系
                  for peak_time in peak_times:
                      precession_data.append({
                          'neuron': neuron_idx,
                          'position': position_trajectory[peak_time],
                          'theta_phase': phases[peak_time],
                          'firing_rate': neuron_activity[peak_time]
                      })

          return precession_data

   **说明**：
   - 绘制：横轴=位置，纵轴=相位
   - 应该看到单调递减的关系（相位随位置提前）

关键概念
--------

**Theta和空间编码的结合**

.. code-block:: text

   Theta节律提供的好处：

   1. 时间窗口划分
      ├─ Theta周期 1：时间 0-125ms
      ├─ Theta周期 2：时间 125-250ms
      └─ 每个周期内编码不同的信息

   2. Phase Precession的计算优势
      位置 = v · t + θ · τ_phase
      其中 τ_phase 是相位偏移，使用 ~30Hz 的"beta振荡"

   3. 序列生成
      - 老鼠奔跑时，位置细胞序列与Theta周期同步
      - 可用于序列学习和回放

**Theta Skipping和Theta Cycling**

两个不同的编码机制：

.. code-block:: text

   Theta Skipping（跳过）：
   - 一个位置场一个Theta周期
   - 低速运动时发生
   - 时间分辨率：位置数 = Theta频率 / 速度

   Theta Cycling（循环）：
   - 一个位置场多个Theta周期
   - 高速运动时发生
   - 交替编码运动轨迹和其他信息

实验变化
--------

**1. 改变Theta频率**

.. code-block:: python

   for theta_freq in [4, 6, 8, 10, 12]:  # 4-12 Hz范围
       theta_input = create_theta_modulated_input(position, t, theta_freq)
       # 观察Phase Precession如何变化

**2. 分析Theta调制强度的影响**

.. code-block:: python

   for modulation_strength in [0.1, 0.3, 0.5, 0.7, 0.9]:
       theta_input = position * (1.0 + modulation_strength * theta_phase)

**3. 测量时间编码能力**

.. code-block:: python

   # 在两个Theta周期内能编码多少个不同位置？
   def measure_temporal_resolution(firing_rates, phases):
       """衡量时间分辨率"""
       # 在一个Theta周期内聚类放电事件
       clusters = cluster_by_theta_phase(phases)
       positions_per_cycle = len(clusters)
       return positions_per_cycle

相关API
-------

- :class:`~src.canns.models.basic.CANN1D` - 支持Theta调制的CANN1D
- :class:`~src.canns.task.tracking.SmoothTracking1D` - 可添加Theta节律

生物学应用
----------

**海马体录音的证据**

- O'Keefe和Recce (1993)发现Position细胞的Phase Precession
- 每个Theta周期内，不同位置细胞序列放电
- 效果像"压缩回放"未来路径

**Theta和学习**

- Theta振荡周期内发生突触可塑性
- STDP与Theta的相位相关
- 支持高效的强化学习

更多资源
--------

- :doc:`path_integration` - 理解基础路径积分
- :doc:`grid_place_cells` - 格点和位置细胞的整合
常见问题
--------

**Q: Phase Precession有什么计算优势？**

A: 它允许在一个Theta周期内编码多个位置。如果Theta频率是8Hz，速度是1m/s，位置字段宽度是40cm，那么一个Theta周期可以编码8个不同的位置！

**Q: 为什么需要Theta节律？**

A:
- 协调不同脑区
- 选择时间窗口进行学习
- 支持前瞻性编码
- 可能与意识相关

**Q: Phase Precession在CANN中如何实现？**

A: 通过以Theta频率调制输入强度。输入在Theta峰值时强，在Theta谷值时弱，导致不同Theta相位的输入强度不同，从而改变放电时间。

下一步
------

1. 分析Phase Precession的强度和持续时间
2. 比较不同速度下的编码
3. 研究Theta和Beta振荡的相互作用
4. 阅读 :doc:`grid_place_cells` 了解更多
