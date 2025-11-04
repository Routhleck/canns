神经元调谐曲线分析
==================

场景描述
--------

你想要深入理解CANN中单个神经元的调谐特性：它们如何对不同空间刺激作出响应，以及这些响应模式如何定义了网络的空间编码。调谐曲线是理解神经表示的关键工具。

你将学到
--------

- 什么是调谐曲线以及为什么重要
- 如何从网络活动计算调谐曲线
- 调谐曲线的数学特性（峰值、宽度、对称性）
- 如何在群体水平分析调谐特性
- 调谐曲线的生物学意义

完整示例
--------

基于一维CANN追踪任务的调谐曲线分析：

.. code-block:: python

   import brainstate
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.optimize import curve_fit
   from canns.models.basic import CANN1D
   from canns.task.tracking import SmoothTracking1D
   from canns.analyzer.plotting import PlotConfigs, tuning_curve
   import brainstate.compile

   # 设置环境
   brainstate.environ.set(dt=0.1)

   # 创建模型
   cann = CANN1D(num=512, z_min=-np.pi, z_max=np.pi)
   cann.init_state()

   # 创建多位置追踪任务以覆盖整个空间
   positions = np.linspace(-np.pi, np.pi, 16)  # 16个位置
   task = SmoothTracking1D(
       cann_instance=cann,
       Iext=tuple(positions),
       duration=(5.,) * 15,  # 每个位置5秒
       time_step=brainstate.environ.get_dt(),
   )
   task.get_data()

   # 定义仿真步骤
   def run_step(t, inputs):
       cann(inputs)
       return cann.r.value

   # 运行仿真
   rs = brainstate.compile.for_loop(
       run_step,
       task.run_steps,
       task.data,
       pbar=brainstate.compile.ProgressBar(10)
   )

   # 计算调谐曲线
   neuron_indices = [64, 128, 256, 384, 448]

   for neuron_idx in neuron_indices:
       tuning = []
       for pos_idx in range(len(positions)):
           start_time = int(pos_idx * 5 / 0.1)  # 每个位置5秒 = 500步
           end_time = int((pos_idx + 1) * 5 / 0.1)
           # 取后半段以避免过渡态
           mid_time = start_time + (end_time - start_time) // 2
           avg_response = np.mean(rs[mid_time:end_time, neuron_idx])
           tuning.append(avg_response)

       # 绘制调谐曲线
       plt.figure(figsize=(10, 4))
       plt.plot(positions, tuning, 'o-', linewidth=2, markersize=8)
       plt.xlabel('刺激位置 (弧度)')
       plt.ylabel('平均发放率 (Hz)')
       plt.title(f'神经元 {neuron_idx} 的调谐曲线')
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(f'tuning_curve_neuron_{neuron_idx}.png')
       plt.close()

   # 群体分析：所有神经元的调谐宽度分布
   tuning_widths = []
   tuning_peaks = []

   for neuron_idx in range(0, 512, 8):  # 每8个神经元取一个
       tuning = []
       for pos_idx in range(len(positions)):
           start_time = int(pos_idx * 5 / 0.1)
           end_time = int((pos_idx + 1) * 5 / 0.1)
           mid_time = start_time + (end_time - start_time) // 2
           avg_response = np.mean(rs[mid_time:end_time, neuron_idx])
           tuning.append(avg_response)

       # 拟合高斯曲线
       def gaussian(x, amp, center, width):
           return amp * np.exp(-(x - center)**2 / (2*width**2))

       try:
           popt, _ = curve_fit(gaussian, positions, tuning, p0=[1, positions[0], 0.5])
           tuning_widths.append(popt[2])
           tuning_peaks.append(popt[0])
       except:
           pass

   print("\\n=== 调谐曲线统计 ===")
   print(f"平均调谐宽度: {np.mean(tuning_widths):.3f} 弧度")
   print(f"调谐宽度标准差: {np.std(tuning_widths):.3f} 弧度")
   print(f"平均峰值响应: {np.mean(tuning_peaks):.3f} Hz")

逐步解析
--------

1. **调谐曲线的数学定义**

   调谐曲线 f(θ) 描述神经元对刺激位置 θ 的响应：

   .. math::

      f(\theta) = \text{平均发放率}(\theta)

   最常见的形式是高斯型：

   .. math::

      f(\theta) = A \cdot \exp\left(\frac{-(\theta - \theta_0)^2}{2\sigma^2}\right)

   其中：
   - A：峰值响应幅度
   - θ₀：首选刺激（最强响应位置）
   - σ：调谐宽度（决定响应的窄或宽）

2. **关键参数解释**

   .. code-block:: text

      响应强度
      ↑
      │         ╱╲
      │        ╱  ╲       高斯曲线
      │ 幅度 A│   ╲
      │      │    ╲
      │      └─────╲───────→ 刺激位置
      │             θ₀
      │      ├─2σ─┤
      │    宽度

   - **幅度 (A)**：神经元的最大响应强度
   - **首选刺激 (θ₀)**：导致最大响应的刺激位置
   - **宽度 (σ)**：调谐的锐锐程度
     - σ 小（<π/10）：锐调谐，高空间选择性
     - σ 中等（π/10-π/6）：中等调谐
     - σ 大（>π/6）：宽调谐，低空间选择性

3. **从网络活动提取调谐曲线**

   .. code-block:: python

      # 步骤1：针对不同刺激位置运行网络
      positions = np.linspace(-π, π, 16)  # 16个测试位置

      # 步骤2：对每个位置，计算平均神经元响应
      for each position:
          稳定期的平均发放率 → 调谐数据点

      # 步骤3：拟合高斯曲线提取参数
      tuning_curve = fit_gaussian(positions, firing_rates)

4. **群体编码的拓扑特性**

   CANN的一个关键特性是拓扑组织：

   .. code-block:: text

      神经元索引 (群体坐标)：
      0   128   256   384   512
      │    │     │     │     │
      │    │     │     │     │
      ↓    ↓     ↓     ↓     ↓
      首选刺激：
      -π  -π/2   0    π/2   π

      相邻的神经元有接近的首选刺激！

运行结果
--------

运行此脚本会生成：

1. **单个神经元的调谐曲线** (5张图像)

   每张图显示一个神经元的调谐特性：
   - X轴：刺激位置（从 -π 到 π）
   - Y轴：平均发放率
   - 曲线形状：高斯曲线，一个明确的峰值

2. **预期结果特征**

   - 神经元64：峰值在 ~ -π (网络的一端)
   - 神经元128：峰值在 ~ -π/2
   - 神经元256：峰值在 ~ 0 (网络中心)
   - 神经元384：峰值在 ~ π/2
   - 神经元448：峰值在 ~ π (网络的另一端)

3. **群体统计**

   .. code-block:: text

      === 调谐曲线统计 ===
      平均调谐宽度: 0.523 弧度  (约30度)
      调谐宽度标准差: 0.045 弧度
      平均峰值响应: 3.821 Hz

关键概念
--------

**调谐曲线的种类**

========== =============== ================== ==========================
调谐类型   宽度范围        首选性              典型应用
========== =============== ================== ==========================
锐调谐     σ < π/10        高度特异化         精细空间编码
中调谐     π/10 < σ < π/6  中等特异化         一般空间编码
宽调谐     σ > π/6         低度特异化         粗糙编码+鲁棒性
========== =============== ================== ==========================

**调谐宽度的生物学含义**

- **锐调谐神经元**：
  - 优点：高空间分辨率
  - 缺点：对噪声敏感，需要更多神经元
  - 见于：视觉皮层中的方向选择性

- **宽调谐神经元**：
  - 优点：鲁棒性强，对噪声不敏感
  - 缺点：空间分辨率低
  - 见于：头方向细胞、位置细胞

**循环空间中的调谐**

在CANN中，空间是循环的（π和-π表示同一方向）：

.. code-block:: text

   线性空间：
   -π──────0──────π
   |              |
   无循环          无循环

   循环空间：
          π ≡ -π
         ╱     ╲
        ╱       ╲
      -π/2     π/2
        ╲       ╱
         ╲     ╱
           0

   在CANN中，边界处的调谐需要特殊处理！

实验变化
--------

**1. 改变刺激测试范围**

.. code-block:: python

   # 只测试网络的一部分（如上半平面）
   positions = np.linspace(0, π, 8)

   # 更高分辨率的测试
   positions = np.linspace(-π, π, 32)

**2. 分析调谐宽度沿网络的变化**

.. code-block:: python

   widths_by_neuron = []
   for neuron_idx in range(512):
       tuning = [compute_tuning(neuron_idx, pos) for pos in positions]
       width = fit_gaussian(positions, tuning)[2]
       widths_by_neuron.append(width)

   plt.figure(figsize=(12, 4))
   plt.plot(widths_by_neuron)
   plt.xlabel('神经元索引')
   plt.ylabel('调谐宽度 (弧度)')
   plt.title('调谐宽度沿网络的分布')
   plt.axhline(np.mean(widths_by_neuron), color='r', linestyle='--', label='平均值')
   plt.legend()
   plt.grid(True, alpha=0.3)

**3. 比较不同网络参数的影响**

.. code-block:: python

   # 改变局部激励范围
   for a_value in [0.3, 0.5, 0.7]:
       cann = CANN1D(num=512, a=a_value)
       # ... 计算并绘制调谐曲线

   # 改变抑制强度
   for J_value in [0.3, 0.5, 0.7]:
       cann = CANN1D(num=512, J=J_value)
       # ... 计算并绘制调谐曲线

**4. 分析调谐曲线的对称性**

.. code-block:: python

   # 检查高斯拟合的质量
   residuals = tuning_data - gaussian_fit
   r_squared = 1 - np.sum(residuals**2) / np.sum((tuning_data - np.mean(tuning_data))**2)
   print(f"拟合质量 (R²): {r_squared:.4f}")

**5. 群体解码：从群体活动重建刺激**

.. code-block:: python

   # 使用最大似然估计从群体活动重建刺激
   def decode_stimulus(population_activity):
       # 假设高斯调谐
       likelihood = np.ones(len(positions))
       for neuron_idx, response in enumerate(population_activity):
           for pos_idx, pos in enumerate(positions):
               tuning_curve = gaussian(pos, *tuning_params[neuron_idx])
               # 泊松模型：P(response|tuning) = exp(-tuning) * tuning^response
               likelihood[pos_idx] *= np.exp(-tuning_curve) * (tuning_curve ** response)

       best_pos = positions[np.argmax(likelihood)]
       return best_pos

相关API
-------

- :class:`~src.canns.models.basic.CANN1D` - 一维CANN模型
- :func:`~src.canns.analyzer.plotting.tuning_curve` - 调谐曲线绘制工具
- :class:`~src.canns.task.tracking.SmoothTracking1D` - 平滑追踪任务

生物学应用
----------

**1. 视觉皮层方向选择性**

哺乳动物视觉皮层的复杂细胞：

- 调谐宽度：30-60度
- 首选刺激：方向（0-360度）
- 应用：CANN模型成功解释了方向柱塔的组织

**2. 空间位置编码**

海马体位置细胞：

- 调谐宽度：20-40 cm
- 首选刺激：空间位置
- 特性：place field的形成和稳定

**3. 头方向编码**

背内侧核的头方向细胞：

- 调谐宽度：30-50度
- 首选刺激：头部方向
- 特性：全球极化框架，与环境对齐

更多资源
--------

- :doc:`tracking_1d` - 基本追踪实验
- :doc:`tracking_2d` - 二维空间编码
- :doc:`../spatial_navigation/path_integration` - 使用调谐曲线进行路径积分
常见问题
--------

**Q: 为什么调谐曲线不完全是高斯形？**

A: 几个可能的原因：
   - 网络噪声或初始条件不稳定
   - 边界效应（靠近 ±π 的神经元）
   - 网络参数不理想
   - 样本量不足

**Q: 如何估计调谐宽度？**

A: 几个方法：

   .. code-block:: python

      # 方法1：半最大宽度 (FWHM)
      peak = max(tuning_curve)
      half_max = peak / 2
      width_fwhm = positions[tuning_curve > half_max][-1] - positions[tuning_curve > half_max][0]

      # 方法2：高斯拟合
      popt = curve_fit(gaussian, positions, tuning_curve)
      sigma = popt[2]  # 高斯标准差

      # 方法3：信息论方法
      information = calculate_mutual_information(population, stimulus)

**Q: 不同神经元的调谐宽度为什么不同？**

A: 在真实的CANN模型中，宽度应该基本相同。不同的原因通常是：
   - 拟合质量问题
   - 网络的边界效应
   - 神经元之间的随机变异性

下一步
------

完成本教程后，推荐：

1. 分析多个神经元的调谐曲线并找出规律
2. 尝试参数变化并观察调谐宽度的变化
