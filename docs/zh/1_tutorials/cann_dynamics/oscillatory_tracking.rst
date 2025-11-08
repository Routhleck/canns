振荡追踪和动态稳定性
====================

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。



场景描述
--------

你想要理解CANN网络如何稳定地追踪快速变化的刺激，并观察网络活动的振荡模式如何随刺激变化。这揭示了CANN的动态稳定性和自适应特性。

你将学到
--------

- 快速变化刺激下的追踪机制
- 网络活动的振荡和衰减模式
- 能量景观的实时动态
- 追踪稳定性和参数的关系
- 吸引子动力学的视觉化

完整示例
--------

.. literalinclude:: ../../../../examples/cann/cann1d_oscillatory_tracking.py
   :language: python
   :linenos:

逐步解析
--------

1. **初始化CANN模型**

   .. code-block:: python

      import brainstate
      from canns.models.basic import CANN1D

      brainstate.environ.set(dt=0.1)

      cann = CANN1D(num=512)
      cann.init_state()

   **说明**：
   - 512个一维排列的神经元
   - dt=0.1ms：时间步长
   - 足够的神经元密度以支持光滑的bump活动

2. **定义快速变化的追踪任务**

   .. code-block:: python

      from canns.task.tracking import SmoothTracking1D

      task_st = SmoothTracking1D(
          cann_instance=cann,
          Iext=(1., 0.75, 2., 1.75, 3.),     # 5个刺激位置（弧度）
          duration=(10., 10., 10., 10.),      # 4个时间段，各10秒
          time_step=brainstate.environ.get_dt(),
      )
      task_st.get_data()

   **说明**：
   - 5个位置的刺激序列：1 → 0.75 → 2 → 1.75 → 3
   - 相邻位置间距约 0.25 弧度（~14度）
   - 10秒持续时间足以观察完整的追踪动态

3. **定义仿真步骤**

   .. code-block:: python

      def run_step(t, inputs):
          """单步仿真：接收刺激输入"""
          cann(inputs)
          return cann.u.value, cann.inp.value

   **说明**：
   - ``cann.u.value``：膜电位（网络状态）
   - ``cann.inp.value``：外部输入电流
   - 返回这两个量用于分析动态

4. **运行编译循环**

   .. code-block:: python

      import brainstate.compile

      us, inps = brainstate.compile.for_loop(
          run_step,
          task_st.run_steps,
          task_st.data,
          pbar=brainstate.compile.ProgressBar(10)
      )

   **说明**：
   - ``us``：所有时间步的膜电位 [时间, 神经元]
   - ``inps``：所有时间步的输入 [时间, 神经元]
   - 总数据点：40秒 × 10Hz = 400时间步

5. **生成动画可视化**

   .. code-block:: python

      from canns.analyzer.plotting import PlotConfigs, energy_landscape_1d_animation

      config = PlotConfigs.energy_landscape_1d_animation(
          time_steps_per_second=100,
          fps=20,
          title='平滑追踪1D',
          xlabel='空间位置',
          ylabel='神经活动',
          repeat=True,
          save_path='oscillatory_tracking.gif',
          show=False
      )

      energy_landscape_1d_animation(
          data_sets={'u': (cann.x, us), 'Iext': (cann.x, inps)},
          config=config
      )

   **说明**：
   - ``data_sets`` 包含两个数据集：膜电位和输入
   - 动画显示时间进展中的两层数据
   - ``cann.x``：神经元的空间位置坐标

运行结果
--------

运行此脚本会生成：

1. **能量景观时间序列动画** (`oscillatory_tracking.gif`)

   - **上层**：输入电流 (Iext) 随时间的变化
   - **下层**：网络膜电位 (u) 形成的bump
   - 颜色变化表示强度变化

2. **预期的动态特征**

   .. code-block:: text

      t=0-10秒：位置1处的追踪
      │ Iext:   [高峰在x=1]
      │ u:      [bump快速形成，在x=1稳定]
      │
      t=10-20秒：位置0.75处的追踪
      │ Iext:   [高峰在x=0.75]
      │ u:      [bump从x=1平滑移动到x=0.75，可能有振荡]
      │
      t=20-30秒：位置2处的追踪
      │ Iext:   [高峰在x=2]
      │ u:      [bump从x=0.75跳跃到x=2，显示最大加速度]
      │
      t=30-40秒：位置1.75→3处的追踪
      │ Iext:   [高峰在x=1.75，然后移到x=3]
      │ u:      [bump继续平滑追踪]

3. **性能指标**

   - 仿真时间：~2-3秒（含编译）
   - 内存使用：~200 MB
   - 生成的GIF大小：~3-5 MB

关键概念
--------

**追踪中的振荡**

当刺激快速移动时，网络会产生振荡：

.. code-block:: text

   刺激阶跃时的bump响应：

   刺激位置：_______●-------●________

   Bump位置：_______●      ╱╱╱╱╱╱___●
                         ╱振荡╱
                        ╱____╱

   - **上升相**：Bump快速移向新位置
   - **过冲**：Bump越过刺激位置
   - **振荡阶段**：在新位置周围振荡
   - **稳定**：最终在刺激位置稳定

**吸引子动力学**

CANN的bump活动是**吸引子**的表现：

1. **稳定吸引子**：bump稳定停留
   - 刺激消失后还能保持活动
   - 实现"记忆"功能

2. **驱动吸引子**：被刺激牵引
   - bump跟随刺激移动
   - 用于"追踪"

3. **竞争吸引子**：多个竞争态（在更大网络中）
   - 代表不同的选择
   - 通过抑制竞争

**稳定性条件**

追踪稳定性取决于：

.. code-block:: python

   稳定性 = f(
       刺激速度,        # 越快越难追踪
       网络时间常数,    # tau越大越难追踪
       局部激励范围,    # a越大越容易追踪
       激励强度,        # A越大越容易追踪
   )

实验变化
--------

**1. 改变刺激速度**

.. code-block:: python

   # 缓慢移动（容易追踪）
   task_st = SmoothTracking1D(
       cann_instance=cann,
       Iext=(0., np.pi/2, np.pi),
       duration=(20., 20.),  # 更长的时间
   )

   # 快速移动（容易产生振荡）
   task_st = SmoothTracking1D(
       cann_instance=cann,
       Iext=(0., np.pi/2, np.pi),
       duration=(2., 2.),  # 更短的时间
   )

**2. 分析追踪延迟**

.. code-block:: python

   import numpy as np

   # 找到bump中心和刺激位置
   bump_peaks = []
   for t in range(len(us)):
       u = us[t]
       peak_idx = np.argmax(u)
       bump_peaks.append(peak_idx)

   # 计算追踪延迟
   stimulus_centers = task_st.Iext_sequence.squeeze()
   delays = bump_peaks - stimulus_centers

   print(f"平均追踪延迟: {np.mean(np.abs(delays)):.2f} 个神经元")

**3. 改变网络参数以优化追踪**

.. code-block:: python

   # 增大局部激励（更容易追踪）
   cann = CANN1D(num=512, a=0.6)

   # 减小时间常数（更快的响应）
   cann = CANN1D(num=512, tau=0.05)

**4. 可视化吸引子景观**

.. code-block:: python

   import matplotlib.pyplot as plt

   # 在没有外部输入时的自发活动
   cann.reset_state()
   cann.u[:] = 0.1  # 小的初始值
   cann.u[256] = 1.0  # 在中心激活一个神经元

   fig, axes = plt.subplots(1, 2, figsize=(12, 4))

   # 无输入的bump稳定性
   for t in range(1000):
       cann(np.zeros(512))
       if t % 100 == 0:
           axes[0].plot(cann.u.value)

   axes[0].set_title("自发bump活动（无输入）")
   axes[0].set_xlabel("神经元索引")
   axes[0].set_ylabel("膜电位")

   # 有输入的bump追踪
   cann.reset_state()
   stimulus = np.zeros(512)
   stimulus[300] = 1.0
   for t in range(1000):
       cann(stimulus)
       if t % 100 == 0:
           axes[1].plot(cann.u.value)

   axes[1].set_title("驱动的bump活动（有输入）")
   axes[1].set_xlabel("神经元索引")

   plt.tight_layout()
   plt.savefig('attractor_dynamics.png')

相关API
-------

- :class:`~src.canns.models.basic.CANN1D` - 一维CANN模型
- :class:`~src.canns.task.tracking.SmoothTracking1D` - 平滑追踪任务
- :func:`~src.canns.analyzer.plotting.energy_landscape_1d_animation` - 能量景观动画

生物学应用
----------

**头方向细胞（Head Direction Cells）**

大脑背内侧核（MEC）的头方向细胞编码动物头部方向。CANN模型成功解释了：

- 细胞的调谐曲线（钟形曲线）
- 头部移动时bump如何追踪
- 多模态输入融合（视觉+前庭感觉）

**空间导航**

啮齿动物进行导航时：

1. 格点细胞（Grid Cells）提供定位信息
2. 位置细胞（Place Cells）编码具体位置
3. CANN机制在位置改变时平滑更新表示

**运动控制**

运动皮层也显示类似的CANN动力学：

- 运动计划中的bump代表计划的运动方向
- 执行运动时bump平滑移动
- 反映了随时间展开的运动轨迹

更多资源
--------

- :doc:`tracking_1d` - 理解基本追踪
- :doc:`tracking_2d` - 二维空间的追踪
- :doc:`../spatial_navigation/index` - CANN在导航中的应用
常见问题
--------

**Q: 为什么bump会振荡？**

A: 当刺激移动时，CANN需要时间重新平衡。振荡是网络寻找新的吸引子态的过程。振荡幅度取决于：
   - 刺激移动的速度（越快越剧烈）
   - 网络的时间常数（越慢越容易振荡）
   - 抑制强度（越强越容易稳定）

**Q: 如何减少追踪延迟？**

A: 几个方法：
   - 增加激励强度 (A)：更快地驱动bump
   - 减小时间常数 (tau)：神经元响应更快
   - 增加局部激励范围 (a)：bump更容易形成
   - 但要避免参数过度调整导致不稳定

**Q: bump会"漂移"吗？**

A: 在某些条件下会产生漂移：
   - 不对称的神经元数量
   - 边界效应（网络边界附近）
   - 噪声累积
   可以通过以下方式检验：在无输入情况下运行网络，检查bump是否随时间偏移

下一步
------

完成本教程后，推荐：

1. 进行上面的实验变化
2. 改变网络参数并观察稳定性变化
3. 分析bump运动速度和加速度
