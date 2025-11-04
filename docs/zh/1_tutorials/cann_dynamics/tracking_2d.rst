二维CANN追踪
============

场景描述
--------

你想要理解CANN如何处理二维空间的刺激，观察神经网络中的"bump"活动模式如何在二维平面上移动。这是从一维扩展到更复杂空间表示的关键一步。

你将学到
--------

- 如何初始化二维CANN模型（CANN2D）
- 二维平滑追踪任务的定义
- 如何在二维空间中可视化神经活动
- 能量景观动画的生成和解释
- 二维空间编码的性质

完整示例
--------

.. literalinclude:: ../../../../examples/cann/cann2d_tracking.py
   :language: python
   :linenos:

逐步解析
--------

1. **初始化二维CANN模型**

   .. code-block:: python

      import brainstate as bst
      from canns.models.basic import CANN2D

      bst.environ.set(dt=0.1)  # 时间步长 0.1ms

      # 创建CANN2D模型
      cann = CANN2D(length=100)  # 100x100的神经元网格
      cann.init_state()

   **说明**：
   - ``length=100``：创建 100×100 = 10,000 个神经元
   - 代表二维空间 [0, length] × [0, length]
   - 每个神经元对应二维坐标 (x, y)

2. **定义二维追踪任务**

   .. code-block:: python

      from canns.task.tracking import SmoothTracking2D

      task_st = SmoothTracking2D(
          cann_instance=cann,
          Iext=([0., 0.], [1., 1.], [0.75, 0.75], [2., 2.], [1.75, 1.75], [3., 3.]),
          duration=(10., 10., 10., 10., 10.),
          time_step=brainstate.environ.get_dt(),
      )
      task_st.get_data()

   **说明**：
   - 刺激在6个位置显示：(0,0) → (1,1) → (0.75,0.75) → (2,2) → (1.75,1.75) → (3,3)
   - 每个位置持续10秒（5个位置 = 50秒总时间）
   - 每个位置代表二维坐标 (x, y)

3. **定义仿真步骤函数**

   .. code-block:: python

      def run_step(t, Iext):
          """单步仿真函数，处理二维刺激"""
          with bst.environ.context(t=t):
              cann(Iext)  # 传入二维刺激
              return cann.u.value, cann.r.value, cann.inp.value

   **说明**：
   - ``Iext`` 是二维刺激 (x, y) 坐标
   - 返回三个量：膜电位、发放率、输入电流
   - 使用 ``context`` 管理时间环境

4. **运行编译循环**

   .. code-block:: python

      import brainstate.compile

      cann_us, cann_rs, inps = brainstate.compile.for_loop(
          run_step,
          task_st.run_steps,      # 时间步索引
          task_st.data,           # 二维刺激序列
          pbar=brainstate.compile.ProgressBar(10)
      )

   **说明**：
   - ``cann_us``：所有时间步的膜电位 [时间, x, y]
   - ``cann_rs``：所有时间步的发放率 [时间, x, y]
   - ``inps``：所有时间步的输入 [时间, x, y]

5. **创建能量景观动画**

   .. code-block:: python

      from canns.analyzer.plotting import PlotConfigs, energy_landscape_2d_animation

      # 配置动画参数
      config = PlotConfigs.energy_landscape_2d_animation(
          time_steps_per_second=100,
          fps=20,
          title='CANN2D编码',
          xlabel='空间 X',
          ylabel='空间 Y',
          clabel='神经活动',
          repeat=True,
          save_path='cann2d_tracking.gif',
          show=False
      )

      # 生成动画
      energy_landscape_2d_animation(
          zs_data=cann_us,
          config=config
      )

   **说明**：
   - ``time_steps_per_second=100``：100个仿真步 = 1秒
   - ``fps=20``：动画播放帧率
   - 生成 GIF 文件显示二维bump的运动

运行结果
--------

运行此脚本会生成：

1. **二维能量景观动画** (`cann2d_tracking.gif`)

   - X轴和Y轴：空间位置
   - Z轴（颜色）：神经活动强度
   - 白色/亮色区域：高活动 bump
   - 深色区域：低活动背景

2. **预期的动画特征**

   - Bump在二维平面上平滑移动
   - Bump的形状接近二维高斯分布
   - 移动轨迹：(0,0) → (1,1) → (0.75,0.75) → (2,2) → (1.75,1.75) → (3,3)
   - 每个位置保持10秒的稳定活动

3. **性能指标**

   - 仿真时间：~5-10秒（含编译）
   - 内存使用：~500 MB（10,000个神经元）
   - 总时间步数：~5000步
   - GIF文件大小：~2-5 MB

关键概念
--------

**二维Bump活动**

二维CANN形成的bump特征：

- Bump是一个**二维高斯形**局部活动区域
- 中心对应刺激位置
- 宽度由神经元连接范围决定
- 周围被强抑制（WSC模式）

.. code-block:: text

   俯视图（从上往下看活动）：

         高活动
           ↑
           |    ★★★
           |   ★★★★★
           |    ★★★★★
           |     ★★★
           |
   低活动  └─────────────→

   侧视图（穿过bump中心）：

            活动
            ↑
            │      ╱╲
            │     ╱  ╲
            │    ╱    ╲
            │___╱      ╲___
            └──────────────→ 位置

**拓扑组织在二维中的扩展**

- 二维CANN保持二维拓扑映射
- 相邻神经元对相邻空间位置有相似响应
- 形成连续的、可导航的表示空间
- 类似大脑中的方向柱塔（orientation columns）和位置编码

**与一维的对比**

==================  ==============  ==============
特性                一维CANN1D      二维CANN2D
==================  ==============  ==============
神经元数             512            10,000 (100²)
表示空间            一条线          二维平面
Bump形状            1D高斯          2D高斯
内存需求            ~100 MB         ~500 MB
计算复杂度          O(n²)           O(n⁴)
==================  ==============  ==============

实验变化
--------

**1. 改变网络大小**

.. code-block:: python

   # 更精细的表示
   cann = CANN2D(length=150)

   # 更粗糙的表示
   cann = CANN2D(length=64)

**2. 改变追踪轨迹**

.. code-block:: python

   # 在方形边界内移动
   task_st = SmoothTracking2D(
       cann_instance=cann,
       Iext=([0., 0.], [5., 0.], [5., 5.], [0., 5.], [0., 0.]),
       duration=(10., 10., 10., 10.),
   )

   # 圆形轨迹
   import numpy as np
   n_points = 8
   angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
   radius = 2.0
   center = 3.0
   points = [[center + radius*np.cos(a), center + radius*np.sin(a)] for a in angles]
   task_st = SmoothTracking2D(
       cann_instance=cann,
       Iext=points,
       duration=(10.,) * (n_points - 1),
   )

**3. 改变神经参数**

.. code-block:: python

   cann = CANN2D(
       length=100,
       tau=0.1,       # 膜时间常数
       a=0.5,         # 局部激励范围
       A=1.2,         # 激励强度
       J0=0.5,        # 背景输入
   )

**4. 分析bump运动速度**

.. code-block:: python

   import numpy as np

   # 计算每个时间步bump的中心
   bump_centers = []
   for t in range(len(cann_us)):
       u = cann_us[t]
       # 找到最大活动的位置
       max_idx = np.unravel_index(np.argmax(u), u.shape)
       bump_centers.append(max_idx)

   # 计算bump的运动速度
   velocities = np.diff(bump_centers, axis=0)
   speed = np.linalg.norm(velocities, axis=1)

相关API
-------

- :class:`~src.canns.models.basic.CANN2D` - 二维CANN模型
- :class:`~src.canns.task.tracking.SmoothTracking2D` - 二维平滑追踪任务
- :func:`~src.canns.analyzer.plotting.energy_landscape_2d_animation` - 二维能量景观动画

更多资源
--------

- :doc:`tracking_1d` - 理解一维基础
- :doc:`tuning_curves` - 分析神经元调谐曲线
- :doc:`../spatial_navigation/index` - 使用二维CANN进行空间导航
常见问题
--------

**Q: 为什么bump会变形？**

A: 几个可能的原因：
   - 网络还未完全收敛
   - 边界效应（靠近 0 或 length）
   - 参数 ``a`` 和 ``A`` 需要调整

**Q: 动画很慢或生成失败？**

A:
   - 减少 ``length`` 降低计算复杂度
   - 增加 ``fps`` 减少帧数
   - 在 GPU 上运行：``JAX_PLATFORM_NAME=gpu``

**Q: 如何分析bump的宽度？**

A: 可以沿着bump中心截面分析，拟合高斯曲线：

   .. code-block:: python

      from scipy.optimize import curve_fit

      # 在bump中心位置截面
      center = np.array(bump_centers[-1])  # 最后时刻的位置
      x_profile = cann_us[-1, int(center[0]), :]

      # 拟合高斯
      def gaussian(x, amp, center, width):
          return amp * np.exp(-(x - center)**2 / (2*width**2))

下一步
------

完成本教程后，推荐：

1. 尝试上面的实验变化，观察二维空间的编码
2. 阅读 :doc:`tuning_curves` 学习如何分析二维调谐曲线
3. 探索 :doc:`../spatial_navigation/index` 的导航应用
4. 研究 :doc:`oscillatory_tracking` 学习动态追踪行为
