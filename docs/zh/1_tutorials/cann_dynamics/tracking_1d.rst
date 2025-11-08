一维CANN追踪
============

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。

场景描述
--------

你想要理解CANN（连续吸引子神经网络）如何响应一维空间的输入，并观察神经网络中"bump"活动模式如何追踪移动的刺激。这是理解CANN基本原理的最好起点。

你将学到
--------

- 如何初始化一维CANN模型（CANN1D）
- 如何定义平滑追踪任务
- 如何使用JAX的编译循环进行高效仿真
- 如何计算和可视化调谐曲线
- 如何分析神经元的响应特性

完整示例
--------

.. literalinclude:: ../../../../examples/cann/cann1d_tuning_curve.py
   :language: python
   :linenos:

逐步解析
--------

1. **环境设置和模型初始化**

   .. code-block:: python

      import brainstate
      import numpy as np
      from canns.models.basic import CANN1D

      # 设置环境参数
      brainstate.environ.set(dt=0.1)  # 时间步长 0.1ms

      # 创建CANN1D模型
      cann = CANN1D(num=512, z_min=-np.pi, z_max=np.pi)
      cann.init_state()

   **说明**：
   - ``num=512``：网络有512个神经元
   - ``z_min=-np.pi, z_max=np.pi``：表示角度空间，从-π到π
   - ``init_state()``：初始化所有状态变量（膜电位、活动等）

2. **定义追踪任务**

   .. code-block:: python

      from canns.task.tracking import SmoothTracking1D

      task = SmoothTracking1D(
          cann_instance=cann,
          Iext=(0., 0., np.pi, 2*np.pi),  # 刺激位置序列
          duration=(2., 20., 20.),          # 每个阶段的持续时间
          time_step=brainstate.environ.get_dt(),
      )
      task.get_data()

   **说明**：
   - 刺激在 4 个位置显示：0, 0, π, 2π
   - 第1个位置持续2秒（预热）
   - 第2、3、4个位置各持续20秒
   - 总仿真时间 = 2 + 20 + 20 + 20 = 62 秒

3. **定义仿真步骤**

   .. code-block:: python

      def run_step(t, inputs):
          """单步仿真函数"""
          cann(inputs)  # 神经网络前向传递
          return cann.r.value, cann.inp.value

   **说明**：
   - 输入接收当前时刻的刺激
   - ``cann.r.value``：神经元发放率（输出）
   - ``cann.inp.value``：输入电流

4. **编译和运行仿真**

   .. code-block:: python

      import brainstate.compile

      rs, inps = brainstate.compile.for_loop(
          run_step,
          task.run_steps,           # 时间步索引
          task.data,                # 输入数据
          pbar=brainstate.compile.ProgressBar(10)  # 进度条
      )

   **说明**：
   - ``for_loop`` 使用 JAX JIT 编译加速
   - 比普通 Python 循环快 2-5 倍
   - ``pbar`` 显示进度条（每10步更新一次）

5. **计算和绘制调谐曲线**

   .. code-block:: python

      from canns.analyzer.plotting import PlotConfigs, tuning_curve

      # 选择要分析的神经元
      neuron_indices = [128, 256, 384]

      # 创建绘图配置
      config = PlotConfigs.tuning_curve(
          num_bins=50,
          pref_stim=cann.x,
          title='一维CANN调谐曲线',
          xlabel='刺激位置 (弧度)',
          ylabel='平均发放率',
          show=False,
          save_path='tuning_curves_1d.png',
          kwargs={'linewidth': 2, 'marker': 'o', 'markersize': 4}
      )

      # 绘制调谐曲线
      tuning_curve(
          stimulus=task.Iext_sequence.squeeze(),
          firing_rates=rs,
          neuron_indices=neuron_indices,
          config=config
      )

   **说明**：
   - 调谐曲线显示神经元对不同刺激的响应
   - 每个神经元应该在某个位置有最强响应（首选刺激）
   - 其他位置的响应逐渐减弱

运行结果
--------

运行此脚本会生成：

1. **调谐曲线图** (`tuning_curves_1d.png`)

   - X轴：刺激位置（弧度）
   - Y轴：平均发放率
   - 每条曲线代表一个神经元的响应
   - 预期：每条曲线都有一个峰值（高斯形状）

2. **预期输出特征**

   - 神经元128：峰值在 ~0 弧度
   - 神经元256：峰值在 ~π 弧度
   - 神经元384：峰值在 ~2π 弧度（= -π）

   这反映了CANN的拓扑组织：相邻的神经元编码相邻的空间位置。

3. **性能指标**

   - 仿真时间：~2-5秒（含编译）
   - 内存使用：~200 MB
   - 时间步数：~6200步

关键概念
--------

**Bump活动**

CANN的特征是形成"bump"状的局部活动模式：

- 最活跃的神经元群体对应刺激位置
- Bump会随刺激移动（追踪）
- Bump的宽度由抑制范围决定

**调谐曲线**

神经元的调谐曲线反映其空间选择性：

- **锐调谐**：窄曲线，仅对特定位置有强响应
- **宽调谐**：宽曲线，对多个位置都有响应
- **高斯形**：最常见的调谐曲线形状

**拓扑组织**

CANN的关键性质：

- 相邻神经元对相邻刺激有高度相似的响应
- 这种拓扑组织是实现连续追踪的基础
- 类似于大脑中的皮层映射（cortical maps）

实验变化
--------

尝试这些修改来深化理解：

**1. 改变网络大小**

.. code-block:: python

   # 更大的网络（更精细的表示）
   cann = CANN1D(num=1024)

   # 更小的网络（更粗糙的表示）
   cann = CANN1D(num=256)

**2. 改变输入刺激**

.. code-block:: python

   # 快速移动的刺激
   task = SmoothTracking1D(
       cann_instance=cann,
       Iext=(0., 2*np.pi, 0.),
       duration=(5., 10., 5.),  # 中间10秒内快速移动
   )

**3. 改变神经元参数**

.. code-block:: python

   cann = CANN1D(
       num=512,
       tau=0.1,           # 更快的时间常数
       a=0.5,             # 改变局部激励范围
       A=1.2,             # 改变激励强度
       J0=0.5,            # 改变背景输入
   )

**4. 分析不同神经元**

.. code-block:: python

   # 分析更多神经元
   neuron_indices = [64, 128, 192, 256, 320, 384, 448]

   # 或者分析所有神经元
   neuron_indices = list(range(0, 512, 32))

相关API
-------

- :class:`~src.canns.models.basic.CANN1D` - 一维CANN模型
- :class:`~src.canns.task.tracking.SmoothTracking1D` - 平滑追踪任务
- :func:`~src.canns.analyzer.plotting.tuning_curve` - 调谐曲线绘制

更多资源
--------

- :doc:`tracking_2d` - 学习二维CANN扩展
- :doc:`../spatial_navigation/index` - 学习基于CANN的空间导航
常见问题
--------

**Q: 为什么调谐曲线不是完美的高斯形？**

A: 这是正常的。实际的调谐曲线受多个因素影响：
   - 网络有限的大小（离散化）
   - 边界效应（环形边界处）
   - 噪声和初始条件

**Q: 如何加快仿真？**

A:
   - 增加 ``dt`` （但会降低精度）
   - 减少 ``num`` （神经元数量）
   - 使用 GPU：设置环境变量 ``JAX_PLATFORM_NAME=gpu``

**Q: 为什么某些神经元的调谐曲线很平？**

A: 可能原因：
   - 网络未完全收敛 （延长 ``duration`` ）
   - 该神经元的首选刺激不在测试范围内
   - 网络参数需要调整

下一步
------

完成本教程后，推荐：

1. 尝试上面的实验变化，观察结果
2. 阅读 :doc:`tracking_2d` 学习二维版本
3. 探索 :doc:`../spatial_navigation/index` 的导航应用