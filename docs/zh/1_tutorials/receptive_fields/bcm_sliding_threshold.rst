BCM规则：滑动阈值的受体场发展
==============================

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。



场景描述
--------

你想要理解BCM（Bienenstock, Cooper, Munro）规则如何实现自适应突触可塑性，通过动态阈值的调整来形成选择性的受体场。BCM规则展示了如何从局部信息产生复杂的特征学习。

你将学到
--------

- BCM规则的数学形式
- 滑动阈值的机制
- 受体场的形成
- M-曲线和可塑性的非线性
- 生物学证据

完整示例
--------

.. literalinclude:: ../../../../examples/brain_inspired/bcm_receptive_fields.py
   :language: python
   :linenos:

逐步解析
--------

1. **BCM规则的核心**

   .. code-block:: python

      # BCM规则：
      # ΔW = η · y · (y - θ) · x
      #
      # 其中：
      # - y：输出
      # - θ：修改阈值（滑动阈值）
      # - x：输入

      def bcm_update(x, y, w, theta, learning_rate=0.01):
          """BCM规则的单步更新"""
          delta_w = learning_rate * y * (y - theta) * x
          w_new = w + delta_w
          return w_new

2. **滑动阈值**

   .. code-block:: python

      # 阈值追踪输出的二阶矩
      # θ = E[y²]

      # 在线估计：
      theta = 0.99 * theta + 0.01 * y**2

      # 含义：
      # - 如果 y > θ：增强权重（LTP）
      # - 如果 y < θ：减弱权重（LTD）
      # - θ 自动调整以平衡学习

3. **M-曲线的S形非线性**

   .. code-block:: python

      import numpy as np
      import matplotlib.pyplot as plt

      # M-曲线：可塑性 vs 输出
      y_values = np.linspace(0, 2, 100)
      theta = 1.0  # 阈值

      learning_curve = y_values * (y_values - theta)

      plt.plot(y_values, learning_curve)
      plt.axhline(0, color='k', linestyle='-', alpha=0.3)
      plt.axvline(theta, color='r', linestyle='--', label=f'θ={theta}')
      plt.xlabel('输出 y')
      plt.ylabel('学习信号 y(y-θ)')
      plt.title('BCM的M-曲线')

4. **特征学习**

   .. code-block:: python

      # 在自然图像上训练
      from canns.trainer import BCMTrainer

      trainer = BCMTrainer(
          input_size=784,      # 28×28图像
          output_size=100,     # 100个学习的特征
          learning_rate=0.001,
          theta_learning_rate=0.01
      )

      # 训练
      for epoch in range(10):
          for batch in image_batches:
              output = trainer.train(batch)

      # 学到的特征应该是方向、颜色、纹理等

关键概念
--------

**M-曲线和可塑性**

.. code-block:: text

   学习信号
        ↑
        │        ╱╲
     LTP│       ╱  ╲
        │      ╱    ╲
      0 │─────╱──────╲─────→ 输出 y
        │    ╱ θ      ╲
     LTD│   ╱          ╲
        │  ╱            ╲

   性质：
   - S形非线性（Sigmoid）
   - 对称于阈值θ
   - 两侧有饱和区域

**稳定性分析**

BCM的收敛性：

- 权重总是稳定的（不爆炸）
- 自动归一化（通过二阶矩）
- 最终形成选择性特征

**与Hebbian的对比**

=============== ============ ==============
特性            Hebbian      BCM
=============== ============ ==============
规则            y·x          y·(y-θ)·x
稳定性          需要正规化    自动稳定
阈值            固定或不变    自适应滑动
学习            无条件        有条件
=============== ============ ==============

实验变化
--------

**1. 不同初始阈值**

.. code-block:: python

   for theta_init in [0.5, 1.0, 2.0, 4.0]:
       trainer = BCMTrainer(theta_init=theta_init)
       learned_features = trainer.train(data)
       # 观察学到的特征如何变化

**2. 学习率效应**

.. code-block:: python

   for lr in [0.0001, 0.001, 0.01, 0.1]:
       trainer = BCMTrainer(learning_rate=lr)
       convergence_time = measure_convergence(trainer)

**3. 输入统计**

.. code-block:: python

   # 高斯输入
   X_gaussian = np.random.randn(1000, 100)

   # 自然图像（有统计结构）
   X_natural = load_natural_images()

   # 比较学到的特征

相关API
-------

- :class:`~src.canns.trainer.BCMTrainer`
- :class:`~src.canns.models.brain_inspired.BCMNeuron`

生物学应用
----------

**视觉皮层**

- 方向选择性的发展
- 眼球竞争和临界期
- 暗视觉中的减敏

**听觉皮层**

- 频率调谐的形成
- 音调选择性

更多资源
--------

- :doc:`orientation_selectivity` - 方向选择性
- :doc:`tuning_visualization` - 调谐曲线可视化
- :doc:`../temporal_learning/stdp_spike_timing` - 对比STDP

常见问题
--------

**Q: BCM与STDP的区别？**

A: BCM是速率编码（频率），STDP是脉冲时序编码（精确时间）。BCM用于特征学习，STDP用于时序学习。

**Q: 为什么需要M-曲线？**

A: S形非线性确保：
   1. 稳定性（不爆炸）
   2. 双向可塑性（LTP和LTD）
   3. 竞争（强输出更强，弱输出更弱）

**Q: 如何选择学习率？**

A: BCM对学习率敏感。通常：
   - 权重学习率：0.001-0.01
   - 阈值学习率：0.01-0.1

下一步
------

1. 在真实图像上训练并分析特征
2. 比较BCM和Oja的学到特征
3. 研究临界期的影响
4. 阅读生物学证据
