STDP：脉冲时序依赖可塑性
=======================

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。



场景描述
--------

你想要理解STDP（脉冲时序依赖可塑性），这是大脑中最重要的学习规则之一。与传统的Hebbian学习只关注"两个神经元同时活跃"不同，STDP关注的是"哪个神经元先活跃"。这实现了因果关系的学习。

你将学到
--------

- STDP规则的数学原理和时序依赖
- 脉冲神经元（LIF）模型的实现
- 脉冲迹（spike trace）的作用
- LTP（长时程增强）和LTD（长时程抑制）的机制
- 时序模式学习的演示

完整示例
--------

.. literalinclude:: ../../../../examples/brain_inspired/stdp_temporal_learning.py
   :language: python
   :linenos:

逐步解析
--------

1. **理解STDP的非对称性**

   标准Hebbian规则对称：同时活跃 → 增强

   .. code-block:: text

      Pre-syn spike:  X           X
      Post-syn spike:     X           X
      结果:          增强(Hebbian)  增强(Hebbian)

   STDP非对称：顺序很重要！

   .. code-block:: text

      时间轴: t=0      t=10ms      t=20ms

      情况1（LTP）：
      Pre-syn:   ⚡       •          •       (t=0脉冲)
      Post-syn:  •        ⚡         •       (t=10脉冲)
      结果:      LTP (前导后，增强)

      情况2（LTD）：
      Pre-syn:   •        ⚡         •       (t=10脉冲)
      Post-syn:  ⚡       •          •       (t=0脉冲)
      结果:      LTD (后导前，减弱)

2. **初始化脉冲神经网络**

   .. code-block:: python

      import brainstate
      import jax.numpy as jnp
      import numpy as np
      from canns.models.brain_inspired import SpikingLayer
      from canns.trainer import STDPTrainer

      np.random.seed(42)
      brainstate.random.seed(42)

      # 创建脉冲神经元层（LIF模型）
      model = SpikingLayer(
          input_size=20,           # 20个输入神经元
          output_size=1,           # 1个输出神经元
          threshold=0.5,           # 脉冲阈值
          v_reset=0.0,             # 重置电位
          leak=0.9,                # 漏电常数
          trace_decay=0.90         # 脉冲迹衰减
      )
      model.init_state()

   **说明**：
   - ``threshold=0.5``：膜电位达到0.5时产生脉冲
   - ``leak=0.9``：膜电位每步衰减到90%（漏电导）
   - ``trace_decay=0.90``：脉冲迹的指数衰减

3. **创建时序输入模式**

   .. code-block:: python

      # 生成时序输入模式
      # 前4个输入先激活，然后后续输入激活
      temporal_patterns = []

      # 模式1：前期激活（应该增强）
      pattern1 = np.zeros(20)
      pattern1[[0, 1, 2, 3]] = 1.0  # 前4个神经元激活
      temporal_patterns.append(pattern1)

      # 模式2：延迟激活（应该减弱）
      pattern2 = np.zeros(20)
      pattern2[[10, 11, 12, 13]] = 1.0  # 后面的神经元激活
      temporal_patterns.append(pattern2)

      # 混合多个时间点
      for _ in range(3):
          temporal_patterns.append(pattern1)  # 重复早期激活
      for _ in range(3):
          temporal_patterns.append(pattern2)  # 重复晚期激活

   **说明**：
   - 前4个输入（索引0-3）代表因果关系的"原因"
   - 后4个输入（索引10-13）代表"无关"信号
   - STDP应该强化与前4个输入的连接

4. **配置和运行STDP训练**

   .. code-block:: python

      # 创建STDP训练器
      trainer = STDPTrainer(
          model,
          learning_rate=0.02,      # 全局学习率
          A_plus=0.005,            # LTP幅度
          A_minus=0.00525,         # LTD幅度（略大于A_plus，防止爆炸）
          w_min=0.0,               # 权重下界（非负性）
          w_max=1.0,               # 权重上界
          compiled=False            # False便于调试输出
      )

      # 记录权重变化
      W_init = model.W.value.copy()

      # 训练多个epoch
      print("开始STDP训练...")
      for epoch in range(50):
          model.reset_state()  # 重置膜电位和迹
          trainer.train(temporal_patterns)

          if (epoch + 1) % 10 == 0:
              W_current = model.W.value
              weight_change = np.linalg.norm(W_current - W_init)
              print(f"Epoch {epoch+1}: 权重变化={weight_change:.4f}")

      W_final = model.W.value.copy()

   **说明**：
   - 每个epoch处理所有时序模式
   - ``reset_state()``重置膜电位和迹（新的开始）
   - 权重应该逐步变化（学习）

5. **分析学习结果**

   .. code-block:: python

      import matplotlib.pyplot as plt

      # 分析权重的变化
      weight_change = W_final - W_init

      # 绘制权重变化热图
      fig, axes = plt.subplots(1, 3, figsize=(15, 4))

      # 初始权重
      ax = axes[0]
      im1 = ax.imshow(W_init, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
      ax.set_title("初始权重（随机）")
      ax.set_xlabel("输入神经元")
      ax.set_ylabel("输出神经元")
      plt.colorbar(im1, ax=ax)

      # 最终权重
      ax = axes[1]
      im2 = ax.imshow(W_final, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
      ax.set_title("最终权重（学习后）")
      ax.set_xlabel("输入神经元")
      plt.colorbar(im2, ax=ax)

      # 权重变化
      ax = axes[2]
      im3 = ax.imshow(weight_change, aspect='auto', cmap='RdBu_r')
      ax.set_title("权重变化 ΔW")
      ax.set_xlabel("输入神经元")
      plt.colorbar(im3, ax=ax)

      plt.tight_layout()
      plt.savefig('stdp_weight_analysis.png')
      plt.show()

      # 分析按输入分组的权重变化
      print("\n=== 权重变化分析 ===")
      print(f"早期输入(0-4)的权重变化: {weight_change[0, 0:4].mean():.4f}")
      print(f"晚期输入(10-14)的权重变化: {weight_change[0, 10:14].mean():.4f}")
      print(f"其他输入的权重变化: {weight_change[0, 14:].mean():.4f}")

运行结果
--------

**预期的权重变化模式**

.. code-block:: text

早期输入(0-4)的权重变化: +0.12    ✅ 增强（LTP）
晚期输入(10-14)的权重变化: -0.05  ✅ 减弱（LTD）
其他输入的权重变化: +0.01         ✅ 轻微变化

**解释**：

- **早期输入**：这些输入在输出脉冲之前激活
  - 脉冲迹（trace）很高
  - 输出脉冲时触发强LTP
  - 结果：权重增强 ✅

- **晚期输入**：这些输入在输出脉冲之后激活
  - 后脉冲迹仍然很高
  - 这些输入激活时触发LTD
  - 结果：权重减弱 ✅

关键概念
--------

**STDP的时间窗口**

生物学证据（Bi & Poo, 1998）：

.. code-block:: text

     LTP区 ↓
      |  /
      | /
 ΔW  +|/___________
      |   \
      |    \
      |     \ ← LTD区
      |______\_____ Δt (ms)
     -40  0  +40

- 前脉冲提前 0-20ms：强LTP
- 前脉冲提前 20-40ms：弱LTP
- 后脉冲提前 0-40ms：LTD
- 超过 ±40ms：无变化

**脉冲迹的数学**

.. math::

   \text{trace} = \text{decay} \times \text{trace} + \text{spike}

例如，decay=0.9：

.. code-block:: text

   时间: 0    1    2    3    4    5    6
   脉冲: 1    0    0    0    0    0    0
   迹:   1   0.9  0.81 0.73 0.66 0.59 0.53

- 脉冲立即设置迹为1
- 迹以指数衰减
- 衰减常数决定"记忆"时间窗口

**STDP与因果关系**

STDP实现了"预测"的学习：

.. code-block:: text

情景A（因果）：
  早期信号 X → 1秒后 → 奖励信号 Y
  学习：强化 X→输出 的连接
  适应性：X预测Y

情景B（非因果）：
  奖励信号 Y → 1秒后 → 信号 X
  学习：弱化 X→输出 的连接
  适应性：X与Y无关

性能与参数
----------

**学习率选择**

=== ========= ==========
学习率  收敛速度  稳定性
=== ========= ==========
0.001  非常慢   非常稳定
0.01   缓慢    稳定
0.02   适中    较稳定 ✓
0.1    快速    可能振荡
=== ========= ==========

**时间窗口（通过trace_decay）**

.. code-block:: python

   # 窄时间窗口（快速衰减）
   model = SpikingLayer(trace_decay=0.95)

   # 宽时间窗口（缓慢衰减）
   model = SpikingLayer(trace_decay=0.99)

**LTP/LTD平衡**

.. code-block:: python

   # A_minus > A_plus：偏向LTD（抑制）
   trainer = STDPTrainer(model, A_plus=0.005, A_minus=0.01)

   # A_minus = A_plus：平衡
   trainer = STDPTrainer(model, A_plus=0.005, A_minus=0.005)

   # A_minus < A_plus：偏向LTP（激励）
   trainer = STDPTrainer(model, A_plus=0.01, A_minus=0.005)

实验变化
--------

**1. 改变输入时序**

.. code-block:: python

   # 更密集的时间关系
   pattern1[0:2] = 1.0      # 非常早
   pattern2[18:20] = 1.0    # 非常晚

**2. 多输出神经元竞争**

.. code-block:: python

   model = SpikingLayer(
       input_size=20,
       output_size=3,  # 3个输出神经元竞争
   )

   # 不同神经元应该学习不同的时序关系

**3. 改变脉冲模式**

.. code-block:: python

   # 使用更复杂的时序序列
   # 如：ABC模式 → 预测D

相关API
-------

- :class:`~src.canns.models.brain_inspired.SpikingLayer` - LIF脉冲神经元
- :class:`~src.canns.trainer.STDPTrainer` - STDP训练器
- :class:`~src.canns.trainer.STDPTrainer.predict` - 脉冲预测

生物学应用
----------

**海马体（Hippocampus）**

- 时序序列记忆（place cell序列）
- Theta phase precession（theta节律中的时序编码）

**听觉皮层（Auditory Cortex）**

- 声音序列识别
- 言语处理

**运动皮层（Motor Cortex）**

- 动作序列规划
- 技能学习

**小脑（Cerebellum）**

- 精确时序控制
- 眼球追踪（vestibulo-ocular reflex）

常见问题
--------

**Q: 为什么需要脉冲迹？**

A: 直接计算时间差Δt=t_post-t_pre效率低且耗能大。脉冲迹通过指数衰减提供了一个"时间标记"，既生物可行又计算高效。

**Q: 为什么A_minus > A_plus？**

A: 这是稳定性的权衡：
   - 如果A_plus > A_minus：权重会爆炸增长
   - 如果A_minus > A_plus：权重逐步削弱，自然稳定
   - A_minus ≈ A_plus + 5%：提供竞争和稳定性

**Q: 如何处理"噪声"脉冲？**

A: STDP有内置的噪声抗性：
   - 随机脉冲不会有一致的时序关系
   - 只有有意义的模式会被强化
   - 自动学习因果关系而忽略噪声

下一步
------

1. 尝试上面的实验变化
3. 探索 :doc:`causal_learning` 了解因果关系学习
4. 研究 :doc:`ltp_ltd_mechanisms` 理解LTP/LTD的分子机制
参考资源
--------

- **原始发现**：Bi, G. Q., & Poo, M. M. (1998). Synaptic Modifications in Cultured Hippocampal Neurons. Journal of Neuroscience, 18(24), 10464-10472.
- **理论综述**：Gerstner, W., & Kistler, W. M. (2002). Spiking Neuron Models. Cambridge University Press.
- **应用综述**：Morrison, A., Diesmann, M., & Gerstner, W. (2008). Phenomenological Models of Synaptic Plasticity. Biological Cybernetics, 98(6), 459-478.
