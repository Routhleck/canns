Hebbian与Anti-Hebbian学习
==========================

场景描述
--------

你想要比较两种对立的学习规则——Hebbian（"同时活跃的神经元连接增强"）和Anti-Hebbian（"同时活跃的神经元连接减弱"），理解它们如何产生完全不同的网络行为和功能。

你将学到
--------

- Hebbian和Anti-Hebbian规则的数学形式
- 它们产生的不同网络行为
- 生物学证据和应用场景
- 学习规则与网络功能的关系

完整示例
--------

.. literalinclude:: ../../../../examples/brain_inspired/hopfield_hebbian_vs_antihebbian.py
   :language: python
   :linenos:

逐步解析
--------

1. **Hebbian学习**

   .. code-block:: python

      # 规则：同时活跃 → 增强连接
      # ΔW = η · y · x^T

      def hebbian_learning(activity, input_signal, learning_rate=0.01):
          """Hebbian学习规则"""
          delta_w = learning_rate * np.outer(activity, input_signal)
          return delta_w

      # 用途：
      # - 关联记忆（联想）
      # - 模式存储
      # - 输出驱动的学习

2. **Anti-Hebbian学习**

   .. code-block:: python

      # 规则：同时活跃 → 减弱连接
      # ΔW = -η · y · x^T

      def antihebbian_learning(activity, input_signal, learning_rate=0.01):
          """Anti-Hebbian学习规则"""
          delta_w = -learning_rate * np.outer(activity, input_signal)
          return delta_w

      # 用途：
      # - 离散代码学习
      # - 竞争网络
      # - 分解/独立成分分析

3. **学习目标的对比**

   .. code-block:: python

      # Hebbian：提取最大方差的方向（PCA）
      # Anti-Hebbian：提取互相独立的方向（ICA）

      # 在Oja规则中的具体化
      oja_rule = "ΔW = η · (y · x^T - y² · W)"
      #                 ↑            ↑
      #            Hebbian     归一化（本质上是Anti-Hebbian）

关键概念
--------

**Hebbian学习的函数**

.. code-block:: text

   关联学习：
   刺激 X（红苹果） ↔ 输出 Y（"红"）
   多次配对 → 强化 X→Y 连接

   未来：看到红色 → 自动"想到"红色概念

**Anti-Hebbian学习的函数**

.. code-block:: text

   竞争学习：
   输入1（高活动） → 抑制输入2
   输入1和输入2分离 → 每个神经元编码特定特征

   结果：稀疏、分散的表示

**生物学证据**

Hebbian：
- 海马体突触强化
- 长期增强（LTP）
- 活动依赖的树突棘形成

Anti-Hebbian：
- 某些小脑突触
- 长期抑制（LTD）
- 竞争性突触剪枝

实验变化
--------

**1. 对比学习速度**

.. code-block:: python

   hebbian_network = train_with_hebbian(data, epochs=100)
   antihebbian_network = train_with_antihebbian(data, epochs=100)

   plt.plot(hebbian_network.losses, label='Hebbian')
   plt.plot(antihebbian_network.losses, label='Anti-Hebbian')

**2. 对比表示质量**

.. code-block:: python

   # Hebbian：重建误差（PCA）
   reconstruction_error_h = np.mean((data - reconstructed_h)**2)

   # Anti-Hebbian：独立性度量（ICA）
   independence_ica = compute_independence_measure(unmixed)

**3. 网络容量对比**

.. code-block:: python

   for num_patterns in [5, 10, 15, 20]:
       h_capacity = test_capacity(hebbian_network, num_patterns)
       ah_capacity = test_capacity(antihebbian_network, num_patterns)

关键概念
--------

**权重矩阵的性质**

Hebbian：
- W 是对称的（W^T = W）
- 特征值都是实数
- 能量函数定义良好

Anti-Hebbian：
- W 通常不对称
- 可能有复特征值
- 可能发生振荡

**学习动力学**

.. code-block:: text

   Hebbian：
   ┌─────────────┐
   │  正反馈     │  → 权重无界增长
   │  强化强信号 │
   └─────────────┘
   需要正规化：W ← W / ||W||

   Anti-Hebbian：
   ┌──────────────┐
   │  负反馈      │  → 自动稳定
   │  抑制强信号  │
   └──────────────┘
   自然有界

相关API
-------

- :class:`~src.canns.trainer.HebbianTrainer`
- :class:`~src.canns.trainer.AntiHebbianTrainer`
- :class:`~src.canns.trainer.OjaTrainer` （Hebbian+正规化）

生物学应用
----------

**Hebbian在学习中**

- 经验依赖的突触强化
- 习惯形成
- 技能学习

**Anti-Hebbian在竞争中**

- 神经元竞争不同特征
- 方向选择性形成
- 特征映射发展

更多资源
--------

- :doc:`hopfield_basics` - Hebbian应用
- :doc:`../unsupervised_learning/oja_pca` - Hebbian + 正规化
常见问题
--------

**Q: 哪种学习规则更好？**

A: 取决于任务：
   - 联想记忆 → Hebbian
   - 特征分离 → Anti-Hebbian
   - 降维 → Hebbian + 正规化

**Q: 大脑使用两种吗？**

A: 是的！不同脑区使用不同规则，甚至同一神经元的不同突触可能使用不同规则。

**Q: 如何选择学习率？**

A: Hebbian需要小学习率+正规化。Anti-Hebbian自然稳定，可以用较大的学习率。

下一步
------

1. 在相同数据上比较两种学习器
2. 分析学习后的权重矩阵
3. 测试在新数据上的泛化能力
4. 探索混合规则
