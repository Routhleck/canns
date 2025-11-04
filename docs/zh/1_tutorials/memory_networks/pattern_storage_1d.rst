一维模式存储和检索
==================

场景描述
--------

你想要理解一维空间中的模式存储机制，这是在实际神经科学实验中最容易实现和测量的配置。

你将学到
--------

- 一维模式的Hebbian存储
- 容量极限和信息论边界
- 检索性能与网络大小的关系
- 实验验证方法

完整示例
--------

.. literalinclude:: ../../../../examples/brain_inspired/hopfield_train_1d.py
   :language: python
   :linenos:

逐步解析
--------

一维模式存储的关键步骤：

1. **生成一维模式**

   .. code-block:: python

      N = 100  # 网络大小
      patterns = [np.random.randn(N) > 0 for _ in range(5)]  # 5个随机模式

2. **Hebbian权重计算**

   .. code-block:: python

      W = np.zeros((N, N))
      for pattern in patterns:
          W += np.outer(pattern, pattern)
      W = W / N
      np.fill_diagonal(W, 0)

3. **测试检索**

   .. code-block:: python

      for pattern_idx, pattern in enumerate(patterns):
          # 创建损坏版本（30%损坏）
          corrupted = pattern.copy()
          corrupted[:30] = 1 - corrupted[:30]

          retrieved = retrieve(corrupted, W)
          accuracy = np.mean(retrieved == pattern)
          print(f"模式{pattern_idx}: 精度 {accuracy:.1%}")

运行结果
--------

一维模式存储的性能曲线：

.. code-block:: text

   检索精度
   ↑
   100%│     ╱──────
       │    ╱
    75%│   ╱
       │  ╱
    50%│ ╱
       │╱
     0%└────────────────→ 存储模式数
       0   5   10   15

关键概念
--------

**信息容量**

一维Hopfield网络的容量：

.. math::

   C \\approx 0.138 \\cdot N

例如：100个神经元可存储~14个模式

**检索复杂度**

异步更新的期望步数：

.. math::

   E[步数] \\propto \\log(N)

实验变化
--------

**1. 改变网络大小**

.. code-block:: python

   for N in [50, 100, 200, 500]:
       patterns = generate_patterns(N, num_patterns=int(0.1*N))
       network = HopfieldNetwork(N)
       success_rate = test_retrieval(network, patterns)

**2. 改变损坏程度**

.. code-block:: python

   for corruption in [0.1, 0.3, 0.5, 0.7]:
       accuracy = test_with_corruption(corruption)

相关API
-------

- :class:`~src.canns.models.brain_inspired.HopfieldNetwork`
- :class:`~src.canns.trainer.HebbianTrainer`

常见问题
--------

**Q: 为什么容量有限？**

A: 因为权重矩阵有限（N²个权重），存储太多模式会导致干扰。

下一步
------

- :doc:`mnist_memory` - 学习实际数据
- :doc:`energy_diagnostics` - 分析能量
