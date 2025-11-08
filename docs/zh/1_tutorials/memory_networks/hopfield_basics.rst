Hopfield网络基础：联想记忆
===========================

场景描述
--------

你想要理解如何使用神经网络来存储和检索记忆。Hopfield网络是实现关联记忆（associative memory）的最简单而优雅的模型：输入一个不完整或损坏的模式，网络能够恢复出完整的存储模式。

你将学到
--------

- Hopfield网络的工作原理
- 能量函数和吸引子动力学
- 模式存储和检索的机制
- 容量限制和干扰
- 生物学实现的可能性

完整示例
--------

基于Hopfield网络的模式存储和检索：

.. code-block:: python

   import numpy as np
   from canns.models.brain_inspired import HopfieldNetwork

   # 创建Hopfield网络
   network = HopfieldNetwork(num_neurons=100)
   network.init_state()

   # 存储模式
   patterns = [
       np.random.randn(100) > 0,  # 模式1：随机二进制
       np.random.randn(100) > 0,  # 模式2
       np.random.randn(100) > 0,  # 模式3
   ]

   # 使用Hebbian规则存储
   for pattern in patterns:
       network.store(pattern)

   # 测试：部分输入→完整输出
   corrupted = patterns[0].copy()
   corrupted[:10] = np.random.rand(10) > 0.5  # 损坏前10个位

   retrieved = network.retrieve(corrupted, steps=100)

   # 检查检索精度
   accuracy = np.mean(retrieved == patterns[0])
   print(f"检索精度: {accuracy:.1%}")

逐步解析
--------

1. **Hopfield网络的结构**

   .. code-block:: python

      # 完全连接的对称网络
      # 神经元总数：N
      # 连接权重：W[i,j] = W[j,i]  (对称)
      # 无自连接：W[i,i] = 0

      import numpy as np

      N = 100  # 神经元数
      W = np.random.randn(N, N)
      W = (W + W.T) / 2  # 对称化
      np.fill_diagonal(W, 0)  # 移除自连接

   **网络拓扑**：

   .. code-block:: text

      ○─────○─────○
      │╲   │╱  │
      │ ╲ ╱ ╲ │
      │  ×   ╲│
      │ ╱ ╲  ╱
      │╱   ╲│
      ○─────○─────○

      - 所有神经元连接到所有其他神经元
      - 权重矩阵：N×N对称矩阵
      - 是递归网络（recurrent）

2. **Hebbian学习规则**

   .. code-block:: python

      # 模式存储：使用Hebbian规则设置权重
      # W = (1/N) * Σ ξ ξ^T  （外积）

      def store_patterns_hebbian(patterns):
          N = len(patterns[0])
          W = np.zeros((N, N))

          for pattern in patterns:
              # 外积：ξ ξ^T
              W += np.outer(pattern, pattern)

          # 归一化和移除自连接
          W = W / N
          np.fill_diagonal(W, 0)

          return W

   **权重意义**：
   - W[i,j] > 0：如果神经元i和j通常一起激活
   - W[i,j] < 0：如果神经元i和j通常相反激活
   - 权重强度反映共激活的频率

3. **能量函数和吸引子**

   .. code-block:: python

      # Hopfield网络的能量函数
      def compute_energy(state, W, b=None):
          """计算网络的能量"""
          # E = -0.5 * s^T W s - b^T s
          energy = -0.5 * np.dot(state, np.dot(W, state))
          if b is not None:
              energy -= np.dot(b, state)
          return energy

      # 关键性质：同步更新时能量单调递减
      # 网络最终收敛到局部最小值（吸引子）

   **能量景观**：

   .. code-block:: text

      能量
      ↑
      │     ╱╲      ╱╲
      │    ╱  ╲    ╱  ╲      吸引子（能量谷）
      │___╱____╱╲__╱____╲____
      │         ╲  ╱
      │          ╲╱
      └─────────────────────→ 神经元状态空间

4. **模式检索过程**

   .. code-block:: python

      def retrieve_pattern(initial_state, W, max_steps=100):
          """从部分输入恢复完整模式"""
          state = initial_state.copy()

          for step in range(max_steps):
              old_state = state.copy()

              # 同步更新：所有神经元同时更新
              activation = np.dot(W, state)
              state = (activation > 0).astype(float)

              # 检查收敛
              if np.array_equal(state, old_state):
                  print(f"收敛于第 {step} 步")
                  break

          return state

   **收敛保证**：
   - Hopfield网络总是收敛
   - 能量函数单调递减
   - 最终状态是吸引子（存储的模式或虚假吸引子）

运行结果
--------

一个成功的检索过程：

.. code-block:: text

   输入（80%损坏）：███░░░░░░░░░░░░░░░░
   第1次迭代后：    ████░░░░░░░░░░░░░░░░
   第5次迭代后：    ████████░░░░░░░░░░░░
   第10次迭代后：   ████████████░░░░░░░░
   第20次迭代后：   ████████████████████ ✓ 完整恢复

关键概念
--------

**吸引子动力学**

Hopfield网络的核心是吸引子：

.. code-block:: text

   吸引子盆地：
   ┌──────────────────────┐
   │   吸引子盆地1        │
   │  ╱───────╲          │
   │ ╱ 存储的  ╲         │
   │╱  模式1   ╲        │
   │           ╲       │
   │  部分模式  → 完整模式1
   │
   └──────────────────────┘

   ┌──────────────────────┐
   │   吸引子盆地2        │
   │  ╱───────╲          │
   │ ╱  存储的  ╲        │
   │╱   模式2   ╲       │
   └──────────────────────┘

**容量和干扰**

Hopfield网络能存储多少模式？

.. code-block:: python

   # 理论容量：α = C/N ≈ 0.138
   # 其中 C 是存储的模式数，N 是神经元数

   N = 100
   max_patterns = int(0.138 * N)  # ~14个模式

   # 超过容量：
   # - 检索错误增加
   # - 虚假吸引子出现
   # - 存储的模式可能无法恢复

**虚假吸引子**

网络可能收敛到非存储的吸引子：

.. code-block:: text

   虚假吸引子示例（存储[1,0] 和 [0,1]）：
   - [1,1]：可能是虚假吸引子（两个存储模式的混合）
   - [0,0]：可能是虚假吸引子
   - 只有 [1,0] 和 [0,1] 是所需的吸引子

实验变化
--------

**1. 改变存储模式数量**

.. code-block:: python

   for num_patterns in [2, 5, 10, 15, 20]:
       patterns = [np.random.randn(100) > 0 for _ in range(num_patterns)]
       network = HopfieldNetwork(100)
       for pattern in patterns:
           network.store(pattern)

       # 测试检索成功率
       success_rate = test_retrieval(network, patterns)
       print(f"{num_patterns} 个模式: 成功率 {success_rate:.1%}")

**2. 分析损坏程度的影响**

.. code-block:: python

   for corruption_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
       corrupted = patterns[0].copy()
       mask = np.random.rand(len(corrupted)) < corruption_level
       corrupted[mask] = 1 - corrupted[mask]

       retrieved = network.retrieve(corrupted)
       accuracy = np.mean(retrieved == patterns[0])
       print(f"损坏{corruption_level:.0%}: 检索精度 {accuracy:.1%}")

**3. 可视化能量景观**

.. code-block:: python

   # 在二维投影上可视化能量
   import matplotlib.pyplot as plt

   x = np.linspace(-1, 1, 100)
   y = np.linspace(-1, 1, 100)
   X, Y = np.meshgrid(x, y)
   Z = np.zeros_like(X)

   for i in range(100):
       for j in range(100):
           state = np.zeros(N)
           state[0] = X[i, j]
           state[1] = Y[i, j]
           Z[i, j] = compute_energy(state, W)

   plt.contourf(X, Y, Z, levels=20, cmap='viridis')
   plt.colorbar(label='能量')
   plt.title('Hopfield网络的能量景观')

相关API
-------

- :class:`~src.canns.models.brain_inspired.HopfieldNetwork` - Hopfield网络
- :class:`~src.canns.trainer.HebbianTrainer` - Hebbian学习器
- :func:`~src.canns.analyzer.spatial.compute_energy` - 能量计算

生物学应用
----------

**大脑中的联想记忆**

- **嗅觉皮层**：气味识别
- **海马体CA3**：背景关联和模式完成
- **前额皮层**：工作记忆维护

**Hopfield的优势**

- 简单的学习规则（Hebbian）
- 自动纠错（部分→完整）
- 生物可实现
- 内容寻址存储

更多资源
--------

- :doc:`pattern_storage_1d` - 一维模式存储
- :doc:`mnist_memory` - MNIST数字的记忆
- :doc:`energy_diagnostics` - 能量分析工具
- :doc:`hebbian_vs_antihebbian` - 不同学习规则的对比
常见问题
--------

**Q: 为什么Hopfield网络总是收敛？**

A: 因为能量函数在每次更新时单调递减（或保持不变）。最终状态必定是局部最小值，即吸引子。这保证了网络的稳定性。

**Q: 虚假吸引子怎么办？**

A: 虚假吸引子是Hopfield网络的基本限制。通过以下方法可以减少：
   - 减少存储的模式数量
   - 使用更好的学习规则（如投影规则）
   - 使用模式的正交化处理

**Q: Hopfield网络能用于什么实际应用？**

A:
   - 错误纠正编码
   - 图像去噪
   - 组合优化（可转化为模式）
   - 关键词匹配
   - 医学诊断系统

下一步
------

1. 尝试存储更多模式并观察检索性能
2. 分析虚假吸引子的性质
3. 比较同步和异步更新的区别
4. 阅读 :doc:`hebbian_vs_antihebbian` 了解不同学习规则
