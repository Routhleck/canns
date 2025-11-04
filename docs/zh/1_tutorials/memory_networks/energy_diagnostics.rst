能量诊断和网络分析
==================

场景描述
--------

你想要理解Hopfield网络的内部工作机制，通过能量函数来诊断网络的行为、识别虚假吸引子、以及优化网络参数。

你将学到
--------

- 能量函数的计算和可视化
- 吸引子盆地的分析
- 虚假吸引子的识别
- 网络调试方法

完整示例
--------

.. literalinclude:: ../../../../examples/brain_inspired/hopfield_energy_diagnostics.py
   :language: python
   :linenos:

逐步解析
--------

1. **能量函数计算**

   .. code-block:: python

      def compute_energy(state, weights, bias=None):
          """E = -0.5 * s^T W s - b^T s"""
          energy = -0.5 * np.dot(state, np.dot(weights, state))
          if bias is not None:
              energy -= np.dot(bias, state)
          return energy

2. **追踪能量随迭代变化**

   .. code-block:: python

      state = initial_state.copy()
      energies = [compute_energy(state, W)]

      for step in range(max_steps):
          state = update(state, W)
          energy = compute_energy(state, W)
          energies.append(energy)

      plt.plot(energies)
      plt.xlabel('迭代步数')
      plt.ylabel('能量')

3. **识别吸引子**

   .. code-block:: python

      def find_attractors(W, num_trials=100):
          """从随机初始化找到所有吸引子"""
          attractors = []

          for trial in range(num_trials):
              init_state = np.random.rand(len(W)) > 0.5
              final_state = retrieve_pattern(init_state, W)
              attractors.append(tuple(final_state))

          unique_attractors = list(set(attractors))
          return unique_attractors

运行结果
--------

能量递减曲线：

.. code-block:: text

   能量
   ↑  ╲
      │  ╲___
      │      ╲___
      │          ╲_____
      │                ╲ ✓ 吸引子（最小值）
      └─────────────────→ 迭代

关键概念
--------

**能量最小化**

Hopfield网络等价于求解：

.. math::

   \\min_s E(s) = -0.5 s^T W s - b^T s

**吸引子盆地**

不同初始化导向不同吸引子：

.. code-block:: text

   状态空间

   盆地1 → 吸引子1（目标模式A）
   盆地2 → 吸引子2（虚假吸引子）
   盆地3 → 吸引子3（目标模式B）

实验变化
--------

**1. 容量vs.虚假吸引子**

.. code-block:: python

   for num_patterns in range(1, 20):
       attractors = find_attractors(W)
       spurious = len(attractors) - num_patterns
       print(f"{num_patterns}个存储 → {spurious}个虚假")

**2. 能量景观可视化**

.. code-block:: python

   # 降维投影后可视化
   from sklearn.decomposition import PCA

   pca = PCA(n_components=2)
   projected_states = pca.fit_transform(all_states)

相关API
-------

- :func:`~src.canns.analyzer.energy_analysis`

下一步
------

- :doc:`hopfield_basics` - 理论基础
- :doc:`hebbian_vs_antihebbian` - 学习规则对比
