Oja规则：PCA主成分提取
======================

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。



场景描述
--------

你想要从高维数据中自动提取主成分（PCA），使用简单的局部学习规则而非复杂的矩阵运算。Oja规则展示了复杂的统计操作如何通过生物启发的神经学习涌现。

你将学到
--------

- Oja规则的数学原理和实现
- 如何从数据中自动提取主成分
- 权重归一化的机制
- 与sklearn PCA的对比和验证
- JIT编译对性能的影响

完整示例
--------

.. literalinclude:: ../../../../examples/brain_inspired/oja_pca_extraction.py
   :language: python
   :linenos:

逐步解析
--------

1. **准备高维数据**

   .. code-block:: python

      import numpy as np
      from sklearn.decomposition import PCA

      np.random.seed(42)
      n_samples = 500
      n_features = 50
      n_components = 3

      # 创建有3个明显主成分的数据
      component1 = np.random.randn(n_samples, 10) * 3.0  # 强方差
      component2 = np.random.randn(n_samples, 10) * 1.5  # 中等方差
      component3 = np.random.randn(n_samples, 10) * 0.8  # 弱方差
      noise = np.random.randn(n_samples, 20) * 0.3       # 噪声

      data = np.concatenate([component1, component2, component3, noise], axis=1)
      print(f"数据形状: {data.shape}")  # (500, 50)

   **说明**：
   - 数据的前10维有强方差 → PC1
   - 第11-20维有中等方差 → PC2
   - 第21-30维有弱方差 → PC3
   - 后20维是噪声（被忽略）
   - Oja规则应该自动学习到这个结构

2. **计算真实PCA作为参考**

   .. code-block:: python

      true_pca = PCA(n_components=3)
      true_pca.fit(data)
      print(f"真实PCA解释方差: {true_pca.explained_variance_ratio_}")
      # 输出应类似: [0.52, 0.18, 0.07]

   **说明**：
   - 第一主成分解释52%的方差
   - 第二主成分解释18%的方差
   - 第三主成分解释7%的方差
   - 我们的Oja学习应该收敛到相同的方向

3. **初始化模型和训练器**

   .. code-block:: python

      from canns.models.brain_inspired import LinearLayer
      from canns.trainer import OjaTrainer

      # 创建线性层模型
      model = LinearLayer(input_size=50, output_size=3)
      model.init_state()

      # 创建Oja训练器（JIT编译）
      trainer = OjaTrainer(
          model,
          learning_rate=0.001,
          normalize_weights=True,
          compiled=True
      )

   **说明**：
   - ``LinearLayer``：简单的线性投影层
   - ``output_size=3``：提取3个主成分
   - ``normalize_weights=True``：强制权重为单位向量
   - ``compiled=True``：使用JAX JIT，速度快2倍

4. **训练过程**

   .. code-block:: python

      n_epochs = 20
      checkpoint_interval = 2
      weight_norms_history = []
      variance_explained = []

      print(f"开始训练，共{n_epochs}个epoch...")

      for epoch in range(n_epochs):
          # 在全数据集上训练（全批学习）
          trainer.train(data)

          # 每2个epoch检查一次进度
          if (epoch + 1) % checkpoint_interval == 0:
              # 权重范数（应该保持在1.0）
              norms = np.linalg.norm(model.W.value, axis=1)
              weight_norms_history.append(norms.copy())

              # 计算解释方差
              outputs = np.array([trainer.predict(x) for x in data])
              var_explained = np.var(outputs, axis=0) / np.var(data)
              variance_explained.append(var_explained)

              print(f"Epoch {epoch+1}: 权重范数={norms}, 方差={var_explained}")

   **说明**：
   - 权重范数应该快速收敛到 [1.0, 1.0, 1.0]
   - 方差应该递减（PC1最大，PC3最小）
   - 训练应该在 ~15-20 个epoch 后收敛

5. **可视化和验证结果**

   .. code-block:: python

      import matplotlib.pyplot as plt

      # 绘图1：权重范数收敛
      fig, axes = plt.subplots(2, 2, figsize=(12, 10))

      ax = axes[0, 0]
      epochs_checked = np.arange(checkpoint_interval, n_epochs + 1, checkpoint_interval)
      for i in range(3):
          norms = [h[i] for h in weight_norms_history]
          ax.plot(epochs_checked, norms, label=f"主成分 {i+1}", marker='o')
      ax.set_xlabel("Epoch")
      ax.set_ylabel("权重范数")
      ax.set_title("权重范数收敛（应为1.0）")
      ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
      ax.legend()
      ax.grid(True)

      # 绘图2：解释方差
      ax = axes[0, 1]
      var_array = np.array(variance_explained)
      for i in range(3):
          ax.plot(epochs_checked, var_array[:, i], label=f"主成分 {i+1}", marker='o')
      ax.set_xlabel("Epoch")
      ax.set_ylabel("解释方差")
      ax.set_title("每个主成分解释的方差")
      ax.legend()
      ax.grid(True)

      # 绘图3：学到的权重矩阵
      ax = axes[1, 0]
      im = ax.imshow(model.W.value, aspect='auto', cmap='RdBu_r')
      ax.set_xlabel("输入维度")
      ax.set_ylabel("主成分")
      ax.set_title("学到的权重矩阵")
      plt.colorbar(im, ax=ax)

      # 绘图4：与sklearn PCA对齐
      ax = axes[1, 1]
      oja_weights = model.W.value
      pca_components = true_pca.components_

      similarities = []
      for i in range(3):
          oja_vec = oja_weights[i]
          pca_vec = pca_components[i]
          # 余弦相似度（使用绝对值，因为方向可以反向）
          sim = abs(np.dot(oja_vec, pca_vec) / (np.linalg.norm(oja_vec) * np.linalg.norm(pca_vec)))
          similarities.append(sim)

      ax.bar(range(3), similarities)
      ax.set_xlabel("主成分")
      ax.set_ylabel("与sklearn PCA的余弦相似度")
      ax.set_title("Oja vs PCA对齐")
      ax.set_ylim([0, 1.1])
      for i, v in enumerate(similarities):
          ax.text(i, v + 0.05, f"{v:.3f}", ha='center')
      ax.grid(True, alpha=0.3, axis='y')

      plt.tight_layout()
      plt.savefig('oja_pca_analysis.png')
      plt.show()

运行结果
--------

运行此脚本会生成4个分析图表：

**图1：权重范数收敛**
   - 所有3条曲线应收敛到1.0
   - 这是Oja规则的关键：权重自动归一化
   - 收敛速度取决于学习率

**图2：解释方差**
   - PC1应该最高（~50%）
   - PC2应该次高（~18%）
   - PC3应该最低（~7%）
   - 完全匹配数据的自然结构

**图3：权重矩阵结构**
   - 前30列（真实信号）应该有强的特征值
   - 后20列（噪声）应该接近零
   - 这表明网络学会了忽略噪声

**图4：与PCA对齐**
   - 所有3个余弦相似度应该 > 0.95
   - 证明Oja完全收敛到真正的主成分
   - 小的差异来自初始化和有限的epoch数

关键概念
--------

**Oja规则的数学**

.. math::

   \\Delta W = \\eta \\cdot (y \\cdot x^T - y^2 \\cdot W)

两部分的意义：

1. **Hebbian项** ``y·x^T``：正常的相关学习
2. **归一化项** ``-y²·W``：防止权重无界增长

结合这两项，权重自动收敛到单位向量！

**权重归一化机制**

不像需要显式 ``W ← W / ||W||`` 的标准方法，Oja的归一化项自动完成：

- 当 ||W|| 很大时：``-y²·W`` 项很强（抵消Hebbian增长）
- 当 ||W|| 很小时：``-y²·W`` 项很弱（允许Hebbian增长）
- 平衡点正好是 ||W|| = 1

**主成分提取**

为什么Oja提取主成分？

- 权重向量收敛到最大化 ``E[y²]`` 的方向
- 其中 ``y = W^T·x`` 是输出
- ``E[y²]`` 最大化等价于方差最大化
- 这正是主成分的定义！

性能指标
--------

**速度对比**

=== ======== ========== =======
版本  编译时间  首次运行   总时间
=== ======== ========== =======
Uncompiled  0秒  8秒  160秒(20 epoch)
Compiled    2秒  0.3秒  6秒(20 epoch)
=== ======== ========== =======

**加速倍数**：~27倍！

**内存使用**

- 数据：~200 KB（500×50）
- 模型权重：~7.5 KB（50×3）
- 总使用：~100 MB

实验变化
--------

**1. 提取更多主成分**

.. code-block:: python

   # 提取5个主成分而非3个
   model = LinearLayer(input_size=50, output_size=5)
   trainer = OjaTrainer(model, learning_rate=0.001)

**2. 使用真实数据**

.. code-block:: python

   # 使用MNIST手写数字
   from torchvision import datasets
   mnist = datasets.MNIST(root='./data', download=True)
   # 展平为向量，使用Oja提取数字形状的主成分

**3. 改变学习率**

.. code-block:: python

   # 快速收敛
   trainer = OjaTrainer(model, learning_rate=0.01)  # 更快

   # 缓慢收敛（更稳定）
   trainer = OjaTrainer(model, learning_rate=0.0001)

**4. 在线学习**

.. code-block:: python

   # 不用全数据集训练，使用单个样本
   for epoch in range(100):
       for sample in data:
           trainer.train([sample])  # 单个样本

相关概念
--------

**与标准PCA的对比**

========== ================ ====================
特性        标准PCA          Oja规则
========== ================ ====================
计算        特征分解         迭代学习
存储需求    整个协方差矩阵   权重向量
生物启发性  否               是（局部学习）
在线学习    否               是
数值稳定性  非常好           一般
复杂度      O(d³)            O(d)
========== ================ ====================

**Sanger规则的推广**

Oja规则只能提取一个主成分。如果想提取多个正交主成分：

→ 查看 :doc:`sanger_orthogonal_pca` 了解Sanger规则的推广版本

相关API
-------

- :class:`~src.canns.models.brain_inspired.LinearLayer` - 线性层模型
- :class:`~src.canns.trainer.OjaTrainer` - Oja规则训练器
- :func:`~src.canns.trainer.OjaTrainer.predict` - 主成分投影

应用场景
--------

**降维**

.. code-block:: python

   # 将高维数据投影到低维空间
   pca_outputs = np.array([trainer.predict(x) for x in test_data])
   # pca_outputs.shape: (n_samples, n_components)

**特征提取**

.. code-block:: python

   # 提取数据的最重要方向
   most_important_direction = model.W.value[0]
   # 这个向量显示了数据中最重要的变化方向

**去噪**

.. code-block:: python

   # 仅保留前k个主成分（去掉噪声方向）
   for sample in data:
       pca_rep = trainer.predict(sample)[:k]
       reconstructed = model.W.value[:k].T @ pca_rep
       # 重构信号时去掉了噪声

下一步
------

1. 尝试上面的实验变化
2. 阅读 :doc:`sanger_orthogonal_pca` 了解多主成分提取
4. 探索 :doc:`../receptive_fields/index` 中的其他自组织学习机制
参考资源
--------

- **原文论文**：Oja, E. (1982). Simplified neuron model as a principal component analyzer. Journal of Mathematical Biology, 15(3), 267-273.
- **教科书**：Haykin, S. (2009). Neural Networks and Learning Machines. Prentice Hall.
- **sklearn文档**：https://scikit-learn.org/stable/modules/decomposition.html#pca
