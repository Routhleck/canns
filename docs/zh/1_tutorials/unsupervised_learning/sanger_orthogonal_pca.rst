Sanger规则：正交PCA多主成分提取
================================

场景描述
--------

你想要从高维数据中提取多个正交的主成分，超越Oja规则只能提取一个主成分的限制。Sanger规则（广义Hebbian算法）通过一个简单的局部学习规则实现多主成分的同时提取。

你将学到
--------

- Sanger规则的数学推导
- 与Oja规则的关系
- 多个正交主成分的提取
- 与标准PCA的收敛性
- 在线学习的应用

完整示例
--------

.. literalinclude:: ../../../../examples/brain_inspired/oja_vs_sanger_comparison.py
   :language: python
   :linenos:

逐步解析
--------

1. **Sanger规则的数学**

   .. code-block:: python

      # Sanger规则（广义Hebbian算法）
      # ΔW = η · (y · x^T - y · (W^T · y) · 1^T)
      #
      # 其中：
      # - y = W^T · x  （输出）
      # - W^T · y     （自反馈）
      # - 1^T         （"Gram-Schmidt"项）

      def sanger_learning(X, W, learning_rate=0.001, epochs=20):
          """Sanger规则实现"""
          N_features, N_components = W.shape

          for epoch in range(epochs):
              for x in X:
                  y = np.dot(W.T, x)  # 输出

                  # Sanger规则的关键：
                  lower_triangular = np.tril(np.dot(y.reshape(-1, 1), y.reshape(1, -1)))
                  delta_w = learning_rate * np.dot(x.reshape(-1, 1), y.reshape(1, -1))
                  delta_w -= learning_rate * np.dot(W, lower_triangular)

                  W += delta_w

          return W

2. **与Oja规则的对比**

   .. code-block:: python

      # Oja规则（单主成分）：
      oja_rule = "ΔW = η · (y · x^T - y² · W)"

      # Sanger规则（多主成分）：
      sanger_rule = "ΔW[i,:] = η · (y[i] · x^T - y[i] · Σ_j<i y[j] · W[j,:])"

      # 含义：
      # - 第一个主成分学习与Oja相同
      # - 后续主成分正交于前面的主成分

3. **测试正交性**

   .. code-block:: python

      # 检查学到的主成分是否正交
      W_learned = train_with_sanger(data)

      for i in range(n_components):
          for j in range(i+1, n_components):
              dot_product = np.dot(W_learned[i], W_learned[j])
              print(f"成分{i}和{j}的内积: {dot_product:.6f}")

   **预期**：内积应接近0（完全正交）

4. **与PCA的收敛**

   .. code-block:: python

      from sklearn.decomposition import PCA

      # 标准PCA
      pca = PCA(n_components=3)
      pca_components = pca.fit_transform(data)

      # Sanger规则学习
      sanger_components = train_with_sanger(data, n_components=3)

      # 比较：应该学到相同的主成分方向
      for i in range(3):
          similarity = np.abs(np.dot(sanger[i], pca.components_[i]))
          print(f"PC{i} 相似度: {similarity:.4f}")  # 应该 > 0.95

运行结果
--------

Sanger规则的学习曲线：

.. code-block:: text

   方差解释率
   ↑
   │  PC1: ════════════════ 52%
   │  PC2: ════════ 18%
   │  PC3: ═══ 7%
   │
   └──────────────────────→ 成分

   收敛时间：10-20个epoch

关键概念
--------

**Gram-Schmidt正交化**

Sanger规则在线实现了Gram-Schmidt过程：

.. math::

   w_i = x_i - \\sum_{j<i} (w_j^T x_i) w_j

对应于：

.. code-block:: text

   新成分 = 原始方向 - 前面成分的投影

**信息论解释**

Sanger规则最大化：

.. math::

   I(y; x) = \\sum_{i=1}^{n} I(y_i; x)

（输出与输入的互信息）

同时满足约束：

.. math::

   y_i \\perp y_j, \\quad i \\neq j

实验变化
--------

**1. 改变主成分数量**

.. code-block:: python

   for n_pc in [1, 2, 3, 5, 10]:
       W = train_sanger(data, n_components=n_pc)
       variance_exp = compute_variance_explained(W, data)
       print(f"主成分数{n_pc}: 方差 {variance_exp:.1%}")

**2. 学习率的影响**

.. code-block:: python

   for lr in [0.0001, 0.001, 0.01, 0.1]:
       W = train_sanger(data, learning_rate=lr)
       convergence_time = measure_convergence(W, target=pca.components_)

**3. 在线学习能力**

.. code-block:: python

   # 流数据上的在线Sanger
   W = np.random.randn(n_features, n_pc) * 0.1

   for batch in data_stream:
       for x in batch:
           y = np.dot(W.T, x)
           W += learning_rate * (np.dot(x.reshape(-1,1), y.reshape(1,-1)) -
                                 np.dot(W, np.tril(np.outer(y, y))))

相关API
-------

- :class:`~src.canns.trainer.SangerTrainer` - Sanger规则训练器
- :class:`~src.canns.trainer.OjaTrainer` - Oja规则（对比）

生物学应用
----------

**多通道感觉处理**

- 视觉：多特征（颜色、方向、空间频率）的同时提取
- 听觉：多频率成分的分离
- 嗅觉：多分子特征的识别

**神经编码的多样性**

皮层神经元似乎组织为多个独立的特征提取器，这可能通过Sanger规则或类似机制实现。

更多资源
--------

- :doc:`oja_pca` - 单主成分提取（基础）
- :doc:`algorithm_comparison` - 与其他方法的比较
常见问题
--------

**Q: Sanger何时优于标准PCA？**

A: 在线场景和计算受限时。Sanger只需计算样本数量的内存，而PCA需要存储整个协方差矩阵（N²内存）。

**Q: 为什么需要Gram-Schmidt项？**

A: 保证提取的主成分正交。没有它，后续成分会重复提取第一主成分的方向。

**Q: 收敛到真PCA需要多久？**

A: 通常20-50个epoch，取决于数据维度和学习率。在线学习时，需要多次遍历数据。

下一步
------

1. 比较Oja和Sanger在相同数据上的性能
2. 测试在线学习能力
3. 应用于真实数据（图像、音频等）
4. 研究非线性扩展（核PCA）

参考资源
--------

- Sanger, T. D. (1989). Optimal unsupervised learning in a single-layer linear feedforward neural network. Neural Networks
- Oja, E., & Karhunen, J. (1985). On stochastic approximation of the eigenvectors and eigenvalues of the expectation of a random matrix. Journal of Mathematical Analysis and Applications
