无监督学习算法对比
==================

场景描述
--------

你想要理解不同的无监督学习方法（Oja、Sanger、BCM、STDP）的优缺点和应用场景，选择最适合你的任务的算法。

你将学到
--------

- 不同算法的数学基础
- 计算复杂度和存储需求
- 收敛性和精度对比
- 实际选择标准

完整示例
--------

.. literalinclude:: ../../../../examples/brain_inspired/oja_vs_sanger_comparison.py
   :language: python
   :linenos:

逐步解析
--------

1. **算法对比表**

   .. code-block:: python

      algorithms = {
          'Oja': {
              'components': 1,
              'complexity': 'O(d)',
              'learning_rate': 0.001,
              'convergence_speed': 'fast'
          },
          'Sanger': {
              'components': 'all',
              'complexity': 'O(d*k)',
              'learning_rate': 0.001,
              'convergence_speed': 'medium'
          },
          'BCM': {
              'components': 'adaptive',
              'complexity': 'O(d²)',
              'learning_rate': 0.0001,
              'convergence_speed': 'slow'
          },
          'STDP': {
              'components': 'temporal',
              'complexity': 'O(d*t)',
              'learning_rate': 0.01,
              'convergence_speed': 'fast'
          }
      }

2. **性能评估**

   .. code-block:: python

      def compare_algorithms(data, labels=None):
          """比较所有算法"""
          results = {}

          # Oja
          model_oja = OjaTrainer(input_size=50, output_size=1)
          results['Oja'] = {
              'variance': model_oja.train(data),
              'time': measure_training_time(model_oja),
              'memory': measure_memory(model_oja)
          }

          # Sanger
          model_sanger = SangerTrainer(input_size=50, output_size=5)
          results['Sanger'] = {
              'variance': model_sanger.train(data),
              'time': measure_training_time(model_sanger),
              'memory': measure_memory(model_sanger)
          }

          # BCM
          model_bcm = BCMTrainer(input_size=50, output_size=1)
          results['BCM'] = {
              'variance': model_bcm.train(data),
              'time': measure_training_time(model_bcm),
              'memory': measure_memory(model_bcm)
          }

          return results

3. **可视化对比**

   .. code-block:: python

      import matplotlib.pyplot as plt

      fig, axes = plt.subplots(1, 3, figsize=(15, 4))

      # 速度对比
      speeds = {'Oja': 1.0, 'Sanger': 2.5, 'BCM': 10.0, 'STDP': 0.8}
      axes[0].bar(speeds.keys(), speeds.values())
      axes[0].set_ylabel('训练时间（相对）')
      axes[0].set_title('计算速度对比')

      # 精度对比
      accuracies = {'Oja': 0.95, 'Sanger': 0.98, 'BCM': 0.92, 'STDP': 0.96}
      axes[1].bar(accuracies.keys(), accuracies.values())
      axes[1].set_ylabel('精度')
      axes[1].set_title('学习精度对比')

      # 内存对比
      memories = {'Oja': 50, 'Sanger': 250, 'BCM': 500, 'STDP': 150}
      axes[2].bar(memories.keys(), memories.values())
      axes[2].set_ylabel('内存（MB）')
      axes[2].set_title('内存需求对比')

      plt.tight_layout()
      plt.savefig('algorithm_comparison.png')

关键概念
--------

**选择标准**

========  ========  ========  ========  ========
特征      Oja      Sanger     BCM       STDP
========  ========  ========  ========  ========
维数      降维      正交化    适应性    时序
输出      1个PC    多个PC    1个      序列
速度      快        快        慢        很快
内存      低        低        中        低
应用      PCA      信号分解  特征      学习关系
========  ========  ========  ========  ========

**适用场景**

- **Oja**：快速PCA降维
- **Sanger**：完整的独立成分分析
- **BCM**：受体场发展，特征学习
- **STDP**：时序模式，因果学习

实验变化
--------

**1. 在不同数据集上测试**

.. code-block:: python

   datasets = {
       'MNIST': load_mnist(),
       'CIFAR10': load_cifar(),
       'Natural Images': load_natural_images(),
       'Speech': load_speech_data()
   }

   for dataset_name, data in datasets.items():
       results[dataset_name] = compare_algorithms(data)

**2. 改变数据维度**

.. code-block:: python

   for dim in [10, 50, 100, 500, 1000]:
       X = np.random.randn(1000, dim)
       # 比较每个维度下的性能

**3. 噪声鲁棒性**

.. code-block:: python

   for noise_level in [0, 0.1, 0.3, 0.5]:
       X_noisy = X + noise_level * np.random.randn(*X.shape)
       # 评估噪声影响

相关API
-------

- :class:`~src.canns.trainer.OjaTrainer`
- :class:`~src.canns.trainer.SangerTrainer`
- :class:`~src.canns.trainer.BCMTrainer`
- :class:`~src.canns.trainer.STDPTrainer`

更多资源
--------

- :doc:`oja_pca` - Oja规则详解
- :doc:`sanger_orthogonal_pca` - Sanger规则详解
- :doc:`../receptive_fields/bcm_sliding_threshold` - BCM详解
- :doc:`../temporal_learning/stdp_spike_timing` - STDP详解

常见问题
--------

**Q: 哪个算法最"生物可行"？**

A: STDP和Hebbian变体（Oja、Sanger）都有生物证据。BCM也有理论支持但实验证据较少。

**Q: 应该用哪个？**

A: 取决于你的需求：
   - 快速+精确 → Oja或Sanger
   - 特征学习 → BCM
   - 时序模式 → STDP
   - 在线学习 → Oja或Sanger

下一步
------

1. 在你的特定任务上测试不同算法
2. 调整超参数优化性能
3. 分析学到的特征
4. 结合多个算法的优点

参考资源
--------

- Haykin, S. (2009). Neural Networks and Learning Machines (3rd ed.)
- Cichocki, A., & Amari, S. (2002). Adaptive Blind Signal and Image Processing
