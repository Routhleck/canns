Unsupervised Learning Algorithm Comparison
==========================================

Scenario Description
--------------------

You want to understand the advantages, disadvantages, and application scenarios of different unsupervised learning methods (Oja, Sanger, BCM, STDP), and select the algorithm most suitable for your task.

What You Will Learn
-------------------

- Mathematical foundations of different algorithms
- Computational complexity and storage requirements
- Convergence and accuracy comparison
- Practical selection criteria

Complete Example
----------------

.. literalinclude:: ../../../../examples/brain_inspired/oja_vs_sanger_comparison.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **Algorithm Comparison Table**

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

2. **Performance Evaluation**

   .. code-block:: python

      def compare_algorithms(data, labels=None):
          """Compare all algorithms"""
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

3. **Visualization Comparison**

   .. code-block:: python

      import matplotlib.pyplot as plt

      fig, axes = plt.subplots(1, 3, figsize=(15, 4))

      # Speed comparison
      speeds = {'Oja': 1.0, 'Sanger': 2.5, 'BCM': 10.0, 'STDP': 0.8}
      axes[0].bar(speeds.keys(), speeds.values())
      axes[0].set_ylabel('Training Time (Relative)')
      axes[0].set_title('Computational Speed Comparison')

      # Accuracy comparison
      accuracies = {'Oja': 0.95, 'Sanger': 0.98, 'BCM': 0.92, 'STDP': 0.96}
      axes[1].bar(accuracies.keys(), accuracies.values())
      axes[1].set_ylabel('Accuracy')
      axes[1].set_title('Learning Accuracy Comparison')

      # Memory comparison
      memories = {'Oja': 50, 'Sanger': 250, 'BCM': 500, 'STDP': 150}
      axes[2].bar(memories.keys(), memories.values())
      axes[2].set_ylabel('Memory (MB)')
      axes[2].set_title('Memory Requirement Comparison')

      plt.tight_layout()
      plt.savefig('algorithm_comparison.png')

Key Concepts
------------

**Selection Criteria**

========  ========  ========  ========  ========
Feature   Oja       Sanger    BCM       STDP
========  ========  ========  ========  ========
Function  Dim Red.  Orthog.   Adaptive  Temporal
Output    1 PC      Multi PC  1         Sequence
Speed     Fast      Fast      Slow      Very Fast
Memory    Low       Low       Medium    Low
App.      PCA       Decomp.   Features  Relations
========  ========  ========  ========  ========

**Applicable Scenarios**

- **Oja**: Fast PCA dimensionality reduction
- **Sanger**: Complete independent component analysis
- **BCM**: Receptive field development, feature learning
- **STDP**: Temporal patterns, causal learning

Experimental Variations
-----------------------

**1. Testing on Different Datasets**

.. code-block:: python

   datasets = {
       'MNIST': load_mnist(),
       'CIFAR10': load_cifar(),
       'Natural Images': load_natural_images(),
       'Speech': load_speech_data()
   }

   for dataset_name, data in datasets.items():
       results[dataset_name] = compare_algorithms(data)

**2. Varying Data Dimensionality**

.. code-block:: python

   for dim in [10, 50, 100, 500, 1000]:
       X = np.random.randn(1000, dim)
       # Compare performance at each dimension

**3. Noise Robustness**

.. code-block:: python

   for noise_level in [0, 0.1, 0.3, 0.5]:
       X_noisy = X + noise_level * np.random.randn(*X.shape)
       # Evaluate noise impact

Related API
-----------

- :class:`~src.canns.trainer.OjaTrainer`
- :class:`~src.canns.trainer.SangerTrainer`
- :class:`~src.canns.trainer.BCMTrainer`
- :class:`~src.canns.trainer.STDPTrainer`

More Resources
--------------

- :doc:`oja_pca` - Oja Rule Explained
- :doc:`sanger_orthogonal_pca` - Sanger Rule Explained
- :doc:`../receptive_fields/bcm_sliding_threshold` - BCM Explained
- :doc:`../temporal_learning/stdp_spike_timing` - STDP Explained

Frequently Asked Questions
--------------------------

**Q: Which algorithm is most "biologically plausible"?**

A: STDP and Hebbian variants (Oja, Sanger) both have biological evidence. BCM also has theoretical support but less experimental evidence.

**Q: Which one should I use?**

A: It depends on your requirements:
   - Fast + Accurate → Oja or Sanger
   - Feature Learning → BCM
   - Temporal Patterns → STDP
   - Online Learning → Oja or Sanger

Next Steps
----------

1. Test different algorithms on your specific task
2. Adjust hyperparameters to optimize performance
3. Analyze learned features
4. Combine strengths of multiple algorithms

Reference Materials
-------------------

- Haykin, S. (2009). Neural Networks and Learning Machines (3rd ed.)
- Cichocki, A., & Amari, S. (2002). Adaptive Blind Signal and Image Processing
