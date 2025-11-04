MNIST数字的联想记忆
===================

场景描述
--------

你想要看到Hopfield网络在真实数据上的表现——存储手写数字图像并从部分图像恢复完整数字。这展示了生物启发模型在实际任务中的能力。

你将学到
--------

- 图像数据的预处理
- 高维模式的存储
- 图像补全任务
- 网络容量的实际限制
- 可视化结果的方法

完整示例
--------

.. literalinclude:: ../../../../examples/brain_inspired/hopfield_train_mnist.py
   :language: python
   :linenos:

逐步解析
--------

1. **MNIST数据准备**

   .. code-block:: python

      from torchvision import datasets
      import numpy as np

      mnist = datasets.MNIST(root='./data', download=True, train=True)
      images = mnist.data.numpy() / 255  # 归一化
      images = images.reshape(len(images), -1)  # 展平为向量
      # 二值化
      binary_images = (images > 0.5).astype(float)

2. **存储MNIST模式**

   .. code-block:: python

      N = 28 * 28  # 784维
      num_images = 10  # 存储10张不同的数字

      selected_images = binary_images[:num_images]
      W = compute_hebbian_weights(selected_images)

3. **从部分图像恢复**

   .. code-block:: python

      # 遮掩上半部分
      corrupted = selected_images[0].copy()
      corrupted[:392] = 0.5 * np.ones(392)  # 遮掩

      recovered = retrieve_pattern(corrupted, W)

      # 可视化
      plot_three_images(selected_images[0], corrupted, recovered)

运行结果
--------

- 成功恢复完整数字（来自50%损坏的输入）
- 但容量有限（通常只能存储3-5个不同数字）

关键概念
--------

**维度诅咒**

高维空间的问题：

.. code-block:: python

   # MNIST：784维
   # 可存储的模式数：~108个（0.138 * 784）
   # 但实际上由于高维性，容量更低

**虚假吸引子**

在高维中虚假吸引子增加：

.. code-block:: text

   2维：虚假吸引子少
   100维：虚假吸引子明显增加
   784维：虚假吸引子很多

实验变化
--------

**1. 改变存储的数字数量**

.. code-block:: python

   for num_digits in [1, 2, 3, 5, 10]:
       success_rates[num_digits] = test_retrieval(num_digits)

**2. 不同的损坏模式**

.. code-block:: python

   # 上半部分损坏
   # 左半部分损坏
   # 随机像素损坏（盐椒噪声）

相关API
-------

- :class:`~src.canns.models.brain_inspired.HopfieldNetwork`

下一步
------

- :doc:`energy_diagnostics` - 分析为什么高维失败
- :doc:`hopfield_basics` - 理论背景
