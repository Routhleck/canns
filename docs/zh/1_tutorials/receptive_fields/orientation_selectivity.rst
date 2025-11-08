方向选择性的发展
================

场景描述
--------

视觉皮层的神经元对特定方向的刺激有强烈反应。你想要理解这种方向选择性如何通过学习机制（如BCM）从无结构的初始连接自然产生。

你将学到
--------

- 方向选择性的神经学基础
- 模型化和学习方向偏好
- 皮层地图的形成
- 竞争和合作的角色

完整示例
--------

基于BCM的方向选择性发展：

.. code-block:: python

   from canns.trainer import BCMTrainer
   import numpy as np
   import matplotlib.pyplot as plt

   # 生成方向刺激
   def create_oriented_gratings(num_orientations=8, size=32):
       """创建不同方向的正弦光栅刺激"""
       stimuli = []
       for orientation in np.linspace(0, np.pi, num_orientations):
           grating = np.sin(np.arange(size) * np.cos(orientation) +
                           np.arange(size)[:, None] * np.sin(orientation))
           grating = (grating - grating.min()) / (grating.max() - grating.min())
           stimuli.append(grating.flatten())
       return np.array(stimuli)

   # 训练
   stimuli = create_oriented_gratings()
   trainer = BCMTrainer(input_size=1024, output_size=100)

   for epoch in range(100):
       for stimulus in stimuli:
           trainer.train(stimulus.reshape(1, -1))

   # 可视化学到的滤波器
   filters = trainer.model.W.value  # [100, 1024]

   fig, axes = plt.subplots(10, 10, figsize=(10, 10))
   for i, ax in enumerate(axes.flat):
       filter_2d = filters[i].reshape(32, 32)
       ax.imshow(filter_2d, cmap='RdBu_r')
       ax.set_xticks([])
       ax.set_yticks([])

   plt.suptitle('学到的方向选择性滤波器')
   plt.tight_layout()
   plt.savefig('orientation_filters.png')

关键概念
--------

**方向调谐曲线**

.. code-block:: text

   响应强度
       ↑
       │    ╱╲
       │   ╱  ╲
       │  ╱    ╲
       │ ╱      ╲
       └─────────→ 方向（0-180°）

   首选方向：~45°
   调谐宽度：~30°

**自组织地图**

相邻神经元学到相似的方向：

.. code-block:: text

   皮层表面：

   0°  15°  30°  45°
   15° 30°  45°  60°
   30° 45°  60°  75°
   45° 60°  75°  90°

   方向柱塔：沿着皮层深度，方向平滑变化

实验变化
--------

**1. 改变刺激复杂性**

.. code-block:: python

   # 简单：纯方向变化
   # 中等：方向+空间频率
   # 复杂：自然图像

**2. 竞争机制**

.. code-block:: python

   # 有侧向抑制
   # 无侧向抑制

   # 观察：抑制如何改善选择性

**3. 临界期**

.. code-block:: python

   # 早期训练（年幼时）
   # 晚期训练（成熟后）

   # 观察：什么时候不能改变偏好

相关API
-------

- :class:`~src.canns.trainer.BCMTrainer`

下一步
------

- :doc:`tuning_visualization` - 可视化调谐特性
- :doc:`../unsupervised_learning/algorithm_comparison` - 对比其他学习规则
