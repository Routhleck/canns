实验室
======

本目录用于存放实验性的扩展和用户自定义内容。

关于此目录
----------

``experiments/`` 目录专门为用户提供一个空间来：

- 测试新的想法和算法
- 开发自定义的分析工具
- 存放正在进行中的研究项目
- 实验性的模型变体

这个目录不包含在 CANNs 核心代码中，但会在文档中被引用和说明。

使用建议
--------

1. **组织结构**

   建议按项目或主题创建子目录：

   .. code-block:: text

      experiments/
      ├── custom_learning_rules/
      │   ├── my_stdp_variant.py
      │   └── README.md
      ├── network_architectures/
      │   ├── modular_cann.py
      │   └── experiments.ipynb
      └── analysis_tools/
          └── custom_analyzer.py

2. **文档化**

   为每个实验项目添加 README 说明：

   - 实验目的
   - 使用的方法
   - 预期结果
   - 当前状态

3. **版本控制**

   可以将此目录加入 ``.gitignore`` 保持私有，或者选择性地提交到仓库。

示例：自定义学习规则
--------------------

.. code-block:: python

   # experiments/custom_learning_rules/my_rule.py
   from canns.trainer import Trainer
   import jax.numpy as jnp

   class MyCustomTrainer(Trainer):
       """我的自定义学习规则"""

       def __init__(self, model, learning_rate=0.01):
           super().__init__(model=model)
           self.learning_rate = learning_rate

       def train(self, train_data):
           # 实现你的学习规则
           for pattern in train_data:
               # ... 自定义逻辑
               pass

分享你的实验
------------

如果你的实验取得了有趣的结果，欢迎：

1. 在 GitHub Issues 中分享
2. 提交 Pull Request 将其纳入核心库
3. 在社区讨论中交流想法

相关资源
--------

- :doc:`../1_tutorials/index` - 学习 CANNs 的基本用法
- `GitHub Issues <https://github.com/your-org/canns/issues>`_ - 讨论新想法
开始实验
--------

创建你的第一个实验项目：

.. code-block:: bash

   cd docs/zh/experiments/
   mkdir my_experiment
   cd my_experiment
   touch experiment.py README.md

然后在 ``experiment.py`` 中开始编写你的代码！
