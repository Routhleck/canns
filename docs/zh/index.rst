CANNs 文档
===========

欢迎来到 CANNs (连续吸引子神经网络) 文档！

.. toctree::
   :maxdepth: 2
   :caption: 内容目录:

   notebooks/index
   ../autoapi/index

介绍
----

CANNs (Continuous Attractor Neural Networks) 是一个专门用于连续吸引子神经网络建模和分析的 Python 库。
本库提供了丰富的工具来构建、训练和分析各种类型的连续吸引子神经网络模型。

主要特性
--------

* 🧠 **多种模型支持**: 支持1D和2D CANN模型，以及层次化网络架构
* 📊 **可视化工具**: 提供丰富的可视化函数用于分析神经网络动态
* 🔬 **分析工具**: 包含spike train分析、发放率计算等实用工具
* 📚 **教程和示例**: 详细的Jupyter notebook教程
* 🎯 **任务支持**: 内置路径积分、跟踪等神经计算任务

快速开始
--------

安装 CANNs:

.. code-block:: bash

   pip install canns

或从源码安装:

.. code-block:: bash

   git clone https://github.com/your-repo/canns.git
   cd canns
   pip install -e .

简单示例:

.. code-block:: python

   import canns
   from canns.models.basic import CANN1D
   
   # 创建一个1D CANN模型
   model = CANN1D(num_neurons=128)
   
   # 运行仿真
   result = model.run(duration=1000)

许可证
------

本项目基于 MIT 许可证开源。

索引和表格
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`