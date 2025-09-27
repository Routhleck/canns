CANNs 文档
===========

.. image:: https://img.shields.io/github/stars/routhleck/canns.svg?style=for-the-badge
   :target: https://github.com/routhleck/canns/stargazers
.. image:: https://img.shields.io/github/license/routhleck/canns.svg?style=for-the-badge
   :target: https://github.com/routhleck/canns/blob/master/LICENSE

欢迎来到 CANNs (连续吸引子神经网络) 文档！本库为构建、训练和分析连续吸引子神经网络提供了统一的高级API。

🚀 **交互式示例**
   在线体验示例：
   
   - |binder| **在 Binder 上运行** (免费，无需设置)
   - |colab| **在 Google Colab 中打开** (需要Google账号)

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/routhleck/canns/HEAD?filepath=docs%2Fzh%2Fnotebooks
   
.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/routhleck/canns/blob/master/docs/zh/notebooks/

📖 **内容目录**

.. toctree::
   :maxdepth: 2
   :caption: 快速入门

   notebooks/01_quick_start
   notebooks/00_design_philosophy

.. toctree::
   :maxdepth: 2
   :caption: 指南

   guide/index

.. toctree::
   :maxdepth: 1
   :caption: 示例

   examples/index
   GitHub 示例 <https://github.com/routhleck/canns/tree/master/examples>

.. toctree::
   :maxdepth: 2
   :caption: API 参考
   
   ../autoapi/index

.. toctree::
   :maxdepth: 2
   :caption: 资源
   :hidden:
   
   GitHub Issues <https://github.com/routhleck/canns/issues>
   讨论区 <https://github.com/routhleck/canns/discussions>

**语言**: `English <../en/>`_ | `中文 <../zh/>`_

关于 CANNs
----------

连续吸引子神经网络 (CANNs) 是一类神经网络模型，其特征是能够在连续状态空间中维持稳定的活动模式。本库提供：

- **丰富的模型库**: 1D/2D CANNs、SFA模型、层次化网络
- **面向任务的设计**: 路径积分、平滑跟踪、自定义任务
- **强大的分析工具**: 实时可视化、统计分析
- **高性能**: 基于JAX的计算，支持GPU/TPU

快速安装
--------

.. code-block:: bash

   # 基础安装 (CPU)
   pip install canns
   
   # GPU支持 (Linux)
   pip install canns[cuda12]
   
   # TPU支持 (Linux)  
   pip install canns[tpu]

快速示例
--------

以下是一个完整的示例，展示如何创建1D CANN、运行平滑跟踪任务并可视化结果：

.. code-block:: python

   import brainstate
   from canns.models.basic import CANN1D
   from canns.task.tracking import SmoothTracking1D
   from canns.analyzer.plotting import PlotConfigs, energy_landscape_1d_animation
   
   # 设置环境并创建1D CANN网络
   brainstate.environ.set(dt=0.1)
   cann = CANN1D(num=512)
   cann.init_state()
   
   # 定义具有多个目标位置的平滑跟踪任务
   task = SmoothTracking1D(
       cann_instance=cann,
       Iext=(1., 0.75, 2., 1.75, 3.),
       duration=(10., 10., 10., 10.),
       time_step=brainstate.environ.get_dt(),
   )
   task.get_data()
   
   # 使用编译循环运行仿真以提高效率
   def run_step(t, inputs):
       cann(inputs)
       return cann.u.value, cann.inp.value
   
   us, inps = brainstate.compile.for_loop(
       run_step, task.run_steps, task.data,
       pbar=brainstate.compile.ProgressBar(10)
   )
   
   # 使用动画可视化结果
   config = PlotConfigs.energy_landscape_1d_animation(
       title='1D CANN 平滑跟踪',
       save_path='tracking_demo.gif'
   )
   energy_landscape_1d_animation(
       {'活动': (cann.x, us), '输入': (cann.x, inps)},
       config=config
   )

社区和支持
----------

- **GitHub 仓库**: https://github.com/routhleck/canns
- **问题报告**: https://github.com/routhleck/canns/issues
- **讨论区**: https://github.com/routhleck/canns/discussions

索引和表格
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
