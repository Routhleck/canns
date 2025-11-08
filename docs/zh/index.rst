CANNs 文档
===========

.. image:: https://badges.ws/badge/status-beta-yellow
   :target: https://github.com/routhleck/canns
   :alt: 状态: Beta

.. image:: https://img.shields.io/pypi/pyversions/canns
   :target: https://pypi.org/project/canns/
   :alt: Python 版本

.. image:: https://badges.ws/maintenance/yes/2025
   :target: https://github.com/routhleck/canns
   :alt: 持续维护

.. image:: https://badges.ws/github/release/routhleck/canns
   :target: https://github.com/routhleck/canns/releases
   :alt: 发行版本

.. image:: https://badges.ws/github/license/routhleck/canns
   :target: https://github.com/routhleck/canns/blob/master/LICENSE
   :alt: 许可证

.. image:: https://badges.ws/github/stars/routhleck/canns?logo=github
   :target: https://github.com/routhleck/canns/stargazers
   :alt: GitHub Stars

.. image:: https://static.pepy.tech/personalized-badge/canns?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/canns
   :alt: 下载量

.. image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/Routhleck/canns
   :alt: 询问 DeepWiki

.. image:: https://badges.ws/badge/Buy_Me_a_Coffee-ff813f?icon=buymeacoffee
   :target: https://buymeacoffee.com/forrestcai6
   :alt: 请我喝咖啡

欢迎使用 CANNs！
----------------

CANNs (连续吸引子神经网络) 是一个强大的神经动力学建模框架，专注于空间认知和神经计算。基于 JAX/BrainState 构建，提供高性能的 GPU/TPU 支持。

你想做什么？
------------

**📊 分析 CANN 动力学**
   理解不同输入如何影响 bump 响应和追踪行为
   → :doc:`1_tutorials/cann_dynamics/index`

**🧭 建模空间导航**
   构建网格细胞、位置细胞和路径积分系统
   → :doc:`1_tutorials/spatial_navigation/index`

**🧠 训练记忆网络**
   实现 Hopfield 联想记忆和模式存储
   → :doc:`1_tutorials/memory_networks/index`

**📈 无监督学习**
   使用 Oja/Sanger 规则提取主成分
   → :doc:`1_tutorials/unsupervised_learning/index`

**👁️ 感受野发展**
   用 BCM 规则训练方向选择性
   → :doc:`1_tutorials/receptive_fields/index`

**⏱️ 时序模式学习**
   使用 STDP 训练脉冲神经网络
   → :doc:`1_tutorials/temporal_learning/index`

**🔬 实验数据分析**
   拟合和分析真实神经记录数据
   → :doc:`1_tutorials/experimental_analysis/index`

**⚙️ 高级工作流**
   构建自动化 Pipeline 和批量处理
   → :doc:`1_tutorials/advanced_workflows/index`

可视化展示
----------

.. raw:: html

   <div align="center">
   <table>
   <tr>
   <td align="center" width="50%" valign="top">
   <h4>1D CANN 平滑追踪</h4>
   <img src="../_static/smooth_tracking_1d.gif" alt="1D CANN 平滑追踪" width="320">
   <br><em>平滑追踪过程中的实时动力学</em>
   </td>
   <td align="center" width="50%" valign="top">
   <h4>2D CANN 群体编码</h4>
   <img src="../_static/CANN2D_encoding.gif" alt="2D CANN 编码" width="320">
   <br><em>空间信息编码模式</em>
   </td>
   </tr>
   <tr>
   <td colspan="2" align="center">
   <h4>Theta 扫描分析</h4>
   <img src="../_static/theta_sweep_animation.gif" alt="Theta 扫描动画" width="600">
   <br><em>网格细胞和方向细胞网络的 theta 节律调制</em>
   </td>
   </tr>
   <tr>
   <td align="center" width="50%" valign="top">
   <h4>Bump 分析</h4>
   <img src="../_static/bump_analysis_demo.gif" alt="Bump 分析演示" width="320">
   <br><em>1D bump 拟合与分析</em>
   </td>
   <td align="center" width="50%" valign="top">
   <h4>环面拓扑分析</h4>
   <img src="../_static/torus_bump.gif" alt="环面 Bump 分析" width="320">
   <br><em>3D 环面可视化与解码</em>
   </td>
   </tr>
   </table>
   </div>

快速开始
--------

安装 CANNs：

.. code-block:: bash

   # 使用 uv (推荐，更快)
   uv pip install canns

   # 或使用 pip
   pip install canns

   # GPU 支持
   pip install canns[cuda12]

运行第一个示例：

.. code-block:: python

   import brainstate
   from canns.models.basic import CANN1D
   from canns.task.tracking import SmoothTracking1D

   # 设置环境
   brainstate.environ.set(dt=0.1)

   # 创建模型
   cann = CANN1D(num=512)
   cann.init_state()

   # 定义追踪任务
   task = SmoothTracking1D(
       cann_instance=cann,
       Iext=(1., 0.75, 2., 1.75, 3.),
       duration=(10., 10., 10., 10.),
       time_step=0.1,
   )
   task.get_data()

   # 运行仿真
   def run_step(t, inputs):
       cann(inputs)
       return cann.u.value

   us = brainstate.compile.for_loop(
       run_step, task.run_steps, task.data
   )

详细教程请参见 :doc:`0_getting_started/quick_start`。


文档导航
--------

.. toctree::
   :maxdepth: 2
   :caption: 快速开始

   0_getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: 场景驱动教程

   1_tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: 资源

   examples/README
   GitHub 仓库 <https://github.com/routhleck/canns>
   GitHub Issues <https://github.com/routhleck/canns/issues>
   讨论区 <https://github.com/routhleck/canns/discussions>

**语言**: `English <../en/index.html>`_ | `中文 <../zh/index.html>`_

关于 CANNs
----------

连续吸引子神经网络 (CANNs) 是一类特殊的神经网络模型，其特征是能够在连续状态空间中维持稳定的"bump"活动模式。这使得它们特别适合建模：

- **空间认知**：位置编码、方向感知
- **工作记忆**：维持短期信息
- **运动控制**：方向和速度的神经表征
- **感知决策**：刺激表征和注意力机制

CANNs 库提供了完整的工具链，从模型构建到训练、分析和可视化。

社区和支持
----------

- **GitHub 仓库**: https://github.com/routhleck/canns
- **问题报告**: https://github.com/routhleck/canns/issues
- **讨论区**: https://github.com/routhleck/canns/discussions
- **文档**: https://canns.readthedocs.io/

贡献
----

欢迎贡献！请查看我们的 `贡献指南 <https://github.com/routhleck/canns/blob/master/CONTRIBUTING.md>`_。

引用
----

如果你在研究中使用了 CANNs，请引用：

.. code-block:: bibtex

   @software{he_2025_canns,
      author       = {He, Sichao},
      title        = {CANNs: Continuous Attractor Neural Networks Toolkit},
      year         = 2025,
      publisher    = {Zenodo},
      version      = {v0.9.0},
      doi          = {10.5281/zenodo.17412545},
      url          = {https://github.com/Routhleck/canns}
   }
