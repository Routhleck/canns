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

CANNs（连续吸引子神经网络工具包）是一个基于大脑仿真生态系统（brainstate、brainunit）构建的 Python 库，它简化了连续吸引子神经网络和相关类脑模型的实验。它提供了即用型模型、任务生成器、分析工具和管道，使神经科学和 AI 研究人员能够快速从想法转变为可重现的仿真。

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

   # 使用 uv（推荐，更快）
   uv pip install canns

   # 或使用 pip
   pip install canns

   # GPU 支持
   pip install canns[cuda12]
   pip install canns[cuda13]


文档导航
--------

.. toctree::
   :maxdepth: 1
   :caption: 介绍

   0_why_canns

.. toctree::
   :maxdepth: 2
   :caption: 快速入门指南

   1_quick_starts/index

.. toctree::
   :maxdepth: 2
   :caption: 核心概念

   2_core_concepts/index

.. toctree::
   :maxdepth: 2
   :caption: 完整详细教程

   3_full_detail_tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: 资源

   GitHub 仓库 <https://github.com/routhleck/canns>
   GitHub Issues <https://github.com/routhleck/canns/issues>
   讨论区 <https://github.com/routhleck/canns/discussions>

**语言**: `English <../en/index.html>`_ | `中文 <../zh/index.html>`_

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

如果您在研究中使用了 CANNs，请引用：

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
