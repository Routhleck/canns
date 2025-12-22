CANNs 文档
====================

.. image:: https://badges.ws/badge/status-beta-yellow
   :target: https://github.com/routhleck/canns
   :alt: Status: Beta

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
-----------------

CANNs（连续吸引子神经网络工具包）是一个基于 BrainPy 构建的 Python 库，BrainPy 是强大的脑动力学编程框架。本工具包简化了连续吸引子神经网络和相关脑启发模型的实验流程。它提供即用型模型、任务生成器、分析工具和流水线——让神经科学和 AI 研究人员能够快速从想法转化为可复现的仿真。

可视化展示
----------

.. raw:: html

   <style>
   .viz-container {
      display: flex;
      flex-direction: column;
      gap: 25px;
      max-width: 1000px;
      margin: 30px auto;
      padding: 20px;
   }
   .viz-row {
      display: flex;
      gap: 20px;
      justify-content: center;
      align-items: stretch;
   }
   .viz-card {
      flex: 1;
      background: #ffffff;
      border: 2px solid #e0e0e0;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
      transition: transform 0.2s, box-shadow 0.2s;
      text-align: center;
   }
   .viz-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
      border-color: #4a90e2;
   }
   .viz-card-wide {
      flex: 1 1 100%;
      background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
      border: 2px solid #b8daff;
      border-radius: 12px;
      padding: 25px;
      box-shadow: 0 6px 16px rgba(74, 144, 226, 0.12);
   }
   .viz-card-wide:hover {
      transform: translateY(-4px);
      box-shadow: 0 10px 24px rgba(74, 144, 226, 0.2);
      border-color: #4a90e2;
   }
   .viz-title {
      color: #2c3e50;
      font-size: 18px;
      font-weight: 600;
      margin: 0 0 15px 0;
      padding-bottom: 10px;
      border-bottom: 2px solid #e8eef5;
   }
   .viz-title-wide {
      color: #1e3a5f;
      font-size: 20px;
      font-weight: 700;
      margin: 0 0 20px 0;
      padding-bottom: 12px;
      border-bottom: 3px solid #4a90e2;
   }
   .viz-caption {
      color: #666;
      font-style: italic;
      font-size: 14px;
      margin-top: 12px;
      line-height: 1.5;
   }
   .viz-img {
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid #e8eef5;
   }
   @media (max-width: 768px) {
      .viz-row {
         flex-direction: column;
      }
   }
   </style>

   <div class="viz-container">
      <div class="viz-row">
         <div class="viz-card">
            <h4 class="viz-title">1D CANN 平滑追踪</h4>
            <img src="../_static/smooth_tracking_1d.gif" alt="1D CANN 平滑追踪" class="viz-img" width="320">
            <p class="viz-caption">平滑追踪过程中的实时动力学</p>
         </div>
         <div class="viz-card">
            <h4 class="viz-title">2D CANN 群体编码</h4>
            <img src="../_static/CANN2D_encoding.gif" alt="2D CANN 编码" class="viz-img" width="320">
            <p class="viz-caption">空间信息编码模式</p>
         </div>
      </div>

      <div class="viz-row">
         <div class="viz-card-wide">
            <h4 class="viz-title-wide">🔬 Theta 扫描分析</h4>
            <img src="../_static/theta_sweep_animation.gif" alt="Theta Sweep Animation" class="viz-img" width="600">
            <p class="viz-caption">网格细胞和方向细胞网络中的 theta 节律调制</p>
         </div>
      </div>

      <div class="viz-row">
         <div class="viz-card">
            <h4 class="viz-title">活动波包分析</h4>
            <img src="../_static/bump_analysis_demo.gif" alt="Bump Analysis Demo" class="viz-img" width="320">
            <p class="viz-caption">1D 活动波包拟合和分析</p>
         </div>
         <div class="viz-card">
            <h4 class="viz-title">环面拓扑分析</h4>
            <img src="../_static/torus_bump.gif" alt="Torus Bump Analysis" class="viz-img" width="320">
            <p class="viz-caption">3D 环面可视化和解码</p>
         </div>
      </div>
   </div>

快速开始
-----------

安装 CANNs：

.. code-block:: bash

   # 使用 uv（推荐，安装更快）
   uv pip install canns

   # 或用 pip
   pip install canns

   # 若需 GPU 支持
   pip install canns[cuda12]
   pip install canns[cuda13]


文档导航
------------------------

.. toctree::
   :maxdepth: 1
   :caption: 简介

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
   :caption: 详细教程

   3_full_detail_tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: 资源

   references
   GitHub 仓库 <https://github.com/routhleck/canns>
   GitHub Issues <https://github.com/routhleck/canns/issues>
   讨论区 <https://github.com/routhleck/canns/discussions>

**语言**: `English <../en/index.html>`_ | `中文 <../zh/index.html>`_

社区和支持
---------------------

- **GitHub 仓库**: https://github.com/routhleck/canns
- **问题追踪**: https://github.com/routhleck/canns/issues
- **讨论区**: https://github.com/routhleck/canns/discussions
- **文档**: https://canns.readthedocs.io/

贡献
------------

欢迎贡献！请查看我们的 `贡献指南 <https://github.com/routhleck/canns/blob/master/CONTRIBUTING.md>`_。

引用
--------

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
