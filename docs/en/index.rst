CANNs Documentation
====================

.. image:: https://badges.ws/badge/status-beta-yellow
   :target: https://github.com/routhleck/canns
   :alt: Status: Beta

.. image:: https://img.shields.io/pypi/pyversions/canns
   :target: https://pypi.org/project/canns/
   :alt: Python Versions

.. image:: https://badges.ws/maintenance/yes/2025
   :target: https://github.com/routhleck/canns
   :alt: Maintained

.. image:: https://badges.ws/github/release/routhleck/canns
   :target: https://github.com/routhleck/canns/releases
   :alt: Release

.. image:: https://badges.ws/github/license/routhleck/canns
   :target: https://github.com/routhleck/canns/blob/master/LICENSE
   :alt: License

.. image:: https://badges.ws/github/stars/routhleck/canns?logo=github
   :target: https://github.com/routhleck/canns/stargazers
   :alt: GitHub Stars

.. image:: https://static.pepy.tech/personalized-badge/canns?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/canns
   :alt: Downloads

.. image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/Routhleck/canns
   :alt: Ask DeepWiki

.. image:: https://badges.ws/badge/Buy_Me_a_Coffee-ff813f?icon=buymeacoffee
   :target: https://buymeacoffee.com/forrestcai6
   :alt: Buy Me a Coffee

Welcome to CANNs!
-----------------

CANNs (Continuous Attractor Neural Networks toolkit) is a Python library built on BrainPy, a powerful framework for brain dynamics programming. It streamlines experimentation with continuous attractor neural networks and related brain-inspired models. The library delivers ready-to-use models, task generators, analysis tools, and pipelines—enabling neuroscience and AI researchers to move quickly from ideas to reproducible simulations.

Visualizations
--------------

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
            <h4 class="viz-title">1D CANN Smooth Tracking</h4>
            <img src="../_static/smooth_tracking_1d.gif" alt="1D CANN Smooth Tracking" class="viz-img" width="320">
            <p class="viz-caption">Real-time dynamics during smooth tracking</p>
         </div>
         <div class="viz-card">
            <h4 class="viz-title">2D CANN Population Encoding</h4>
            <img src="../_static/CANN2D_encoding.gif" alt="2D CANN Encoding" class="viz-img" width="320">
            <p class="viz-caption">Spatial information encoding patterns</p>
         </div>
      </div>

      <div class="viz-row">
         <div class="viz-card-wide">
            <h4 class="viz-title-wide">🔬 Theta Sweep Analysis</h4>
            <img src="../_static/theta_sweep_animation.gif" alt="Theta Sweep Animation" class="viz-img" width="600">
            <p class="viz-caption">Theta rhythm modulation in grid and direction cell networks</p>
         </div>
      </div>

      <div class="viz-row">
         <div class="viz-card">
            <h4 class="viz-title">Bump Analysis</h4>
            <img src="../_static/bump_analysis_demo.gif" alt="Bump Analysis Demo" class="viz-img" width="320">
            <p class="viz-caption">1D bump fitting and analysis</p>
         </div>
         <div class="viz-card">
            <h4 class="viz-title">Torus Topology Analysis</h4>
            <img src="../_static/torus_bump.gif" alt="Torus Bump Analysis" class="viz-img" width="320">
            <p class="viz-caption">3D torus visualization and decoding</p>
         </div>
      </div>
   </div>

Quick Start
-----------

Install CANNs:

.. code-block:: bash

   # Using uv (recommended for faster installs)
   uv pip install canns

   # Or use pip
   pip install canns

   # For GPU support
   pip install canns[cuda12]
   pip install canns[cuda13]


Documentation Navigation
------------------------

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   0_why_canns

.. toctree::
   :maxdepth: 2
   :caption: Quick Start Guides

   1_quick_starts/index

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   2_core_concepts/index

.. toctree::
   :maxdepth: 2
   :caption: Full Detail Tutorials

   3_full_detail_tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Resources

   references
   GitHub Repository <https://github.com/routhleck/canns>
   GitHub Issues <https://github.com/routhleck/canns/issues>
   Discussions <https://github.com/routhleck/canns/discussions>

**Language**: `English <../en/index.html>`_ | `中文 <../zh/index.html>`_

Community and Support
---------------------

- **GitHub Repository**: https://github.com/routhleck/canns
- **Issue Tracker**: https://github.com/routhleck/canns/issues
- **Discussions**: https://github.com/routhleck/canns/discussions
- **Documentation**: https://canns.readthedocs.io/

Contributing
------------

Contributions are welcome! Please check our `Contribution Guidelines <https://github.com/routhleck/canns/blob/master/CONTRIBUTING.md>`_.

Citation
--------

If you use CANNs in your research, please cite:

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
