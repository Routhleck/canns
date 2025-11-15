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

CANNs (Continuous Attractor Neural Networks toolkit) is a Python library built on top of the Brain Simulation Ecosystem (brainstate, brainunit) that streamlines experimentation with continuous attractor neural networks and related brain-inspired models. It delivers ready-to-use models, task generators, analysis tools, and pipelines so neuroscience and AI researchers can move from ideas to reproducible simulations quickly.

Visualizations
--------------

.. raw:: html

   <div align="center">
   <table>
   <tr>
   <td align="center" width="50%" valign="top">
   <h4>1D CANN Smooth Tracking</h4>
   <img src="../_static/smooth_tracking_1d.gif" alt="1D CANN Smooth Tracking" width="320">
   <br><em>Real-time dynamics during smooth tracking</em>
   </td>
   <td align="center" width="50%" valign="top">
   <h4>2D CANN Population Encoding</h4>
   <img src="../_static/CANN2D_encoding.gif" alt="2D CANN Encoding" width="320">
   <br><em>Spatial information encoding patterns</em>
   </td>
   </tr>
   <tr>
   <td colspan="2" align="center">
   <h4>Theta Sweep Analysis</h4>
   <img src="../_static/theta_sweep_animation.gif" alt="Theta Sweep Animation" width="600">
   <br><em>Theta rhythm modulation in grid and direction cell networks</em>
   </td>
   </tr>
   <tr>
   <td align="center" width="50%" valign="top">
   <h4>Bump Analysis</h4>
   <img src="../_static/bump_analysis_demo.gif" alt="Bump Analysis Demo" width="320">
   <br><em>1D bump fitting and analysis</em>
   </td>
   <td align="center" width="50%" valign="top">
   <h4>Torus Topology Analysis</h4>
   <img src="../_static/torus_bump.gif" alt="Torus Bump Analysis" width="320">
   <br><em>3D torus visualization and decoding</em>
   </td>
   </tr>
   </table>
   </div>

Quick Start
-----------

Install CANNs:

.. code-block:: bash

   # Using uv (recommended, faster)
   uv pip install canns

   # Or using pip
   pip install canns

   # GPU support
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
