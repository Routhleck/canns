CANNs Documentation
===================

.. image:: https://badges.ws/badge/status-beta-yellow
   :target: https://github.com/routhleck/canns
   :alt: Status: Beta

.. image:: https://img.shields.io/pypi/pyversions/canns
   :target: https://pypi.org/project/canns/
   :alt: Python Version

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
   :alt: Stars

.. image:: https://static.pepy.tech/personalized-badge/canns?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/canns
   :alt: Downloads

.. image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/Routhleck/canns
   :alt: Ask DeepWiki

.. image:: https://badges.ws/badge/Buy_Me_a_Coffee-ff813f?icon=buymeacoffee
   :target: https://buymeacoffee.com/forrestcai6
   :alt: Buy Me a Coffee

Welcome to the CANNs (Continuous Attractor Neural Networks) documentation! This library provides a unified, high-level API for building, training, and analyzing continuous attractor neural networks.

Visual Gallery
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
   <br><em>Grid cell and head direction networks with theta rhythm modulation</em>
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

📖 **Table of Contents**

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guide/index

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guide/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index
   GitHub Examples <https://github.com/routhleck/canns/tree/master/examples>

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   ../autoapi/index

.. toctree::
   :maxdepth: 2
   :caption: Resources
   :hidden:
   
   GitHub Issues <https://github.com/routhleck/canns/issues>
   Discussions <https://github.com/routhleck/canns/discussions>

**Language**: `English <../en/>`_ | `中文 <../zh/>`_

About CANNs
-----------

Continuous Attractor Neural Networks (CANNs) are a class of neural network models characterized by their ability to maintain stable activity patterns in continuous state spaces. This library provides:

- **Rich Model Library**: 1D/2D CANNs, SFA models, hierarchical networks
- **Task-Oriented Design**: Path integration, smooth tracking, custom tasks
- **Powerful Analysis Tools**: Real-time visualization, statistical analysis
- **High Performance**: JAX-based computation with GPU/TPU support

Quick Installation
------------------

.. code-block:: bash

   # Basic installation (CPU)
   pip install canns
   
   # GPU support (Linux)
   pip install canns[cuda12]
   
   # TPU support (Linux)  
   pip install canns[tpu]

Quick Example
-------------

Here's a complete example showing how to create a 1D CANN, run a smooth tracking task, and visualize the results:

.. code-block:: python

   import brainstate
   from canns.models.basic import CANN1D
   from canns.task.tracking import SmoothTracking1D
   from canns.analyzer.plotting import PlotConfigs, energy_landscape_1d_animation
   
   # Set up environment and create 1D CANN network
   brainstate.environ.set(dt=0.1)
   cann = CANN1D(num=512)
   cann.init_state()
   
   # Define smooth tracking task with multiple target positions
   task = SmoothTracking1D(
       cann_instance=cann,
       Iext=(1., 0.75, 2., 1.75, 3.),
       duration=(10., 10., 10., 10.),
       time_step=brainstate.environ.get_dt(),
   )
   task.get_data()
   
   # Run simulation with compiled loop for efficiency
   def run_step(t, inputs):
       cann(inputs)
       return cann.u.value, cann.inp.value
   
   us, inps = brainstate.compile.for_loop(
       run_step, task.run_steps, task.data,
       pbar=brainstate.compile.ProgressBar(10)
   )
   
   # Visualize results with animation
   config = PlotConfigs.energy_landscape_1d_animation(
       title='1D CANN Smooth Tracking',
       save_path='tracking_demo.gif'
   )
   energy_landscape_1d_animation(
       {'Activity': (cann.x, us), 'Input': (cann.x, inps)},
       config=config
   )

Community and Support
---------------------

- **GitHub Repository**: https://github.com/routhleck/canns
- **Issue Reports**: https://github.com/routhleck/canns/issues
- **Discussions**: https://github.com/routhleck/canns/discussions

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
