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

CANNs (Continuous Attractor Neural Networks) is a powerful neural dynamics modeling framework focused on spatial cognition and neural computation. Built on JAX/BrainState, it provides high-performance GPU/TPU support.

What would you like to do?
--------------------------

**📊 Analyze CANN Dynamics**
   Understand how different inputs affect bump responses and tracking behavior
   → :doc:`1_tutorials/cann_dynamics/index`

**🧭 Model Spatial Navigation**
   Build grid cells, place cells, and path integration systems
   → :doc:`1_tutorials/spatial_navigation/index`

**🧠 Train Memory Networks**
   Implement Hopfield associative memory and pattern storage
   → :doc:`1_tutorials/memory_networks/index`

**📈 Unsupervised Learning**
   Extract principal components using Oja/Sanger rules
   → :doc:`1_tutorials/unsupervised_learning/index`

**👁️ Develop Receptive Fields**
   Train orientation-selective neurons with BCM rule
   → :doc:`1_tutorials/receptive_fields/index`

**⏱️ Learn Temporal Patterns**
   Train spiking neural networks using STDP
   → :doc:`1_tutorials/temporal_learning/index`

**🔬 Analyze Experimental Data**
   Fit and analyze real neural recording data
   → :doc:`1_tutorials/experimental_analysis/index`

**⚙️ Advanced Workflows**
   Build automated pipelines and batch processing
   → :doc:`1_tutorials/advanced_workflows/index`

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

Run your first example:

.. code-block:: python

   import brainstate
   from canns.models.basic import CANN1D
   from canns.task.tracking import SmoothTracking1D

   # Set environment
   brainstate.environ.set(dt=0.1)

   # Create model
   cann = CANN1D(num=512)
   cann.init_state()

   # Define tracking task
   task = SmoothTracking1D(
       cann_instance=cann,
       Iext=(1., 0.75, 2., 1.75, 3.),
       duration=(10., 10., 10., 10.),
       time_step=0.1,
   )
   task.get_data()

   # Run simulation
   def run_step(t, inputs):
       cann(inputs)
       return cann.u.value

   us = brainstate.compile.for_loop(
       run_step, task.run_steps, task.data
   )

See :doc:`0_getting_started/01_quick_start` for detailed tutorials.


Documentation Navigation
------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   0_getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: Scenario-Driven Tutorials

   1_tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Resources

   examples/README
   GitHub Repository <https://github.com/routhleck/canns>
   GitHub Issues <https://github.com/routhleck/canns/issues>
   Discussions <https://github.com/routhleck/canns/discussions>

**Language**: `English <../en/index.html>`_ | `中文 <../zh/index.html>`_

About CANNs
-----------

Continuous Attractor Neural Networks (CANNs) are a special class of neural network models characterized by their ability to maintain stable "bump" activity patterns in continuous state spaces. This makes them particularly suitable for modeling:

- **Spatial Cognition**: Location encoding, direction sensing
- **Working Memory**: Maintaining short-term information
- **Motor Control**: Neural representations of direction and velocity
- **Perceptual Decision Making**: Stimulus representation and attention mechanisms

The CANNs library provides a complete toolchain, from model construction to training, analysis, and visualization.

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
