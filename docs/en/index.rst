CANNs Documentation
===================

.. image:: https://img.shields.io/github/stars/routhleck/canns.svg?style=for-the-badge
   :target: https://github.com/routhleck/canns/stargazers
.. image:: https://img.shields.io/github/license/routhleck/canns.svg?style=for-the-badge
   :target: https://github.com/routhleck/canns/blob/master/LICENSE

Welcome to the CANNs (Continuous Attractor Neural Networks) documentation! This library provides a unified, high-level API for building, training, and analyzing continuous attractor neural networks.

🚀 **Interactive Examples**
   Try the examples interactively:
   
   - |binder| **Run on Binder** (Free, no setup required)
   - |colab| **Open in Google Colab** (Google account required)

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/routhleck/canns/HEAD?filepath=docs%2Fen%2Fnotebooks
   
.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/routhleck/canns/blob/master/docs/en/notebooks/

📖 **Table of Contents**

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   notebooks/01_introduction
   notebooks/03_core_concepts

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:
   
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

.. code-block:: python

   import brainstate
   from canns.models.basic import CANN1D
   from canns.task.tracking import SmoothTracking1D
   
   # Create 1D CANN network
   cann = CANN1D(num=512)
   cann.init_state()
   
   # Define smooth tracking task
   task = SmoothTracking1D(
       cann_instance=cann,
       Iext=(1., 0.75, 2., 1.75, 3.),
       duration=(10., 10., 10., 10.),
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