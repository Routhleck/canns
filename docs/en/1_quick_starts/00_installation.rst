Installation Guide
==================

.. grid:: 2

    .. grid-item-card::  🚀 Quick Start
       :link: installation-with-uv-recommended
       :link-type: ref

       Install using the ultra-fast ``uv`` package manager.

    .. grid-item-card::  📦 Standard Pip
       :link: installation-with-pip
       :link-type: ref

       Install using the standard Python ``pip`` tool.

.. note::
   **Requirement:** Python 3.11 or higher.

Installation
------------

Choose your preferred package manager:

.. tab-set::

    .. tab-item:: Using uv (Recommended)
       :sync: uv

       First, ensure you have `uv installed <https://github.com/astral-sh/uv>`_.

       .. code-block:: bash

          # Standard CPU
          uv pip install [ANONYMOUS_PROJECT]

          # With Acceleration
          uv pip install "[ANONYMOUS_PROJECT][cuda12]"   # NVIDIA CUDA 12
          uv pip install "[ANONYMOUS_PROJECT][tpu]"      # Google TPU

    .. tab-item:: Using pip
       :sync: pip

       .. code-block:: bash

          # Standard CPU
          pip install [ANONYMOUS_PROJECT]

          # With Acceleration
          pip install "[ANONYMOUS_PROJECT][cuda12]"   # NVIDIA CUDA 12
          pip install "[ANONYMOUS_PROJECT][tpu]"      # Google TPU

    .. tab-item:: From Source
       :sync: source

       .. code-block:: bash

          git clone [ANONYMOUS_REPO].git
          cd [ANONYMOUS_PROJECT]
          pip install -e .

Verify Installation
-------------------

.. code-block:: python

   import [ANONYMOUS_PROJECT]
   print(f"✅ Successfully installed [ANONYMOUS_PROJECT] version {[ANONYMOUS_PROJECT].__version__}")

.. seealso::
   Ready to go? Check out the :doc:`First Steps Guide <01_build_model>`.
