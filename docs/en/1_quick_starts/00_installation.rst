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
          uv pip install canns

          # With Acceleration
          uv pip install "canns[cuda12]"   # NVIDIA CUDA 12
          uv pip install "canns[tpu]"      # Google TPU

    .. tab-item:: Using pip
       :sync: pip

       .. code-block:: bash

          # Standard CPU
          pip install canns

          # With Acceleration
          pip install "canns[cuda12]"   # NVIDIA CUDA 12
          pip install "canns[tpu]"      # Google TPU

    .. tab-item:: From Source
       :sync: source

       .. code-block:: bash

          git clone https://github.com/routhleck/canns.git
          cd canns
          pip install -e .

Verify Installation
-------------------

.. code-block:: python

   import canns
   print(f"✅ Successfully installed canns version {canns.__version__}")

.. seealso::
   Ready to go? Check out the :doc:`First Steps Guide <01_build_model>`.
