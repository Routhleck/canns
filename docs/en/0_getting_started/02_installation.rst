Installation Guide
===================

Environment Requirements
------------------------

- Python 3.11 or higher

Installation with uv (Recommended)
----------------------------------

``uv`` is a fast Python package manager:

.. code-block:: bash

   # Install uv (if not already installed)
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Install canns (CPU version)
   uv pip install canns

   # Or specify an accelerator
   uv pip install canns[cuda12]   # NVIDIA CUDA 12
   uv pip install canns[tpu]      # Google TPU

Installation with pip
---------------------

.. code-block:: bash

   # CPU version
   pip install canns

   # Specify an accelerator
   pip install canns[cuda12]   # NVIDIA CUDA 12
   pip install canns[tpu]      # Google TPU

Installation from Source
------------------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/routhleck/canns.git
   cd canns

   # Method 1: Using uv
   uv sync --all-extras

   # Method 2: Using pip
   pip install -e "."

Verify Installation
-------------------

.. code-block:: python

   import canns
   print(canns.__version__)
   print("✅ Installation successful!")

Next Steps
----------

- :doc:`01_quick_start`
