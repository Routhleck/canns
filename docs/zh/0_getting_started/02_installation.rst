安装指南
========

环境要求
--------

- Python 3.11 或更高版本

使用 uv 安装（推荐）
--------------------

``uv`` 是一个快速的 Python 包管理器：

.. code-block:: bash

   # 安装 uv（如未安装）
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # 安装 canns（CPU 版本）
   uv pip install canns

   # 或指定加速器
   uv pip install canns[cuda12]   # NVIDIA CUDA 12
   uv pip install canns[tpu]      # Google TPU

使用 pip 安装
-------------

.. code-block:: bash

   # CPU 版本
   pip install canns

   # 指定加速器
   pip install canns[cuda12]   # NVIDIA CUDA 12
   pip install canns[tpu]      # Google TPU

从源码安装
---------

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/routhleck/canns.git
   cd canns

   # 方法1：使用 uv
   uv sync --all-extras

   # 方法2：使用 pip
   pip install -e "."

验证安装
--------

.. code-block:: python

   import canns
   print(canns.__version__)
   print("✅ 安装成功！")

下一步
------

- :doc:`01_quick_start`
