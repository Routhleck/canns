安装指南
========

.. grid:: 2

    .. grid-item-card::  🚀 快速开始
       :link: installation-with-uv-recommended
       :link-type: ref

       使用超快的 ``uv`` 包管理器进行安装。

    .. grid-item-card::  📦 标准 Pip
       :link: installation-with-pip
       :link-type: ref

       使用标准的 Python ``pip`` 工具进行安装。

.. note::
   **要求**：Python 3.11 或更高版本。

安装
----

选择您喜欢的包管理器：

.. tab-set::

    .. tab-item:: 使用 uv（推荐）
       :sync: uv

       首先，确保您已安装 `uv <https://github.com/astral-sh/uv>`_。

       .. code-block:: bash

          # 标准 CPU 版本
          uv pip install [ANONYMOUS_PROJECT]

          # 使用加速器
          uv pip install "[ANONYMOUS_PROJECT][cuda12]"   # NVIDIA CUDA 12
          uv pip install "[ANONYMOUS_PROJECT][tpu]"      # Google TPU

    .. tab-item:: 使用 pip
       :sync: pip

       .. code-block:: bash

          # 标准 CPU 版本
          pip install [ANONYMOUS_PROJECT]

          # 使用加速器
          pip install "[ANONYMOUS_PROJECT][cuda12]"   # NVIDIA CUDA 12
          pip install "[ANONYMOUS_PROJECT][tpu]"      # Google TPU

    .. tab-item:: 从源码安装
       :sync: source

       .. code-block:: bash

          git clone [ANONYMOUS_REPO].git
          cd [ANONYMOUS_PROJECT]
          pip install -e .

验证安装
--------

.. code-block:: python

   import [ANONYMOUS_PROJECT]
   print(f"✅ 成功安装 [ANONYMOUS_PROJECT] 版本 {[ANONYMOUS_PROJECT].__version__}")

.. seealso::
   准备好了吗？查看 :doc:`第一步指南 <01_build_model>`。
