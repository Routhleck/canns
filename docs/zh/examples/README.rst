示例代码
========

CANNs 提供了丰富的示例代码，涵盖各种使用场景。所有示例代码位于项目根目录的 ``examples/`` 文件夹中。

代码组织结构
------------

示例代码按功能分为三个主要类别：

.. code-block:: text

   examples/
   ├── cann/                    # CANN 模型示例
   │   ├── cann1d_*.py
   │   ├── cann2d_*.py
   │   ├── hierarchical_*.py
   │   ├── navigation_*.py
   │   └── theta_sweep_*.py
   ├── brain_inspired/          # 脑启发学习算法
   │   ├── hopfield_*.py
   │   ├── oja_*.py
   │   ├── bcm_*.py
   │   └── stdp_*.py
   └── pipeline/                # 高级工作流
       └── *_pipeline.py

CANN 模型示例
-------------

**基础追踪任务**：

- ``cann1d_tuning_curve.py`` - 一维 CANN 调谐曲线
- ``cann1d_oscillatory_tracking.py`` - 振荡性追踪
- ``cann2d_tracking.py`` - 二维 CANN 追踪

**空间导航**：

- ``hierarchical_path_integration.py`` - 分层网络路径积分
- ``navigation_complex_environment.py`` - 复杂环境导航
- ``theta_sweep_place_cell_network.py`` - 位置细胞 Theta sweep
- ``theta_sweep_grid_cell_network.py`` - 网格细胞 Theta sweep

**轨迹处理**：

- ``import_external_trajectory.py`` - 导入外部轨迹数据

脑启发学习示例
--------------

**Hopfield 网络**：

- ``hopfield_train.py`` - 基本训练（图像）
- ``hopfield_train_1d.py`` - 一维模式存储
- ``hopfield_train_mnist.py`` - MNIST 数字记忆
- ``hopfield_energy_diagnostics.py`` - 能量分析
- ``hopfield_hebbian_vs_antihebbian.py`` - 学习规则对比

**无监督学习**：

- ``oja_pca_extraction.py`` - Oja 规则 PCA
- ``oja_vs_sanger_comparison.py`` - Oja 与 Sanger 对比

**感受野发展**：

- ``bcm_receptive_fields.py`` - BCM 方向选择性

**时序学习**：

- ``stdp_temporal_learning.py`` - STDP 时序模式学习

Pipeline 示例
-------------

- ``advanced_theta_sweep_pipeline.py`` - 高级 Theta sweep 工作流
- ``theta_sweep_from_external_data.py`` - 使用外部数据的 Pipeline

运行示例
--------

所有示例都可以直接运行：

.. code-block:: bash

   # 从项目根目录运行
   python examples/cann/cann2d_tracking.py

   # 或从 examples 目录运行
   cd examples/brain_inspired/
   python oja_pca_extraction.py

大多数示例会生成可视化结果（PNG 或 GIF 文件）。

示例与教程的对应关系
--------------------

每个教程都对应一个或多个示例文件：

==================== ========================================
教程                 对应示例
==================== ========================================
CANN 动力学          ``examples/cann/cann*_tracking.py``
空间导航             ``examples/cann/hierarchical_*.py``
记忆网络             ``examples/brain_inspired/hopfield_*.py``
无监督学习           ``examples/brain_inspired/oja_*.py``
感受野发展           ``examples/brain_inspired/bcm_*.py``
时序学习             ``examples/brain_inspired/stdp_*.py``
高级工作流           ``examples/pipeline/*.py``
==================== ========================================

修改和扩展示例
--------------

示例代码设计为易于修改和扩展：

1. **参数调整**

   大多数示例在文件开头定义了关键参数，可以直接修改。

2. **代码复用**

   复制示例作为你自己项目的起点：

   .. code-block:: bash

      cp examples/cann/cann2d_tracking.py my_project.py
      # 然后修改 my_project.py

3. **组合使用**

   可以混合不同示例中的技术：

   - 使用 CANN 模型 + 自定义追踪任务
   - 使用 Hopfield 网络 + 新的分析器
   - 结合多个学习规则

获取帮助
--------

如果示例运行遇到问题：

1. 检查是否安装了所有依赖：``make install``
2. 查看对应的教程文档获取详细解释
3. 在 `GitHub Issues <https://github.com/your-org/canns/issues>`_ 提问

相关文档
--------

- :doc:`../1_tutorials/index` - 详细的教程文档
- :doc:`../0_getting_started/quick_start` - 快速开始指南
开始探索
--------

从简单的示例开始：

1. :doc:`../1_tutorials/cann_dynamics/tracking_1d` → ``cann1d_tuning_curve.py``
2. :doc:`../1_tutorials/memory_networks/hopfield_basics` → ``hopfield_train.py``
3. :doc:`../1_tutorials/unsupervised_learning/oja_pca` → ``oja_pca_extraction.py``

逐步学习更复杂的场景！
