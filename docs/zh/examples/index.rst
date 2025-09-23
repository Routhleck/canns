示例速览
========

以下列出了仓库中常用的示例脚本与笔记本。所有示例均位于项目根目录
``examples/`` 下，可本地运行，也可以通过 README 中提供的 Binder / Colab
按键直接在线体验。

.. list-table:: 精选示例
   :header-rows: 1
   :widths: 30 70

   * - 路径
     - 说明
   * - ``examples/brain_inspired/hopfield_train.py``
     - 使用统一的 ``HebbianTrainer`` 训练 ``AmariHopfieldNetwork``，在含噪图像上执行模式恢复。
   * - ``examples/brain_inspired/hopfield_train_mnist.py``
     - 将 MNIST 字符存储到 Hopfield 网络中，展示同一训练流程在真实数据集上的表现。
   * - ``examples/cann/cann1d_oscillatory_tracking.py``
     - 在一维 CANN 中运行振荡跟踪，并利用绘图工具生成能量景观动画。
   * - ``examples/cann/cann2d_tracking.py``
     - 演示二维 CANN 的平滑跟踪，并通过配置式绘图导出能量景观动画。
   * - ``examples/experimental_cann1d_analysis.py``
     - 载入 ROI 活动，调用实验数据分析器拟合 1D bump，并导出逐帧 GIF。
   * - ``examples/experimental_cann2d_analysis.py``
     - 对二维实验数据执行 spike embedding、UMAP 和 TDA 分析，并生成环面可视化。
   * - ``examples/pipeline/theta_sweep_from_external_data.py``
     - 导入外部轨迹并运行高级 ``ThetaSweepPipeline``，完成方向/网格细胞分析。
   * - ``examples/pipeline/advanced_theta_sweep_pipeline.py``
     - 展示 theta-sweep 流水线的完整参数配置，适合高级用户参考。

更多脚本可查看 `GitHub 示例目录
<https://github.com/routhleck/canns/tree/master/examples>`_。
