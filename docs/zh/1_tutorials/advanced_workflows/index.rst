高级工作流
==========

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。



场景描述
--------

当你需要处理更复杂的场景时，CANNs 提供了高级工具来简化工作流程。本系列教程将教你：

- 构建端到端的 Theta sweep pipeline
- 导入和使用外部轨迹数据
- 完全定制化的参数配置
- 批量处理和并行计算

你将学到
--------

1. ``ThetaSweepPipeline`` 的完整使用
2. 如何导入外部轨迹数据
3. 参数的精细调优
4. 工作流的自动化和优化
5. 结果的批量生成和管理

教程列表
--------

.. toctree::
   :maxdepth: 1

   building_pipelines
   external_trajectories

适用人群
--------

- 需要批量处理数据的研究人员
- 进行参数扫描和优化的学生
- 开发自动化分析流程的工程师
- 需要自定义工作流的高级用户

前置知识
--------

- 熟悉 CANNs 基本概念（完成前面的教程）
- Python 高级特性（装饰器、上下文管理器等）
- 命令行工具使用
- 基本的并行计算概念

Pipeline 架构
-------------

``ThetaSweepPipeline`` 统一了以下步骤：

1. **轨迹处理**

   - 导入外部数据或生成轨迹
   - 轨迹平滑和插值
   - 计算速度和方向

2. **网络配置**

   - 方向细胞网络
   - 网格细胞网络
   - Theta 节律参数

3. **仿真执行**

   - 高效的 JAX 编译循环
   - 进度监控
   - 中间结果保存

4. **结果分析**

   - 活动热图生成
   - 统计分析
   - 动画可视化

高级特性
--------

**参数定制化**
~~~~~~~~~~~~~~~

完全控制所有网络和任务参数：

.. code-block:: python

   pipeline = ThetaSweepPipeline(
       trajectory_data=positions,
       times=times,
       direction_cell_params={
           "num": 100,
           "adaptation_strength": 15,
           "noise_strength": 0.0,
       },
       grid_cell_params={
           "num_gc_x": 100,
           "mapping_ratio": 0.85,
       },
       theta_params={
           "theta_strength_hd": 1.0,
           "theta_cycle_len": 100.0,
       },
   )

**外部轨迹导入**
~~~~~~~~~~~~~~~~

支持多种格式：

- NumPy 数组
- CSV 文件
- HDF5 文件
- 自定义格式

**批量处理**
~~~~~~~~~~~~

使用 Python 脚本自动化：

.. code-block:: python

   for params in parameter_grid:
       pipeline = ThetaSweepPipeline(**params)
       results = pipeline.run(
           output_dir=f"results_{params['id']}",
           save_animation=True,
       )

性能优化
--------

- **JAX JIT 编译**：加速仿真循环
- **GPU 加速**：支持 CUDA 后端
- **并行处理**：多参数组并行
- **内存优化**：流式数据处理

实际应用
--------

- **参数扫描**：系统性探索参数空间
- **模型比较**：对比不同配置的性能
- **重现实验**：使用真实轨迹重现行为
- **假设检验**：测试理论预测

最佳实践
--------

1. **模块化设计**：将工作流分解为独立步骤
2. **版本控制**：使用 Git 管理配置和代码
3. **文档化**：记录参数选择的原因
4. **验证**：检查中间结果的合理性
5. **备份**：定期保存重要结果

开始学习
--------

从 :doc:`building_pipelines` 开始，构建你的第一个自动化工作流！
