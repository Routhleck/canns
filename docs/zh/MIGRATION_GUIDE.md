# 文档迁移指南

本文档说明如何完成从旧结构到新场景驱动结构的迁移。

## ✅ 已完成的工作

### 1. 目录结构创建

- ✅ `getting_started/` - 快速开始目录
- ✅ `tutorials/` - 8个场景驱动教程目录
- ✅ `api_reference/` - API参考目录
- ✅ `experiments/` - 实验室目录
- ✅ `examples/` - 示例代码说明

### 2. 核心文件创建

- ✅ `index.rst` - 重写主页（场景驱动）
- ✅ `getting_started/index.rst` - 快速开始首页
- ✅ `getting_started/installation.rst` - 安装指南
- ✅ `getting_started/00_design_philosophy.rst` - 设计哲学（从notebook转换）
- ✅ `getting_started/01_quick_start.rst` - 快速开始（从notebook转换）
- ✅ `tutorials/index.rst` - 教程首页
- ✅ 8个scenario的 `index.rst`（每个场景一个）
- ✅ `api_reference/index.rst` - API参考首页
- ✅ `experiments/README.rst` - 实验室说明
- ✅ `examples/README.rst` - 示例代码说明

## 🔄 待完成的工作

### 1. 迁移 API 参考文档

需要从 `guide/` 迁移到 `api_reference/`：

```bash
# 从 guide/ 目录复制并修改
cp guide/models.rst api_reference/models.rst
cp guide/tasks.rst api_reference/tasks.rst
cp guide/analyzer.rst api_reference/analyzers.rst
cp guide/pipeline.rst api_reference/pipeline.rst
```

#### 重写 `api_reference/trainers.rst`

当前 `guide/trainer.rst` 已过时，需要重写以包含所有新的训练器：

- HebbianTrainer
- AntiHebbianTrainer
- **OjaTrainer** ⭐ 新增
- **BCMTrainer** ⭐ 新增
- **SangerTrainer** ⭐ 新增
- **STDPTrainer** ⭐ 新增

参考 `examples/brain_inspired/README.md` 中的详细说明。

### 2. 创建教程内容文件

为每个scenario创建具体的教程文件。每个scenario有4-6个教程文件需要创建。

#### 示例：`tutorials/cann_dynamics/tracking_1d.rst`

```rst
一维CANN追踪
============

场景描述
--------
你想要理解CANN如何响应一维空间的输入，并追踪bump的移动。

你将学到
--------
- 如何初始化CANN1D模型
- 如何定义平滑追踪任务
- 如何使用for_loop进行编译加速
- 如何生成调谐曲线

完整示例
--------
.. literalinclude:: ../../../examples/cann/cann1d_tuning_curve.py
   :language: python
   :linenos:

逐步解析
--------
1. **环境设置**
   ...

2. **模型初始化**
   ...

[继续...]
```

#### 所有需要创建的教程文件：

**CANN Dynamics** (cann_dynamics/)
- tracking_1d.rst
- tracking_2d.rst
- tuning_curves.rst
- oscillatory_tracking.rst

**Spatial Navigation** (spatial_navigation/)
- path_integration.rst
- hierarchical_network.rst
- theta_modulation.rst
- grid_place_cells.rst
- complex_environments.rst

**Memory Networks** (memory_networks/)
- hopfield_basics.rst
- pattern_storage_1d.rst
- mnist_memory.rst
- energy_diagnostics.rst
- hebbian_vs_antihebbian.rst

**Unsupervised Learning** (unsupervised_learning/)
- oja_pca.rst
- sanger_orthogonal_pca.rst
- algorithm_comparison.rst

**Receptive Fields** (receptive_fields/)
- bcm_sliding_threshold.rst
- orientation_selectivity.rst
- tuning_visualization.rst

**Temporal Learning** (temporal_learning/)
- stdp_spike_timing.rst
- causal_learning.rst
- ltp_ltd_mechanisms.rst

**Experimental Analysis** (experimental_analysis/)
- bump_fitting_1d.rst
- bump_fitting_2d.rst
- data_preprocessing.rst

**Advanced Workflows** (advanced_workflows/)
- building_pipelines.rst
- external_trajectories.rst
- parameter_customization.rst

### 3. 更新 Sphinx 配置

检查 `docs/conf.py` 是否需要更新：

- 确保 `toctree` 配置正确
- 添加新的路径（如果需要）
- 更新任何硬编码的路径

### 4. 构建和验证

```bash
cd docs
make clean
make html

# 或使用项目的make命令
cd /Users/sichaohe/Documents/GitHub/canns
make docs
```

检查：
- 所有链接是否正常
- 图片是否加载
- 交叉引用是否工作
- 没有Sphinx警告

## 📋 教程编写模板

使用以下模板创建新教程：

```rst
教程标题
========

场景描述
--------
[1-2句话描述用户想完成什么任务]

你将学到
--------
- [学习要点1]
- [学习要点2]
- [学习要点3]

完整示例
--------
.. literalinclude:: ../../../examples/xxx/example.py
   :language: python
   :linenos:

逐步解析
--------

1. **第一步标题**

   [解释代码]

   .. code-block:: python

      # 关键代码片段
      cann = CANN1D(num=512)
      cann.init_state()

2. **第二步标题**

   [继续解释]

运行结果
--------
运行此示例会生成：

.. image:: path/to/result.png
   :width: 600px

[解释结果]

相关API
-------
- :class:`canns.models.basic.CANN1D`
- :func:`canns.analyzer.plotting.tuning_curve`

下一步
------
- :doc:`tracking_2d` - 扩展到二维
- :doc:`../spatial_navigation/index` - 学习空间导航
```

## 🔗 交叉引用指南

### 链接到其他教程

```rst
:doc:`../cann_dynamics/tracking_1d`
:doc:`../spatial_navigation/hierarchical_network`
:doc:`../../api_reference/models`
```

### 链接到 API

```rst
:class:`canns.models.basic.CANN1D`
:func:`canns.analyzer.plotting.tuning_curve`
:mod:`canns.trainer`
```

### 链接到示例代码

```rst
.. literalinclude:: ../../../examples/cann/cann1d_tracking.py
   :language: python
   :lines: 10-20
   :emphasize-lines: 3, 5
```

## 📊 优先级建议

### 高优先级（立即完成）

1. **API参考迁移**
   - 迁移 models.rst, tasks.rst, analyzer.rst, pipeline.rst
   - 重写 trainers.rst（最重要，添加新trainer）

2. **创建关键教程**
   - cann_dynamics/tracking_1d.rst（最简单的入门）
   - unsupervised_learning/oja_pca.rst（展示新trainer）
   - temporal_learning/stdp_spike_timing.rst（展示新trainer）

3. **构建验证**
   - 运行 `make docs`
   - 修复所有错误和警告

### 中优先级（第二阶段）

1. 完成所有CANN dynamics教程
2. 完成所有brain-inspired教程（memory, unsupervised, receptive_fields, temporal）
3. 添加更多可视化和图表

### 低优先级（可选）

1. Spatial navigation详细教程（已有例子）
2. Experimental analysis教程
3. Advanced workflows教程

## 🛠️ 自动化工具建议

可以写脚本批量生成教程骨架：

```python
# generate_tutorial_stubs.py
scenarios = {
    "cann_dynamics": ["tracking_1d", "tracking_2d", "tuning_curves", "oscillatory_tracking"],
    # ...
}

for scenario, tutorials in scenarios.items():
    for tutorial in tutorials:
        # 生成rst文件骨架
        pass
```

## ✅ 验证清单

完成后检查：

- [ ] 所有8个scenario有index.rst
- [ ] 每个scenario至少有1个完整教程
- [ ] API参考文档完整（models, trainers, analyzers, tasks, pipeline）
- [ ] trainers.rst包含所有6个trainer
- [ ] 主index.rst场景链接都能正常工作
- [ ] `make docs` 无错误
- [ ] 文档可以正常访问和导航
- [ ] 所有交叉引用正常工作

## 📞 需要帮助？

- 查看现有的 `examples/brain_inspired/README.md` 获取trainer详细说明
- 参考 Sphinx 官方文档：https://www.sphinx-doc.org/
- 查看 reStructuredText 语法：https://docutils.sourceforge.io/rst.html
