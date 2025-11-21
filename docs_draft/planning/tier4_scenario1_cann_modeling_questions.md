# Tier 4: Scenario 1 - CANN Modeling and Simulation - Planning Questions

**Status**: 🔴 Awaiting your answers
**Target Audience**: Researchers, Engineers, Graduate students
**Estimated Total Reading Time**: 3-4 hours (7 tutorials)
**Writing Style**: Hands-on, progressive, complete runnable examples

---

## 📋 Module Overview

Scenario 1 是 Tier 4 的核心，覆盖 CANN 建模和仿真的完整工作流：

**基础工作流 (1→2→3→4)**：构成必要的基础知识链
**高级应用 (5/6/7)**：针对特定研究方向的进阶内容

---

## 🎯 The 7 Tutorials

### 基础工作流 (Foundation)
1. 实现CANN模型与调用内置CANN模型
2. Task数据生成与CANN模型仿真模拟
3. 分析方法对仿真结果进行可视化分析
4. 不同CANN参数的影响

### 高级应用 (Advanced)
5. Hierarchical Path Integration Network
6. Theta Sweep System Model (HD + Grid Cell)
7. Theta Sweep Place Cell Network

---

## ❓ Questions for Each Tutorial

---

## Tutorial 1: 实现CANN模型与调用内置CANN模型

### Context
这是整个 Scenario 1 的入口，需要让用户理解 CANNs 库的模型构建基础。

### Q1.1: BrainState 介绍的深度

用户需要了解 BrainState 才能理解 CANN 模型。应该介绍多深？

- **选项A**: 最简介绍（5分钟）- 只说"基于BrainState"，直接给链接
- **选项B**: 关键概念（10分钟）- Dynamics, State, HiddenState, environ.set(dt=...)
- **选项C**: 完整基础（20分钟）- 包括 for_loop, compile, random 等

**Your Answer:**
```
B 关键概念
```

---

### Q1.2: CANN1D 实现解析的深度

以 CANN1D 为例解析模型实现，应该展示哪些内容？

- [x] `__init__` 参数说明
- [x] `make_conn()` 连接矩阵生成
- [x] `get_stimulus_by_pos()` 刺激生成
- [x] `init_state()` 状态初始化
- [x] `update()` 动力学更新
- [ ] 数学公式（tau * du/dt = ...）

**Your Answer:**
```
参数这里只是简单过下，详细的可以去看后面有个不同参数的影响章节
```

---

### Q1.3: 内置模型一览

最后需要展示内置的丰富模型。应该如何展示？

- **选项A**: 简单列表 + 一句话描述
- **选项B**: 表格对比（维度、特性、适用场景）
- **选项C**: 链接到 Tier 3 Core Concepts 的 Model Collections

**Your Answer:**
```
link到tier3吧，这里就说3种类型的model
```

---

### Q1.4: 教程时长目标

这个入门教程应该多长？

- **选项A**: 15-20 分钟（精简，快速入门）
- **选项B**: 25-30 分钟（适中，有一定深度）
- **选项C**: 35-45 分钟（详细，全面覆盖）

**Your Answer:**
```
适中（不要让初学者看到这个教程就觉得很难懂，我们这里就是面向初学者的最简单的教程），不过我感觉这里可能主要介绍brainstate的使用（可以让用户直接跳到如何调用内置模型）
```

---

## Tutorial 2: Task数据生成与CANN模型仿真模拟

### Context
教用户如何生成任务数据并与模型结合进行仿真。

### Q2.1: Task 示例选择

您提到用 Population Coding 作为最简单的示例。需要确认：

- 是否只展示 PopulationCoding1D？
- 还是也简单提及 TemplateMatching1D 和 SmoothTracking1D 的区别？

**Your Answer:**
```
只展示 PopulationCoding1D，可以简单提及，说明下会在后面内容中进行不同task的展示
```

---

### Q2.2: Task 参数说明的详细程度

PopulationCoding1D 有多个参数（duration, dt, stimulus_pos 等）。应该：

- **选项A**: 只展示必要参数，其他用默认值
- **选项B**: 解释所有参数的含义
- **选项C**: 用表格总结参数 + 简单说明

**Your Answer:**
```
B所有，因为这个参数还算少的，并且都比较重要
```

---

### Q2.3: brainstate.for_loop 仿真方法

这是核心内容。应该如何讲解？

- [x] for_loop 的基本用法
- [ ] 与普通 Python for 循环的对比
- [x] JIT 编译的好处
- [ ] ProgressBar 使用
- [x] 返回值处理（如何获取 u, r 历史）

**Your Answer:**
```
可以加个跳转到brainstate的readthedocs: https://brainstate.readthedocs.io/tutorials/transforms/05_loops_conditions.html
```

---

### Q2.4: 完整示例的复杂度

最终的完整示例应该包含多少组件？

- **选项A**: 最简（Model + Task + for_loop，~30行）
- **选项B**: 标准（+ 简单可视化，~50行）
- **选项C**: 完整（+ 多种可视化，~80行）

**Your Answer:**
```
A，因为这里还不涉及analyzer，所以先不进行可视化，可以print下各种信息吧
```

---

## Tutorial 3: 分析方法对仿真结果进行可视化分析

### Context
这是一个较大的教程，需要涵盖 analyzer 模块的各种方法。

### Q3.1: 1D 分析方法覆盖范围

analyzer 模块中的 1D 方法，应该全部涵盖还是选择性介绍？

请标记应该包含的方法：
- [ ] plot_population_activity - 群体活动热图
- [ ] plot_tuning_curve - 调谐曲线
- [ ] plot_bump_position - bump 位置追踪
- [ ] plot_energy_landscape - 能量景观
- [ ] animate_dynamics - 动态动画
- [ ] 其他？请补充

**Your Answer:**
```
这里可能有些方法命名不太对，我这里就没有勾选错误的方法，应该是以下这些：
- energy_landscape_1d_static
- energy_landscape_1d_animation
- raster_plot
- average_firing_rate_plot
- population_activity_heatmap
- tuning_curve
```

---

### Q3.2: 不同 tracking task 的 energy landscape

您提到用不同的 tracking task 来展示不同的 energy landscape。具体用哪几种？

- [x] PopulationCoding1D
- [x] TemplateMatching1D
- [x] SmoothTracking1D
- [ ] OscillatoryTracking1D

**Your Answer:**
```
对的，这里我记得是没有实现oscillatory tracking的，因为就是smooth tracking没有区别，只不过oscillatory tracking是他的一种而已
```

---

### Q3.3: CANN2D 的 2D plot 方法

最后要展示 2D 对应的方法。应该包含哪些？

- [ ] plot_population_activity_2d
- [x] plot_firing_field
- [ ] animate_dynamics_2d
- [ ] 其他？

**Your Answer:**
```
还应该有：
- energy_landscape_2d_static
- energy_landscape_2d_animation
```

---

### Q3.4: PlotConfig 系统

是否在这个教程中介绍 PlotConfig 的使用？

- **选项A**: 不介绍，直接用函数参数
- **选项B**: 简单提及，给出示例
- **选项C**: 详细介绍 PlotConfig.for_animation() 等方法

**Your Answer:**
```
简单提及吧，其实完全可以重新写一个介绍plotconfig章节，可以加在todo中
```

---

### Q3.5: 教程时长目标

这个教程内容较多，应该多长？

- **选项A**: 30-40 分钟
- **选项B**: 45-60 分钟
- **选项C**: 拆分成多个子教程

**Your Answer:**
```
如果内容过长的话可以考虑拆分
```

---

## Tutorial 4: 不同CANN参数的影响

### Context
系统性探索 CANN1D 参数对模型行为的影响。

### Q4.1: 参数探索范围

CANN1D 有以下主要参数，应该探索哪些？

- [x] num - 神经元数量（分辨率）
- [x] tau - 时间常数
- [x] tau_v - SFA 时间常数（如果用 CANN1D_SFA）
- [x] k - 全局抑制强度
- [x] a - 连接宽度
- [x] A - 外部输入强度
- [x] J0 - 突触连接强度

**Your Answer:**
```
因为篇幅有限，可以每个简单修改下看看变化
```

---

### Q4.2: 参数效果展示方式

应该如何展示参数效果？

- [x] 单参数扫描（固定其他，变化一个）
- [ ] 参数组合效果（如 k vs J0）
- [ ] 稳定性边界探索
- [ ] 与理论预测对比

**Your Answer:**
```
[Check items and add notes]
```

---

### Q4.3: 可视化固定使用

您提到固定使用 oscillation tracking + energy landscape。是否需要补充其他可视化？

**Your Answer:**
```
应该并不是oscillation tracking，是smooth tracking
```

---

### Q4.4: 参数调优指南

是否在最后提供"参数调优指南"或"常见问题排查"？

- 如：bump 消失怎么办？振荡不稳定怎么办？

**Your Answer:**
```
暂时不需要提供
```

---

## Tutorial 5: Hierarchical Path Integration Network

### Context
从这里开始是高级应用，介绍层级路径积分网络。

### Q5.1: 模型实现解析深度

应该多详细地解析模型实现？

- **选项A**: 高层概述（架构图 + 组件功能）
- **选项B**: 中等深度（关键代码片段 + 解释）
- **选项C**: 完整解析（类似 Tutorial 1 的 CANN1D）

**Your Answer:**
```
高层概述，详情可以链接到文章
@article{chu2025localized,
  title={Localized Space Coding and Phase Coding Complement Each Other to Achieve Robust and Efficient Spatial Representation},
  author={Chu, Tianhao and Wu, Yuling and Qiu, Wentao and Jiang, Zihao and Burgess, Neil and Hong, Bo and Wu, Si},
  journal={bioRxiv},
  pages={2025--09},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

---

### Q5.2: Navigation Task 介绍

需要介绍 Navigation Task 的使用。应该包含：

- [x] ClosedLoopNavigation vs OpenLoopNavigation 区别
- [x] 轨迹数据格式
- [x] 与 Tracking Task 的差异
- [ ] 环境设置（RatInABox 集成？）

**Your Answer:**
```
这里是使用canns-lib中的生成方法，可以等价于ratinabox，不过有700倍的效率提升
```

---

### Q5.3: 分析可视化

最终展示 firing heatmap。还需要其他可视化吗？

- [x] 各层网络的 firing heatmap
- [ ] 轨迹叠加显示
- [ ] 路径积分误差分析
- [ ] 其他？

**Your Answer:**
```
[Check items]
```

---

### Q5.4: 教程时长

这个高级教程应该多长？

- **选项A**: 30-40 分钟
- **选项B**: 45-60 分钟

**Your Answer:**
```
A
```

---

## Tutorial 6: Theta Sweep System Model (HD + Grid Cell)

### Context
介绍 Theta Sweep 系统模型，包括头方向细胞和格子细胞网络。

### Q6.1: HD Cell 和 Grid Cell 的介绍

应该如何组织这两个网络的介绍？

- **选项A**: 先 HD Cell，再 Grid Cell（顺序学习）
- **选项B**: 并行对比（共同点和差异）
- **选项C**: 以 Grid Cell 为主，HD Cell 简单提及

**Your Answer:**
```
A
```

---

### Q6.2: Theta Sweep 机制解释

需要解释 theta sweep 的神经科学背景吗？

- **选项A**: 简单一段话（theta 节律、相位进动）
- **选项B**: 详细解释 + 参考文献
- **选项C**: 跳过背景，直接展示使用

**Your Answer:**
```
A ，然后link到具体文献。我记得这里每个类的docstring都是有reference，你可以看下
```

---

### Q6.3: 分析可视化

应该展示哪些可视化？

- [ ] Theta 相位-位置关系图
- [ ] Grid cell 六边形模式
- [ ] Spike train 可视化
- [ ] 动态动画
- [ ] 其他？

**Your Answer:**
```
可以参考examples/cann/theta_sweep_grid_cell_network.py，几个静态图与最终的一个动态图
```

---

## Tutorial 7: Theta Sweep Place Cell Network

### Context
介绍位置细胞网络的 theta sweep。

### Q7.1: 与 Tutorial 6 的关系

这个教程与 Tutorial 6 有重叠。应该如何处理？

- **选项A**: 独立完整（有些重复但自包含）
- **选项B**: 建立在 Tutorial 6 基础上（假设已学过）
- **选项C**: 与 Tutorial 6 合并成一个大教程

**Your Answer:**
```
独立完整
```

---

### Q7.2: Place Cell 特有内容

Place Cell 相比 HD/Grid Cell 有什么特有内容需要强调？

**Your Answer:**
```
不太需要强调，这个是另外的部分
```

---

### Q7.3: 完整仿真流程展示

最终应该展示完整的仿真流程吗？（从数据生成到最终分析）

**Your Answer:**
```
是的
```

---

## Section: Cross-Cutting Questions

### QX.1: 代码风格

所有 7 个教程的代码应该如何呈现？

- **选项A**: 渐进式构建（每步加一点）
- **选项B**: 完整代码 + 分块解释
- **选项C**: 混合（简单教程用A，复杂用B）

**Your Answer:**
```
C
```

---

### QX.2: 理论 vs 实践比例

对于基础教程 (1-4) 和高级教程 (5-7)，理论解释的比例应该不同吗？

- 基础教程 (1-4): ___% 理论, ___% 代码
- 高级教程 (5-7): ___% 理论, ___% 代码

**Your Answer:**
```
可能不太需要理论解释，更多的是如何使用我们的package，就是可能需要再基础教程中进行各种方法与类的解释
```

---

### QX.3: 错误处理和调试

是否在教程中包含"常见错误"或"调试技巧"部分？

**Your Answer:**
```
暂时不太需要，调试技巧的话可以给下with jax.disable_jit()的方法，就是不用jax jit来去跑，可以进行debug
```

---

### QX.4: 练习题

是否在每个教程末尾提供练习题？

- **选项A**: 不提供
- **选项B**: 1-2个简单修改练习
- **选项C**: 3-5个不同难度的练习

**Your Answer:**
```
不提供
```

---

### QX.5: 教程之间的导航

应该如何引导用户从一个教程到下一个？

- **选项A**: 简单的"下一步"链接
- **选项B**: "您学到了什么"总结 + "下一步"
- **选项C**: 包含"如果您想..."的分支建议

**Your Answer:**
```
A,C
```

---

### QX.6: 实施优先级

如果需要分阶段实施，哪些教程应该首先完成？

请排序（1 = 最高优先级）：
- [ ] Tutorial 1: 实现CANN模型
- [ ] Tutorial 2: Task与仿真
- [ ] Tutorial 3: 分析可视化
- [ ] Tutorial 4: 参数影响
- [ ] Tutorial 5: Hierarchical Network
- [ ] Tutorial 6: Theta Sweep HD/Grid
- [ ] Tutorial 7: Theta Sweep Place Cell

**Your Answer:**
```
按顺序
```

---

## ✅ Next Steps After Answering

完成答案后：
1. 保存此文件
2. 告诉我您已完成
3. 我将基于您的答案生成 Scenario 1 的教程
4. 审查迭代后，继续 Scenario 3 和 4

---

**Tips for Answering**:
- 考虑您的目标用户（研究生、博后、工程师）
- 平衡全面性和实用性
- 可以说"跳过这个"或"与另一个合并"
- 重点关注能帮助用户真正理解和使用的内容
