# Tier 3: Core Concepts - Planning Questions

**Status**: 🔴 Awaiting your answers
**Target Audience**: Engineers/Developers, Graduate students, Cross-domain collaborators
**Estimated Reading Time per Topic**: 15-20 minutes
**Writing Style**: Conceptual, explanatory, linking theory to practice

---

## 📋 Section Overview

The "Core Concepts" tier provides **in-depth explanations** of library design and components. These are NOT how-to guides (that's Tier 2), but rather **conceptual foundations** that help users understand:
- Why the library is designed this way
- How different components work together
- When to use which approach
- The theoretical background behind implementations

**Key difference from other tiers**:
- Tier 1 (Why CANNs): Motivation and value proposition
- Tier 2 (Basic Intro): Practical how-to guides
- **Tier 3 (Core Concepts)**: Deep conceptual understanding
- Tier 4 (Full Details): Complete API reference with examples

---

## 🎯 The 5 Core Concept Topics

Based on your outline (`docs-arch.md`), Tier 3 covers:

1. **Overview (Design Philosophy)** - Architecture, module organization, design principles
2. **Model Collections** - Basic CANNs, Hybrid models, Brain-Inspired models
3. **Task Generators** - Tracking, navigation, population coding paradigms
4. **Analysis Methods** - Model Analyzer, Data Analyzer, RNN Dynamics Analysis
5. **Brain-Inspired Training** - Learning rules, trainer framework

---

## Topic 1: Overview & Design Philosophy

### Context
This reorganizes the existing `00_design_philosophy.rst` (661 lines) into a more focused overview that ties everything together.

### Q1.1: What are the core design principles of the library?
The existing design philosophy explains modules but doesn't highlight **key principles**. What principles should we emphasize?

Examples:
- Separation of concerns (models ≠ tasks ≠ analyzers)
- BrainState integration for dynamics
- Extensibility through base classes
- JAX-first for performance
- Other principles?

**Your Answer:**
```
主要说Separation of concerns和Extensibility through base classes吧
```

---

### Q1.2: How should we explain the module hierarchy?
Current doc has a flat list of modules. Should we show:
- **Dependency graph** (which modules depend on others)?
- **Workflow diagram** (typical usage flow)?
- **Layered architecture** (low-level to high-level)?

**Your Answer:**
```
workflow diagram可能比较好些
```

---

### Q1.3: What should be preserved from current design_philosophy.rst?
The current document covers:
- Module overview (models, task, analyzer, trainer, pipeline)
- Usage examples
- Extension guides

**Your Answer:**
```
尽量都保留吧，然后看看有什么improve的
```

---

### Q1.4: How much technical depth for Overview?
Should Overview include:
- Code examples showing module interaction?
- Technical implementation details?
- Or just high-level concepts with links to other topics?

**Your Answer:**
```
just high-level concepts with links to other topics
```

---

## Topic 2: Model Collections

### Context
Explains the three model categories: Basic CANN, Hybrid (TODO), Brain-Inspired

### Q2.1: What makes each model category distinct?
Help readers understand when to use which:
- **Basic CANN Models**: When to use? What problems do they solve?
- **Hybrid Models**: What's the concept? (even if TODO, explain the vision)
- **Brain-Inspired Models**: How do they differ from Basic CANNs?

**Your Answer:**
```
模型模块实现不同维度的CANN基础模型及其变体，脑启发模型以及CANN混合模型。该模块是本库的基础，可以与其他的模块来进行交互来实现各种场景的应用。
这里根据不同的模型类型进行分类：
Basic Models (canns.models.basic) 基础的CANNs模型及其各个变体。
Brain-Inspired Models (canns.models.brain_inspired) 类脑模型。
Hybrid Models (canns.models.hybrid) CANN与ANN或其他的混合模型。
在这里主要依赖Brain simulation ecosystem中的brainstate来实现各个模型。brainstate 是 Brain Simulation Ecosystem 中面向动力系统的核心框架，底层基于 JAX/BrainUnit。它提供 brainstate.nn.Dynamics 抽象、State/HiddenState/ParamState 状态容器以及 brainstate.environ 统一的时间步长管理，与 brainstate.compile.for_loop、brainstate.random 等工具一起，让我们可以写出既可 JIT 编译又支持自动微分的神经网络动力学。借助这些接口，CANN 模型只需描述变量与更新方程，时间推进、并行化和随机数管理都由 brainstate 负责，从而显著降低实现成本。
```

---

### Q2.2: Should we explain the BaseCANN abstraction?
The library has `BaseCANN` as parent class for CANN1D/2D.
- Explain the abstract methods (`cell_coords`, `f_r`, `f_u`, `f_r_given_u`)?
- Show how inheritance works?
- Or keep it high-level?

**Your Answer:**
```
每个模型都继承自canns.models.basic.BasicModel或canns.models.basic.BasicModelGroup类，并实现了以下主要方法：
在基础模型中需要完成的主要工作：
继承 canns.models.basic.BasicModel 或 BasicModelGroup，在 __init__ 中调用父类构造（例如 super().__init__(math.prod(shape), **kwargs)）并保存好 shape、varshape 等维度信息；
实现 make_conn() 生成连接矩阵，并在构造函数里赋值给 self.conn_mat（可参考 src/canns/models/basic/cann.py 中的高斯核实现）；
实现 get_stimulus_by_pos(pos)，根据特征空间的位置返回外部刺激，供任务模块调用；
在 init_state() 注册 brainstate.HiddenState/State（常见的有 self.u、self.r、self.inp），确保更新函数能够直接读写；
在 update(inputs) 中写出单步动力学，记得乘以 brainstate.environ.get_dt() 维持数值稳定；
需要暴露诊断量或轴信息时，通过属性/方法返回（如 self.x、self.rho），供任务、分析器和流水线重用。
对于脑启发模型
每个模型都继承自canns.models.brain_inspired.BrainInspiredModel或canns.models.brain_inspired.BrainInspiredModelGroup类，并实现了
若要扩展脑启发模型（继承 BrainInspiredModel 或 BrainInspiredModelGroup），请确保：
在 init_state() 中至少注册状态向量（默认 self.s）和连接权重 self.W，其中 self.W 建议使用 brainstate.ParamState 以便 Hebbian 学习直接写入；
如果权重属性名称不是 W，重写 weight_attr 以便 HebbianTrainer 能找到；
实现 update(...) 与 energy 属性，确保训练器可以运行通用预测循环并判定收敛；
需要定制 Hebbian 规则时实现 apply_hebbian_learning(patterns)，否则可以完全依赖训练器的通用实现；
若模型支持动态尺寸调整，可重写 resize(num_neurons, preserve_submatrix=True)，参考 src/canns/models/brain_inspired/hopfield.py 中的做法。
```

---

### Q2.3: How to explain model variants (e.g., CANN1D vs CANN1D_SFA)?
- Focus on **conceptual differences** (SFA adds adaptation)?
- Show **when to choose** each variant?
- Include **parameter comparison**?

**Your Answer:**
```
就是增加新的特性
```

---

### Q2.4: Hierarchical models (grid cells, place cells)?
These are special:
- Part of Basic Models but more complex
- Should they have dedicated explanation?
- How to explain the hierarchy concept?

**Your Answer:**
```
其也是相当于CANN的变体，思想与思路与基础CANN一致，不过实现有些区别
```

---

## Topic 3: Task Generators

### Context
Explain the task generation philosophy and available paradigms

### Q3.1: What's the key concept users need to understand about tasks?
Tasks are more than just "data generators". What's the deeper concept?
- Experimental paradigm abstraction?
- Model-task coupling philosophy?
- Reproducibility and standardization?

**Your Answer:**
```
任务模块主要用于生成、保存、读取、导入和可视化各种CANN任务。该模块提供了多种预定义的任务类型，并允许用户自定义任务以满足特定需求。
```

---

### Q3.2: How should we organize task types?
Current categories:
- Tracking (smooth, oscillatory)
- Closed-loop navigation
- Open-loop navigation
- Population coding

Should we organize by:
- **Cognitive function** (spatial navigation, memory encoding)?
- **Input pattern** (static, dynamic, feedback-driven)?
- **Use case** (research, benchmarking, teaching)?

**Your Answer:**
```
这里暂时就两类，Tracking任务和Navigation任务，然后其中tracking又分为
- population coding
- template matching
- smooth tracking
navigation的话分成
- closed loop navigation
- open loop navigation
```

---

### Q3.3: How much detail on task-model coupling?
Some tasks need `cann_instance` (like SmoothTracking1D).
- Explain **why** this coupling exists (get_stimulus_by_pos)?
- Show **when** coupling is necessary vs. optional?
- Discuss trade-offs?

**Your Answer:**
```
目前只有tracking task是需要传入model来获取对应的stimulus的，因为基本的CANN model都是这样子来进行输入的，我们想要做到更user-firendly，所以暂时需要coupling，对于navigation就不需要了，因为可能我们需要提供更多的data信息（比如速度、角度等等）然后让用户自行判断来去使用。
```

---

### Q3.4: Should we explain trajectory import?
The library can import external trajectories.
- Just mention it exists?
- Explain use cases (real experimental data)?
- Show conceptual workflow?

**Your Answer:**
```
可以简单提一下，不用特别去详细说明，这是后面要做的事情
```

---

## Topic 4: Analysis Methods

### Context
Covers Model Analyzer, Data Analyzer, and RNN Dynamics Analysis

### Q4.1: Model Analyzer vs. Data Analyzer - key distinction?
Help users understand which to use when:
- Model Analyzer: Analyzing **simulation outputs**?
- Data Analyzer: Analyzing **experimental recordings**?
- What's the philosophical difference?

**Your Answer:**
```
是的，不过model analyzer主要是对我们现在的一些CANN model的输出做一些分析可视化，然后data analyzer主要是对实验数据（一般可能是spike train或者是firing rate）以及可以生成这一类的虚拟数据来去进行分析可视化
```

---

### Q4.2: PlotConfig design philosophy?
Why did we create PlotConfig instead of just function arguments?
- Reusability?
- Type safety?
- Configuration sharing?

Should we explain this design choice?

**Your Answer:**
```
简单提一下PlotConfig吧
```

---

### Q4.3: RNN Dynamics Analysis - scope?
Your outline mentions:
- Slow and fixed points analysis

Is this:
- For analyzing CANN models as RNNs?
- For analyzing arbitrary trained RNNs?
- Both?

**Your Answer:**
```
暂时只用于分析RNN model
```

---

### Q4.4: Topological Data Analysis (TDA)?
The library has TDA tools (UMAP, persistent homology).
- Explain **why** TDA for CANNs (detecting torus structure)?
- Show **when** to use it?
- Keep it high-level or include math?

**Your Answer:**
```
是的，我们cann-lib提供了加速的ripser持续同调方法，但对于降维工具，我们这里没有重新实现，用户可以自行使用，我们可能在某些tda中会有调用外部方法，因为grid cell是有torus structure的，然后可能有一些拓扑结构能够用CANN来去构建，所以我们希望有这样的工具来去探索数据中有没有attractor structure
```

---

## Topic 5: Brain-Inspired Training

### Context
Learning rules (Hebbian, STDP, BCM) and the Trainer framework

### Q5.1: What's the unifying concept of brain-inspired learning?
Beyond "local vs. global", what ties these rules together?
- Biological plausibility?
- Unsupervised learning?
- Synaptic plasticity mechanisms?

**Your Answer:**
```
应该是activity-dependent plasticity
```

---

### Q5.2: How much neuroscience background?
Different learning rules have neuroscience origins:
- Hebbian: "Neurons that fire together wire together"
- STDP: Spike-timing dependent plasticity
- BCM: Bienenstock-Cooper-Munro rule

Should we:
- Explain the neuroscience briefly?
- Just describe algorithmic behavior?
- Link to external neuroscience resources?

**Your Answer:**
```
只是大概简单说下吧，这部分还是主要是如何统一去用trainer这个module
```

---

### Q5.3: Trainer abstraction - design rationale?
Why separate `Trainer` from `Model`?
- Separation of concerns?
- Swappable learning rules?
- Unified API?

**Your Answer:**
```
训练模块提供了统一的接口，用于训练和评估类脑模型。

用户可以通过继承canns.trainer.Trainer类来创建自定义的训练器。需要实现以下主要方法：
若要实现新的训练器，需继承 canns.trainer.Trainer 并：
在构造函数中保存目标模型及进度显示配置；
实现 train(self, train_data)，定义参数更新策略；
实现 predict(self, pattern, *args, **kwargs)，给出单样本推理逻辑，必要时使用 predict_batch封装批量推理；
遵循默认的 configure_progress 约定，让用户可以打开/关闭进度条或编译模式；
当训练器需要与特定模型协作时，约定好公共属性名（如权重、状态向量）以保证互操作性。
```

---

### Q5.4: Comparison with deep learning training?
Should we explicitly contrast:
- Hebbian vs. Backpropagation?
- Local vs. Global learning?
- When to use which?

Or assume readers already understand deep learning?

**Your Answer:**
```
这个感觉不用说什么，没必要解释太多
```

---

## Cross-Cutting Questions

### QX.1: Depth vs. Breadth balance?
Core Concepts should be:
- **Broad** survey of all components?
- **Deep** dive into fewer key topics?
- **Balanced** - moderate depth across all topics?

**Your Answer:**
```
balanced吧，最好用户易懂地介绍
```

---

### QX.2: Code examples in Core Concepts?
Should these conceptual docs include:
- **No code** - pure concepts and diagrams?
- **Code snippets** - to illustrate concepts?
- **Full examples** - like Tier 2 but more annotated?

**Your Answer:**
```
这里就不要提代码了，可以说具体的module或者是class中的属性
```

---

### QX.3: Diagrams and visualizations?
Would diagrams help? Which types:
- **Architecture diagrams** (module relationships)?
- **Workflow diagrams** (data flow)?
- **Conceptual diagrams** (e.g., attractor landscape)?
- **UML/class diagrams**?

**Your Answer:**
```
workflow可以根据下tier2中的几个how来去展示
```

---

### QX.4: Cross-references to Tier 2 and Tier 4?
How should Core Concepts link to other tiers?
- Forward links to Tier 4 (Full Details)?
- Back links to Tier 2 (Basic Intro)?
- "For hands-on guide see..., for complete API see..."?

**Your Answer:**
```
暂时先留着mark吧，以后都完成后再统一加
```

---

### QX.5: Comparison with other frameworks?
Should Core Concepts compare design choices with:
- Other neural network libraries (PyTorch, TensorFlow)?
- Other neuroscience simulation tools (NEST, Brian2)?
- Or focus only on CANNs library design?

**Your Answer:**
```
暂时不要提了
```

---

## 📝 Document Length Guidelines

**Target**: Each of the 5 topics should be ~1500-2500 words
- Longer than Tier 2 (more depth)
- Shorter than Tier 4 (not exhaustive reference)
- Readable in 15-20 minutes

Is this appropriate?

**Your Answer:**
```
主要讲核心部分，尽量精简，应该和tier2差不多，而且这部分应该不会有什么代码，所以可能还比tier2短
```

---

## 📚 Relationship to Existing Design Philosophy

The current `00_design_philosophy.rst` is comprehensive (661 lines). How should we handle it?

**Option 1**: Break it into all 5 Core Concept topics
- Overview gets intro + module list
- Each module gets its own topic (Models → Topic 2, Tasks → Topic 3, etc.)

**Option 2**: Keep it as "Overview" and create new focused docs for other topics
- Preserve current design_philosophy mostly intact
- Add 4 new topic-specific documents

**Option 3**: Hybrid approach
- Streamline overview to essentials
- Expand with new focused sections per topic
- Some content reused, some new

**Your Answer:**
```
感觉不用太动这个
```

---

## ✅ Next Steps After Answering

Once you've completed your answers:
1. Save this file
2. Let me know you're done
3. I'll generate draft documentation for all 5 Core Concept topics
4. We'll review together and iterate as needed

---

**Tips for Answering**:
- Think about what YOU needed when learning the library
- Consider different reader backgrounds (student, researcher, engineer)
- Balance between accessibility and technical depth
- Remember: This is "concepts", not "tutorials" or "API reference"
- Focus on the "why" and "when", not just the "how"
