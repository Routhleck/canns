# Tier 2: Basic Intro (How-To Guides) - Planning Questions

**Status**: 🔴 Awaiting your answers
**Target Audience**: Engineers/Developers, Graduate students, Cross-domain collaborators
**Estimated Reading Time per Guide**: 10-15 minutes
**Writing Style**: Practical, task-oriented, step-by-step

---

## 📋 Section Overview

The "Basic Intro" tier consists of **5 focused how-to guides** that help users accomplish common tasks quickly. These are NOT comprehensive tutorials (that's Tier 4), but rather **quick-start guides** for specific goals.

### Key Principles for These Guides:
- **Goal-oriented**: Start with "You want to do X? Here's how."
- **Minimal but complete**: Show the essential steps, skip edge cases
- **Working examples**: Every guide should have a runnable code snippet
- **Bridge to deeper docs**: Link to Core Concepts and Full Details for more

---

## 🎯 The 5 How-To Guides

Based on your outline, we'll create:

1. **How to build CANN model?** - Create and initialize basic CANN models
2. **How to generate task data?** - Generate stimuli and task environments
3. **How to analyze CANN model?** - Visualize and understand model dynamics
4. **How to analyze experimental data?** - Apply CANN analysis to real data
5. **How to train brain-inspired model?** - Train models with Hebbian learning

---

## ❓ Questions for Each Guide

Please answer the questions below for each of the 5 guides. Your answers will shape the content.

---

# Guide 1: How to Build CANN Model?

## Context
This guide should help users create their first CANN model in minutes. It should cover the basics of instantiation, initialization, and running a simple forward pass.

### Q1.1: What should users learn from this guide?
Choose the scope (can select multiple aspects):
- [x] Creating 1D CANN
- [ ] Creating 2D CANN
- [x] Understanding key parameters (num, k, tau, etc.)
- [x] Initializing model state
- [x] Running a single forward pass
- [ ] Common mistakes to avoid

**Your Answer:**
```
就介绍如何构建模型吧，不过我这里是使用的brainstate来去实现的，更多的是简单介绍brainstate的模型构建方法，然后一个简单的CANN1D是怎么构建起来的，然后是应该如何调用
```

---

### Q1.2: What's the simplest working example?
What's the minimal code that creates and runs a CANN model?
- Should it be CANN1D or CANN2D?
- Should it include visualization, or just model creation?
- What parameters should we show vs. use defaults?

**Your Answer:**
```
examples/cann/cann1d_oscillatory_tracking.py
```

---

### Q1.3: What are the 3 most common mistakes beginners make?
When you've seen students or new users build CANNs, what do they get wrong?
- Forgetting to call `init_state()`?
- Wrong parameter values?
- Not setting `brainstate.environ.set(dt=...)`?

**Your Answer:**
```
我觉得你上面列的就很好
1. 忘记init_state()
2. 没有设置brainstate.environ.set(dt=...)
```

---

### Q1.4: What should users do AFTER reading this guide?
Where should they go next?
- Link to "How to generate task data?"
- Link to Core Concepts > Model Collections?
- Link to Full Details > CANN1D/CANN2D?

**Your Answer:**
```
上面列的就挺好的，不过link full details要转到Model Collections，以及需要自己动手构建model的要link到brainstate的tutorial中
```

---

# Guide 2: How to Generate Task Data?

## Context
This guide shows users how to create stimuli and task environments for their CANN models—tracking tasks, navigation environments, population coding, etc.

### Q2.1: Which task types should this guide cover?
The library has multiple task modules. Which should we include in this basic guide?
- [x] Smooth tracking (1D/2D)
- [ ] Population coding
- [ ] Closed-loop navigation
- [ ] Open-loop navigation
- [ ] Importing external trajectories

**Your Answer:**
```
就拿smooth tracking的1D举例吧，其实大致用法都类似，都是需要实例化、get_data()，然后拿到这个task中的一些attribute或者是data
```

---

### Q2.2: What's the simplest task example?
What's the easiest task to demonstrate?
- SmoothTracking1D with fixed Iext values?
- Something else?

**Your Answer:**
```
同样是examples/cann/cann1d_oscillatory_tracking.py中就有用到
```

---

### Q2.3: How should we explain task data structure?
Users need to understand what `task.data` contains and how to use it.
- Should we show the data shape/format?
- Should we explain `run_steps` and `time_step`?
- How much detail is appropriate for a quick guide?

**Your Answer:**

```
对可以show下，因为未来我希望将我们这些markdown转换为notebook，所以是可以运行来进行展示的
```

---

### Q2.4: What's the connection between tasks and models?
How should we explain the relationship?
- Tasks generate inputs for models
- Models consume task data in simulation loops
- Tasks can be model-agnostic or model-specific (cann_instance parameter)

**Your Answer:**
```
对的可以在这里简单说下你上述的几条
```

---

### Q2.5: What should users do AFTER reading this guide?
**Your Answer:**
```
继续阅读如何分析CANN model，然后Link to Core Concepts > Task Generators和Full Details > Task Generators
```

---

# Guide 3: How to Analyze CANN Model?

## Context
This guide covers visualizing and understanding CANN dynamics—energy landscapes, tuning curves, bump dynamics, etc.

### Q3.1: What analysis methods should this guide include?
The analyzer module has many tools. Which are essential for beginners?
- [x] Energy landscape visualization (1D/2D)
- [ ] Tuning curves
- [ ] Bump tracking/decoding
- [ ] Firing fields
- [ ] Animation vs. static plots
- [ ] PlotConfig system

**Your Answer:**
```
就简单介绍一个energy landscape的static plot和animation gif吧
```

---

### Q3.2: Should we show the PlotConfig approach?
The library has both old-style function calls and new PlotConfig dataclasses.
- Show both?
- Only PlotConfig (recommended way)?
- Only old-style (simpler for beginners)?

**Your Answer:**
```
Only PlotConfig，然后只在这里简单介绍，要link到Full Detail > Analysis Methods > Model Analyzer > Plot Config
```

---

### Q3.3: What's the most impressive but simple visualization?
Which visualization best demonstrates CANN behavior for a first-time user?
- 1D energy landscape animation?
- 2D bump tracking?
- Something else?

**Your Answer:**

```
1D的oscillatory tracking，可以考虑model再加上sfa的plot来对比下
```

---

### Q3.4: Should this guide include analysis of results?
Beyond generating plots, should we explain what to look for?
- How to tell if a CANN is working correctly?
- What a healthy bump looks like?
- Common issues (diffusion, instability)?

**Your Answer:**
```
就直接最终在notebook运行代码展示吧
```

---

### Q3.5: What should users do AFTER reading this guide?
**Your Answer:**
```
继续阅读如何分析Experimental Data，然后Link to Core Concepts > Analysis Methods > Model Analyzer 和Full Details > Analysis Methods > Model Analyzer
```

---

# Guide 4: How to Analyze Experimental Data?

## Context
This guide shows how to apply CANN analysis methods to real experimental data (e.g., neural recordings, behavioral trajectories).

### Q4.1: What types of experimental data can be analyzed?
Based on the library's capabilities:
- [x] Neural spike data (bump fitting)
- [ ] Behavioral trajectories (place field analysis)
- [x] Time series data (topology analysis)
- [x] RNN Model trajectories
- [ ] Other?

**Your Answer:**

```
这里可以先暂时说下该文档等待校准，可能之后再去修正
```

---

### Q4.2: What's the simplest example of experimental data analysis?
What's the easiest analysis to demonstrate?
- 1D bump fitting to synthetic "experimental" data?
- 2D place field analysis?
- Something else?

**Your Answer:**
```
1D bump fitting to real experimental data
```

---

### Q4.3: Should we provide sample data?
Do we need to include example datasets, or should users provide their own?
- Include synthetic data that mimics real experiments?
- Point to public datasets?
- Assume users have their own data?

**Your Answer:**

```
这里应该有我们自己上传的一些数据，可以使用canns.data中的方法来获取下示例data
```

---

### Q4.4: What's the relationship to model analysis (Guide 3)?
How are analyzing models vs. experimental data similar/different?
- Similar tools (bump fitting, tuning curves)?
- Different workflows?
- When to use which?

**Your Answer:**
```
Totally Different workflows, especially for neuroscientists who have experimental data
```

---

### Q4.5: What are the key steps in experimental data analysis?
What's the typical workflow?
1. Load data
2. Preprocess/format
3. Apply CANN analysis
4. Interpret results

**Your Answer:**
```
Load Data -> Change to proper Data Input (if needed) -> processing -> analysis result and visualization
```

---

### Q4.6: What should users do AFTER reading this guide?
**Your Answer:**
```
继续阅读如何训练Brain-Inspired Model，然后Link to Core Concepts > Analysis Methods > Data Analyzer 和Full Details > Analysis Methods > Data Analyzer
```

---

# Guide 5: How to Train Brain-Inspired Model?

## Context
This guide introduces training models with Hebbian learning and the Trainer framework.

### Q5.1: Which models should this guide cover?
The library has multiple trainable models:
- [x] Amari-Hopfield networks
- [ ] Linear feedforward models
- [ ] Spike-based (LIF) models
- [ ] Other brain-inspired models

**Your Answer:**
```
仅说下最简单的AmariHopfield network吧
```

---

### Q5.2: What's the simplest training example?
What's the easiest training task to demonstrate?
- Hopfield pattern storage?
- Hebbian weight adaptation?
- Something else?

**Your Answer:**
```
如何用hebbian方法来训练，从而实现对image的记忆
```

---

### Q5.3: How much should we explain about Hebbian learning?
This is a basic guide, not a neuroscience textbook.
- Brief explanation of Hebbian principle?
- Just show the API without theory?
- Link to external resources?

**Your Answer:**

```
Brief
```

---

### Q5.4: Should we show the Trainer framework?
The library has `HebbianTrainer` and abstract `Trainer` base.
- Show HebbianTrainer usage?
- Explain the Trainer abstraction?
- Just focus on model.train() methods?

**Your Answer:**

```
最好通过设计哲学中的内容来介绍下trainer的大致用法，然后这里应该不是model.train()而是trainer.train
```

---

### Q5.5: What's different about training CANNs vs. ANNs?
Users coming from deep learning need to understand the paradigm shift.
- No backpropagation?
- Local learning rules?
- Unsupervised learning?

**Your Answer:**
```
这里简单介绍下就好
```

---

### Q5.6: What should users do AFTER reading this guide?
**Your Answer:**
```
继续看Core Concepts，然后Link to Core Concepts > Brain-Inspired Training 和Full Details > Brain-Inspired Training
```

---

# Cross-Cutting Questions

These apply to all 5 guides:

### QX.1: Should each guide be standalone or build on previous ones?
- **Standalone**: Each guide is independent, users can read any order
- **Sequential**: Guides build on each other (1→2→3→4→5)

**Your Answer:**

```
并不太需要有顺序，可能前三个还有些相互使用的循序渐进，不过后两个就完全是另外的内容
```

---

### QX.2: How much code should each guide contain?
- One main example + variations?
- Multiple independent examples?
- Just code snippets without full context?

**Your Answer:**
```
就是一个example，然后可以展示各种变量的数据结构用于说明（notebook的形式这样子）
```

---

### QX.3: Should we include "Common Pitfalls" sections?
For each guide, should we have a section on common mistakes?

**Your Answer:**
```
暂时先不要了
```

---

### QX.4: Should examples be self-contained or reference examples/?
Should code be:
- Complete and runnable in the docs?
- Abbreviated with links to `examples/` directory?
- Mix of both?

**Your Answer:**
```
直接可以runnable的
```

---

### QX.5: Language considerations?
Remember we're doing English first. Should these guides:
- Use simple English (international audience)?
- Include technical terms in both English and Chinese?
- Just focus on English for now?

**Your Answer:**
```
simple English
```

---

## 📝 Additional Notes

### Overall Structure Suggestion
Each guide could follow this template:
1. **Goal** - "By the end of this guide, you'll be able to..."
2. **Prerequisites** - "You should have completed..."
3. **Quick Example** - Minimal working code
4. **Step-by-Step Explanation** - Break down the example
5. **Common Variations** - Other common use cases
6. **Next Steps** - Where to go from here

Do you want to follow this structure, or prefer something different?

**Your Answer:**
```
我建议随意一些，尽量让读者更便捷地进行了解，结构最好根据每个intro来去改编
```

---

## ✅ Next Steps After Answering

Once you've completed your answers:
1. Save this file
2. Let me know you're done
3. I'll generate draft documentation for all 5 guides
4. We'll review together and iterate as needed

---

**Tips for Answering**:
- These are **basic guides**, not comprehensive tutorials
- Focus on the **most common use case** for each topic
- Keep it **practical**—users want to accomplish tasks, not read theory
- Think about what YOU wish you had when starting with CANNs
- It's okay to say "skip this" or "cover later" for complex topics
