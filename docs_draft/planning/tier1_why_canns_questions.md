# Tier 1: Why CANNs? - Planning Questions

**Status**: 🔴 Awaiting your answers
**Target Audience**: Engineers/Developers, Graduate students, Cross-domain collaborators
**Estimated Reading Time**: 5-10 minutes
**Writing Style**: Motivating, accessible, practical

---

## 📋 Section Overview

The "Why CANNs?" section is the **front door** of your documentation. It needs to:
- Convince readers why CANNs matter scientifically and practically
- Help them quickly assess if this library fits their needs
- Provide context without overwhelming technical details
- Bridge from "I heard about this" to "I want to try this"

This is NOT a technical deep-dive (that comes later). Think of it as your "elevator pitch" expanded into a compelling narrative.

---

## ❓ Questions to Answer

Please answer the following questions in the space provided after each question. Your answers will guide the documentation content.

### 1. Scientific Motivation

**Q1.1**: What is the core neuroscience/computational problem that CANNs address?
- What brain functions or neural phenomena do CANNs model?
- Why are traditional neural networks insufficient for these problems?

**Your Answer:**
```
CANNs是一个专门用于简化连续吸引子神经网络研究的Python库。CANNs有着内置的可调用模型、任务生成工具，使神经科学和AI研究人员能够快速从理论概念转向可重复的仿真实验，以及针对CANN模型的分析工具和实验得来的数据进行CANN分析。
```

---

**Q1.2**: What are the key theoretical advantages of CANNs?
- What makes the "continuous attractor" property special?
- How do CANNs relate to biological neural circuits?

**Your Answer:**
```
连续吸引子神经网络(Continuous Attractor Neural Networks, CANNs)作为研究核心具有多方面的独特优势。首先，从理论角度看，CANNs能够自然地表达和处理连续状态空间中的信息，这与大脑处理连续物理世界信息的方式高度一致。与传统的RNN相比，CANNs具有更强的动力学稳定性和更清晰的理论基础，能够形成稳定的神经活动模式（吸引子状态），这些状态可以在没有外部输入的情况下维持，从而实现信息的短期记忆功能。相比于Transformer等注意力机制模型，CANNs的计算机制更接近生物神经系统的实际工作方式，特别是在空间认知、导航等任务中表现出与大脑神经元群体活动高度相似的特性。
在应用方面，CANNs在空间表征、路径积分、头朝向编码等脑启发计算任务上有着天然的优势。例如，CANNs可以模拟海马体的位置细胞、内嗅皮层的网格细胞以及头朝向细胞等神经元群体的活动模式，为理解大脑空间认知机制提供了重要工具。此外，CANNs在处理连续变化的时空信息、维持工作记忆、执行序列学习等任务上也展现出独特的能力，这些特性使其成为连接神经科学与人工智能的重要桥梁。

当前研究和应用CANNs面临的主要瓶颈和挑战包括：首先，缺乏标准化的实现工具，导致研究者需要从零开始构建模型，增加了研究门槛和复现难度；其次，CANNs的训练和参数调优相对复杂，特别是在确保形成稳定吸引子状态方面存在挑战；第三，现有的神经网络框架主要针对深度学习优化，对CANNs等动力学系统的特殊需求支持不足；第四，缺乏统一的评估基准和任务环境，使得不同研究之间的比较变得困难；最后，理论分析工具不足，难以深入理解CANNs的内部工作机制和动力学特性。
```

---

### 2. Practical Use Cases

**Q2.1**: Who should use this library? (Be specific about roles/backgrounds)

- Computational neuroscientists studying...?
- Engineers building...?
- Students learning about...?

**Your Answer:**
```
随着连续吸引子神经网络的概念在前沿领域的火热，神经科学家都希望能够对他们的实验数据进行分析，看是否有吸引子存在，以及尝试使用CANN来去根据他们的数据进行建模验证。预计未来随着CANN的逐步进展，就像transformer一样，是需要指导工程师们进行统一的开发与工程实践标准。而学生则是需要一个便捷的工具，而不需要完全从头来去实现一个CANN，通过简单的修改参数以及模型修改，就能给他们带来学习的热情和兴趣。
```

---

**Q2.2**: What are 3-5 concrete research/application scenarios where CANNs excel?
- Please provide specific examples (e.g., "modeling head direction cells in rodent navigation", "building bio-inspired path integration systems")
- For each scenario, briefly note what problem it solves

**Your Answer:**
```
Scenario 1: Theta Sweep Modeling and Analysis
- Problem solved: 对于Theta Sweep，我们目前有一系列进展：用一个统一的模型，A-CANN（连续吸引子神经网络CANN+神经元活动的自适应Adaptation），成功解释了不同实验所发现的大脑海马神经元在静止或睡眠时所展现的丰富序列放电模式，包括静止、扩散、超扩散等，而这些序列活动都具有重要认知功能。CANN已经被广泛用于海马神经网络的建模，该工作的核心贡献是发现：CANN在引入Adaptation，这一神经系统的普遍性质后，adaptation作为单一变量，能够解释大量貌似差别巨大的神经元群的序列放电活动，从而为理解记忆编码与提取的神经机制提供了全新框架。在CANNs中，我们提供了一系列Theta Sweep的可调用的Model（Head Direction Network, Grid Cell Network, Place Cell Network），以及针对Theta Sweep需要的可视化分析方法，让该领域的科研人员便捷的对这项重要工作进行following。

Scenario 2: Speedup Simulation Time
- Problem solved: 对于建模CANN中常用的任务数据生成以及拓扑分析方法（可能未来有更多的场景），我们发布了canns-lib来去进行加速，canns-lib是一个基于Rust的加速库，为CANNs Python包提供优化的计算后端。canns-lib在拓扑数据分析方面表现出色，其Ripser模块在54个基准测试中实现了平均1.13倍的加速（峰值1.82倍），同时保持与ripser.py 100%的结果匹配。在空间导航模块方面，通过与RatInABox保持完全API兼容，实现了相对于纯Python参考实现约700倍的运行时加速，感知明显。

Scenario 3: 参考Q2.1，在教学方面是极大的进步
- Problem solved: 目前在CANN建模的课程中，之前都是用BrainPy给几个简单example，BrainPy内置模型并没有包含CANN，所以基本上每次学生都需要自己进行代码的实现。

[Add more if needed]
```

---

### 3. Library Advantages

**Q3.1**: Why should someone use THIS CANNs library instead of implementing from scratch?

- What are the key features/benefits?
- What pain points does it solve for researchers?

**Your Answer:**
```
这对于大部分研究人员是极大地效率提升，我们通过整理整合各类CANN模型实现、模型分析方法、数据分析方法、统一的任务生成工具，来为该领域进行革命性地效率提升。
```

---

**Q3.2**: What are the technical foundations that make this library powerful?

- JAX-based computation?
- BrainX/BrainState integration?
- GPU/TPU support?
- Other key technical advantages?

**Your Answer:**
```
基于BrainState的高效JIT编译与简单地建模语法，通过canns-lib的rust-based加速库针对场景进行性能提升，以及丰富地分析工具
```

---

### 4. Comparison with Alternatives

**Q4.1**: What other tools/frameworks do researchers currently use for similar work?

- List 2-4 alternatives (e.g., custom MATLAB code, other neural network libraries, specific CANN implementations)

**Your Answer:**
```
目前还真没有，我的思路就是做CANN领域的huggingface的transformer。
目前，CANNs领域确实存在类似于transformers出现前NLP领域的"各自为战"状况。不同研究组使用不同的实现方式，缺乏统一的标准和共享平台，导致研究成果难以比较和复用。这种状况主要表现在：模型实现多样且不兼容，缺乏标准化的接口和数据格式，复现他人工作困难，以及缺乏共享的预训练模型和评估基准。canns库将通过以下方式改变这一现状：建立统一的模型实现标准和接口规范；提供共享平台，促进预训练模型的交流；设计标准化的任务环境和评估流程；构建活跃的开发者和研究者社区。通过这些努力，canns有望成为CANNs研究的中心枢纽，类似于transformers之于NLP领域的地位，从而加速整个领域的发展。
```

---

**Q4.2**: How does this library compare? (Honest assessment)
- What does this library do better?
- What trade-offs exist?
- When should someone use an alternative instead?

**Your Answer:**
```
显然现在并没有比较对象
```

---

### 5. Success Stories & Validation

**Q5.1**: Are there published papers, projects, or case studies using this library?

- If yes, list them with brief descriptions
- If no, what validation exists (e.g., reproducing known results)?

**Your Answer:**

```
这个项目还处在开发中，只发展了4个月，目前组内有使用我们的package
```

---

**Q5.2**: What specific scientific results or benchmarks demonstrate the library's effectiveness?

- Any quantitative comparisons?
- Reproductions of classic CANN studies?
- Novel findings enabled by the library?

**Your Answer:**
```
canns-lib中有对比标准实现方法package的benchmark：

1.
High-performance implementation of the Ripser algorithm for computing Vietoris-Rips persistence barcodes.

Performance Highlights
Mean speedup: 1.13x across 54 benchmarks vs ripser.py
Peak speedup: Up to 1.82x on certain datasets
Memory efficiency: 1.01x memory ratio (stable usage)
Perfect accuracy: 100% match with ripser.py results

2.
Accelerated reimplementation of RatInABox environments and agents with PyO3/
Rust. Supports solid and periodic boundaries, arbitrary polygons, holes, and
thigmotaxis wall-following.

#### Performance Snapshot

The spatial backend delivers ~700× runtime speedups vs. the pure-Python
reference when integrating long trajectories.  Benchmarked with
`benchmarks/spatial/step_scaling_benchmark.py` (`dt=0.02`, repeats=1).

| Steps | RatInABox Runtime | canns-lib Runtime | Speedup |
|------:|------------------:|------------------:|--------:|
| 10²   | 0.020 s | <0.001 s | 477× |
| 10³   | 0.190 s | <0.001 s | 713× |
| 10⁴   | 1.928 s | 0.003 s | 732× |
| 10⁵   | 19.481 s | 0.027 s | 718× |
| 10⁶   | 192.775 s | 0.266 s | 726× |
```

---

### 6. Getting Started Preview

**Q6.1**: What can a user accomplish in 10 minutes with this library?
- This will be a teaser for the "Quick Start" section
- Something impressive but achievable

**Your Answer:**

```
我觉得以下这两个example可能很好地显示了任务生成->模型调用->分析可视化:
examples/cann/cann1d_oscillatory_tracking.py
examples/cann/cann2d_tracking.py
```

---

**Q6.2**: What's a compelling visual example to show?
- What animation/plot best demonstrates CANNs in action?
- (We can reference existing examples or create new ones)

**Your Answer:**

```
README中的可视化展示就ok，暂时先用这个，后续可以考虑再增添一些东西
```

---

## 📝 Writing Guidelines for This Section

When I generate the documentation from your answers, I will:

1. **Start with a hook**: Lead with the most compelling problem/application
2. **Use the "Problem → Solution → Benefit" pattern**
3. **Include 1-2 visual examples** early to make it concrete
4. **Keep it under 1500 words** - this is high-level motivation
5. **End with a clear call-to-action** pointing to Quick Start
6. **Avoid**:
   - Heavy mathematics (save for Core Concepts)
   - Implementation details (save for tutorials)
   - Assuming readers know what CANNs are (start from basics)

---

## 📚 Reference Materials

For context while answering, you may want to review:
- Current Design Philosophy doc: `/docs/en/0_getting_started/00_design_philosophy.rst`
- Example scripts in: `/examples/`
- Published papers using CANNs (if any)
- README.md in project root

---

## ✅ Next Steps After Answering

Once you've completed your answers:
1. Save this file
2. Let me know you're done
3. I'll read your answers and generate the draft documentation
4. We'll review together and iterate as needed

---

**Tips for Answering**:
- Be honest and specific
- It's okay to say "I'm not sure" - we can refine together
- Think about what YOU wished you knew before starting to use/develop this library
- Consider different reader backgrounds (student vs. engineer vs. researcher)
