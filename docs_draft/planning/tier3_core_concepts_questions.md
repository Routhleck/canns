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
[What are the 3-5 key design principles that guided library development?]
```

---

### Q1.2: How should we explain the module hierarchy?
Current doc has a flat list of modules. Should we show:
- **Dependency graph** (which modules depend on others)?
- **Workflow diagram** (typical usage flow)?
- **Layered architecture** (low-level to high-level)?

**Your Answer:**
```
[Which visualization/organization would be most helpful?]
```

---

### Q1.3: What should be preserved from current design_philosophy.rst?
The current document covers:
- Module overview (models, task, analyzer, trainer, pipeline)
- Usage examples
- Extension guides

**Your Answer:**
```
[Which sections are essential and should stay in Overview?
Which sections should move to other Core Concept topics?]
```

---

### Q1.4: How much technical depth for Overview?
Should Overview include:
- Code examples showing module interaction?
- Technical implementation details?
- Or just high-level concepts with links to other topics?

**Your Answer:**
```
[Level of technical detail desired]
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
Basic CANN Models:
[Purpose and use cases]

Hybrid Models:
[Concept and vision]

Brain-Inspired Models:
[Key differences and when to use]
```

---

### Q2.2: Should we explain the BaseCANN abstraction?
The library has `BaseCANN` as parent class for CANN1D/2D.
- Explain the abstract methods (`cell_coords`, `f_r`, `f_u`, `f_r_given_u`)?
- Show how inheritance works?
- Or keep it high-level?

**Your Answer:**
```
[Yes/No, and if yes, how much detail?]
```

---

### Q2.3: How to explain model variants (e.g., CANN1D vs CANN1D_SFA)?
- Focus on **conceptual differences** (SFA adds adaptation)?
- Show **when to choose** each variant?
- Include **parameter comparison**?

**Your Answer:**
```
[Approach for explaining variants]
```

---

### Q2.4: Hierarchical models (grid cells, place cells)?
These are special:
- Part of Basic Models but more complex
- Should they have dedicated explanation?
- How to explain the hierarchy concept?

**Your Answer:**
```
[How to handle hierarchical models?]
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
[Core concept of task generators]
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
[Preferred organization principle]
```

---

### Q3.3: How much detail on task-model coupling?
Some tasks need `cann_instance` (like SmoothTracking1D).
- Explain **why** this coupling exists (get_stimulus_by_pos)?
- Show **when** coupling is necessary vs. optional?
- Discuss trade-offs?

**Your Answer:**
```
[Detail level for task-model relationship]
```

---

### Q3.4: Should we explain trajectory import?
The library can import external trajectories.
- Just mention it exists?
- Explain use cases (real experimental data)?
- Show conceptual workflow?

**Your Answer:**
```
[How to handle trajectory import topic?]
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
[Clear distinction between the two analyzers]
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
[Yes/No, and reasoning for PlotConfig]
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
[Scope of RNN dynamics analysis]
```

---

### Q4.4: Topological Data Analysis (TDA)?
The library has TDA tools (UMAP, persistent homology).
- Explain **why** TDA for CANNs (detecting torus structure)?
- Show **when** to use it?
- Keep it high-level or include math?

**Your Answer:**
```
[How to handle TDA explanation?]
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
[Unifying theme for brain-inspired training]
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
[Level of neuroscience explanation]
```

---

### Q5.3: Trainer abstraction - design rationale?
Why separate `Trainer` from `Model`?
- Separation of concerns?
- Swappable learning rules?
- Unified API?

**Your Answer:**
```
[Reasoning for Trainer design]
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
[Include comparison? If yes, how much detail?]
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
[Preferred balance]
```

---

### QX.2: Code examples in Core Concepts?
Should these conceptual docs include:
- **No code** - pure concepts and diagrams?
- **Code snippets** - to illustrate concepts?
- **Full examples** - like Tier 2 but more annotated?

**Your Answer:**
```
[Code inclusion strategy]
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
[Which diagram types would be most valuable?]
```

---

### QX.4: Cross-references to Tier 2 and Tier 4?
How should Core Concepts link to other tiers?
- Forward links to Tier 4 (Full Details)?
- Back links to Tier 2 (Basic Intro)?
- "For hands-on guide see..., for complete API see..."?

**Your Answer:**
```
[Linking strategy]
```

---

### QX.5: Comparison with other frameworks?
Should Core Concepts compare design choices with:
- Other neural network libraries (PyTorch, TensorFlow)?
- Other neuroscience simulation tools (NEST, Brian2)?
- Or focus only on CANNs library design?

**Your Answer:**
```
[Include comparisons? Which frameworks?]
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
[Feedback on target length]
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
[Which option do you prefer? Or another approach?]
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
