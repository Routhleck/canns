# Why CANNs?

**Estimated Reading Time**: 8 minutes
**Target Audience**: Neuroscientists, AI researchers, engineers, and students interested in brain-inspired computing

---

## The Challenge: Modeling Continuous Neural Representations

How does your brain know where you are in a room? How do neurons in the hippocampus and entorhinal cortex maintain stable representations of your position, head direction, and navigation path—even when external cues are absent? These questions touch on one of neuroscience's most fascinating phenomena: **continuous attractor neural networks** (CANNs).

Unlike traditional neural networks that process discrete inputs and outputs, the brain must handle **continuous state spaces**—position, orientation, speed, and other variables that change smoothly over time. CANNs provide a computational framework for understanding how neural populations encode, maintain, and update these continuous representations through stable activity patterns called "attractors."

<div align="center">
<img src="../../docs/_static/smooth_tracking_1d.gif" width="400">
<p><em>A 1D CANN tracking a smoothly moving stimulus, demonstrating stable bump dynamics</em></p>
</div>

Despite decades of theoretical progress, CANNs remain challenging to work with:
- **No standardized implementations** – researchers build models from scratch for each study
- **Fragmented tools** – task generation, model simulation, and analysis require disparate codebases
- **Reproducibility barriers** – comparing results across studies is difficult without shared infrastructure
- **Steep learning curve** – students must implement complex dynamics before exploring ideas

**This is where the CANNs library comes in.**

---

## What Makes CANNs Special?

### Biological Realism Meets Theoretical Elegance

**Continuous Attractor Neural Networks** possess unique properties that bridge neuroscience and AI:

1. **Stable Continuous Representations**
   CANNs naturally maintain stable activity patterns (attractors) across continuous state spaces. Unlike Recurrent Neural Networks (RNNs) that require careful tuning, CANNs have strong theoretical foundations ensuring stability—activity bumps persist without external input, enabling short-term memory and robust encoding.

2. **Brain-Inspired Dynamics**
   Compared to attention-based models like Transformers, CANNs operate through mechanisms closer to biological neural circuits. They excel at modeling:
   - **Place cells** in the hippocampus (spatial position encoding)
   - **Grid cells** in the entorhinal cortex (periodic spatial maps)
   - **Head direction cells** (angular orientation)
   - **Working memory** networks (persistent activity)

3. **Continuous State Space Processing**
   Traditional deep learning models discretize the world. CANNs process continuous variables natively—matching how brains handle smooth changes in position, orientation, and sensory stimuli.

4. **Path Integration and Navigation**
   CANNs perform path integration naturally: integrating velocity signals over time to track position without external landmarks—a core computation in rodent navigation and human spatial cognition.

<div align="center">
<img src="../../docs/_static/theta_sweep_animation.gif" width="600">
<p><em>Theta sweep dynamics in grid cell and head direction networks</em></p>
</div>

### Recent Breakthrough: A-CANN and Theta Sweep Phenomena

A major recent advance combined **CANNs with neural adaptation** (A-CANN) to explain diverse hippocampal sequence replay patterns during rest and sleep. By introducing adaptation—a universal neural property—as a single control variable, researchers unified seemingly disparate phenomena: stationary replay, diffusive sequences, and super-diffusive sweeps.

This work demonstrates CANN's power: **simple, biologically plausible mechanisms can explain complex neural dynamics with profound implications for memory encoding and retrieval.**

---

## Who Should Use This Library?

The CANNs library serves three main communities:

### 🔬 Computational Neuroscientists
Continuous attractor networks are gaining traction in systems neuroscience. Researchers want to:
- **Analyze experimental data** for attractor signatures
- **Build CANN models** to validate hypotheses against neural recordings
- **Reproduce and extend** published CANN studies efficiently

### 🛠️ Engineers & Developers
As CANNs mature, they require **standardized development practices**—similar to how Transformers revolutionized NLP with consistent APIs and shared infrastructure. Engineers need unified tools to:
- Implement bio-inspired navigation and memory systems
- Benchmark CANN architectures systematically
- Deploy CANN-based applications in robotics and AI

### 🎓 Students & Educators
Learning CANNs shouldn't require implementing complex dynamics from scratch. Students benefit from:
- **Ready-to-use models** for hands-on exploration
- **Clear examples** demonstrating key concepts
- **Modifiable code** to experiment with parameters and architectures

**Without standardized tools, each group reinvents the wheel. The CANNs library changes that.**

---

## Key Application Scenarios

### 1. Theta Sweep Modeling and Analysis

**The Challenge**: Hippocampal neurons exhibit rich sequential firing patterns during rest and sleep—stationary, diffusive, super-diffusive—with important cognitive functions. Understanding these "theta sweeps" is central to memory research.

**The Solution**: The A-CANN framework (CANN + neural adaptation) explains these diverse patterns through a single variable. This library provides:
- **Pre-built models**: `HeadDirectionNetwork`, `GridCellNetwork`, `PlaceCellNetwork`
- **Specialized visualization**: Theta sweep animation and analysis tools
- **Reproducible pipelines**: `ThetaSweepPipeline` orchestrates simulation, analysis, and plotting

**Impact**: Researchers can immediately build on this work without reimplementing models and analysis tools.

<div align="center">
<img src="../../docs/_static/CANN2D_encoding.gif" width="400">
<p><em>2D spatial encoding patterns in CANN networks</em></p>
</div>

### 2. Education and Research Training

**The Challenge**: Teaching CANNs traditionally requires students to implement models from scratch each semester. This consumes weeks that could be spent on scientific exploration.

**The Solution**: With this library, students can:
- Instantiate CANN models in **3 lines of code**
- Generate task data (smooth tracking, population coding) with **minimal setup**
- Visualize dynamics with **built-in analysis tools**

**Impact**: Educators report that students now focus on **understanding mechanisms** rather than debugging implementations.

### 3. High-Performance Simulation

**The Challenge**: Long simulations and large-scale experiments (e.g., parameter sweeps, topological data analysis) are computationally expensive.

**The Solution**: The companion **`canns-lib`** Rust library provides:
- **700× speedup** for spatial navigation tasks vs. pure Python (RatInABox-compatible API)
- **1.13× average, 1.82× peak speedup** for topological analysis (Ripser algorithm)
- **Perfect accuracy** – 100% result matching with reference implementations
- **GPU/TPU support** – via JAX/BrainState backends

**Impact**: What once took hours now runs in minutes. Researchers can explore parameter spaces and analyze datasets at scale.

| Simulation Steps | Pure Python | canns-lib (Rust) | Speedup |
|------------------|-------------|------------------|---------|
| 10²              | 0.020 s     | <0.001 s         | 477×    |
| 10⁴              | 1.928 s     | 0.003 s          | 732×    |
| 10⁶              | 192.775 s   | 0.266 s          | 726×    |

---

## Why This Library? The Unified Ecosystem Advantage

### The Problem: A Fragmented Landscape

Currently, CANN research resembles **NLP before Transformers**—each lab uses custom code, diverse implementations, and incompatible formats. This fragmentation causes:
- **Reinvention overhead**: Researchers re-implement basics repeatedly
- **Reproducibility issues**: Comparing studies requires reverse-engineering code
- **Slow progress**: No shared models, benchmarks, or best practices

### The Vision: CANNs as the "Hugging Face Transformers" of Attractor Networks

Just as Hugging Face standardized Transformer usage, **the CANNs library aims to unify CANN research**:

1. **Standardized Model Zoo**
   - Pre-built 1D/2D CANNs, SFA variants, hierarchical networks
   - Brain-inspired models: Hopfield networks, spike-based (LIF) models
   - Hybrid architectures combining CANNs with ANNs

2. **Unified Task API**
   - Smooth tracking, population coding, closed/open-loop navigation
   - Import experimental trajectories directly
   - Consistent data formats across tasks

3. **Complete Analysis Pipeline**
   - Energy landscapes, tuning curves, firing fields, spike embeddings
   - Topological data analysis (UMAP, TDA, persistent homology)
   - Theta sweep and RNN dynamics analysis

4. **Extensible Architecture**
   - Base classes (`BasicModel`, `Task`, `Trainer`, `Pipeline`) for custom components
   - Built on BrainState for JAX-powered JIT compilation and autodifferentiation
   - GPU/TPU acceleration out of the box

5. **Community & Sharing**
   - Open-source foundation for model and benchmark sharing
   - Unified evaluation protocols
   - Growing ecosystem of examples and tutorials

---

## Technical Foundations

What makes this library powerful?

### 🚀 Performance Through BrainState + Rust
- **BrainState integration**: High-level dynamics API with JAX's JIT compilation, automatic differentiation, and GPU/TPU support
- **canns-lib acceleration**: Rust-powered hot paths for task generation and topological analysis
- **Efficient compilation**: Write models in simple Python, run at C++ speeds

### 🧩 Comprehensive Toolchain
- **Models**: 1D/2D CANNs, hierarchical networks, SFA variants, brain-inspired models
- **Tasks**: Tracking, navigation, population coding, trajectory import
- **Analyzers**: Visualization, TDA, bump fitting, dynamics analysis
- **Trainers**: Hebbian learning, prediction workflows
- **Pipelines**: End-to-end workflows (e.g., theta sweeps) in single calls

### 🔬 Research-Grade Quality
- **Validated implementations**: Models reproduce published results
- **Comprehensive testing**: Pytest suite covering key behaviors
- **Active development**: Regular updates, bug fixes, community contributions

---

## Current Status & Future Directions

**Development Stage**: The library has been under active development for 4 months and is currently in **beta** (v0.x). It is being used internally by our research group, and we're actively expanding features based on user feedback.

**Validation**:
- ✅ Models reproduce established CANN behaviors
- ✅ Performance benchmarks show significant speedups (canns-lib)
- ✅ Growing collection of working examples across models and tasks

**Roadmap**:
- Expand brain-inspired model collection (recurrent networks, spike-based models)
- Add hybrid CANN-ANN architectures
- Develop comprehensive benchmarking suite
- Build community-contributed model zoo
- Publish companion paper documenting library design

**Limitations** (we believe in transparency):
- Beta software – APIs may evolve based on feedback
- Documentation is actively expanding (your contributions welcome!)
- Limited pre-trained models currently (we're building this out)
- Smaller community compared to mature deep learning frameworks (but growing!)

---

## Next Steps: Dive In!

### Quick Start
Ready to build your first CANN? Jump to our **[Quick Start Guide](link-to-quick-start)** for a hands-on walkthrough in <10 minutes.

### Installation
```bash
pip install canns          # CPU version
pip install canns[cuda12]  # GPU support (Linux)
```

### Learn More
- **[Core Concepts](link-to-core-concepts)**: Understand the library's design philosophy
- **[Basic Tutorials](link-to-basic-tutorials)**: Step-by-step guides for common tasks
- **[Full API Documentation](link-to-api-docs)**: Complete reference for all models and methods
- **[Examples Gallery](link-to-examples)**: Ready-to-run scripts demonstrating key features

### Get Involved
- 🐛 **Report issues**: [GitHub Issues](https://github.com/routhleck/canns/issues)
- 💬 **Ask questions**: [GitHub Discussions](https://github.com/routhleck/canns/discussions)
- 🤝 **Contribute**: Check our [Contributing Guide](link-to-contributing)
- ⭐ **Star the repo**: [github.com/routhleck/canns](https://github.com/routhleck/canns)

---

## Summary: Why CANNs?

| **Question** | **Answer** |
|-------------|------------|
| **What problem does it solve?** | Unifies fragmented CANN research with standardized models, tasks, and analysis tools |
| **Who is it for?** | Neuroscientists analyzing data, engineers building systems, students learning dynamics |
| **What makes it unique?** | First comprehensive CANN library—complete ecosystem from models to visualization |
| **How fast is it?** | 700× speedup for navigation, GPU/TPU support, JIT compilation |
| **Can I trust it?** | Validated against published results, actively developed, open source |
| **Where do I start?** | `pip install canns` → Quick Start Guide → Build your first CANN in 10 minutes |

**The era of fragmented CANN implementations is ending. The era of unified, reproducible, accessible attractor network research is beginning.**

**Let's build it together.**

---

*Have questions or suggestions for this documentation? Open an issue or discussion on [GitHub](https://github.com/routhleck/canns)!*
