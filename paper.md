---
title: "CANNs: Continuous Attractor Neural Networks Toolkit with ASA for Attractor Structure Analysis"
tags:
  - Python
  - continuous attractor neural networks
  - computational neuroscience
  - topological data analysis
authors:
  - name: Sichao He
    orcid: 0009-0009-4564-1753
    affiliation: 1
  - name: Aiersi Tuerhong
    orcid: 0009-0006-4603-4734
    affiliation: 2
  - name: Shangjun She
    orcid: 0009-0008-4490-7612
    affiliation: 3
  - name: Tianhao Chu  
    orcid: 0000-0001-9910-9361
    affiliation: 3
  - name: Yuling Wu
    orcid: 0009-0001-5303-5413
    affiliation: 1
  - name: Junfeng Zuo
    affiliation: 1
  - name: Si Wu
    orcid: 0000-0001-9650-6935
    corresponding: true
    affiliation: "1, 3, 4, 5, 6"
      
affiliations:
  - name: Peking-Tsinghua Center for Life Sciences, Academy for Advanced Interdisciplinary Studies, Peking University, Beijing, China
    index: 1
  - name: College of Mathematics and Statistics, Chongqing University, Chongqing, China
    index: 2
  - name: School of Psychological and Cognitive Sciences, Peking University, Beijing, China
    index: 3
  - name: School of Psychology and Cognitive Sciences, Peking University, Beijing, China
    index: 4
  - name: PKU-IDG/McGovern Institute for Brain Research, Peking University, Beijing, China
    index: 5
  - name: Center of Quantitative Biology, Peking University, Beijing, China
    index: 6

date: 16 January 2026
bibliography: docs/refs/references.bib
---
# Summary

CANNs (Continuous Attractor Neural Networks toolkit) is a Python library built on BrainPy, a powerful framework for brain dynamics 
programming. It streamlines experimentation with continuous attractor neural networks and related brain-inspired models. The library 
delivers ready-to-use models, task generators, analysis tools, and pipelines—enabling neuroscience and AI researchers to move quickly 
from ideas to reproducible simulations.


# Statement of need

Continuous Attractor Neural Networks (CANNs) provide a theoretical framework for understanding how the brain encodes continuous
variables—such as spatial position, head direction, and movement trajectories—through stable neural activity patterns. These models
explain key phenomena in hippocampal place cells [@o1971hippocampus], entorhinal grid cells [@hafting2005microstructure], and head
direction systems [@taube1990head]. Despite their importance, CANN research suffers from fragmentation: researchers implement models
from scratch, use incompatible codebases, and face significant reproducibility barriers. This lack of standardization slows progress
and creates steep learning curves for newcomers.

CANNs addresses this gap by providing a unified Python toolkit built on BrainPy [@wang2023brainpy]. It delivers: (1) standardized
implementations of CANN and related brain-inspired models, including canonical frameworks (Wu-Amari-Wong [@amari1977dynamics; @wu2008dynamics],
adaptation-augmented CANNs [@mi2014spike; @li2025dynamics], grid cell networks [@burak2009accurate]), alongside additional attractor
architectures; (2) integrated task generation, simulation, and analysis pipelines; and (3) high-performance computation via JAX JIT
compilation and optional Rust acceleration. By standardizing workflows—analogous to Hugging Face
Transformers in deep learning—this library accelerates reproducible research and lowers barriers for computational neuroscientists,
AI engineers, and students exploring attractor dynamics.

# Software design

![Layer hierarchy of the CANNs library showing five levels: Application (Pipeline orchestration), Functional (Task, Trainer, Analyzer, Utils modules), Core Models (CANN implementations), Foundation (BrainPy/JAX and Rust FFI backends), and Hardware (CPU/GPU/TPU support).\label{fig:architecture}](images/architecture.png)

The CANNs library follows a modular architecture (\autoref{fig:architecture}) guided by two core principles: **separation of concerns** and **extensibility through
base classes**. The design separates functional responsibilities into five independent modules: (1) **Models** (`canns.models`) define
neural network dynamics; (2) **Tasks** (`canns.task`) generate experimental paradigms and input data; (3) **Analyzers** (`canns.analyzer`)
provide visualization and analysis tools; (4) **Trainers** (`canns.trainer`) implement learning rules for brain-inspired models; and
(5) **Pipeline** (`canns.pipeline`) orchestrates complete experimental workflows.

Each module focuses on a single responsibility—models don't generate input data, tasks don't analyze results, and analyzers don't modify
parameters. This separation ensures maintainability, testability, and extensibility. All major components inherit from abstract base classes
(`BasicModel`, `BrainInspiredModel`, `Trainer`) that define standard interfaces, enabling users to create custom implementations that
seamlessly integrate with the built-in ecosystem.

The library supports four distinct research workflows: (1) CANN modeling and simulation for studying attractor dynamics; (2) data analysis
for processing experimental neural recordings; (3) brain-inspired learning with biologically plausible plasticity rules (Hebbian, STDP,
BCM); and (4) end-to-end pipelines for automated parameter sweeps and reproducible experiments. All models inherit from BrainPy's
`DynamicalSystem` base class [@wang2023brainpy], leveraging JAX's JIT compilation for GPU/TPU acceleration while maintaining simple
Python APIs. For operations where Python overhead is significant, the companion `canns-lib` Rust library provides optional accelerated
backends—notably achieving 400× speedup for spatial navigation tasks and 1.13-1.82× speedup for topological data analysis—without
requiring code structure changes.

# Related Works

Computational neuroscience has established simulators for spiking neural networks, including NEST [@Gewaltig:NEST]
for large-scale network models, Brian 2 [@stimberg2019brian] for rapid prototyping with user-friendly syntax, and
NEURON [@hines1997neuron] for detailed biophysical simulations. However, these general-purpose tools lack specialized support
for continuous attractor dynamics and require significant implementation effort for CANN-specific workflows. While individual CANN
implementations exist (e.g., [cann_base](https://github.com/fccaa/cann_base) on GitHub), they remain fragmented, lab-specific codebases
without standardized APIs, comprehensive task generation, or analysis tools—paralleling the pre-standardization era of deep learning before
frameworks like Hugging Face Transformers [@wolf-etal-2020-transformers] unified model definitions and usage patterns.

CANNs builds upon BrainPy [@wang2023brainpy], a modern brain dynamics framework leveraging
JAX [@jax2018github] for JIT compilation and autodifferentiation. BrainPy provides the foundational infrastructure—state
management, time stepping, and GPU/TPU acceleration—that CANNs extends with CANN-specific abstractions: standardized model implementations,
task-generation APIs, analysis pipelines, and Rust-accelerated backends. This layered approach mirrors successful deep learning ecosystems:
PyTorch [@Ansel_PyTorch_2_Faster_2024] and TensorFlow [@Abadi_TensorFlow_Large-scale_machine_2015] provide low-level tensor operations, while domain-specific
libraries (e.g., Transformers [[@wolf-etal-2020-transformers]] for NLP, torchvision [torchvision2016] for computer vision) deliver standardized high-level
components. CANNs fills this role for continuous attractor research, reducing fragmentation and accelerating reproducible science.

# AI usage disclosure

AI-assisted tools were used during the development of this software in limited, auxiliary capacities. Sourcery AI was employed for
automated code quality reviews and refactoring suggestions in pull requests. Large language models (including Claude and ChatGPT) assisted
with documentation writing, including docstring generation, tutorial content drafting, and technical writing refinement. However, all core
library code—including model implementations, task generators, analyzers, and algorithmic components—was written by human developers. All
AI-generated content was reviewed, validated, and edited by the authors to ensure technical accuracy and consistency with project standards.
The software architecture, design decisions, and scientific contributions represent original human intellectual work.

# Acknowledgements

We thank Aiersi Tuerhong and Shangjun She for their collaborative development contributions to the library implementation. We are grateful
to Tianhao Chu, Yuling Wu, and Junfeng Zuo for valuable discussions, feedback, and guidance throughout the project development. We especially
thank Si Wu for overall scientific guidance and mentorship. We acknowledge the BrainPy development team for providing the foundational
framework upon which this library is built, and the broader open-source community for tools and libraries that enabled this work.

# References