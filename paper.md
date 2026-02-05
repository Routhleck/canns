---
title: "CANNs: Continuous Attractor Neural Networks Toolkit"
tags:
  - Python
  - continuous attractor neural networks
  - computational neuroscience
  - topological data analysis
authors:
  - name: Sichao He
    orcid: 0009-0009-4564-1753
    affiliation: "1, 2, 3, 4"
  - name: Aiersi Tuerhong
    orcid: 0009-0006-4603-4734
    affiliation: 5
  - name: Shangjun She
    orcid: 0009-0008-4490-7612
    affiliation: "1, 2, 3, 4"
  - name: Tianhao Chu
    orcid: 0000-0001-9910-9361
    affiliation: "1, 2, 3, 4"
  - name: Yuling Wu
    orcid: 0009-0001-5303-5413
    affiliation: "1, 2, 3, 4"
  - name: Junfeng Zuo
    affiliation: "1, 2, 3, 4"
  - name: Si Wu
    orcid: 0000-0001-9650-6935
    corresponding: true
    affiliation: "1, 2, 3, 4"
      
affiliations:
  - name: School of Psychological and Cognitive Sciences, Peking University, Beijing, China
    index: 1
  - name: Peking-Tsinghua Center for Life Sciences, Academy for Advanced Interdisciplinary Studies, Peking University, Beijing, China
    index: 2
  - name: PKU-IDG/McGovern Institute for Brain Research, Peking University, Beijing, China
    index: 3
  - name: Center of Quantitative Biology, Peking University, Beijing, China
    index: 4
  - name: College of Mathematics and Statistics, Chongqing University, Chongqing, China
    index: 5

date: 19 January 2026
bibliography: paper.bib
---
# Summary

CANNs (Continuous Attractor Neural Networks toolkit) is a Python library built on BrainPy, a powerful framework for brain dynamics 
programming. It streamlines experimentation with continuous attractor neural networks and related brain-inspired models. The library 
delivers ready-to-use models, task generators, analysis tools, and pipelines—enabling neuroscience and AI researchers to move quickly 
from ideas to reproducible simulations.


# Statement of need

Continuous Attractor Neural Networks (CANNs) are a canonical neural circuit model for information processing in the brain and are receiving increasing attention in the fields of neuroscience and brain-inspired computing.   They not only provide a  mathematical model for understanding how the brain encodes continuous variables, such as spatial position, head direction, and moving direction, but also serve as a theoretical framework for elucidating how the brain processes abstract features and relationships. Known examples include that CANNs successfully
explain key phenomena in hippocampal place cells [@o1971hippocampus], entorhinal grid cells [@hafting2005microstructure], and head
direction systems [@taube1990head]. Despite the importance, the CANN research suffers from fragmentation: researchers implement models
from scratch, use incompatible codebases, and face significant reproducibility barriers. This lack of standardization slows progress
and creates steep learning curves for newcomers.

CANNs toolkit addresses this gap by providing a unified Python toolkit built on a user-friendly and efficient programming framework BrainPy [@wang2023brainpy; @wang2025integrating]. It delivers: (1) standardized
implementations of CANNs and related brain-inspired models, including mathematically tractable CANN models [@amari1977dynamics; @wu2008dynamics],
adaptation-augmented CANN models [@mi2014spike; @li2025dynamics], grid cell networks [@burak2009accurate], alongside additional attractor
architectures; (2) integrated task generation, simulation, and analysis pipelines; (3) high-performance computation via JAX JIT
compilation and optional Rust acceleration. By standardizing the workflow—analogous to Hugging Face Transformers in deep learning—this library accelerates reproducible research and lowers barriers for computational neuroscientists, AI engineers, and students exploring attractor dynamics.

# Software design

![Layer hierarchy of the CANNs library showing five levels: Application (Pipeline orchestration), Functional (Task, Trainer, Analyzer, Utils modules), Core Models (CANN implementations), Foundation (BrainPy/JAX and Rust FFI backends), and Hardware (CPU/GPU/TPU support).\label{fig:architecture}](docs/_static/architecture.png)

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
for processing experimental neural recordings; (3) brain-inspired learning with biologically plausible plasticity rules; and (4) end-to-end pipelines for automated parameter sweeps and reproducible experiments. All models inherit from BrainPy's
`DynamicalSystem` base class, leveraging JAX's JIT compilation for GPU/TPU acceleration while maintaining simple
Python APIs. For operations where Python overhead is significant, the companion `canns-lib` Rust library provides optional accelerated
backends that can substantially improve performance in internal benchmarks for spatial navigation tasks and topological data analysis,
without requiring code structure changes.

# State of the field

While general-purpose neural network simulators like NEST [@Gewaltig:NEST], Brian 2 [@stimberg2019brian], and NEURON [@hines1997neuron] exist, they lack specialized support for continuous attractor networks. Existing CANN implementations remain fragmented, lab-specific codebases without standardized APIs or comprehensive tooling.

CANNs builds upon BrainPy [@wang2023brainpy; @wang2025integrating], a powerful brain dynamics framework leveraging JAX [@jax2018github] for JIT compilation and GPU/TPU acceleration. CANNs extends BrainPy with CANN-specific abstractions: standardized model implementations, task-generation APIs, analysis pipelines, and optional Rust-accelerated backends for performance-critical operations.

# Research impact statement

CANNs provides full-detail, runnable modeling tutorials that reproduce recent CANN-related studies, packaged as standardized pipelines
with consistent inputs, analysis steps, and visualizations. This allows researchers to verify published results and compare alternative
mechanisms under a shared experimental setup, reducing the need for lab-specific reimplementations. The reproduced workflows include
spike-frequency adaptation [@mi2014spike; @li2025dynamics] and theta-sweep dynamics in head-direction/grid and place-cell systems [@ji2025phase; @ji2025systems; @chu2024firing], offering trusted baselines for new modeling and benchmarking studies.

# AI usage disclosure

AI-assisted tools were used for code quality reviews and documentation writing. All core library code was written by human developers, and AI-generated content was reviewed and validated by the authors.

# Acknowledgements

This work was supported by the National Natural Science Foundation of China (no. T2421004 to S.W.),the National Key Research and Development Program of China (2024YFF1206500),  the Science and Technology Innovation 2030-Brain Science and Brain-inspired Intelligence Project (no. 2021ZD0200204, S.W.).

# References
