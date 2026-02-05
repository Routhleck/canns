CANNs Documentation / CANNs 文档
=================================

Welcome to the CANNs (Continuous Attractor Neural Networks) documentation!

欢迎来到 CANNs（连续吸引子神经网络）文档！

Choose Your Language / 选择语言
================================

.. grid:: 2

    .. grid-item-card:: 🇺🇸 English
        :link: en/index
        :link-type: doc
        :class-header: bg-light
        
        Complete documentation in English
        
        * API Reference
        * Tutorials & Examples  
        * User Guide

    .. grid-item-card:: 🇨🇳 中文
        :link: zh/index
        :link-type: doc
        :class-header: bg-light
        
        中文版文档
        
        * API 参考（链接至英文版）
        * 教程和示例
        * 用户指南

.. toctree::
   :maxdepth: 2
   :caption: Languages / 语言:
   :hidden:

   English <en/index>
   中文 <zh/index>

About CANNs
-----------

CANNs (Continuous Attractor Neural Networks toolkit) is a research toolkit built on BrainPy and JAX, with optional Rust-accelerated ``canns-lib`` for selected performance-critical routines. It bundles model collections, task generators, analyzers, trainers, and the ASA pipeline (GUI/TUI) so you can run simulations and analyze results in a consistent workflow.

CANNs（Continuous Attractor Neural Networks toolkit）是基于 BrainPy 与 JAX 构建的研究工具库，并可选使用 Rust 加速库 ``canns-lib`` 优化部分性能敏感例程。它集成模型集合、任务生成器、分析器、训练器与 ASA 流水线（GUI/TUI），以统一工作流完成仿真与分析。
