Scenario 4: End-to-End Research Workflow
======================================

Complete the full research workflow—from data loading to analysis and visualization—via the interactive ASA GUI (Attractor Structure Analyzer), without needing in-depth knowledge of model implementation details.

.. note::
   It is recommended to prioritize using the **ASA GUI**. The ASA TUI is an earlier version and will no longer be maintained; it is provided only for transitional use.

Tutorials
---------

.. toctree::
   :maxdepth:  '', 1
   :caption: Research Pipeline

   03_asa_gui
   01_asa_tui
   02_model_gallery_tui

Overview
--------

This scenario demonstrates an end-to-end analysis pipeline based on both the ASA GUI and ASA TUI, targeting experimental neuroscientists and researchers. It provides a graphical interface to perform preprocessing, TDA, decoding, and result browsing.

**Tutorial 1: End-to-End Analysis with ASA GUI**

- Complete preprocessing and analysis via a PySide6 graphical interface
- Supports TDA / CohoMap / PathCompare / CohoSpace / FR / FRM / GridScore
- Preview results in dedicated tabs and quickly open the output directory

**Tutorial 2: End-to-End Analysis with ASA TUI (Legacy)**

- Interactive interface for data preparation, preprocessing, analysis, and result export
- Supports dual input modes: ASA ``.npz`` and Neuron + Trajectory
- Built-in support for TDA / CohoMap / PathCompare / CohoSpace / FR / FRM / GridScore
- Results are automatically archived with logs and previews provided

**Tutorial 3: Model Gallery TUI**

- Replicates the 5×3 analysis layout from ``canns-experiments/figure2``
- One-click generation of canonical visualizations for CANN1D / CANN2D / GridCell
- Unified management of result previews and output directories

Who Should Use This Pipeline?
-----------------------------

**Highly Suitable For**:

- Experimental neuroscientists without extensive coding expertise
- Rapid prototyping and exploratory analysis
- Standardized processing of multiple datasets
- Generating publication-quality figures
- Teaching and demonstrations

**Consider Manual Approaches When**:

- Implementing non-standard model architectures
- Developing novel analysis methods
- Requiring fine-grained control over each step
- Extending pipeline functionality

Learning Path
-------------

**Quick Start**:

1. Prepare ASA or Neuron + Trajectory data
2. Launch ASA GUI and select input files
3. Run with default preprocessing and analysis parameters
4. Inspect outputs in the Results directory and adjust parameters as needed

**Advanced Usage**:

- Batch process multiple sessions (by switching working directories and input files)
- Fine-tune parameters for TDA, decoding, and visualization
- Feed generated intermediate results into custom analyses
- Integrate with existing experimental workflows

Prerequisites
-------------

- Basic Python knowledge
- Familiarity with your experimental data format (spike/x/y/t)
- Ability to run commands in a terminal

Estimated Time
--------------

- Tutorial 1: 30–40 minutes
- Setting up with your own data: 15–30 minutes
- Total: 70 minutes

Pipeline Features
-----------------

The ASA GUI provides:

- **Interactive Workflow** — GUI-based preprocessing and analysis
- **Automatic Data Validation** — Checks input format and missing fields
- **TDA + Decoding** — Persistent homology, phase decoding, and comparison
- **Visualization Suite** — CohoMap / CohoSpace / FR / FRM / GridScore
- **Result Archiving** — Automatic per-dataset directory organization
- **Logging and Caching** — Execution logs and stage-wise cache reuse

Data Input Formats
------------------

Two input types are supported:

- **ASA ``.npz``**: Contains ``spike`` / ``t`` (optionally ``x`` / ``y``)
- **Neuron + Traj ``.npz``**: Neuron file contains ``spike``; trajectory file contains ``x`` / ``y`` / ``t``

Next Steps
----------

After completing this scenario:

- Apply ASA GUI to your real experimental data
- Build custom analyses on top of generated intermediate results
- Extend functionality by referencing ``canns.pipeline.asa``
- Contribute new analysis modules to the library