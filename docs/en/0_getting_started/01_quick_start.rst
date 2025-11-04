Quick Start
===========

This note outlines the key points of CANNs design philosophy to help you quickly familiarize yourself with the library structure.

Before diving into the full design philosophy documentation or source code, you can treat it as a high-level checklist.

Module Overview
---------------

- ``model`` built-in model package.

  - ``basic`` core CANN models and their variants.
  - ``brain_inspired`` various brain-inspired models.
  - ``hybrid`` models that combine CANNs with other architectures (e.g., ANNs).

- ``task`` task tools for generating, persisting, importing, and visualizing stimuli.
- ``analyzer`` analysis tools focused on model and data visualization.

  - ``model analyzer`` model analysis functions such as energy landscape, firing rate, and tuning curves.
  - ``data analyzer`` analysis pipelines for experimental data or virtual RNN dynamics.

- ``trainer`` unified training and inference interfaces.
- ``pipeline`` end-to-end pipelines that connect the above modules.

Quick Overview of Model Module
-------------------------------

``models``
~~~~~~~~~~

Overview
^^^^^^^^

The model module implements foundational CANNs in different dimensions, brain-inspired variants, and hybrid models, which form the core of the entire library and work in coordination with other modules to cover diverse scenarios.

The implementations are grouped by type:

- Basic Models (:mod:`~src.canns.models.basic`) standard CANN structures and their extensions.
- Brain-Inspired Models (:mod:`~src.canns.models.brain_inspired`)
  brain-inspired network implementations.
- Hybrid Models (:mod:`~src.canns.models.hybrid`) hybrid models that combine CANNs with mechanisms such as
  ANNs.

These models rely on the `Brain Simulation
Ecosystem <https://brainmodeling.readthedocs.io/index.html>`__\ , especially
`brainstate <https://brainstate.readthedocs.io>`__\ .\ ``brainstate``
is built on JAX/BrainUnit and provides ``brainstate.nn.Dynamics``
abstraction, ``State``/``HiddenState``/``ParamState``
containers, unified time-stepping control via ``brainstate.environ``, and
utilities such as ``brainstate.compile.for_loop`` and ``brainstate.random``.
With these components, models only need to describe state variables and update rules,
while brainstate handles time advancement, parallelization, and random number management,
thus reducing implementation costs.

Quick Overview of Task Module
-----------------------------

:mod:`~src.canns.task`
~~~~~~~~~~~~~~~~~

Overview
^^^^^^^^

The task module (:mod:`~src.canns.task`) handles the generation, saving, loading, importing, and visualization of CANN-related stimuli.
It provides multiple preset tasks while allowing extensions for specific requirements.
Major types include :class:`~src.canns.task.tracking.SmoothTracking1D`, :class:`~src.canns.task.tracking.SmoothTracking2D`, and
:class:`~src.canns.task.open_loop_navigation.OpenLoopNavigationTask`.

Quick Overview of Analyzer Module
----------------------------------

:mod:`~src.canns.analyzer`
~~~~~~~~~~~~~~~~~~~~~

Overview
^^^^^^^^

The analyzer module (:mod:`~src.canns.analyzer`) provides visualization and statistical analysis tools for CANN
models and experimental data, covering both model analysis and data analysis.
Key components include the :class:`~src.canns.analyzer.plotting.PlotConfigs` configuration system and visualization functions such as
:func:`~src.canns.analyzer.plotting.energy_landscape_1d_animation` and
:func:`~src.canns.analyzer.plotting.energy_landscape_2d_animation`.

Trainer Highlights
------------------

:mod:`~src.canns.trainer`
~~~~~~~~~~~~~~~~~~~~

Overview
^^^^^^^^

The trainer module (:mod:`~src.canns.trainer`) provides unified training and evaluation interfaces.
Currently centered on Hebbian learning with extensibility for other strategies in the future.
Core types include :class:`~src.canns.trainer.HebbianTrainer`,
:class:`~src.canns.trainer.OjaTrainer`, :class:`~src.canns.trainer.SangerTrainer`,
:class:`~src.canns.trainer.BCMTrainer`, and :class:`~src.canns.trainer.STDPTrainer`, among others.

Pipeline Overview
-----------------

:mod:`~src.canns.pipeline`
~~~~~~~~~~~~~~~~~~~~~

Overview
^^^^^^^^

The pipeline module (:mod:`~src.canns.pipeline`) connects models, tasks, analyzers, and trainers into end-to-end workflows,
enabling common requirements to be completed with minimal code.
Core pipelines include :class:`~src.canns.pipeline.ThetaSweepPipeline` for spatial navigation and theta sweep analysis.

Next Steps
----------

- Read the
  `Design Philosophy <https://routhleck.com/canns/zh/notebooks/00_design_philosophy.html>`__
  for complete design principles.
- Browse the `examples/  <https://github.com/Routhleck/canns/tree/master/examples>`__ directory to learn practical usage of each module.
- Customize your own components following the extension guidelines provided in each section.