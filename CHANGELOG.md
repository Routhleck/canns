# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.7] - 2026-01-19

### Added
- Initial paper draft (`paper.md`) describing CANNs toolkit motivation and design
- Comprehensive docstrings with usage examples across all major modules (PR #81)
- New test file `tests/analyzer/visualization/test_backend.py` with 6 tests for multiprocessing context
- Expanded `docs/refs/references.bib` with additional related works

### Changed
- Enhanced module-level docstrings for package namespaces with clearer documentation
- Improved docstrings for model classes (CANN, grid cell, hierarchical, Hopfield, linear, spiking)
- Expanded task module documentation with step-by-step workflows and examples
- Updated analyzer module docstrings with concrete usage examples
- Modified `get_multiprocessing_context()` to return tuple `(context, method_name)` (PR #82)

### Fixed
- Fixed AttributeError in multiprocessing context on Linux with JAX (PR #82)
- Updated 5 call sites in visualization modules to unpack multiprocessing context tuple
- Added deterministic tests using monkeypatch for JAX-dependent behavior

## [0.12.6] - 2026-01-14

### Added
- Unified parallel animation backend system with imageio and matplotlib support (#79)
- Centralized backend selection with smart auto-detection
- Parallel frame rendering for 3-4x speedup on GIF and 2-3x on MP4
- New `backend.py` module for backend selection logic
- Extended parallel rendering functions in `rendering.py`
- CHANGELOG.md documenting all project releases
- CODE_OF_CONDUCT.md establishing community standards
- CONTRIBUTING.md with comprehensive development guidelines
- `render_backend`, `render_workers`, `render_start_method` options to PlotConfig

### Changed
- Refactored `energy_plots.py`, `spatial_plots.py`, `theta_sweep_plots.py` to use unified backend
- Updated imageio dependency to `imageio[ffmpeg]>=2.37.0`
- Updated project license to Apache-2.0 in pyproject.toml

### Fixed
- Improved animation backend selection and fallback logic with helpful error messages

## [0.12.5] - 2026-01-14

### Added
- Add rng_seed parameter for reproducible navigation tasks (#78)

### Changed
- Enhance spatial plot heatmap with label and title options
- Remove try-except blocks from plotting functions

### Fixed
- Fix formatting issues and refine text descriptions (#76)
- Add finalize_figure helper and mixed-mode save config (#77)

## [0.12.4] - 2026-01-13

### Changed
- Refactor and expand visualization and metrics docs
- Update README files and tutorial titles
- Update CANNs documentation with torus grid cell model

### Fixed
- Fix animation save to support MP4 format with unified writer

## [0.12.3] - 2026-01-12

### Added
- Grid Cell Velocity Model Enhancement: Path Integration and Spatial Analysis (#74)

## [0.12.2] - 2026-01-11

### Fixed
- Fix animation save to support MP4 format with unified writer

## [0.12.1] - 2026-01-10

### Changed
- Comprehensive Animation Performance Optimization (11.2x Speedup) (#73)

## [0.12.0] - 2026-01-09

### Changed
- Refactor analyzer module: Separate metrics and visualization (#72)

## [0.11.1] - 2026-01-08

### Added
- Add Grid Cell 2D model and relative analysis methods (#71)

### Changed
- Clarify WAW model terminology in documentation (#70)
- Updates the documentation for CANNs to improve terminology consistency, clarity, and accuracy (#69)
- Polish Documentation (#68)
- Migrate slow_points examples to modern BrainPy API (#66)

### Fixed
- Update README files with badge improvements
- Fix badge links and formatting in README.md

## [0.11.0] - 2026-01-07

### Added
- Add citation system and improve documentation (#67)
- Add Chinese RNN fixed point analysis FlipFlop tutorial
- Add FlipFlop RNN fixed point analysis tutorial
- Add Tier 4 Full Detail Tutorials (EN + ZH) (#63)

### Changed
- Migrate from brainstate/brainunit to brainpy (#65)
- Disable PDF and ePub formats in ReadTheDocs config
- Update CANNs library description in README files
- Enhance README with Rust backend details
- Remove docs_draft workspace and draft files
- Update docs versioning logic and toctree depth
- Complete Tier 3 Core Concepts documentation (EN + ZH) (#62)

### Fixed
- Remove redundant warning and fix markdown formatting in tutorials
- Improve Jupyter notebook animation rendering with autoplay (#61)

## [0.10.0] - 2026-01-06

### Added
- Add automatic Jupyter notebook HTML rendering for matplotlib animations (#59)
- Add fixed point finder for RNN analysis (#42)
- Add AntiHebbianTrainer for pattern unlearning (#50)
- Add 1D continuous Hopfield training example
- Add configurable parameters to hierarchical models and enhance path integration visualization (#48)
- Add citation and DOI information to README files
- Add DOI badge to README
- Create CITATION.cff
- Create RELEASE_TEMPLATE.md
- Add place cell theta sweep and refactor navigation tasks (#47)
- Add closed-loop geodesic tools & rename open-loop navigation (#44)
- Add imageio backend for animations (#43)
- Add 'Buy Me a Coffee' badge to README (#41)
- Add English guide documentation with autoapi and GitHub links (#38)
- Add Chinese guide documentation with autoapi and GitHub links (#37)
- Add logo to README files
- Add visual gallery to README

### Changed
- Complete documentation refactor: New Quick Starts series and API docs (#60)
- Brain-inspired learning rules with JIT compilation (Oja, Sanger, BCM, STDP, Hopfield analyzer) (#55)
- Restructure modules: consolidate data utilities and clarify module organization (#54)
- Vectorize performance bottlenecks in CANN analysis code (#51)
- Remove ratinabox in favor of canns_lib backends (#52)
- Refactor spatial analysis utilities to analyzer module (#49)
- Improve theta sweep animation title layout (#43)
- Enhance docstrings for core model, task, pipeline, and trainer modules (#40)
- Enhance README with new badges and links (#39)
- Update documentation links to external HTML pages
- Refresh notebooks and example navigation (#35)

### Fixed
- Fix TDA: ensure H₀ bars displayed correctly after shuffle visualization (#57)
- Exercise closed-loop cache behaviour in tests (#46)

## [0.9.3] - 2025-12-20

### Added
- Add citation and DOI information to README files
- Add DOI badge to README
- Create CITATION.cff

## [0.9.2] - 2025-12-19

### Added
- Add AntiHebbianTrainer for pattern unlearning (#50)
- Add 1D continuous Hopfield training example

## [0.9.1] - 2025-12-18

### Changed
- Refactor spatial analysis utilities to analyzer module (#49)

## [0.9.0] - 2025-12-17

### Added
- Add configurable parameters to hierarchical models and enhance path integration visualization (#48)

## [0.8.3] - 2025-12-16

### Changed
- Vectorize performance bottlenecks in CANN analysis code (#51)

## [0.8.2] - 2025-12-15

### Changed
- Remove ratinabox in favor of canns_lib backends (#52)

## [0.8.1] - 2025-12-14

### Changed
- Restructure modules: consolidate data utilities and clarify module organization (#54)

## [0.8.0] - 2025-12-13

### Added
- Brain-inspired learning rules with JIT compilation (Oja, Sanger, BCM, STDP, Hopfield analyzer) (#55)

### Changed
- Complete documentation restructure with bilingual translations (#56)

### Fixed
- Fix TDA: ensure H₀ bars displayed correctly after shuffle visualization (#57)

## [0.7.1] - 2025-12-12

### Added
- Add fixed point finder for RNN analysis (#42)

### Changed
- Add development status warnings to all tutorial files

## [0.7.0] - 2025-12-11

### Added
- Add automatic Jupyter notebook HTML rendering for matplotlib animations (#59)

### Changed
- Complete documentation refactor: New Quick Starts series and API docs (#60)

### Fixed
- Improve Jupyter notebook animation rendering with autoplay (#61)

## [0.6.2] - 2025-12-10

### Added
- Add place cell theta sweep and refactor navigation tasks (#47)

## [0.6.1] - 2025-12-09

### Added
- Add closed-loop geodesic tools & rename open-loop navigation (#44)

### Changed
- Exercise closed-loop cache behaviour in tests (#46)

## [0.6.0] - 2025-12-08

### Added
- Add imageio backend for animations
- Add 'Buy Me a Coffee' badge to README (#41)

### Changed
- Improve theta sweep animation title layout (#43)
- Enhance docstrings for core model, task, pipeline, and trainer modules (#40)

## [0.5.1] - 2025-12-07

### Added
- Add English guide documentation with autoapi and GitHub links (#38)
- Add Chinese guide documentation with autoapi and GitHub links (#37)

### Changed
- Enhance README with new badges and links (#39)

## [0.5.0] - 2025-12-06

### Added
- Add logo to README files
- Add visual gallery to README
- Add Trainer base class (#34)
- Add ThetaSweepPipeline with memory optimization and advanced examples (#31)
- Add import_data method for external trajectory import (#30)
- Add theta sweep models with optimized animation and spatial navigation (#29)

### Changed
- Restore plotting docstrings (#36)
- Update CANN2D encoding GIF in documentation
- Update documentation links to external HTML pages
- Refresh notebooks and example navigation (#35)
- Reorganize plotting module structure (#33)
- Add shared pipeline base (#32)

## [0.4.1] - 2025-12-05

### Added
- Implement AmariHopfieldNetwork with flexible activation functions and enhanced progress reporting (#26)

### Changed
- Trainer: unify Hebbian training/prediction via Trainer; simplify progress with tqdm; remove model-level predict and model-specific Hebbian; add optional resize; update examples/docs (#27)
- Hopfield: add threshold term to energy; compiled predict by default; MNIST example updates (#28)

## [0.4.0] - 2025-12-04

### Added
- Add circular coordinate decoding and 3D torus visualization (#21)
- Add experimental data utilities and 1D CANN bump fitting (#19)

### Changed
- Unified plotting configuration system with specialized config classes (#22)
- Refactor spatial navigation, modernize plot configs and type annotations (#25)
- Integrate canns-ripser with progress bar support (#24)
- Update LICENSE (#23)
- Refactor CANN1D module and Implement CANN2D module (#20)

### Fixed
- Fix documentation website: SVG favicon and API navigation (#17)
- Fix Sphinx build issues and add GitHub Pages deployment (#15)
- Enable GitHub Pages auto-deployment and fix CI security issues (#16)

## [0.3.0] - 2025-12-03

### Added
- Add z-score normalization to firing rate utils (#12)
- Add tuning curve plot method (#11)
- Add some visualization methods, utility functions and their tests (#10)

### Changed
- Complete documentation overhaul: interactive notebooks, multilingual support, and automated deployment (#13)
- Refactor for_loop calls for readability in examples

### Removed
- Delete CORE_CONCEPTS.md (#14)

## [0.2.0] - 2025-12-02

### Added
- Add Hierarchical Path Integration Model (#6)
- Add Path Integration Task (#7)
- Add Tracking1D tasks and detailed docstring (#3)
- Add Tracking2d and Refactor basic models (#5)
- Add CANN2D and CANN2D SFA models (#4)
- Add CANN1D_SFA model and update example (#2)
- Add issue templates

### Changed
- Optimize hierarchical model and Fix some bugs (#9)
- Refactor tasks (#8)
- Fix Hierarchical Model (#7)
- Update logo to SVG in README and add SVG asset
- Update README.md

### Removed
- Delete canns.py

## [0.1.0] - 2025-12-01

### Added
- Initial release
- Basic structure template
- Core application structure

[0.12.5]: https://github.com/routhleck/canns/compare/v0.12.4...v0.12.5
[0.12.4]: https://github.com/routhleck/canns/compare/v0.12.3...v0.12.4
[0.12.3]: https://github.com/routhleck/canns/compare/v0.12.2...v0.12.3
[0.12.2]: https://github.com/routhleck/canns/compare/v0.12.1...v0.12.2
[0.12.1]: https://github.com/routhleck/canns/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/routhleck/canns/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/routhleck/canns/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/routhleck/canns/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/routhleck/canns/compare/v0.9.3...v0.10.0
[0.9.3]: https://github.com/routhleck/canns/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/routhleck/canns/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/routhleck/canns/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/routhleck/canns/compare/v0.8.3...v0.9.0
[0.8.3]: https://github.com/routhleck/canns/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/routhleck/canns/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/routhleck/canns/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/routhleck/canns/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/routhleck/canns/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/routhleck/canns/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/routhleck/canns/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/routhleck/canns/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/routhleck/canns/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/routhleck/canns/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/routhleck/canns/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/routhleck/canns/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/routhleck/canns/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/routhleck/canns/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/routhleck/canns/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/routhleck/canns/releases/tag/v0.1.0