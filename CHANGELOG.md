# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-02-06

### Added
- CLI module (`src/canns/__main__.py`) for command-line interface support
- Comprehensive test coverage for CLI, lazy imports, data loaders, and utilities
- Integration tests for pipeline launcher
- Unit tests for task base classes and benchmarking utilities
- 1D CANN smooth-tracking example in documentation

### Changed
- Restructured test suite into `unit/` and `integration/` directories (PR #96)
- Optimized CI pipeline for faster test execution (PR #96)
- Relocated documentation images from `images/` to `docs/_static/`
- Updated status badge to "stable" and maintenance year to 2026
- Enhanced installation instructions with accelerator options (cuda12, cuda13, tpu, gui)
- Expanded architecture diagram caption with five-layer hierarchy description
- Normalized pip install examples by removing unnecessary quotes
- Updated documentation URL to https://routhleck.com/canns
- Added Zenodo DOI badge for v1.0.0 release
- Updated citation information with expanded author list and affiliations

### Fixed
- Improved test organization and maintainability
- Updated all documentation image references to new paths

## [1.0.0] - 2026-02-02

### Added
- Architecture diagram (`docs/_static/architecture.png`) explaining modular design principles
- Comprehensive analyzer visualization showcase (`docs/_static/analyzer-display.png`, 4.7MB)
- ASA GUI preview animation (`docs/_static/asa-gui.gif`, 898KB)

### Changed
- **Production Ready Milestone**: v1.0.0 marks API stability and production readiness
- Complete README restructure (both English and Chinese) with improved project overview
- Enhanced Quick Start example with better comments and complete workflow demonstration
- Updated status badge from "beta" to "stable" in README.md and README_zh.md
- Improved visual gallery presentation with richer analyzer capability showcase
- Reorganized README sections: Overview, Architecture, Core Features, Analyzer Visuals, Quick Start

## [0.15.1] - 2026-02-01

### Changed
- Refactored ASA module: removed legacy aliases and consolidated utilities (PR #95)
- Replaced custom Gaussian filter with scipy public API
- Added comprehensive docstrings to key private functions

### Fixed
- Reduced PyPI package size by excluding unnecessary files

## [0.15.0] - 2026-01-31

### Added
- New CohoMap modules for topological feature extraction (PR #94)
  - `cohomap.py`: Core CohoMap computation
  - `cohomap_scatter.py`: Scatter plot visualization (renamed from `cohomap.py`)
  - `cohomap_vectors.py`: Vector field visualization with 4-panel layouts
- New CohoSpace modules for enhanced trajectory analysis (PR #94)
  - `cohospace_scatter.py`: Refactored scatter plotting for 1D/2D trajectories
  - `cohospace_phase_centers.py`: Phase center analysis and alignment
- BTN (Border-to-Neighbor) analysis for spatial boundary detection (PR #94)
  - `btn.py`: Border cell classification and spatial analysis
  - `btn_plots.py`: Comprehensive BTN visualization tools
- `plot_internal_position_trajectory()` function for grid cell bump tracking (PR #93)
- `PlotConfigs.internal_position_trajectory_static` configuration helper (PR #93)
- New English tutorial notebooks (PR #92)
  - `01_asa_pipeline.ipynb`: Comprehensive ASA pipeline tutorial
  - `02_cann1d_bump_fit.ipynb`: CANN1D bump fitting guide
  - `04_cell_classification.ipynb`: Cell classification tutorial
- Pipeline documentation pages (PR #92)
  - `01_asa_tui.rst`: ASA TUI (legacy) guide
  - `02_model_gallery_tui.rst`: Model Gallery TUI guide
  - `03_asa_gui.rst`: ASA GUI (primary) guide
- New example scripts (PR #94)
  - `btn_example.py`: BTN analysis demonstration
  - `cohomap_example.py`: Basic CohoMap computation
  - `cohomap_vectors_example.py`: Vector field visualization
  - `cohospace_example.py`: CohoSpace trajectory analysis
  - `cohospace_phase_centers_example.py`: Phase center alignment
- Sidebar auto-scroll functionality with `sidebar-current.js` (1f8a6cc)
- TDA literature references (Vaupel2023, Gardner2022) in documentation (PR #92)
- Active page highlighting in documentation sidebar (1f8a6cc)

### Changed
- Migrated API documentation from `src.canns` to `canns` namespace (1f8a6cc)
  - Updated 150+ autoapi RST files for consistent import paths
  - Improved documentation discoverability and user experience
- Refactored `cohospace.py` by extracting specialized functions into dedicated modules (PR #94)
  - Improved code maintainability and reduced complexity
  - Backward compatible - existing functions remain available
- Enhanced documentation navigation with auto-scroll and visual highlighting (1f8a6cc)
- Updated ASA documentation to prioritize GUI as primary interface, TUI as legacy (PR #92)
- Improved `GridCellNetwork` parameters for better demonstrations (PR #93)
  - Stronger `adaptation_strength` and `mapping_ratio`
  - Updated theta sweep example workflow
- Optimized theta sweep animation GIF (2.9MB → 1.5MB) (PR #93)
- Renamed example scripts for clarity (PR #94)
  - `cohomap.py` → `cohomap_scatter.py`
  - `cohospace1d.py` → `cohospace_scatter1d.py`
  - `cohospace2d.py` → `cohospace_scatter2d.py`
- Enhanced tutorial notebook formatting and metadata (33bad11, f4bf5a4)
- Added `apply_language()` method documentation to ASA GUI analysis modes (33bad11)
- Improved cell classification tutorial with better code outputs and examples (f4bf5a4)

### Fixed
- Added `strict=True` to zip() calls for better error detection (PR #94)

## [0.14.3] - 2026-01-28

### Added
- Dataset/URL input support in ASA GUI preprocess page (e6a8937)
- Built-in dataset selection with session and filename controls
- Direct URL input for fetching remote datasets with automatic download and validation
- Automatic format conversion and validation for fetched datasets

### Changed
- Enhanced preprocess page UI with comprehensive dataset loading interface
- Updated dependencies to include `requests` library for HTTP functionality (gui and dev-gui extras)

## [0.14.2] - 2026-01-28

### Added
- Multilingual tooltip support for ASA GUI with Chinese and English localization (3064508)
- `apply_language()` method to all analysis mode classes for dynamic tooltip setting
- Tooltips for all 8 analysis modes: CohoMap, CohoSpace, Decode, FR, FRM, Grid Score, Path Compare, and TDA
- Tooltips for preprocessing controls in preprocess page
- QToolTip styling to dark and light theme stylesheets

### Changed
- Enhanced ASA GUI user experience with localized interface guidance
- Improved tooltip appearance with consistent styling across themes

## [0.14.1] - 2026-01-28

### Added
- Official logo resources (logo.svg and logo.ico) to ASA GUI application (3ddaf12)
- Resources directory to package data in pyproject.toml

### Changed
- Updated README.md and README_zh.md to recommend ASA GUI as primary pipeline interface (3ddaf12)
- Reorganized pipeline tutorial documentation to prioritize ASA GUI (42ec942)
- Enhanced resource loading logic in ASA GUI to use bundled resources (3ddaf12)
- Marked ASA TUI as legacy interface in documentation (42ec942)
- Improved tutorial titles, ordering, and descriptions in pipeline documentation (42ec942)

## [0.14.0] - 2026-01-28

### Added
- Complete PySide6-based ASA GUI application with 10+ analysis modes (PR #91)
  - Interactive desktop interface with drag-and-drop file loading
  - Light/dark theme support with modern UI design
  - Real-time progress tracking and result visualization
  - CLI entry points: `canns-gui` and `python -m canns.pipeline.asa_gui`
  - Comprehensive Chinese tutorial with screenshots (`docs/zh/3_full_detail_tutorials/04_pipeline/03_asa_gui.rst`)
- Cell classification module for grid cells and head direction cells (PR #89)
  - Grid cell analysis: gridness score, spatial rate maps, autocorrelogram features
  - Head direction cell analysis: directional tuning, mean vector length, Rayleigh test
  - Leiden clustering for grid module identification
  - Complete utilities: circular statistics, correlation, geometry, image processing
  - Visualization tools for grid plots and HD plots
  - Tutorial notebook (`docs/zh/examples/02_data_analysis/04_cell_classification.ipynb`)
  - Example scripts in `examples/cell_classification/`
- ASA core references section to documentation (PR #90)
- Application icon (`images/logo.ico`)
- 12 new documentation images for ASA GUI modes

### Changed
- Refactored ASA API with explicit 1D/2D function separation (PR #89)
  - `plot_cohospace_trajectory()` → `plot_cohospace_trajectory_1d()` / `plot_cohospace_trajectory_2d()`
  - `align_coords_to_position()` → `align_coords_to_position_1d()` / `align_coords_to_position_2d()`
  - `compare_paths()` → `compare_paths_1d()` / `compare_paths_2d()`
- Optimized `embed_spike_trains` binning and smoothing performance (PR #89)
- Modernized type annotations to Python 3.10+ syntax in cell classification modules (PR #89)
- Improved import ordering and code consistency across ASA GUI (PR #91)
- Updated example scripts to use new 1D/2D API variants

### Added Dependencies
- **PySide6 (>=6.8.1)**: Qt6 bindings for GUI (optional: `pip install canns[gui]`)
- **scikit-image (>=0.24.0)**: Image processing for spatial analysis
- **h5py (>=3.12.1)**: HDF5 file support for MATLAB data
- **igraph (>=0.11.8)**: Graph algorithms for clustering
- **leidenalg (>=0.10.2)**: Leiden clustering algorithm

### Breaking Changes
- ASA cohospace and path functions split into explicit 1D/2D variants
- Old function names deprecated (will be removed in v0.15.0)

### Technical Improvements
- Added `standardize` option to TDAConfig for normalized persistence diagrams
- Enhanced warnings with stacklevel for better debugging
- Improved test assertions in `test_embed_spike_trains_basic`

## [0.13.2] - 2026-01-26

### Added
- Model Gallery TUI with interactive terminal interface for browsing and running model examples (PR #88)
- New gallery module with app, runner, and state management (`src/canns/pipeline/gallery/`)
- CLI entry points: `python -m canns.pipeline.gallery` and `canns-gallery` command
- Pipeline launcher module (`src/canns/pipeline/launcher.py`) for unified pipeline access
- ASA TUI tutorial with comprehensive Chinese documentation (PR #87)
- Model Gallery TUI tutorial (`docs/zh/3_full_detail_tutorials/04_pipeline/02_model_gallery_tui.rst`)
- Left-Right dataset loading utilities: `get_left_right_data_session()`, `get_left_right_npz()`, `load_left_right_npz()`
- New static images for ASA TUI and Gallery TUI documentation (7 PNG files)
- Extensive API documentation for gallery modules and launcher

### Changed
- Updated pipeline documentation structure with new tutorial organization (PR #87)
- Enhanced `src/canns/data/datasets.py` with Left-Right dataset support
- Updated `pyproject.toml` with new entry points for gallery and launcher
- Improved pipeline module exports in `__init__.py`

### Removed
- Deprecated base pipeline class (`src/canns/pipeline/_base.py`) and associated tests
- Obsolete theta sweep pipeline notebook (`docs/zh/3_full_detail_tutorials/04_pipeline/01_theta_sweep_pipeline.ipynb`)

## [0.13.1] - 2026-01-26

### Added
- Comprehensive ASA pipeline documentation with Chinese tutorials (PR #85)
- New CANN1D bump fitting tutorial (`docs/zh/examples/02_data_analysis/02_cann1d_bump_fit.ipynb`) (PR #85)
- New ASA pipeline tutorial (`docs/zh/examples/02_data_analysis/01_asa_pipeline.ipynb`) (PR #85)
- New example script `fly_roi_bump_fit.py` for Fly ROI bump analysis (PR #85)
- Static images for ASA documentation (barcode, cohomap, cohospace, FRM, spike train visualizations) (PR #86)

### Changed
- Refactored `cann1d.py` to `fly_roi.py` with modernized API (PR #85)
  - Renamed function `cann1d_bump_fits` to `roi_bump_fits`
  - Updated to use `CANN1DPlotConfig` for configuration
- Updated data analysis documentation structure and formatting (PR #86)
- Improved ASA module documentation with clearer examples and output descriptions (PR #86)
- Reformatted imports and function signatures in ASA modules for better readability (PR #85)

### Removed
- Deprecated legacy CANN1D analysis module (`src/canns/analyzer/data/legacy/cann1d.py`) (PR #85, #86)
- Deprecated legacy CANN2D analysis module (`src/canns/analyzer/data/legacy/cann2d.py`) (PR #86)
- Legacy experimental analysis script `experimental_cann1d_analysis.py` (PR #85)
- Legacy data analysis documentation from auto-generated API reference (PR #86)
- Obsolete theta sweep pipeline documentation (PR #85)

### Fixed
- Updated test file `test_experimental_data_cann1d.py` to use new `fly_roi` module (PR #85)
- Clarified `save_path` behavior in decode documentation (PR #85)

## [0.13.0] - 2026-01-26

### Added
- New ASA (Attractor State Analysis) pipeline with Textual-based terminal user interface (PR #84)
- Complete `canns.analyzer.data.asa` submodule for experimental data analysis (PR #83)
- Cohomological analysis tools (CohoMap, CohoSpace alignment)
- Advanced circular coordinate decoding methods with multiple versions
- Topological Data Analysis (TDA) utilities with persistent homology
- Embedding and dimensionality reduction tools (PCA, UMAP)
- Comprehensive plotting functions for ASA analysis types
- New example scripts: cohomap.py, cohospace.py, firing_field_map.py, path_compare.py, tda_vis.py
- Research paper draft (paper.md) with comprehensive bibliography (paper.bib)
- Architecture diagram (docs/_static/architecture.png)
- GitHub Actions workflow for draft PDF generation
- CLI entry point for ASA TUI: `python -m canns.pipeline.asa`
- Image preview and external viewer support in TUI
- Dataset-specific result directories with log file handling

### Changed
- Refactored analyzer.data module with new ASA submodule structure (PR #83)
- Updated type hints to modern Python syntax (PEP 604) throughout ASA modules
- Improved error handling and validation in decoding functions
- Enhanced preprocessing config with user and default value merging
- Moved legacy experimental analysis scripts to `legacy` folder
- Updated paper.md references and toolkit description

### Removed
- Legacy theta sweep pipeline module (`src/canns/pipeline/theta_sweep.py`)
- Theta sweep example scripts (advanced_theta_sweep_pipeline.py, theta_sweep_from_external_data.py)

### Fixed
- Circular coordinate assignment in decoding functions to prevent unintended values
- Auto-filtering support in cohospace alignment when activity data lengths mismatch

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

[1.0.1]: https://github.com/routhleck/canns/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/routhleck/canns/compare/v0.15.1...v1.0.0
[0.15.1]: https://github.com/routhleck/canns/compare/v0.15.0...v0.15.1
[0.15.0]: https://github.com/routhleck/canns/compare/v0.14.3...v0.15.0
[0.14.3]: https://github.com/routhleck/canns/compare/v0.14.2...v0.14.3
[0.14.2]: https://github.com/routhleck/canns/compare/v0.14.1...v0.14.2
[0.14.1]: https://github.com/routhleck/canns/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/routhleck/canns/compare/v0.13.2...v0.14.0
[0.13.2]: https://github.com/routhleck/canns/compare/v0.13.1...v0.13.2
[0.13.1]: https://github.com/routhleck/canns/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/routhleck/canns/compare/v0.12.7...v0.13.0
[0.12.7]: https://github.com/routhleck/canns/compare/v0.12.6...v0.12.7
[0.12.6]: https://github.com/routhleck/canns/compare/v0.12.5...v0.12.6
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
