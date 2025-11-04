Experimental Data Analysis
==========================

Scenario Description
--------------------

CANNs provides specialized tools for analyzing real neural recording data, particularly for fitting and visualizing CANN bump activity. This series of tutorials will teach you:

- How to extract bump parameters from experimental data
- 1D and 2D bump fitting methods
- Data preprocessing and quality control
- Animated visualization of bump evolution

What You Will Learn
-------------------

1. Loading and preprocessing experimental data
2. 1D bump fitting algorithms
3. 2D bump fitting and parameter estimation
4. Time series data processing
5. Visualization and animation generation of results

Tutorial List
-------------

.. toctree::
   :maxdepth: 1

   bump_fitting_1d
   bump_fitting_2d

Target Audience
---------------

- Neuroscientists analyzing calcium imaging data
- Researchers working with electrophysiological recordings
- Students needing to fit neural activity patterns

Prerequisites
-------------

- Basic signal processing knowledge
- Neural recording methods (calcium imaging, electrophysiology, etc.)
- Python data analysis libraries (NumPy, SciPy)

Analysis Workflow
-----------------

**Typical data analysis workflow**:

1. **Data Loading**

   - Import ROI activity data
   - Check data format and completeness

2. **Preprocessing**

   - Denoising and filtering
   - Baseline correction
   - Normalization

3. **Bump Fitting**

   - Parameter initialization
   - Optimization algorithm selection
   - Fitting quality assessment

4. **Results Visualization**

   - Static plots
   - Animation generation
   - Statistical analysis

Supported Data Formats
----------------------

- **Calcium imaging data**: ROI time series
- **Electrophysiological data**: Multi-channel recordings
- **Population coding**: Neuronal cluster activity

Key Features
------------

``bump_fits()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~

Core bump fitting function supporting:

- Automatic bump number detection
- Multi-bump fitting
- Parameter uncertainty estimation

``create_1d_bump_animation()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate animations of bump evolution:

- Support for GIF and MP4 formats
- Customizable frame rate and resolution
- Progress bar display

``CANN1DPlotConfig`` Configuration Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified plotting configuration:

.. code-block:: python

   config = CANN1DPlotConfig.for_bump_animation(
       show=False,
       save_path="bump_analysis.gif",
       nframes=100,
       fps=10
   )

Practical Application Cases
----------------------------

- **Working memory experiments**: Analyze bump maintenance during delay periods
- **Spatial navigation**: Track bump movement in position coding
- **Attention research**: Observe the effects of attention modulation on bumps
- **Developmental studies**: Compare bump characteristics across different developmental stages

Getting Started
---------------

Start with :doc:`bump_fitting_1d` to learn how to analyze your experimental data!
