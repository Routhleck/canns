Scenario 4: End-to-End Research Workflows
==========================================

High-level pipelines for complete research workflows—from data loading to analysis and visualization—without requiring detailed knowledge of model implementation.

Tutorials
---------

.. toctree::
   :maxdepth: 1
   :caption: Research Pipelines

   01_theta_sweep_pipeline

Overview
--------

This scenario demonstrates streamlined workflows for common research tasks using pre-built pipelines. Perfect for experimental neuroscientists and researchers who want to analyze data quickly without diving into implementation details.

**Tutorial 1: Theta Sweep Pipeline**

- Complete theta sweep analysis in one line
- Loading trajectory data from various sources
- Automatic simulation and visualization
- Customizable parameters for advanced users
- Batch processing multiple datasets

Who Should Use Pipelines?
--------------------------

**Perfect for**:

- Experimental neuroscientists without deep coding expertise
- Rapid prototyping and exploratory analysis
- Standardized processing of multiple datasets
- Publication-quality figure generation
- Teaching and demonstrations

**Consider manual approach when**:

- Implementing non-standard model architectures
- Developing new analysis methods
- Need fine-grained control over every step
- Extending pipeline functionality

Learning Path
-------------

**Quick Start**:

1. Prepare your trajectory data (positions + timestamps)
2. Run the pipeline with default parameters
3. Examine generated plots and animations
4. Customize parameters as needed

**Advanced Usage**:

- Custom post-processing of simulation data
- Batch processing multiple sessions
- Parameter sweeps and optimization
- Integration with existing analysis workflows

Prerequisites
-------------

- Basic Python knowledge
- Understanding of your experimental data format
- Trajectory data (position over time)

Estimated Time
--------------

- Tutorial 1: 30-35 minutes
- Setting up for your own data: 15-30 minutes
- Total: 60 minutes

Pipeline Features
-----------------

The ThetaSweepPipeline provides:

- **Automatic data validation**—Checks data format and quality
- **Network simulation**—Direction cells and grid cells
- **Theta modulation**—Speed-dependent oscillations
- **Visualization suite**—Trajectory plots, population activity, animations
- **Raw data export**—For custom analysis
- **Flexible configuration**—From simple to advanced usage

Data Input Formats
------------------

Supported formats for trajectory data:

- CSV files
- NumPy arrays (`.npy`)
- MATLAB files (`.mat`)
- Pandas DataFrames
- DeepLabCut output
- Bonsai tracking output
- Custom formats (with preprocessing)

Next Steps
----------

After completing this scenario:

- Apply pipelines to your own experimental data
- Explore custom analysis using raw simulation outputs
- Learn implementation details in Scenario 1 for customization
- Contribute new pipelines to the library
