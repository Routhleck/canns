Advanced Workflows
==================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scenario Description
--------------------

When you need to handle more complex scenarios, CANNs provides advanced tools to simplify workflows. This tutorial series will teach you:

- Build end-to-end Theta sweep pipelines
- Import and use external trajectory data
- Fully customizable parameter configuration
- Batch processing and parallel computation

What You Will Learn
-------------------

1. Complete usage of ``ThetaSweepPipeline``
2. How to import external trajectory data
3. Fine-tuning of parameters
4. Workflow automation and optimization
5. Batch result generation and management

Tutorial List
-------------

.. toctree::
   :maxdepth: 1

   building_pipelines
   external_trajectories

Intended Audience
-----------------

- Researchers who need to process data in batches
- Students conducting parameter sweeps and optimization
- Engineers developing automated analysis pipelines
- Advanced users requiring customized workflows

Prerequisites
-------------

- Familiarity with CANNs basic concepts (completing previous tutorials)
- Advanced Python features (decorators, context managers, etc.)
- Command-line tool usage
- Basic parallel computing concepts

Pipeline Architecture
---------------------

``ThetaSweepPipeline`` unifies the following steps:

1. **Trajectory Processing**

   - Import external data or generate trajectories
   - Trajectory smoothing and interpolation
   - Compute velocity and direction

2. **Network Configuration**

   - Head direction cell network
   - Grid cell network
   - Theta rhythm parameters

3. **Simulation Execution**

   - Efficient JAX compilation loop
   - Progress monitoring
   - Intermediate result saving

4. **Result Analysis**

   - Activity heatmap generation
   - Statistical analysis
   - Animation visualization

Advanced Features
-----------------

**Parameter Customization**
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full control over all network and task parameters:

.. code-block:: python

   pipeline = ThetaSweepPipeline(
       trajectory_data=positions,
       times=times,
       direction_cell_params={
           "num": 100,
           "adaptation_strength": 15,
           "noise_strength": 0.0,
       },
       grid_cell_params={
           "num_gc_x": 100,
           "mapping_ratio": 0.85,
       },
       theta_params={
           "theta_strength_hd": 1.0,
           "theta_cycle_len": 100.0,
       },
   )

**External Trajectory Import**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Support for multiple formats:

- NumPy arrays
- CSV files
- HDF5 files
- Custom formats

**Batch Processing**
~~~~~~~~~~~~~~~~~~~~

Automation using Python scripts:

.. code-block:: python

   for params in parameter_grid:
       pipeline = ThetaSweepPipeline(**params)
       results = pipeline.run(
           output_dir=f"results_{params['id']}",
           save_animation=True,
       )

Performance Optimization
------------------------

- **JAX JIT Compilation**: Accelerate simulation loops
- **GPU Acceleration**: Support for CUDA backend
- **Parallel Processing**: Parallel multi-parameter sweeps
- **Memory Optimization**: Streaming data processing

Practical Applications
----------------------

- **Parameter Sweeps**: Systematically explore parameter space
- **Model Comparison**: Compare performance of different configurations
- **Reproduce Experiments**: Reproduce behavior using real trajectories
- **Hypothesis Testing**: Test theoretical predictions

Best Practices
--------------

1. **Modular Design**: Decompose workflows into independent steps
2. **Version Control**: Use Git to manage configurations and code
3. **Documentation**: Document the rationale behind parameter choices
4. **Validation**: Verify the reasonableness of intermediate results
5. **Backup**: Regularly save important results

Getting Started
---------------

Start with :doc:`building_pipelines` to build your first automated workflow!
