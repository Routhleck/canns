Example Code
============

CANNs provides a rich collection of example code covering various use cases. All example code is located in the ``examples/`` folder in the project root directory.

Code Organization Structure
---------------------------

Example code is organized into three main categories by functionality:

.. code-block:: text

   examples/
   ├── cann/                    # CANN model examples
   │   ├── cann1d_*.py
   │   ├── cann2d_*.py
   │   ├── hierarchical_*.py
   │   ├── navigation_*.py
   │   └── theta_sweep_*.py
   ├── brain_inspired/          # Brain-inspired learning algorithms
   │   ├── hopfield_*.py
   │   ├── oja_*.py
   │   ├── bcm_*.py
   │   └── stdp_*.py
   └── pipeline/                # Advanced workflows
       └── *_pipeline.py

CANN Model Examples
-------------------

**Basic Tracking Tasks**:

- ``cann1d_tuning_curve.py`` - 1D CANN tuning curve
- ``cann1d_oscillatory_tracking.py`` - Oscillatory tracking
- ``cann2d_tracking.py`` - 2D CANN tracking

**Spatial Navigation**:

- ``hierarchical_path_integration.py`` - Hierarchical network path integration
- ``navigation_complex_environment.py`` - Navigation in complex environments
- ``theta_sweep_place_cell_network.py`` - Place cell Theta sweep
- ``theta_sweep_grid_cell_network.py`` - Grid cell Theta sweep

**Trajectory Processing**:

- ``import_external_trajectory.py`` - Import external trajectory data

Brain-Inspired Learning Examples
---------------------------------

**Hopfield Networks**:

- ``hopfield_train.py`` - Basic training (images)
- ``hopfield_train_1d.py`` - 1D pattern storage
- ``hopfield_train_mnist.py`` - MNIST digit memory
- ``hopfield_energy_diagnostics.py`` - Energy analysis
- ``hopfield_hebbian_vs_antihebbian.py`` - Learning rule comparison

**Unsupervised Learning**:

- ``oja_pca_extraction.py`` - Oja rule PCA
- ``oja_vs_sanger_comparison.py`` - Oja vs Sanger comparison

**Receptive Field Development**:

- ``bcm_receptive_fields.py`` - BCM directional selectivity

**Temporal Learning**:

- ``stdp_temporal_learning.py`` - STDP temporal pattern learning

Pipeline Examples
-----------------

- ``advanced_theta_sweep_pipeline.py`` - Advanced Theta sweep workflow
- ``theta_sweep_from_external_data.py`` - Pipeline using external data

Running Examples
----------------

All examples can be run directly:

.. code-block:: bash

   # Run from project root directory
   python examples/cann/cann2d_tracking.py

   # Or run from examples directory
   cd examples/brain_inspired/
   python oja_pca_extraction.py

Most examples will generate visualization results (PNG or GIF files).

Tutorial-to-Example Mapping
---------------------------

Each tutorial corresponds to one or more example files:

==================== ========================================
Tutorial             Corresponding Examples
==================== ========================================
CANN Dynamics        ``examples/cann/cann*_tracking.py``
Spatial Navigation   ``examples/cann/hierarchical_*.py``
Memory Networks      ``examples/brain_inspired/hopfield_*.py``
Unsupervised Learn.  ``examples/brain_inspired/oja_*.py``
Receptive Fields     ``examples/brain_inspired/bcm_*.py``
Temporal Learning    ``examples/brain_inspired/stdp_*.py``
Advanced Workflows   ``examples/pipeline/*.py``
==================== ========================================

Modifying and Extending Examples
---------------------------------

Example code is designed to be easy to modify and extend:

1. **Parameter Adjustment**

   Most examples define key parameters at the beginning of the file and can be modified directly.

2. **Code Reuse**

   Copy examples as a starting point for your own projects:

   .. code-block:: bash

      cp examples/cann/cann2d_tracking.py my_project.py
      # Then modify my_project.py

3. **Combined Usage**

   Mix techniques from different examples:

   - Use CANN models + custom tracking tasks
   - Use Hopfield networks + new analyzers
   - Combine multiple learning rules

Getting Help
------------

If you encounter issues running examples:

1. Check that all dependencies are installed: ``make install``
2. Refer to the corresponding tutorial documentation for detailed explanations
3. Ask questions on `GitHub Issues <https://github.com/your-org/canns/issues>`_

Related Documentation
---------------------

- :doc:`../1_tutorials/index` - Detailed tutorial documentation
- :doc:`../0_getting_started/01_quick_start` - Quick start guide

Start Exploring
---------------

Begin with simple examples:

1. :doc:`../1_tutorials/cann_dynamics/tracking_1d` → ``cann1d_tuning_curve.py``
2. :doc:`../1_tutorials/memory_networks/hopfield_basics` → ``hopfield_train.py``
3. :doc:`../1_tutorials/unsupervised_learning/oja_pca` → ``oja_pca_extraction.py``

Gradually explore more complex scenarios!
