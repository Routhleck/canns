API Reference
=============

This section contains the complete API documentation for the CANNs library, automatically generated from the source code docstrings.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   models
   tasks
   analyzer
   misc

Quick Navigation
----------------

**Core Components:**

* :doc:`models` - Neural network models (CANN1D, CANN2D, Hierarchical networks)
* :doc:`tasks` - Task definitions (Tracking, Path integration)  
* :doc:`analyzer` - Analysis tools (Visualization, Statistical analysis)
* :doc:`misc` - Utilities (Benchmarking, Helper functions)

**Key Classes:**

* :class:`canns.models.basic.CANN1D` - One-dimensional CANN
* :class:`canns.models.basic.CANN2D` - Two-dimensional CANN
* :class:`canns.task.tracking.SmoothTracking1D` - 1D smooth tracking task
* :class:`canns.task.tracking.SmoothTracking2D` - 2D smooth tracking task
* :class:`canns.models.basic.HierarchicalNetwork` - Hierarchical network model

**Key Functions:**

* :func:`canns.analyzer.visualize.energy_landscape_1d_animation` - 1D energy landscape animation
* :func:`canns.analyzer.visualize.energy_landscape_2d_animation` - 2D energy landscape animation
* :func:`canns.analyzer.utils.z_score_normalization` - Z-score normalization utility

Search and Index
-----------------

* :ref:`genindex` - General index
* :ref:`modindex` - Module index
* :ref:`search` - Search page