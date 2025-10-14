src.canns.pipeline.theta_sweep
==============================

.. py:module:: src.canns.pipeline.theta_sweep

.. autoapi-nested-parse::

   Theta Sweep Pipeline for External Trajectory Analysis

   This module provides a high-level pipeline for experimental scientists to analyze
   their trajectory data using CANN theta sweep models without needing to understand
   the underlying implementation details.



Classes
-------

.. autoapisummary::

   src.canns.pipeline.theta_sweep.ThetaSweepPipeline


Functions
---------

.. autoapisummary::

   src.canns.pipeline.theta_sweep.batch_process_trajectories
   src.canns.pipeline.theta_sweep.load_trajectory_from_csv


Module Contents
---------------

.. py:class:: ThetaSweepPipeline(trajectory_data, times = None, env_size = 2.0, dt = 0.001, direction_cell_params = None, grid_cell_params = None, theta_params = None, spatial_nav_params = None)

   Bases: :py:obj:`src.canns.pipeline._base.Pipeline`


   High-level pipeline for theta sweep analysis of external trajectory data.

   This pipeline abstracts the complex workflow of running CANN theta sweep models
   on experimental trajectory data, making it accessible to researchers who want
   to analyze neural responses without diving into implementation details.

   .. rubric:: Example

   ```python
   # Simple usage - just provide trajectory data
   pipeline = ThetaSweepPipeline(
       trajectory_data=positions,  # shape: (n_steps, 2)
       times=times                 # shape: (n_steps,)
   )

   results = pipeline.run(output_dir="my_results/")
   print(f"Animation saved to: {results['animation_path']}")
   ```

   Initialize the theta sweep pipeline.

   :param trajectory_data: Position coordinates with shape (n_steps, 2) for 2D trajectories
   :param times: Optional time array with shape (n_steps,). If None, uniform time steps will be used
   :param env_size: Environment size (assumes square environment)
   :param dt: Simulation time step
   :param direction_cell_params: Parameters for DirectionCellNetwork. If None, uses defaults
   :param grid_cell_params: Parameters for GridCellNetwork. If None, uses defaults
   :param theta_params: Parameters for theta modulation. If None, uses defaults
   :param spatial_nav_params: Additional parameters for OpenLoopNavigationTask. If None, uses defaults


   .. py:method:: run(output_dir = 'theta_sweep_results', save_animation = True, save_plots = True, show_plots = False, animation_fps = 10, animation_dpi = 120, verbose = True)

      Run the complete theta sweep pipeline.

      :param output_dir: Directory to save output files
      :param save_animation: Whether to save the theta sweep animation
      :param save_plots: Whether to save analysis plots
      :param show_plots: Whether to display plots interactively
      :param animation_fps: Frame rate for animation
      :param animation_dpi: DPI for animation output
      :param verbose: Whether to print progress messages

      :returns: Dictionary containing paths to generated files and analysis data



   .. py:attribute:: direction_cell_params


   .. py:attribute:: direction_network
      :value: None



   .. py:attribute:: dt
      :value: 0.001



   .. py:attribute:: env_size
      :value: 2.0



   .. py:attribute:: grid_cell_params


   .. py:attribute:: grid_network
      :value: None



   .. py:attribute:: spatial_nav_params


   .. py:attribute:: spatial_nav_task
      :value: None



   .. py:attribute:: theta_params


   .. py:attribute:: times


   .. py:attribute:: trajectory_data


.. py:function:: batch_process_trajectories(trajectory_list, output_base_dir = 'batch_results', **kwargs)

   Process multiple trajectories in batch.

   :param trajectory_list: List of (trajectory_data, times) tuples or trajectory_data arrays
   :param output_base_dir: Base directory for batch results
   :param \*\*kwargs: Additional parameters passed to ThetaSweepPipeline

   :returns: Dictionary mapping trajectory indices to results


.. py:function:: load_trajectory_from_csv(filepath, x_col = 'x', y_col = 'y', time_col = 'time', **kwargs)

   Load trajectory data from CSV file and run theta sweep analysis.

   :param filepath: Path to CSV file
   :param x_col: Column name for x coordinates
   :param y_col: Column name for y coordinates
   :param time_col: Column name for time data (optional)
   :param \*\*kwargs: Additional parameters passed to ThetaSweepPipeline

   :returns: Dictionary containing analysis results and file paths


