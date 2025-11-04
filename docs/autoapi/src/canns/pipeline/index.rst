src.canns.pipeline
==================

.. py:module:: src.canns.pipeline

.. autoapi-nested-parse::

   CANNs Pipeline Module

   High-level pipelines for common analysis workflows, designed to make CANN models
   accessible to experimental researchers without requiring detailed knowledge of
   the underlying implementations.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/pipeline/theta_sweep/index


Classes
-------

.. autoapisummary::

   src.canns.pipeline.Pipeline
   src.canns.pipeline.ThetaSweepPipeline


Functions
---------

.. autoapisummary::

   src.canns.pipeline.batch_process_trajectories
   src.canns.pipeline.load_trajectory_from_csv


Package Contents
----------------

.. py:class:: Pipeline

   Bases: :py:obj:`abc.ABC`


   Abstract base class for CANNs pipelines.

   Pipelines orchestrate multi-step workflows (data preparation, model execution,
   visualization, etc.). This base class standardizes how we manage results and
   output directories so derived pipelines can focus on domain-specific logic.


   .. py:method:: get_results()

      Return stored results or raise if the pipeline has not been executed.



   .. py:method:: has_results()

      Check whether the pipeline has already produced results.



   .. py:method:: prepare_output_dir(output_dir, *, create = True)

      Validate and optionally create the output directory for derived pipelines.



   .. py:method:: reset()

      Reset stored state so the pipeline can be executed again cleanly.



   .. py:method:: run(*args, **kwargs)
      :abstractmethod:


      Execute the pipeline and return a mapping of generated artifacts.



   .. py:method:: set_results(results)

      Store pipeline results and return them for convenient chaining.



   .. py:attribute:: output_dir
      :type:  pathlib.Path | None
      :value: None



   .. py:attribute:: results
      :type:  dict[str, Any] | None
      :value: None



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


