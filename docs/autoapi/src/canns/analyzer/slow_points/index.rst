src.canns.analyzer.slow_points
==============================

.. py:module:: src.canns.analyzer.slow_points

.. autoapi-nested-parse::

   Fixed point finder for BrainPy RNN models.

   This module provides tools for identifying and analyzing fixed points
   in recurrent neural networks using JAX/BrainPy.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/analyzer/slow_points/checkpoint/index
   /autoapi/src/canns/analyzer/slow_points/finder/index
   /autoapi/src/canns/analyzer/slow_points/fixed_points/index
   /autoapi/src/canns/analyzer/slow_points/visualization/index


Classes
-------

.. autoapisummary::

   src.canns.analyzer.slow_points.FixedPointFinder
   src.canns.analyzer.slow_points.FixedPoints


Functions
---------

.. autoapisummary::

   src.canns.analyzer.slow_points.load_checkpoint
   src.canns.analyzer.slow_points.plot_fixed_points_2d
   src.canns.analyzer.slow_points.plot_fixed_points_3d
   src.canns.analyzer.slow_points.save_checkpoint


Package Contents
----------------

.. py:class:: FixedPointFinder(rnn_model, method = 'joint', max_iters = 5000, tol_q = 1e-12, tol_dq = 1e-20, lr_init = 1.0, lr_factor = 0.95, lr_patience = 5, lr_cooldown = 0, do_compute_jacobians = True, do_decompose_jacobians = True, tol_unique = 0.001, do_exclude_distance_outliers = True, outlier_distance_scale = 10.0, do_rerun_q_outliers = False, outlier_q_scale = 10.0, max_n_unique = np.inf, final_q_threshold = 1e-08, dtype = 'float32', verbose = True, super_verbose = False, n_iters_per_print_update = 100)

   Find and analyze fixed points in RNN dynamics.

   This class implements an optimization-based approach to finding fixed points
   in recurrent neural networks. It uses gradient descent to minimize the
   objective q = 0.5 * ||x - F(x, u)||^2, where F is the RNN transition function.

   The implementation is compatible with BrainPy RNN models and uses JAX for
   automatic differentiation and optimization.

   Initialize the FixedPointFinder.

   :param rnn_model: A BrainPy RNN model with __call__(inputs, hidden) signature.
   :param method: Optimization method ('joint' or 'sequential').
   :param max_iters: Maximum optimization iterations.
   :param tol_q: Tolerance for q value convergence.
   :param tol_dq: Tolerance for change in q value.
   :param lr_init: Initial learning rate.
   :param lr_factor: Learning rate reduction factor.
   :param lr_patience: Patience for learning rate scheduler.
   :param lr_cooldown: Cooldown for learning rate scheduler.
   :param do_compute_jacobians: Whether to compute Jacobians.
   :param do_decompose_jacobians: Whether to eigendecompose Jacobians.
   :param tol_unique: Tolerance for identifying unique fixed points.
   :param do_exclude_distance_outliers: Whether to exclude distance outliers.
   :param outlier_distance_scale: Scale for distance outlier detection.
   :param do_rerun_q_outliers: Whether to rerun optimization on q outliers.
   :param outlier_q_scale: Scale for q outlier detection.
   :param max_n_unique: Maximum number of unique fixed points to keep.
   :param dtype: Data type for computations.
   :param verbose: Print high-level status updates.
   :param super_verbose: Print per-iteration updates.
   :param n_iters_per_print_update: Print frequency during optimization.


   .. py:method:: find_fixed_points(state_traj, inputs, n_inits = 1024, noise_scale = 0.0, valid_bxt = None, cond_ids = None)

      Find fixed points from sampled RNN states.

      :param state_traj: [n_batch x n_time x n_states] trajectory of RNN states.
      :param inputs: [1 x n_inputs] or [n_inits x n_inputs] constant inputs.
      :param n_inits: Number of initial states to sample.
      :param noise_scale: Std dev of Gaussian noise added to sampled states.
      :param valid_bxt: [n_batch x n_time] boolean mask for valid samples.
      :param cond_ids: [n_inits,] condition IDs for each initialization.

      :returns: FixedPoints object with unique fixed points.
                all_fps: FixedPoints object with all fixed points before filtering.
      :rtype: unique_fps



   .. py:attribute:: do_compute_jacobians
      :value: True



   .. py:attribute:: do_decompose_jacobians
      :value: True



   .. py:attribute:: do_exclude_distance_outliers
      :value: True



   .. py:attribute:: do_rerun_q_outliers
      :value: False



   .. py:attribute:: final_q_threshold


   .. py:attribute:: lr_cooldown
      :value: 0



   .. py:attribute:: lr_factor


   .. py:attribute:: lr_init


   .. py:attribute:: lr_patience
      :value: 5



   .. py:attribute:: max_iters
      :value: 5000



   .. py:attribute:: max_n_unique


   .. py:attribute:: method
      :value: 'joint'



   .. py:attribute:: n_iters_per_print_update
      :value: 100



   .. py:attribute:: outlier_distance_scale


   .. py:attribute:: outlier_q_scale


   .. py:attribute:: rng


   .. py:attribute:: rnn_model


   .. py:attribute:: super_verbose
      :value: False



   .. py:attribute:: tol_dq


   .. py:attribute:: tol_q


   .. py:attribute:: tol_unique


   .. py:attribute:: verbose
      :value: True



.. py:class:: FixedPoints(xstar = None, F_xstar = None, x_init = None, inputs = None, qstar = None, dq = None, n_iters = None, J_xstar = None, dFdu = None, eigval_J_xstar = None, eigvec_J_xstar = None, is_stable = None, cond_id = None, tol_unique = 0.001, dtype=np.float32)

   Container for storing and manipulating fixed points.

   This class stores fixed points found by the FixedPointFinder algorithm,
   along with associated metadata like Jacobians, eigenvalues, and stability.

   .. attribute:: xstar

      [n x n_states] array of fixed point states.

   .. attribute:: F_xstar

      [n x n_states] array of states after one RNN step from xstar.

   .. attribute:: x_init

      [n x n_states] array of initial states used for optimization.

   .. attribute:: inputs

      [n x n_inputs] array of constant inputs during optimization.

   .. attribute:: qstar

      [n,] array of final q values (optimization objective).

   .. attribute:: dq

      [n,] array of change in q at the last optimization step.

   .. attribute:: n_iters

      [n,] array of iteration counts for each optimization.

   .. attribute:: J_xstar

      [n x n_states x n_states] array of Jacobians dF/dx at fixed points.

   .. attribute:: dFdu

      [n x n_states x n_inputs] array of Jacobians dF/du at fixed points.

   .. attribute:: eigval_J_xstar

      [n x n_states] complex array of eigenvalues.

   .. attribute:: eigvec_J_xstar

      [n x n_states x n_states] complex array of eigenvectors.

   .. attribute:: is_stable

      [n,] bool array indicating stability (max |eigenvalue| < 1).

   .. attribute:: cond_id

      [n,] array of condition IDs (optional).

   .. attribute:: tol_unique

      Tolerance for identifying unique fixed points.

   .. attribute:: dtype

      NumPy dtype for data storage.

   Initialize a FixedPoints object.

   :param xstar: Fixed point states [n x n_states].
   :param F_xstar: States after one RNN step [n x n_states].
   :param x_init: Initial states [n x n_states].
   :param inputs: Constant inputs [n x n_inputs].
   :param qstar: Final q values [n,].
   :param dq: Change in q at last step [n,].
   :param n_iters: Iteration counts [n,].
   :param J_xstar: Jacobians dF/dx [n x n_states x n_states].
   :param dFdu: Jacobians dF/du [n x n_states x n_inputs].
   :param eigval_J_xstar: Eigenvalues [n x n_states] (complex).
   :param eigvec_J_xstar: Eigenvectors [n x n_states x n_states] (complex).
   :param is_stable: Stability flags [n,].
   :param cond_id: Condition IDs [n,].
   :param tol_unique: Tolerance for uniqueness detection.
   :param dtype: NumPy data type for storage.


   .. py:method:: __getitem__(idx)

      Index into the fixed points.

      :param idx: Integer index, slice, or array of indices.

      :returns: A new FixedPoints object containing the indexed subset.



   .. py:method:: __len__()

      Return the number of fixed points.



   .. py:method:: decompose_jacobians(verbose = False)

      Compute eigendecomposition of Jacobians and determine stability.

      Computes eigenvalues and eigenvectors for self.J_xstar and determines
      stability based on whether max |eigenvalue| < 1.

      :param verbose: Whether to print status messages.



   .. py:method:: get_unique()

      Identify and return unique fixed points.

      Uniqueness is determined by Euclidean distance in the concatenated
      (xstar, inputs) space. Among duplicates, keeps the one with lowest qstar.

      :returns: A new FixedPoints object containing only unique fixed points.



   .. py:method:: print_summary()

      Print a summary of the fixed points.



   .. py:attribute:: F_xstar
      :value: None



   .. py:attribute:: J_xstar
      :value: None



   .. py:attribute:: cond_id
      :value: None



   .. py:attribute:: dFdu
      :value: None



   .. py:attribute:: dq
      :value: None



   .. py:attribute:: dtype


   .. py:attribute:: eigval_J_xstar
      :value: None



   .. py:attribute:: eigvec_J_xstar
      :value: None



   .. py:property:: has_decomposed_jacobians
      :type: bool


      Check if Jacobians have been decomposed.


   .. py:attribute:: inputs
      :value: None



   .. py:attribute:: is_stable
      :value: None



   .. py:attribute:: n_iters
      :value: None



   .. py:attribute:: qstar
      :value: None



   .. py:attribute:: tol_unique


   .. py:attribute:: x_init
      :value: None



   .. py:attribute:: xstar
      :value: None



.. py:function:: load_checkpoint(model, filepath)

   Load model parameters from a checkpoint file using BrainPy checkpointing.

   :param model: BrainPy model to load parameters into.
   :param filepath: Path to the checkpoint file.

   :returns: True if checkpoint was loaded successfully, False otherwise.

   .. rubric:: Example

   >>> from canns.analyzer.slow_points import load_checkpoint
   >>> if load_checkpoint(rnn, "my_model.msgpack"):
   ...     print("Loaded successfully")
   ... else:
   ...     print("No checkpoint found")
   Loaded checkpoint from: my_model.msgpack
   Loaded successfully


.. py:function:: plot_fixed_points_2d(fixed_points, state_traj, config = None, plot_batch_idx = None, plot_start_time = 0)

   Plot fixed points and trajectories in 2D using PCA.

   :param fixed_points: FixedPoints object containing analysis results.
   :param state_traj: State trajectories [n_batch x n_time x n_states].
   :param config: Plot configuration. If None, uses default config.
   :param plot_batch_idx: Batch indices to plot trajectories. If None, plots first 30.
   :param plot_start_time: Starting time index for trajectory plotting.

   :returns: matplotlib Figure object.

   .. rubric:: Example

   >>> from canns.analyzer.slow_points import plot_fixed_points_2d, FixedPoints
   >>> from canns.analyzer.visualization import PlotConfig
   >>> config = PlotConfig(
   ...     title="Fixed Points Analysis",
   ...     figsize=(10, 8),
   ...     save_path="fps_2d.png"
   ... )
   >>> fig = plot_fixed_points_2d(unique_fps, hiddens, config=config)


.. py:function:: plot_fixed_points_3d(fixed_points, state_traj, config = None, plot_batch_idx = None, plot_start_time = 0)

   Plot fixed points and trajectories in 3D using PCA.

   :param fixed_points: FixedPoints object containing analysis results.
   :param state_traj: State trajectories [n_batch x n_time x n_states].
   :param config: Plot configuration. If None, uses default config.
   :param plot_batch_idx: Batch indices to plot trajectories. If None, plots first 30.
   :param plot_start_time: Starting time index for trajectory plotting.

   :returns: matplotlib Figure object.

   .. rubric:: Example

   >>> from canns.analyzer.slow_points import plot_fixed_points_3d, FixedPoints
   >>> from canns.analyzer.visualization import PlotConfig
   >>> config = PlotConfig(
   ...     title="Fixed Points 3D",
   ...     figsize=(12, 10),
   ...     save_path="fps_3d.png"
   ... )
   >>> fig = plot_fixed_points_3d(unique_fps, hiddens, config=config)


.. py:function:: save_checkpoint(model, filepath)

   Save model parameters to a checkpoint file using BrainPy checkpointing.

   :param model: BrainPy model to save.
   :param filepath: Path to save the checkpoint file.

   .. rubric:: Example

   >>> from canns.analyzer.slow_points import save_checkpoint
   >>> save_checkpoint(rnn, "my_model.msgpack")
   Saved checkpoint to: my_model.msgpack


