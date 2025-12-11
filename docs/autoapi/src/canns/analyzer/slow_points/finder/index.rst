src.canns.analyzer.slow_points.finder
=====================================

.. py:module:: src.canns.analyzer.slow_points.finder

.. autoapi-nested-parse::

   Fixed point finder for BrainPy RNN models.



Classes
-------

.. autoapisummary::

   src.canns.analyzer.slow_points.finder.FixedPointFinder


Module Contents
---------------

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



