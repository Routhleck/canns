src.canns.analyzer.slow_points.fixed_points
===========================================

.. py:module:: src.canns.analyzer.slow_points.fixed_points

.. autoapi-nested-parse::

   FixedPoints data container class for storing fixed point analysis results.



Classes
-------

.. autoapisummary::

   src.canns.analyzer.slow_points.fixed_points.FixedPoints


Module Contents
---------------

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



