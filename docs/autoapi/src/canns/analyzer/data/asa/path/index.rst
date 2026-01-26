src.canns.analyzer.data.asa.path
================================

.. py:module:: src.canns.analyzer.data.asa.path


Functions
---------

.. autoapisummary::

   src.canns.analyzer.data.asa.path.align_coords_to_position
   src.canns.analyzer.data.asa.path.apply_angle_scale
   src.canns.analyzer.data.asa.path.as_1d
   src.canns.analyzer.data.asa.path.draw_base_parallelogram
   src.canns.analyzer.data.asa.path.find_coords_matrix
   src.canns.analyzer.data.asa.path.find_times_box
   src.canns.analyzer.data.asa.path.interp_coords_to_full
   src.canns.analyzer.data.asa.path.load_npz_any
   src.canns.analyzer.data.asa.path.parse_times_box_to_indices
   src.canns.analyzer.data.asa.path.resolve_time_slice
   src.canns.analyzer.data.asa.path.skew_transform
   src.canns.analyzer.data.asa.path.snake_wrap_trail_in_parallelogram
   src.canns.analyzer.data.asa.path.unwrap_container


Module Contents
---------------

.. py:function:: align_coords_to_position(t_full, x_full, y_full, coords2, use_box, times_box, interp_to_full)

   Align decoded coordinates to the original (x, y, t) trajectory.

   :param t_full: Full-length trajectory arrays of shape (T,).
   :type t_full: np.ndarray
   :param x_full: Full-length trajectory arrays of shape (T,).
   :type x_full: np.ndarray
   :param y_full: Full-length trajectory arrays of shape (T,).
   :type y_full: np.ndarray
   :param coords2: Decoded coordinates of shape (K, 2) or (T, 2).
   :type coords2: np.ndarray
   :param use_box: Whether to use ``times_box`` to align to the original trajectory.
   :type use_box: bool
   :param times_box: Time indices or timestamps corresponding to ``coords2`` when ``use_box=True``.
   :type times_box: np.ndarray | None
   :param interp_to_full: If True, interpolate decoded coords back to full length; otherwise return a subset.
   :type interp_to_full: bool

   :returns: ``(t_aligned, x_aligned, y_aligned, coords_aligned, tag)`` where ``tag`` describes
             the alignment path used.
   :rtype: tuple

   .. rubric:: Examples

   >>> t, x, y, coords2, tag = align_coords_to_position(  # doctest: +SKIP
   ...     t_full, x_full, y_full, coords2,
   ...     use_box=True, times_box=decoding["times_box"], interp_to_full=True
   ... )
   >>> coords2.shape[1]
   2


.. py:function:: apply_angle_scale(coords2, scale)

   Convert angle units to radians before wrapping.

   :param coords2: Angle array of shape (T, 2) in the given ``scale``.
   :type coords2: np.ndarray
   :param scale: ``rad``  : already in radians.
                 ``deg``  : degrees -> radians.
                 ``unit`` : unit circle in [0, 1] -> radians.
                 ``auto`` : infer unit circle if values look like [0, 1].
   :type scale: {"rad", "deg", "unit", "auto"}

   :returns: Angles in radians.
   :rtype: np.ndarray

   .. rubric:: Examples

   >>> apply_angle_scale([[0.25, 0.5]], "unit")  # doctest: +SKIP


.. py:function:: as_1d(x)

   Convert input to a 1D numpy array (robust to 0-d object containers).


.. py:function:: draw_base_parallelogram(ax)

.. py:function:: find_coords_matrix(dec, coords_key = None, prefer_box_fallback = False)

   Find a decoded coords matrix (N,D>=2) in decoding dict.

   IMPORTANT: to match your original test1 behavior, we ALWAYS prefer a true (N,2)
   angles matrix (e.g. key 'coords') if it exists, even if you set --use-box.

   Only when no (N,2) matrix exists do we fall back to >=2-col matrices (coordsbox, etc.).


.. py:function:: find_times_box(dec)

   Try to find a 'times_box' / keep-index vector in decoding dict.


.. py:function:: interp_coords_to_full(idx_map, coords2, T_full)

   Interpolate (K,2) circular coords back to full length (T_full,2).


.. py:function:: load_npz_any(path)

   Load .npz into a plain dict (allow_pickle=True).


.. py:function:: parse_times_box_to_indices(times_box, t_full)

   Convert times_box to integer indices into t_full.


.. py:function:: resolve_time_slice(t, tmin, tmax, imin, imax)

   Return [i0,i1) slice bounds using either index bounds or time bounds.


.. py:function:: skew_transform(theta_2d)

   Map (theta1,theta2) to skew coordinates used in the base parallelogram.


.. py:function:: snake_wrap_trail_in_parallelogram(xy_base, e1, e2)

   Insert NaNs when the trail wraps across the torus fundamental domain.


.. py:function:: unwrap_container(x)

   Unwrap 0-d object arrays that store a python object.


