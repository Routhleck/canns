src.canns.analyzer.visualization.core.backend
=============================================

.. py:module:: src.canns.analyzer.visualization.core.backend

.. autoapi-nested-parse::

   Unified animation backend selection and management.

   This module provides a centralized system for choosing the optimal rendering backend
   (imageio vs matplotlib) based on file format, available dependencies, and user preferences.



Classes
-------

.. autoapisummary::

   src.canns.analyzer.visualization.core.backend.AnimationBackend
   src.canns.analyzer.visualization.core.backend.BackendSelection


Functions
---------

.. autoapisummary::

   src.canns.analyzer.visualization.core.backend.emit_backend_warnings
   src.canns.analyzer.visualization.core.backend.get_imageio_writer_kwargs
   src.canns.analyzer.visualization.core.backend.get_multiprocessing_context
   src.canns.analyzer.visualization.core.backend.get_optimal_worker_count
   src.canns.analyzer.visualization.core.backend.select_animation_backend


Module Contents
---------------

.. py:class:: AnimationBackend(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Available animation rendering backends.


   .. py:attribute:: AUTO
      :value: 'auto'



   .. py:attribute:: IMAGEIO
      :value: 'imageio'



   .. py:attribute:: MATPLOTLIB
      :value: 'matplotlib'



.. py:class:: BackendSelection

   Result of backend selection process.


   .. py:attribute:: backend
      :type:  Literal['imageio', 'matplotlib']

      The selected backend.


   .. py:attribute:: reason
      :type:  str

      Why this backend was selected.


   .. py:attribute:: supports_parallel
      :type:  bool

      Whether this backend supports parallel rendering.


   .. py:attribute:: warnings
      :type:  list[str]

      Any warnings or suggestions for the user.


.. py:function:: emit_backend_warnings(warnings_list, stacklevel = 2)

   Emit all backend selection warnings.


.. py:function:: get_imageio_writer_kwargs(save_path, fps)

   Get appropriate kwargs for imageio.get_writer() based on file format.

   :param save_path: Output file path
   :param fps: Frames per second

   :returns: Tuple of (writer_kwargs, mode) where mode is for get_writer()

   .. rubric:: Example

   >>> kwargs, mode = get_imageio_writer_kwargs("output.gif", 10)
   >>> writer = imageio.get_writer("output.gif", mode=mode, **kwargs)


.. py:function:: get_multiprocessing_context(prefer_fork = False)

   Get appropriate multiprocessing context for this platform.

   :param prefer_fork: Whether to prefer 'fork' over 'spawn' (Linux only)

   :returns: Tuple of (multiprocessing context, method name) or (None, None) if unavailable


.. py:function:: get_optimal_worker_count()

   Get optimal number of parallel workers for this system.

   :returns: Number of workers (cpu_count - 1, minimum 1)


.. py:function:: select_animation_backend(save_path, requested_backend = None, check_imageio_plugins = True)

   Select the optimal animation rendering backend.

   :param save_path: Output file path (determines format).
   :param requested_backend: Backend preference ('imageio', 'matplotlib', 'auto', or None).
   :param check_imageio_plugins: Whether to verify imageio can write the format.

   :returns: BackendSelection with backend choice and metadata.

   .. rubric:: Examples

   >>> from canns.analyzer.visualization.core.backend import select_animation_backend
   >>> selection = select_animation_backend("output.mp4")
   >>> print(selection.backend in {"imageio", "matplotlib"})
   True


