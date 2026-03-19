[ANONYMOUS_PROJECT].task
==========

.. py:module:: [ANONYMOUS_PROJECT].task


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/[ANONYMOUS_PROJECT]/task/closed_loop_navigation/index
   /autoapi/[ANONYMOUS_PROJECT]/task/navigation_base/index
   /autoapi/[ANONYMOUS_PROJECT]/task/open_loop_navigation/index
   /autoapi/[ANONYMOUS_PROJECT]/task/tracking/index


Attributes
----------

.. autoapisummary::

   [ANONYMOUS_PROJECT].task.INT32_MAX


Classes
-------

.. autoapisummary::

   [ANONYMOUS_PROJECT].task.GeodesicDistanceResult
   [ANONYMOUS_PROJECT].task.MovementCostGrid


Package Contents
----------------

.. py:class:: GeodesicDistanceResult

   .. py:attribute:: accessible_indices
      :type:  numpy.ndarray


   .. py:attribute:: cost_grid
      :type:  MovementCostGrid


   .. py:attribute:: distances
      :type:  numpy.ndarray


.. py:class:: MovementCostGrid

   .. py:method:: get_cell_index(pos)

      Get the geodesic index of the grid cell containing the given position.

      This method is JAX-compatible and can be used inside jitted functions.

      :param pos: (x, y) coordinates of the position.

      :returns: Index of the grid cell in the accessible_indices array, or -1 if
                the position is out of bounds or in an impassable cell.

      .. note::

         Returns -1 (instead of None) for JAX compatibility. The caller should
         check for negative values to detect invalid positions.



   .. py:attribute:: accessible_indices
      :type:  numpy.ndarray | None
      :value: None



   .. py:property:: accessible_mask
      :type: numpy.ndarray



   .. py:attribute:: costs
      :type:  numpy.ndarray


   .. py:attribute:: dx
      :type:  float


   .. py:attribute:: dy
      :type:  float


   .. py:property:: shape
      :type: tuple[int, int]



   .. py:property:: x_centers
      :type: numpy.ndarray



   .. py:attribute:: x_edges
      :type:  numpy.ndarray


   .. py:property:: y_centers
      :type: numpy.ndarray



   .. py:attribute:: y_edges
      :type:  numpy.ndarray


.. py:data:: INT32_MAX

