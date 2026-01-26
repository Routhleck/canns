src.canns.utils
===============

.. py:module:: src.canns.utils

.. autoapi-nested-parse::

   General utilities for CANNs.

   This namespace provides small helpers that don't fit into a specific domain,
   such as benchmarking utilities.

   .. rubric:: Examples

   >>> from canns.utils import benchmark
   >>>
   >>> @benchmark(runs=3)
   ... def add():
   ...     return 1 + 1
   >>>
   >>> add()



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/utils/benchmark/index


Functions
---------

.. autoapisummary::

   src.canns.utils.benchmark


Package Contents
----------------

.. py:function:: benchmark(runs=10)

