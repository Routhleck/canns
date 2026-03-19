[ANONYMOUS_PROJECT].utils
===========

.. py:module:: [ANONYMOUS_PROJECT].utils

.. autoapi-nested-parse::

   General utilities for [ANONYMOUS_PROJECT].

   This namespace provides small helpers that don't fit into a specific domain,
   such as benchmarking utilities.

   .. rubric:: Examples

   >>> from [ANONYMOUS_PROJECT].utils import benchmark
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

   /autoapi/[ANONYMOUS_PROJECT]/utils/benchmark/index


Functions
---------

.. autoapisummary::

   [ANONYMOUS_PROJECT].utils.benchmark


Package Contents
----------------

.. py:function:: benchmark(runs=10)

