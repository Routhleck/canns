canns
=====

.. py:module:: canns

.. autoapi-nested-parse::

   Top-level package for CANNs.

   This module exposes the main namespaces so you can import them directly from
   ``canns`` (for example, ``canns.data`` or ``canns.trainer``). It also provides
   simple version metadata.

   .. rubric:: Examples

   >>> import canns
   >>> print(canns.__version__)
   >>> print(canns.version_info)
   >>> print(list(canns.data.DATASETS))



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/canns/analyzer/index
   /autoapi/canns/data/index
   /autoapi/canns/models/index
   /autoapi/canns/pipeline/index
   /autoapi/canns/task/index
   /autoapi/canns/trainer/index
   /autoapi/canns/typing/index
   /autoapi/canns/utils/index


Attributes
----------

.. autoapisummary::

   canns.__version__
   canns.version_info


Package Contents
----------------

.. py:data:: __version__

   Human-readable package version string.

   This is usually derived from the installed package metadata. When that
   information is unavailable, it falls back to ``"unknown"``.

   .. rubric:: Examples

   >>> import canns
   >>> print(canns.__version__)

.. py:data:: version_info

   Version information as a tuple.

   The tuple typically follows ``(major, minor, patch)``. A development or
   documentation build may return a fallback value instead.

   .. rubric:: Examples

   >>> import canns
   >>> print(canns.version_info)

