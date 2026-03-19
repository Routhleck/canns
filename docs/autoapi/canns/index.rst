[ANONYMOUS_PROJECT]
=====

.. py:module:: [ANONYMOUS_PROJECT]

.. autoapi-nested-parse::

   Top-level package for [ANONYMOUS_PROJECT].

   This module exposes the main namespaces so you can import them directly from
   ``[ANONYMOUS_PROJECT]`` (for example, ``[ANONYMOUS_PROJECT].data`` or ``[ANONYMOUS_PROJECT].trainer``). It also provides
   simple version metadata.

   .. rubric:: Examples

   >>> import [ANONYMOUS_PROJECT]
   >>> print([ANONYMOUS_PROJECT].__version__)
   >>> print([ANONYMOUS_PROJECT].version_info)
   >>> print(list([ANONYMOUS_PROJECT].data.DATASETS))



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/[ANONYMOUS_PROJECT]/analyzer/index
   /autoapi/[ANONYMOUS_PROJECT]/data/index
   /autoapi/[ANONYMOUS_PROJECT]/models/index
   /autoapi/[ANONYMOUS_PROJECT]/pipeline/index
   /autoapi/[ANONYMOUS_PROJECT]/task/index
   /autoapi/[ANONYMOUS_PROJECT]/trainer/index
   /autoapi/[ANONYMOUS_PROJECT]/typing/index
   /autoapi/[ANONYMOUS_PROJECT]/utils/index


Attributes
----------

.. autoapisummary::

   [ANONYMOUS_PROJECT].__version__
   [ANONYMOUS_PROJECT].version_info


Package Contents
----------------

.. py:data:: __version__

   Human-readable package version string.

   This is usually derived from the installed package metadata. When that
   information is unavailable, it falls back to ``"unknown"``.

   .. rubric:: Examples

   >>> import [ANONYMOUS_PROJECT]
   >>> print([ANONYMOUS_PROJECT].__version__)

.. py:data:: version_info

   Version information as a tuple.

   The tuple typically follows ``(major, minor, patch)``. A development or
   documentation build may return a fallback value instead.

   .. rubric:: Examples

   >>> import [ANONYMOUS_PROJECT]
   >>> print([ANONYMOUS_PROJECT].version_info)

