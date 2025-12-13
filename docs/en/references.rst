==========
References
==========

This page lists all references cited throughout the CANNs documentation.

.. note::
   To cite these references in your documentation or notebooks, use the ``:cite:`` role.
   For example: ``:cite:`wu2008dynamics``` renders as [Wu08].

Complete Bibliography
=====================

.. bibliography::
   :style: alpha
   :all:
   :list: enumerated

How to Cite References
======================

In RST Files
------------

Use the ``:cite:`` role in your text:

.. code-block:: rst

   The dynamics of continuous attractors were analyzed by :cite:`wu2008dynamics`.
   Foundational work includes :cite:`amari1977dynamics` and :cite:`wu2016continuous`.

In Jupyter Notebooks
--------------------

Use the same ``:cite:`` role in markdown cells:

.. code-block:: markdown

   The dynamics of continuous attractors were analyzed by :cite:`wu2008dynamics`.
   Grid cells were discovered by :cite:`hafting2005microstructure`.

Adding New References
=====================

To add new references to the bibliography:

1. Open ``docs/refs/references.bib``
2. Add your BibTeX entry following the existing format
3. Use a consistent citation key format: ``authorYEARkeyword`` (e.g., ``wu2008dynamics``)
4. Cite the reference using ``:cite:`citationkey```
5. The reference will automatically appear in this bibliography

For more information, see the `sphinxcontrib-bibtex documentation <https://sphinxcontrib-bibtex.readthedocs.io/>`_.
