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
   :all:
   :style: unsrt

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

**Important**: In Jupyter notebooks, you must use **raw cells** with reStructuredText format, not markdown cells.

1. Create a raw cell (Cell → Cell Type → Raw)
2. Set the cell metadata to indicate RST format:

   .. code-block:: json

      {
        "raw_mimetype": "text/restructuredtext"
      }

3. Write RST content with citations:

   .. code-block:: rst

      This is a paragraph with citations :cite:p:`amari1977dynamics,wu2008dynamics`.

4. Add a bibliography directive at the end of the notebook (in another raw RST cell):

   .. code-block:: rst

      References
      ----------

      .. bibliography::
         :cited:
         :style: alpha

**Citation Styles**:

- ``:cite:p:`key``` - Parenthetical: (Author, Year) - entire citation is clickable
- ``:cite:t:`key``` - Textual: Author [Year] - only year is clickable

**Example**: See ``docs/en/0_why_canns.ipynb`` for a complete working example.

Adding New References
=====================

To add new references to the bibliography:

1. Open ``docs/refs/references.bib``
2. Add your BibTeX entry following the existing format
3. Use a consistent citation key format: ``authorYEARkeyword`` (e.g., ``wu2008dynamics``)
4. Cite the reference using ``:cite:`citationkey```
5. The reference will automatically appear in this bibliography

For more information, see the `sphinxcontrib-bibtex documentation <https://sphinxcontrib-bibtex.readthedocs.io/>`_.
