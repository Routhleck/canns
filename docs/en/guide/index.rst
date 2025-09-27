Example Handbook Overview
=========================

This guide organizes the repository examples by theme so you can jump straight to a
reference implementation. Each chapter lists the matching ``examples/`` path, the primary
modules involved, the artefacts the script produces, and ideas for extending it.

Running Notes
-------------

- Run ``make install`` first to prepare dependencies, then execute a script with
  ``uv run python <example.py>``.
- Examples that emit GIF/PNG/NPZ files write to the example directory (or the project
  root) by default; tweak the script arguments if you need a different location.
- Plots that normally pop up a GUI window default to ``show=False`` so they work on headless
  machines—enable the flag if you want interactive figures.

Example Categories
------------------

.. toctree::
   :maxdepth: 2
   :caption: Chapters

   architecture
   models
   trainer
   tasks
   analyzer
   pipeline
   workflows
