Example Map
===========

Use this quick map to find the script that matches your goal:

- **First impressions** – Start with the demos in :doc:`models`.
  `cann1d_oscillatory_tracking.py <https://github.com/Routhleck/canns/blob/master/examples/cann/cann1d_oscillatory_tracking.py>`_
  and `cann2d_tracking.py <https://github.com/Routhleck/canns/blob/master/examples/cann/cann2d_tracking.py>`_
  animate 1D/2D CANN bumps side by side.
- **Hebbian memory** – Head to :doc:`trainer` for
  `hopfield_train.py <https://github.com/Routhleck/canns/blob/master/examples/brain_inspired/hopfield_train.py>`_
  and `hopfield_train_mnist.py <https://github.com/Routhleck/canns/blob/master/examples/brain_inspired/hopfield_train_mnist.py>`_.
  They show how to drive :class:`~src.canns.trainer.HebbianTrainer` with image and digit data.
- **Navigation & tasks** – Examples that integrate trajectories, external data, or path integration live in :doc:`tasks`.
- **Experimental analysis** – If you need Hugging Face datasets, bump fitting, or topology, jump to :doc:`analyzer`.
- **End-to-end pipelines** – Theta sweep automation, file exports, and batch workflows are summarised in :doc:`pipeline`.

Ready-to-run setup
------------------

1. **Create the environment**

   .. code-block:: bash

      make install

2. **Run a script with uv** (example below launches
   `cann1d_oscillatory_tracking.py <https://github.com/Routhleck/canns/blob/master/examples/cann/cann1d_oscillatory_tracking.py>`_):

   .. code-block:: bash

      uv run python examples/cann/cann1d_oscillatory_tracking.py

3. **Review artefacts** – Most scripts emit GIFs/PNGs/NPZ files. The filename is usually printed near the end of the script.

Notebook shortcuts
------------------

In addition to scripts, ``docs/en/notebooks/01_quick_start.ipynb`` and
``docs/en/notebooks/00_design_philosophy.ipynb`` provide interactive walkthroughs.
The README links to Binder and Colab for zero-install exploration.

What’s next
-----------

The remaining chapters (:doc:`models` through :doc:`pipeline`) follow a consistent format:
script overview → reasoning → key API references → extension ideas. Feel free to dip into
the sections you need or read straight through for a complete tour of the toolkit.
