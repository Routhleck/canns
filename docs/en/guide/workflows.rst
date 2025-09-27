Putting It Together
===================

Once you have sampled the individual demos, combine them into a workflow that fits your project.

Pick modules as needed
----------------------

- **Model baselines → tuning** – Start with the 1D/2D CANN demos in :doc:`models`, then revisit the
  Hopfield examples in :doc:`trainer` to observe Hebbian effects before and after training.
- **Real trajectories → pipelines** – Import recordings with :doc:`tasks` (see
  `import_external_trajectory.py <https://github.com/Routhleck/canns/blob/master/examples/cann/import_external_trajectory.py>`_),
  store them as ``.npz``, and feed them into the theta-sweep pipeline in :doc:`pipeline`.
- **Experimental validation** – Compare pipeline outputs against the ROI/TDA analyses in :doc:`analyzer` to confirm that
  simulated and recorded data share the same structure.

Rapid prototyping checklist
---------------------------

1. **Copy a template** – Duplicate the closest example into ``examples/your_topic/``.
2. **Adjust configuration** –

   - Modify ``PlotConfigs``/``ThetaSweepPipeline`` dictionaries for new visual outputs.
   - Pass overrides on the command line via ``uv run python`` and ``--help`` if the script exposes arguments.
3. **Record artefacts** – Print or log the generated file paths so downstream automation can pick them up.

Common pitfalls
---------------

- **Slow animations** – Reduce ``time_steps_per_second`` or ``fps``. On headless servers install ``imageio[ffmpeg]`` for faster encoding.
- **Missing dependencies** – ``ModuleNotFoundError`` means you should ``uv add <package>``. The earlier chapters list the extra packages
you might need.
- **Download hiccups** – Place the required files manually or retry; the ``load_*`` helpers always check the local cache first.

Next steps
----------

Add your own scripts and extend this guide so that every example has a matching doc entry.
When you open a PR, follow the pattern in ``docs/en/examples/index.rst`` (or mirror the structure used here).
