src.canns.pipeline
==================

.. py:module:: src.canns.pipeline

.. autoapi-nested-parse::

   CANNs pipeline entrypoints.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/pipeline/asa/index
   /autoapi/src/canns/pipeline/gallery/index


Classes
-------

.. autoapisummary::

   src.canns.pipeline.ASAApp


Functions
---------

.. autoapisummary::

   src.canns.pipeline.asa_main


Package Contents
----------------

.. py:class:: ASAApp

   Bases: :py:obj:`textual.app.App`


   Main TUI application for ASA analysis.

   Create an instance of an app.

   :param driver_class: Driver class or `None` to auto-detect.
                        This will be used by some Textual tools.
   :param css_path: Path to CSS or `None` to use the `CSS_PATH` class variable.
                    To load multiple CSS files, pass a list of strings or paths which
                    will be loaded in order.
   :param watch_css: Reload CSS if the files changed. This is set automatically if
                     you are using `textual run` with the `dev` switch.
   :param ansi_color: Allow ANSI colors if `True`, or convert ANSI colors to RGB if `False`.

   :raises CssPathError: When the supplied CSS path(s) are an unexpected type.


   .. py:method:: action_back_to_preprocess()

      Go back to preprocessing page.



   .. py:method:: action_change_workdir()

      Change working directory.



   .. py:method:: action_continue_to_analysis()

      Continue from preprocessing to analysis page.



   .. py:method:: action_help()

      Show help screen.



   .. py:method:: action_quit()

      Quit the application.



   .. py:method:: action_refresh()

      Refresh the UI.



   .. py:method:: action_run_action()

      Run current page action (Continue or Run Analysis).



   .. py:method:: action_run_analysis()

      Run analysis on preprocessed data.



   .. py:method:: action_stop()

      Request cancellation of the running worker.



   .. py:method:: append_log_file(message)

      Append log message to file for easy copying.



   .. py:method:: apply_preset_params()

      Apply preset defaults to analysis inputs.



   .. py:method:: check_terminal_size()

      Check terminal size and show warning if too small.



   .. py:method:: collect_analysis_params()

      Collect analysis parameters from UI into state.



   .. py:method:: compose()

      Compose the main UI layout.



   .. py:method:: log_message(message)

      Add log message.



   .. py:method:: on_button_pressed(event)

      Handle button presses.



   .. py:method:: on_checkbox_changed(event)

      Handle checkbox changes.



   .. py:method:: on_directory_tree_file_selected(event)

      Handle file selection from tree.



   .. py:method:: on_mount()

      Handle app mount event.



   .. py:method:: on_resize(event)

      Handle terminal resize events.



   .. py:method:: on_select_changed(event)

      Handle select changes.



   .. py:method:: on_workdir_selected(path)

      Handle workdir selection.



   .. py:method:: on_worker_state_changed(event)

      Handle worker state changes.



   .. py:method:: set_run_status(message, status_class = None)

      Update run status label and styling.



   .. py:method:: update_analysis_params_visibility()

      Show params for the selected analysis mode.



   .. py:method:: update_cohospace_controls()

      Enable/disable CohoSpace controls based on mode.



   .. py:method:: update_decode_controls()

      Enable/disable decode controls based on decode version.



   .. py:method:: update_pathcompare_controls()

      Enable/disable PathCompare controls based on mode.



   .. py:method:: update_progress(percent)

      Update progress bar.



   .. py:method:: update_workdir_label()

      Update the workdir label.



   .. py:attribute:: BINDINGS

      The default key bindings.


   .. py:attribute:: CSS_PATH
      :value: 'styles.tcss'


      File paths to load CSS from.


   .. py:attribute:: MIN_HEIGHT
      :value: 30



   .. py:attribute:: MIN_WIDTH
      :value: 100



   .. py:attribute:: RECOMMENDED_HEIGHT
      :value: 40



   .. py:attribute:: RECOMMENDED_WIDTH
      :value: 120



   .. py:attribute:: TITLE
      :value: 'Attractor Structure Analyzer (ASA)'


      A class variable to set the *default* title for the application.

      To update the title while the app is running, you can set the [title][textual.app.App.title] attribute.
      See also [the `Screen.TITLE` attribute][textual.screen.Screen.TITLE].


   .. py:attribute:: current_page
      :value: 'preprocess'



   .. py:attribute:: current_worker
      :type:  textual.worker.Worker
      :value: None



   .. py:attribute:: runner


   .. py:attribute:: state


.. py:function:: asa_main()

   Entry point for canns-tui command.


