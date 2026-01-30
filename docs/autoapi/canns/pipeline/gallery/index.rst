canns.pipeline.gallery
======================

.. py:module:: canns.pipeline.gallery

.. autoapi-nested-parse::

   Model gallery TUI.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/canns/pipeline/gallery/__main__/index
   /autoapi/canns/pipeline/gallery/app/index
   /autoapi/canns/pipeline/gallery/runner/index
   /autoapi/canns/pipeline/gallery/state/index


Classes
-------

.. autoapisummary::

   canns.pipeline.gallery.GalleryApp


Functions
---------

.. autoapisummary::

   canns.pipeline.gallery.main


Package Contents
----------------

.. py:class:: GalleryApp

   Bases: :py:obj:`textual.app.App`


   Main TUI application for the model gallery.

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


   .. py:method:: action_change_workdir()


   .. py:method:: action_refresh()


   .. py:method:: action_run()


   .. py:method:: check_terminal_size()


   .. py:method:: collect_params()


   .. py:method:: compose()

      Yield child widgets for a container.

      This method should be implemented in a subclass.



   .. py:method:: log_message(message)


   .. py:method:: on_button_pressed(event)


   .. py:method:: on_mount()


   .. py:method:: on_resize(event)


   .. py:method:: on_select_changed(event)


   .. py:method:: on_workdir_selected(path)


   .. py:method:: on_worker_state_changed(event)


   .. py:method:: set_run_status(message, status_class = None)


   .. py:method:: update_progress(percent)


   .. py:method:: update_workdir_label()


   .. py:attribute:: BINDINGS

      The default key bindings.


   .. py:attribute:: CSS_PATH
      :value: 'styles.tcss'


      File paths to load CSS from.


   .. py:attribute:: MIN_HEIGHT
      :value: 28



   .. py:attribute:: MIN_WIDTH
      :value: 100



   .. py:attribute:: RECOMMENDED_HEIGHT
      :value: 36



   .. py:attribute:: RECOMMENDED_WIDTH
      :value: 120



   .. py:attribute:: TITLE
      :value: 'CANNs Model Gallery'


      A class variable to set the *default* title for the application.

      To update the title while the app is running, you can set the [title][textual.app.App.title] attribute.
      See also [the `Screen.TITLE` attribute][textual.screen.Screen.TITLE].


   .. py:attribute:: current_worker
      :type:  textual.worker.Worker | None
      :value: None



   .. py:attribute:: runner


   .. py:attribute:: state


.. py:function:: main()

   Entry point for the model gallery TUI.


