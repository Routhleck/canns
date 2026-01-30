canns.pipeline.launcher
=======================

.. py:module:: canns.pipeline.launcher

.. autoapi-nested-parse::

   Launcher for selecting ASA or Gallery TUI.



Classes
-------

.. autoapisummary::

   canns.pipeline.launcher.ModePicker


Functions
---------

.. autoapisummary::

   canns.pipeline.launcher.main


Module Contents
---------------

.. py:class:: ModePicker(driver_class = None, css_path = None, watch_css = False, ansi_color = False)

   Bases: :py:obj:`textual.app.App`\ [\ :py:obj:`str | None`\ ]


   Select which TUI to run.

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


   .. py:method:: compose()

      Yield child widgets for a container.

      This method should be implemented in a subclass.



   .. py:method:: on_button_pressed(event)


   .. py:attribute:: BINDINGS

      The default key bindings.


   .. py:attribute:: CSS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
             Screen {
                 align: center middle;
             }
         
             #card {
                 width: 68;
                 height: 18;
                 border: thick $accent;
                 background: $surface;
                 padding: 2;
             }
         
             #title {
                 text-style: bold;
                 margin-bottom: 1;
             }
         
             #subtitle {
                 color: $text-muted;
                 margin-bottom: 2;
             }
         
             Button {
                 width: 100%;
                 margin: 1 0;
             }
             """

      .. raw:: html

         </details>



      Inline CSS, useful for quick scripts. This is loaded after CSS_PATH,
      and therefore takes priority in the event of a specificity clash.


   .. py:attribute:: TITLE
      :value: 'CANNs TUI Launcher'


      A class variable to set the *default* title for the application.

      To update the title while the app is running, you can set the [title][textual.app.App.title] attribute.
      See also [the `Screen.TITLE` attribute][textual.screen.Screen.TITLE].


.. py:function:: main()

   Entry point for the unified canns-tui launcher.


