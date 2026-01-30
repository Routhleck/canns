canns.pipeline.asa.screens
==========================

.. py:module:: canns.pipeline.asa.screens

.. autoapi-nested-parse::

   Modal screens for ASA TUI.

   This module provides modal overlays for directory selection, help, and error display.



Classes
-------

.. autoapisummary::

   canns.pipeline.asa.screens.ErrorScreen
   canns.pipeline.asa.screens.HelpScreen
   canns.pipeline.asa.screens.TerminalSizeWarning
   canns.pipeline.asa.screens.WorkdirScreen


Module Contents
---------------

.. py:class:: ErrorScreen(title, message, **kwargs)

   Bases: :py:obj:`textual.screen.ModalScreen`


   Modal screen for displaying errors.

   Initialize the screen.

   :param name: The name of the screen.
   :param id: The ID of the screen in the DOM.
   :param classes: The CSS classes for the screen.


   .. py:method:: compose()

      Called by Textual to create child widgets.

      This method is called when a widget is mounted or by setting `recompose=True` when
      calling [`refresh()`][textual.widget.Widget.refresh].

      Note that you don't typically need to explicitly call this method.

      .. rubric:: Example

      ```python
      def compose(self) -> ComposeResult:
          yield Header()
          yield Label("Press the button below:")
          yield Button()
          yield Footer()
      ```



   .. py:method:: on_button_pressed(event)


   .. py:attribute:: DEFAULT_CSS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
             ErrorScreen {
                 align: center middle;
             }
         
             ErrorScreen > Container {
                 width: 60;
                 height: 20;
                 border: thick $error;
                 background: $surface;
                 padding: 2;
             }
         
             #error-message {
                 color: $error;
                 overflow-y: scroll;
             }
             """

      .. raw:: html

         </details>



      Default TCSS.


   .. py:attribute:: error_message


   .. py:attribute:: error_title


.. py:class:: HelpScreen(name = None, id = None, classes = None)

   Bases: :py:obj:`textual.screen.ModalScreen`


   Modal screen showing help and key bindings.

   Initialize the screen.

   :param name: The name of the screen.
   :param id: The ID of the screen in the DOM.
   :param classes: The CSS classes for the screen.


   .. py:method:: compose()

      Called by Textual to create child widgets.

      This method is called when a widget is mounted or by setting `recompose=True` when
      calling [`refresh()`][textual.widget.Widget.refresh].

      Note that you don't typically need to explicitly call this method.

      .. rubric:: Example

      ```python
      def compose(self) -> ComposeResult:
          yield Header()
          yield Label("Press the button below:")
          yield Button()
          yield Footer()
      ```



   .. py:method:: on_button_pressed(event)


   .. py:method:: on_key(event)


   .. py:attribute:: DEFAULT_CSS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
             HelpScreen {
                 align: center middle;
             }
         
             HelpScreen > Container {
                 width: 70;
                 height: 25;
                 border: thick $primary;
                 background: $surface;
                 padding: 2;
             }
         
             #help-content {
                 overflow-y: scroll;
             }
             """

      .. raw:: html

         </details>



      Default TCSS.


   .. py:attribute:: HELP_TEXT
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
         ASA TUI - Terminal User Interface for ASA Analysis
         
         KEY BINDINGS:
           Ctrl-W    Change working directory
           Ctrl-R    Run analysis
           F5        Refresh previews
           ?         Show this help
           Esc       Quit application
           Tab       Navigate between panels
         
         ANALYSIS MODULES:
           TDA           Topological Data Analysis
           CohoMap       Cohomology Map (requires TDA)
           PathCompare   Trajectory Comparison (requires CohoMap)
           CohoSpace     Cohomology Space Visualization (requires CohoMap)
           FR            Firing Rate Heatmap
           FRM           Single Neuron Firing Rate Map
           GridScore     Grid Cell Analysis
         
         WORKFLOW:
           1. Select working directory (Ctrl-W)
           2. Choose input mode (ASA or Neuron+Traj)
           3. Load files
           4. Configure preprocessing
           5. Select analysis mode
           6. Set parameters
           7. Run analysis (Ctrl-R)
           8. View results
         
         TERMINAL REQUIREMENTS:
           Minimum size: 100 cols × 30 rows
           Recommended: 120 cols × 40 rows
           Tip: Use smaller font size for better display
         
           If display is incomplete, try:
           - Reduce terminal font size
           - Maximize terminal window
           - Use fullscreen mode
         
         Press any key to close...
             """

      .. raw:: html

         </details>




.. py:class:: TerminalSizeWarning(current_width, current_height, **kwargs)

   Bases: :py:obj:`textual.screen.ModalScreen`


   Warning screen for insufficient terminal size.

   Initialize the screen.

   :param name: The name of the screen.
   :param id: The ID of the screen in the DOM.
   :param classes: The CSS classes for the screen.


   .. py:method:: compose()

      Called by Textual to create child widgets.

      This method is called when a widget is mounted or by setting `recompose=True` when
      calling [`refresh()`][textual.widget.Widget.refresh].

      Note that you don't typically need to explicitly call this method.

      .. rubric:: Example

      ```python
      def compose(self) -> ComposeResult:
          yield Header()
          yield Label("Press the button below:")
          yield Button()
          yield Footer()
      ```



   .. py:method:: on_button_pressed(event)


   .. py:method:: on_key(event)


   .. py:attribute:: DEFAULT_CSS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
             TerminalSizeWarning {
                 align: center middle;
             }
         
             TerminalSizeWarning > Container {
                 width: 50;
                 height: 15;
                 border: thick $warning;
                 background: $surface;
                 padding: 2;
             }
         
             #warning-message {
                 color: $warning;
                 text-align: center;
             }
             """

      .. raw:: html

         </details>



      Default TCSS.


   .. py:attribute:: current_height


   .. py:attribute:: current_width


.. py:class:: WorkdirScreen(name = None, id = None, classes = None)

   Bases: :py:obj:`textual.screen.ModalScreen`\ [\ :py:obj:`pathlib.Path`\ ]


   Modal screen for selecting working directory.

   Initialize the screen.

   :param name: The name of the screen.
   :param id: The ID of the screen in the DOM.
   :param classes: The CSS classes for the screen.


   .. py:method:: compose()

      Called by Textual to create child widgets.

      This method is called when a widget is mounted or by setting `recompose=True` when
      calling [`refresh()`][textual.widget.Widget.refresh].

      Note that you don't typically need to explicitly call this method.

      .. rubric:: Example

      ```python
      def compose(self) -> ComposeResult:
          yield Header()
          yield Label("Press the button below:")
          yield Button()
          yield Footer()
      ```



   .. py:method:: on_button_pressed(event)


   .. py:attribute:: DEFAULT_CSS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
             WorkdirScreen {
                 align: center middle;
             }
         
             WorkdirScreen > Container {
                 width: 80;
                 height: 30;
                 border: thick $accent;
                 background: $surface;
             }
         
             #workdir-tree {
                 height: 1fr;
             }
             """

      .. raw:: html

         </details>



      Default TCSS.


