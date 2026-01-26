src.canns.pipeline.asa.widgets
==============================

.. py:module:: src.canns.pipeline.asa.widgets

.. autoapi-nested-parse::

   Custom Textual widgets for ASA TUI.

   This module provides reusable UI components for the ASA analysis interface.



Classes
-------

.. autoapisummary::

   src.canns.pipeline.asa.widgets.ImagePreview
   src.canns.pipeline.asa.widgets.LogViewer
   src.canns.pipeline.asa.widgets.ParamGroup


Module Contents
---------------

.. py:class:: ImagePreview(image_path = None, **kwargs)

   Bases: :py:obj:`textual.containers.Vertical`


   Widget for previewing images in the terminal using climage.

   Initialize a Widget.

   :param \*children: Child widgets.
   :param name: The name of the widget.
   :param id: The ID of the widget in the DOM.
   :param classes: The CSS classes for the widget.
   :param disabled: Whether the widget is disabled or not.
   :param markup: Enable content markup?


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


   .. py:method:: on_resize(event)


   .. py:method:: update_image(path)

      Update the previewed image.



   .. py:attribute:: DEFAULT_CSS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
             ImagePreview {
                 height: auto;
                 min-height: 20;
                 border: solid $accent;
                 padding: 1;
             }
             #preview-content {
                 width: 100%;
                 height: auto;
             }
             #preview-scroll {
                 height: 1fr;
             }
             #preview-controls Button {
                 margin: 0 1 0 0;
             }
             #preview-arrows Button {
                 margin: 0 1 0 0;
             }
             #preview-controls, #preview-arrows {
                 height: auto;
             }
             """

      .. raw:: html

         </details>



      Default TCSS.


   .. py:attribute:: image_path
      :value: None



.. py:class:: LogViewer(**kwargs)

   Bases: :py:obj:`textual.containers.Vertical`


   Widget for displaying log messages.

   Initialize a Widget.

   :param \*children: Child widgets.
   :param name: The name of the widget.
   :param id: The ID of the widget in the DOM.
   :param classes: The CSS classes for the widget.
   :param disabled: Whether the widget is disabled or not.
   :param markup: Enable content markup?


   .. py:method:: add_log(message)

      Add a log message.



   .. py:method:: clear()

      Clear all log messages.



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



   .. py:attribute:: DEFAULT_CSS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
             LogViewer {
                 height: 10;
                 border: solid $primary;
                 padding: 1;
                 overflow-y: scroll;
             }
             """

      .. raw:: html

         </details>



      Default TCSS.


   .. py:attribute:: log_lines
      :value: []



.. py:class:: ParamGroup(title, **kwargs)

   Bases: :py:obj:`textual.containers.Vertical`


   Widget for grouping related parameters.

   Initialize a Widget.

   :param \*children: Child widgets.
   :param name: The name of the widget.
   :param id: The ID of the widget in the DOM.
   :param classes: The CSS classes for the widget.
   :param disabled: Whether the widget is disabled or not.
   :param markup: Enable content markup?


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



   .. py:attribute:: DEFAULT_CSS
      :value: Multiline-String

      .. raw:: html

         <details><summary>Show Value</summary>

      .. code-block:: python

         """
             ParamGroup {
                 border: round $secondary;
                 padding: 1;
                 margin: 1 0;
                 height: auto;
                 width: 100%;
             }
             """

      .. raw:: html

         </details>



      Default TCSS.


   .. py:attribute:: title


