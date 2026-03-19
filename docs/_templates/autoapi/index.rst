API Reference
=============

This page contains auto-generated API reference documentation [#f1]_.

.. toctree::
   :titlesonly:

   {% for page in pages %}
   {% if page.include_path.endswith('/autoapi/[ANONYMOUS_PROJECT]/index') or page.include_path.endswith('/autoapi/src/[ANONYMOUS_PROJECT]/index') %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}


.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
