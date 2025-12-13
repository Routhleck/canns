========
参考文献
========

本页列出了 CANNs 文档中引用的所有参考文献。

.. note::
   要在文档或笔记本中引用这些参考文献,请使用 ``:cite:`` 角色。
   例如:``:cite:`wu2008dynamics``` 渲染为 [Wu08]。

完整文献列表
============

.. bibliography::
   :style: alpha
   :all:
   :list: enumerated

如何引用文献
============

在 RST 文件中
-------------

在文本中使用 ``:cite:`` 角色:

.. code-block:: rst

   连续吸引子的动力学由 :cite:`wu2008dynamics` 分析。
   基础性工作包括 :cite:`amari1977dynamics` 和 :cite:`wu2016continuous`。

在 Jupyter 笔记本中
-------------------

在 Markdown 单元格中使用相同的 ``:cite:`` 角色:

.. code-block:: markdown

   连续吸引子的动力学由 :cite:`wu2008dynamics` 分析。
   网格细胞由 :cite:`hafting2005microstructure` 发现。

添加新参考文献
==============

要向文献库添加新参考文献:

1. 打开 ``docs/refs/references.bib``
2. 按照现有格式添加您的 BibTeX 条目
3. 使用一致的引用键格式:``作者年份关键词``(例如 ``wu2008dynamics``)
4. 使用 ``:cite:`引用键``` 引用参考文献
5. 参考文献将自动出现在此文献列表中

更多信息请参阅 `sphinxcontrib-bibtex 文档 <https://sphinxcontrib-bibtex.readthedocs.io/>`_。
