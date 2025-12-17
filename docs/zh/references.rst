==========
参考文献
==========

本页列出了 CANNs 文档中引用的所有参考文献。

.. note::
   要在文档或笔记本中引用这些参考文献，请使用 ``:cite:`` 角色。
   例如：``:cite:`wu2008dynamics``` 渲染为 [Wu08]。

完整参考文献列表
=====================

.. bibliography::
   :all:
   :style: unsrt

如何引用文献
======================

在 RST 文件中
------------

在文本中使用 ``:cite:`` 角色：

.. code-block:: rst

   连续吸引子的动力学由 :cite:`wu2008dynamics` 分析。
   基础性工作包括 :cite:`amari1977dynamics` 和 :cite:`wu2016continuous`。

在 Jupyter 笔记本中
--------------------

**重要提示**：在 Jupyter 笔记本中，您必须使用 **raw 单元格**，并设置为 reStructuredText 格式，而不是 markdown 单元格。

1. 创建一个 raw 单元格（Cell → Cell Type → Raw）
2. 设置单元格元数据以指示 RST 格式：

   .. code-block:: json

      {
        "raw_mimetype": "text/restructuredtext"
      }

3. 编写包含引用的 RST 内容：

   .. code-block:: rst

      这是一个包含引用的段落 :cite:p:`amari1977dynamics,wu2008dynamics`.

4. 在笔记本末尾添加参考文献列表指令（在另一个 raw RST 单元格中）：

   .. code-block:: rst

      参考文献
      --------

      .. bibliography::
         :cited:
         :style: alpha

**引用样式**：

- ``:cite:p:`key``` - 括号式引用：（作者，年份） - 整个引用都可点击
- ``:cite:t:`key``` - 文本式引用：作者 [年份] - 只有年份可点击

**示例**：查看 ``docs/en/0_why_canns.ipynb`` 获取完整的工作示例。

添加新参考文献
=====================

要向文献库添加新参考文献：

1. 打开 ``docs/refs/references.bib``
2. 按照现有格式添加您的 BibTeX 条目
3. 使用一致的引用键格式：``作者年份关键词``（例如 ``wu2008dynamics``）
4. 使用 ``:cite:`引用键``` 引用参考文献
5. 参考文献将自动出现在此文献列表中

更多信息请参阅 `sphinxcontrib-bibtex 文档 <https://sphinxcontrib-bibtex.readthedocs.io/>`_。
