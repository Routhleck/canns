示例手册导览
============

本指南按“脚本示例”划分章节，帮助你快速找到仓库中的参考实现。每个章节都会列出
对应的 ``examples/`` 路径、主要依赖模块、运行后的产物，以及可继续探索的延伸思路。

运行提示
--------

- 建议先执行 ``make install`` 构建依赖，再使用 ``uv run python <example.py>`` 运行脚本。
- 若示例会写入 GIF/PNG/NPZ 等文件，默认保存在示例目录或根目录；可根据脚本参数改写。
- 需要交互输出的示例（Matplotlib/SVG）在无图形界面环境下默认 ``show=False``，可手动开启。

示例分类
--------

.. toctree::
   :maxdepth: 2
   :caption: 示例章节

   architecture
   models
   trainer
   tasks
   analyzer
   pipeline
   workflows
