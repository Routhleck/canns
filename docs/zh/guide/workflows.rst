组合与自定义
==========

完成各示例后，你可以按照以下思路组合脚本，形成适合自身研究的问题驱动流程。

按需挑选模块
------------

- **模型评估 → 调参**：先运行 :doc:`models` 中的 1D/2D CANN 示例，确认参数范围；再切换到
  :doc:`trainer` 的 Hopfield 例子，对比 Hebb 学习前后的活动特性。
- **真实轨迹 → 流水线**：利用 :doc:`tasks` 的 `import_external_trajectory.py <https://github.com/Routhleck/canns/blob/master/examples/cann/import_external_trajectory.py>`_ 导入实验路径，保存
  为 ``.npz``；随后在 :doc:`pipeline` 中读取并执行 theta sweep 全流程。
- **实验验证**：将流水线产生的活动矩阵与 :doc:`analyzer` 中的 ROI/TDA 分析脚本对照，验证仿真与
  实验数据在拓扑结构上的一致性。

快速构建新示例
--------------

1. **复制模板**：从最接近需求的脚本拷贝到 ``examples/your_topic/``。
2. **修改配置**：
   - 调整 ``PlotConfigs``、``ThetaSweepPipeline`` 等配置字典即可获得新的图像输出。
   - 借助 ``uv run python`` 与 ``--help`` （若脚本支持 argparse）在命令行传参。
3. **记录输出**：建议在脚本末尾打印生成文件路径，便于自动化流水线上游消费。

常见排错
--------

- **动画生成缓慢**：
  - 降低 ``time_steps_per_second`` 或 ``fps``。
  - 在服务器环境中确保安装 ``imageio[ffmpeg]`` 以启用更快的渲染后端。
- **缺少依赖**：若看到 ``ModuleNotFoundError``，使用 ``uv add <package>`` 即时补齐；指南各章均注明
  额外依赖。
- **数据下载失败**：检查网络或预先在本地放置所需数据文件，``load_*`` 系列函数会优先使用本地缓存。

下一步
------

欢迎根据自身课题补充新的示例脚本，并在此指南中追加说明，保持“示例 → 文档”一一对应。
提交 PR 时可参考 ``docs/zh/examples/index.rst`` 的列表风格或本章的结构。
