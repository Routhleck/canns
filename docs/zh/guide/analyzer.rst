实验数据分析示例
==============

当需要处理真实神经数据或生成高阶可视化时，可参考 ``examples/experimental_*`` 系列脚本。
这些示例展示了如何结合 :mod:`canns.analyzer.experimental_data <src.canns.analyzer.experimental_data>`、:class:`PlotConfig <src.canns.analyzer.plotting.config.PlotConfig>` 以及外部依赖
（UMAP、TDA、Numba 等）完成完整的分析流程。

``experimental_cann1d_analysis.py``
--------------------------------

- **路径**：`examples/experimental_cann1d_analysis.py <https://github.com/Routhleck/canns/blob/master/examples/experimental_cann1d_analysis.py>`_
- **数据来源**：通过 ``load_roi_data()`` 自动下载/读取 Hugging Face 上的 ROI 数据。
- **分析步骤**：
  1. 调用 ``bump_fits`` 执行 MCMC 拟合，估计每帧 bump 参数。
  2. 使用 ``CANN1DPlotConfig.for_bump_animation`` 定义动画标题、帧率、亮度限制。
  3. 调用 ``create_1d_bump_animation`` 生成 GIF，可控制帧数 (``nframes``) 与进度条显示。
- **输出**：``bump_analysis_demo.gif``，同时在终端打印拟合统计信息。
- **延伸**：
- 将 ``n_steps``、``n_roi`` 调整为实验数据对应值。
  - 若部署在没有 Numba 的环境，可留意脚本开头的提示（会自动退回纯 NumPy 实现）。

``experimental_cann2d_analysis.py``
--------------------------------

- **路径**：`examples/experimental_cann2d_analysis.py <https://github.com/Routhleck/canns/blob/master/examples/experimental_cann2d_analysis.py>`_
- **数据来源**：``load_grid_data()`` 下载 2D 网格细胞数据（含尖峰与位置）。
- **分析步骤**：
  1. 通过 ``SpikeEmbeddingConfig`` 控制平滑与速度过滤，调用 ``embed_spike_trains`` 获取嵌入矩阵。
  2. 使用 UMAP (``umap.UMAP``) 进行降维，并配合 ``plot_projection`` 输出 3D 投影图。
  3. ``tda_vis`` + ``TDAConfig`` 计算持久同调，得到环面结构的拓扑证据。
  4. ``decode_circular_coordinates`` 进行相位解码，最后 ``plot_3d_bump_on_torus`` 生成环面动画。
- **输出**：``experimental_cann2d_analysis_torus.gif`` 及若干图表。
- **延伸**：
- 修改 ``tda_config`` 中的 ``do_shuffle``、``num_shuffles`` 进行统计检验。
  - 使用 ``save_path`` 参数持久化投影图、拓扑条形图等。

工具与依赖
----------

- 如果脚本首次运行时提示缺少数据，将自动创建 ``~/.canns/data`` 并缓存下载结果。
- 需要额外库：UMAP（``umap-learn``）、Ripser（``canns_lib.ripser``，随 ``canns-lib`` 提供）、Numba、Matplotlib。
  若未安装，可在 `make install` 完成后追加：
  
  .. code-block:: bash

     uv add umap-learn canns-lib numba

- ``PlotConfig``/``CANN2DPlotConfig`` 均支持 ``show=False``，便于在服务器上离线渲染。
