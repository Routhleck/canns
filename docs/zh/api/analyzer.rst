分析工具
========

分析模块提供了用于分析神经网络数据和可视化结果的工具。

实用工具
--------

.. automodule:: canns.analyzer.utils
   :members:
   :undoc-members:
   :show-inheritance:

可视化工具
--------

.. automodule:: canns.analyzer.visualize
   :members:
   :undoc-members:
   :show-inheritance:

主要功能
--------

数据处理:
- :func:`canns.analyzer.utils.spike_train_to_firing_rate` - 将spike train转换为发放率
- :func:`canns.analyzer.utils.firing_rate_to_spike_train` - 将发放率转换为spike train
- :func:`canns.analyzer.utils.normalize_firing_rates` - 归一化发放率数据

可视化:
- :func:`canns.analyzer.visualize.raster_plot` - 绘制raster图
- :func:`canns.analyzer.visualize.tuning_curve` - 绘制调谐曲线
- :func:`canns.analyzer.visualize.energy_landscape_1d_animation` - 1D能量景观动画
- :func:`canns.analyzer.visualize.energy_landscape_2d_animation` - 2D能量景观动画