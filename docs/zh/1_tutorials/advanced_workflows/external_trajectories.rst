集成外部轨迹数据
================

.. warning::

   ⚠️ **重要提示**：本文档部分内容仍在开发和验证中，可能存在不完善之处。建议仅用于参考，重要项目前请与开发团队确认相关功能的完整性。



场景描述
--------

你有动物的真实运动轨迹（来自视频追踪、位置芯片等），想要用它来驱动神经网络模拟，研究网络如何编码真实行为中的动物位置。

你将学到
--------

- 轨迹数据的加载和处理
- 轨迹与模型的同步
- 实时和离线仿真
- 多模态数据融合
- 行为状态识别

完整示例
--------

.. code-block:: python

   import numpy as np
   from canns.task.open_loop_navigation import OpenLoopNavigationTask

   class ExternalTrajectoryTask(OpenLoopNavigationTask):
       def __init__(self, trajectory_file, dt=0.05):
           """从文件加载外部轨迹"""
           self.data = self.load_trajectory(trajectory_file)
           self.dt = dt

       def load_trajectory(self, file):
           """支持多种格式"""
           if file.endswith('.csv'):
               return np.loadtxt(file, delimiter=',', skiprows=1)
           elif file.endswith('.npz'):
               return np.load(file)['position']
           elif file.endswith('.mat'):
               import scipy.io as sio
               return sio.loadmat(file)['position']

       def get_input_at_time(self, t):
           """获取特定时刻的轨迹"""
           idx = int(t / self.dt)
           if idx < len(self.data):
               return self.data[idx]
           return self.data[-1]

   # 使用
   task = ExternalTrajectoryTask('animal_trajectory.csv')
   network = HierarchicalNetwork(num_module=5, num_place=30)

   for t in np.arange(0, task.duration, task.dt):
       position = task.get_input_at_time(t)
       velocity = np.gradient(task.data[int(t/task.dt)])

       network(velocity=velocity, loc=position)

关键概念
--------

**轨迹数据格式**

- CSV：时间,x,y 列
- NPZ：numpy压缩格式
- H5：HDF5格式
- MAT：MATLAB格式

**同步问题**

- 采样率匹配
- 时间对齐
- 坐标系转换

实验变化
--------

**1. 多个动物的轨迹**

.. code-block:: python

   for animal_id in range(num_animals):
       trajectory = load_animal_trajectory(animal_id)
       # 分别分析每个动物

**2. 多模态数据融合**

.. code-block:: python

   # 位置 + 脑成像
   # 位置 + 行为视频
   # 位置 + 神经记录

相关API
-------

- :class:`~src.canns.task.open_loop_navigation.ExternalTrajectoryTask`

下一步
------

- :doc:`parameter_customization` - 参数优化
- :doc:`../spatial_navigation/complex_environments` - 处理复杂环境
