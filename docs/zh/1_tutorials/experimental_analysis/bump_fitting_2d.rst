二维Bump拟合和分析
==================

场景描述
--------

从二维神经活动数据（如脑成像或多电极阵列）中提取bump的性质，进行定量分析和可视化。

你将学到
--------

- 二维高斯模型拟合
- 椭圆bump的参数
- 方向和偏心率的估计
- 热图生成和可视化

完整示例
--------

.. code-block:: python

   import numpy as np
   from scipy.optimize import curve_fit
   import matplotlib.pyplot as plt

   def gaussian_bump_2d(coords, amplitude, center_x, center_y, sigma_x, sigma_y, angle):
       """二维高斯bump模型"""
       x, y = coords
       cos_a = np.cos(angle)
       sin_a = np.sin(angle)

       x_rot = cos_a * (x - center_x) + sin_a * (y - center_y)
       y_rot = -sin_a * (x - center_x) + cos_a * (y - center_y)

       return amplitude * np.exp(-(x_rot**2/(2*sigma_x**2) + y_rot**2/(2*sigma_y**2)))

   # 模拟2D数据
   x = np.linspace(-5, 5, 50)
   y = np.linspace(-5, 5, 50)
   X, Y = np.meshgrid(x, y)
   coords = np.array([X.ravel(), Y.ravel()])

   true_params = (1.0, 0.5, 0.2, 0.8, 0.5, 0.3)  # amplitude, cx, cy, sx, sy, angle
   Z = gaussian_bump_2d(coords, *true_params).reshape(50, 50)
   Z_noisy = Z + 0.05 * np.random.randn(50, 50)

   # 拟合
   popt, _ = curve_fit(
       lambda c, a, cx, cy, sx, sy, ang: gaussian_bump_2d(c, a, cx, cy, sx, sy, ang),
       coords, Z_noisy.ravel(),
       p0=true_params
   )

   # 可视化
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   # 原始数据
   axes[0].imshow(Z_noisy, cmap='viridis')
   axes[0].set_title('观察到的数据')

   # 拟合结果
   Z_fit = gaussian_bump_2d(coords, *popt).reshape(50, 50)
   axes[1].imshow(Z_fit, cmap='viridis')
   axes[1].set_title('拟合结果')

   # 残差
   residuals = Z_noisy - Z_fit
   axes[2].imshow(residuals, cmap='RdBu_r')
   axes[2].set_title('残差')

   plt.tight_layout()
   plt.savefig('bump_fitting_2d.png')

关键概念
--------

**二维高斯的参数**

- A：幅度
- (μ_x, μ_y)：中心
- (σ_x, σ_y)：X和Y方向的标准差
- θ：方向角

**椭圆形Bump**

当 σ_x ≠ σ_y 时，bump是椭圆形的。

长轴方向由角度θ指定。

实验变化
--------

**1. 不同形状的Bump**

.. code-block:: python

   # 圆形：σ_x = σ_y
   # 椭圆形：σ_x < σ_y
   # 细长：σ_x << σ_y

**2. 多个Bump拟合**

.. code-block:: python

   # 高斯混合模型
   # GMM for multiple bumps

相关API
-------

- :func:`scipy.optimize.minimize`

下一步
------

- :doc:`data_preprocessing` - 数据准备
- :doc:`../cann_dynamics/tuning_curves` - 与调谐曲线的对比
