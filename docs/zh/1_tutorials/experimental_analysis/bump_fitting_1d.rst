一维Bump拟合和分析
==================

场景描述
--------

从实验记录中，你看到神经活动形成"bump"形状的活动模式。你想要从原始数据中提取bump的性质（位置、宽度、幅度），进行定量分析。

你将学到
--------

- 高斯拟合方法
- Bump参数的鲁棒性估计
- 置信区间计算
- 残差分析和质量评估

完整示例
--------

.. code-block:: python

   import numpy as np
   from scipy.optimize import curve_fit
   from scipy.stats import linregress

   def gaussian_bump(x, amplitude, center, width):
       """高斯bump模型"""
       return amplitude * np.exp(-(x - center)**2 / (2*width**2))

   # 模拟数据
   x = np.linspace(-np.pi, np.pi, 100)
   true_params = (1.0, 0.5, 0.3)  # amplitude, center, width
   y = gaussian_bump(x, *true_params) + 0.05 * np.random.randn(100)

   # 拟合
   popt, pcov = curve_fit(gaussian_bump, x, y, p0=[1, 0, 0.3])

   # 参数估计
   amplitude, center, width = popt
   amplitude_err, center_err, width_err = np.sqrt(np.diag(pcov))

   print(f"幅度: {amplitude:.3f} ± {amplitude_err:.3f}")
   print(f"位置: {center:.3f} ± {center_err:.3f}")
   print(f"宽度: {width:.3f} ± {width_err:.3f}")

   # 质量评估
   y_fit = gaussian_bump(x, *popt)
   residuals = y - y_fit
   r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

   print(f"拟合质量（R²）: {r_squared:.4f}")

关键概念
--------

**高斯模型**

.. math::

   f(x) = A \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)

参数：
- A：幅度（bump的高度）
- μ：中心（bump的位置）
- σ：标准差（bump的宽度）

**FWHM（半最大全宽度）**

.. code-block:: python

   FWHM = 2.355 * sigma  # 约2.35倍标准差

实验变化
--------

**1. 不同噪声水平**

.. code-block:: python

   for noise_level in [0.01, 0.05, 0.1, 0.2]:
       y_noisy = y_true + noise_level * np.random.randn(len(y_true))
       popt, _ = curve_fit(gaussian_bump, x, y_noisy)
       # 评估参数误差

**2. 非高斯bump**

.. code-block:: python

   # 测试Lorentzian模型
   # Voigt模型
   # Lorenz + Gaussian混合

相关API
-------

- :func:`scipy.optimize.curve_fit`
- :class:`~src.canns.analyzer.spatial.compute_bump_stats`

下一步
------

- :doc:`bump_fitting_2d` - 二维情况
- :doc:`data_preprocessing` - 数据清理
