2D Bump Fitting and Analysis
=============================

Scene Description
-----------------

Extract properties of bumps from 2D neural activity data (such as brain imaging or multi-electrode arrays), and perform quantitative analysis and visualization.

What You Will Learn
--------------------

- 2D Gaussian model fitting
- Parameters of elliptical bumps
- Estimation of orientation and eccentricity
- Heatmap generation and visualization

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from scipy.optimize import curve_fit
   import matplotlib.pyplot as plt

   def gaussian_bump_2d(coords, amplitude, center_x, center_y, sigma_x, sigma_y, angle):
       """2D Gaussian bump model"""
       x, y = coords
       cos_a = np.cos(angle)
       sin_a = np.sin(angle)

       x_rot = cos_a * (x - center_x) + sin_a * (y - center_y)
       y_rot = -sin_a * (x - center_x) + cos_a * (y - center_y)

       return amplitude * np.exp(-(x_rot**2/(2*sigma_x**2) + y_rot**2/(2*sigma_y**2)))

   # Simulate 2D data
   x = np.linspace(-5, 5, 50)
   y = np.linspace(-5, 5, 50)
   X, Y = np.meshgrid(x, y)
   coords = np.array([X.ravel(), Y.ravel()])

   true_params = (1.0, 0.5, 0.2, 0.8, 0.5, 0.3)  # amplitude, cx, cy, sx, sy, angle
   Z = gaussian_bump_2d(coords, *true_params).reshape(50, 50)
   Z_noisy = Z + 0.05 * np.random.randn(50, 50)

   # Fitting
   popt, _ = curve_fit(
       lambda c, a, cx, cy, sx, sy, ang: gaussian_bump_2d(c, a, cx, cy, sx, sy, ang),
       coords, Z_noisy.ravel(),
       p0=true_params
   )

   # Visualization
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   # Observed data
   axes[0].imshow(Z_noisy, cmap='viridis')
   axes[0].set_title('Observed Data')

   # Fitting result
   Z_fit = gaussian_bump_2d(coords, *popt).reshape(50, 50)
   axes[1].imshow(Z_fit, cmap='viridis')
   axes[1].set_title('Fitting Result')

   # Residuals
   residuals = Z_noisy - Z_fit
   axes[2].imshow(residuals, cmap='RdBu_r')
   axes[2].set_title('Residuals')

   plt.tight_layout()
   plt.savefig('bump_fitting_2d.png')

Key Concepts
------------

**Parameters of 2D Gaussian**

- A: Amplitude
- (μ_x, μ_y): Center
- (σ_x, σ_y): Standard deviation in X and Y directions
- θ: Orientation angle

**Elliptical Bump**

When σ_x ≠ σ_y, the bump is elliptical.

The direction of the major axis is specified by the angle θ.

Experimental Variations
-----------------------

**1. Bumps with Different Shapes**

.. code-block:: python

   # Circular: σ_x = σ_y
   # Elliptical: σ_x < σ_y
   # Elongated: σ_x << σ_y

**2. Multiple Bump Fitting**

.. code-block:: python

   # Gaussian mixture model
   # GMM for multiple bumps

Related API
-----------

- :func:`scipy.optimize.minimize`

Next Steps
----------

- :doc:`data_preprocessing` - Data preparation
- :doc:`../cann_dynamics/tuning_curves` - Comparison with tuning curves