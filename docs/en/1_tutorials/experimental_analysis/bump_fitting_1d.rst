1D Bump Fitting and Analysis
=============================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scenario Description
--------------------

From experimental records, you observe neural activity forming a "bump" shaped activity pattern. You want to extract the properties of the bump (location, width, amplitude) from raw data for quantitative analysis.

What You Will Learn
--------------------

- Gaussian fitting methods
- Robust estimation of bump parameters
- Confidence interval calculation
- Residual analysis and quality assessment

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from scipy.optimize import curve_fit
   from scipy.stats import linregress

   def gaussian_bump(x, amplitude, center, width):
       """Gaussian bump model"""
       return amplitude * np.exp(-(x - center)**2 / (2*width**2))

   # Simulated data
   x = np.linspace(-np.pi, np.pi, 100)
   true_params = (1.0, 0.5, 0.3)  # amplitude, center, width
   y = gaussian_bump(x, *true_params) + 0.05 * np.random.randn(100)

   # Fitting
   popt, pcov = curve_fit(gaussian_bump, x, y, p0=[1, 0, 0.3])

   # Parameter estimation
   amplitude, center, width = popt
   amplitude_err, center_err, width_err = np.sqrt(np.diag(pcov))

   print(f"Amplitude: {amplitude:.3f} ± {amplitude_err:.3f}")
   print(f"Location: {center:.3f} ± {center_err:.3f}")
   print(f"Width: {width:.3f} ± {width_err:.3f}")

   # Quality assessment
   y_fit = gaussian_bump(x, *popt)
   residuals = y - y_fit
   r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

   print(f"Fitting quality (R²): {r_squared:.4f}")

Key Concepts
------------

**Gaussian Model**

.. math::

   f(x) = A \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)

Parameters:
- A: Amplitude (height of the bump)
- μ: Center (location of the bump)
- σ: Standard deviation (width of the bump)

**FWHM (Full Width at Half Maximum)**

.. code-block:: python

   FWHM = 2.355 * sigma  # approximately 2.35 times the standard deviation

Experimental Variations
------------------------

**1. Different Noise Levels**

.. code-block:: python

   for noise_level in [0.01, 0.05, 0.1, 0.2]:
       y_noisy = y_true + noise_level * np.random.randn(len(y_true))
       popt, _ = curve_fit(gaussian_bump, x, y_noisy)
       # Evaluate parameter errors

**2. Non-Gaussian Bumps**

.. code-block:: python

   # Test Lorentzian model
   # Voigt model
   # Lorentz + Gaussian mixture

Related APIs
------------

- :func:`scipy.optimize.curve_fit`
- :class:`~src.canns.analyzer.spatial.compute_bump_stats`

Next Steps
----------

- :doc:`bump_fitting_2d` - 2D case
- :doc:`data_preprocessing` - Data cleaning
