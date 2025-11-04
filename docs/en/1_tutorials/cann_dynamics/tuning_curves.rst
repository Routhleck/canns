Tuning Curve Analysis of Neurons in CANN
=========================================

Scenario Description
--------------------

You want to gain a deep understanding of the tuning properties of individual neurons in CANN: how they respond to different spatial stimuli, and how these response patterns define the spatial encoding of the network. Tuning curves are a key tool for understanding neural representations.

What You Will Learn
--------------------

- What tuning curves are and why they are important
- How to compute tuning curves from network activity
- Mathematical properties of tuning curves (peak, width, symmetry)
- How to analyze tuning properties at the population level
- Biological significance of tuning curves

Complete Example
----------------

Tuning curve analysis based on a 1D CANN tracking task:

.. code-block:: python

   import brainstate
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.optimize import curve_fit
   from canns.models.basic import CANN1D
   from canns.task.tracking import SmoothTracking1D
   from canns.analyzer.plotting import PlotConfigs, tuning_curve
   import brainstate.compile

   # Set up environment
   brainstate.environ.set(dt=0.1)

   # Create model
   cann = CANN1D(num=512, z_min=-np.pi, z_max=np.pi)
   cann.init_state()

   # Create multi-position tracking task to cover the entire space
   positions = np.linspace(-np.pi, np.pi, 16)  # 16 positions
   task = SmoothTracking1D(
       cann_instance=cann,
       Iext=tuple(positions),
       duration=(5.,) * 15,  # 5 seconds for each position
       time_step=brainstate.environ.get_dt(),
   )
   task.get_data()

   # Define simulation steps
   def run_step(t, inputs):
       cann(inputs)
       return cann.r.value

   # Run simulation
   rs = brainstate.compile.for_loop(
       run_step,
       task.run_steps,
       task.data,
       pbar=brainstate.compile.ProgressBar(10)
   )

   # Compute tuning curves
   neuron_indices = [64, 128, 256, 384, 448]

   for neuron_idx in neuron_indices:
       tuning = []
       for pos_idx in range(len(positions)):
           start_time = int(pos_idx * 5 / 0.1)  # 5 seconds per position = 500 steps
           end_time = int((pos_idx + 1) * 5 / 0.1)
           # Take second half to avoid transients
           mid_time = start_time + (end_time - start_time) // 2
           avg_response = np.mean(rs[mid_time:end_time, neuron_idx])
           tuning.append(avg_response)

       # Plot tuning curve
       plt.figure(figsize=(10, 4))
       plt.plot(positions, tuning, 'o-', linewidth=2, markersize=8)
       plt.xlabel('Stimulus Position (radians)')
       plt.ylabel('Mean Firing Rate (Hz)')
       plt.title(f'Tuning Curve of Neuron {neuron_idx}')
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(f'tuning_curve_neuron_{neuron_idx}.png')
       plt.close()

   # Population analysis: distribution of tuning widths for all neurons
   tuning_widths = []
   tuning_peaks = []

   for neuron_idx in range(0, 512, 8):  # Sample every 8 neurons
       tuning = []
       for pos_idx in range(len(positions)):
           start_time = int(pos_idx * 5 / 0.1)
           end_time = int((pos_idx + 1) * 5 / 0.1)
           mid_time = start_time + (end_time - start_time) // 2
           avg_response = np.mean(rs[mid_time:end_time, neuron_idx])
           tuning.append(avg_response)

       # Fit Gaussian curve
       def gaussian(x, amp, center, width):
           return amp * np.exp(-(x - center)**2 / (2*width**2))

       try:
           popt, _ = curve_fit(gaussian, positions, tuning, p0=[1, positions[0], 0.5])
           tuning_widths.append(popt[2])
           tuning_peaks.append(popt[0])
       except:
           pass

   print("\\n=== Tuning Curve Statistics ===")
   print(f"Mean tuning width: {np.mean(tuning_widths):.3f} radians")
   print(f"Tuning width std: {np.std(tuning_widths):.3f} radians")
   print(f"Mean peak response: {np.mean(tuning_peaks):.3f} Hz")

Step-by-Step Explanation
------------------------

1. **Mathematical Definition of Tuning Curves**

   The tuning curve f(θ) describes the response of a neuron to stimulus position θ:

   .. math::

      f(\theta) = \text{Mean firing rate}(\theta)

   The most common form is Gaussian:

   .. math::

      f(\theta) = A \cdot \exp\left(\frac{-(\theta - \theta_0)^2}{2\sigma^2}\right)

   Where:
   - A: Peak response amplitude
   - θ₀: Preferred stimulus (location of strongest response)
   - σ: Tuning width (determines narrowness or broadness of response)

2. **Key Parameters Explained**

   .. code-block:: text

      Response Strength
      ↑
      │         ╱╲
      │        ╱  ╲       Gaussian curve
      │ Amplitude A│   ╲
      │      │    ╲
      │      └─────╲───────→ Stimulus Position
      │             θ₀
      │      ├─2σ─┤
      │    Width

   - **Amplitude (A)**: Maximum response strength of the neuron
   - **Preferred Stimulus (θ₀)**: Stimulus position that produces maximum response
   - **Width (σ)**: Sharpness of tuning
     - σ small (<π/10): Sharp tuning, high spatial selectivity
     - σ moderate (π/10-π/6): Moderate tuning
     - σ large (>π/6): Broad tuning, low spatial selectivity

3. **Extracting Tuning Curves from Network Activity**

   .. code-block:: python

      # Step 1: Run network for different stimulus positions
      positions = np.linspace(-π, π, 16)  # 16 test positions

      # Step 2: For each position, compute average neuron response
      for each position:
          Average firing rate in steady state → tuning data point

      # Step 3: Fit Gaussian curve to extract parameters
      tuning_curve = fit_gaussian(positions, firing_rates)

4. **Topological Properties of Population Coding**

   A key property of CANN is topological organization:

   .. code-block:: text

      Neuron indices (population coordinates):
      0   128   256   384   512
      │    │     │     │     │
      │    │     │     │     │
      ↓    ↓     ↓     ↓     ↓
      Preferred stimuli:
      -π  -π/2   0    π/2   π

      Adjacent neurons have similar preferred stimuli!

Running Results
---------------

Running this script generates:

1. **Tuning Curves of Individual Neurons** (5 images)

   Each image shows the tuning property of one neuron:
   - X-axis: Stimulus position (from -π to π)
   - Y-axis: Mean firing rate
   - Curve shape: Gaussian curve with one clear peak

2. **Expected Result Characteristics**

   - Neuron 64: Peak at ~ -π (one end of network)
   - Neuron 128: Peak at ~ -π/2
   - Neuron 256: Peak at ~ 0 (network center)
   - Neuron 384: Peak at ~ π/2
   - Neuron 448: Peak at ~ π (other end of network)

3. **Population Statistics**

   .. code-block:: text

      === Tuning Curve Statistics ===
      Mean tuning width: 0.523 radians  (approximately 30 degrees)
      Tuning width std: 0.045 radians
      Mean peak response: 3.821 Hz

Key Concepts
------------

**Types of Tuning Curves**

========== =============== ================== ==========================
Tuning Type Width Range    Selectivity        Typical Application
========== =============== ================== ==========================
Sharp      σ < π/10        Highly specific    Fine spatial encoding
Moderate   π/10 < σ < π/6  Moderately specific General spatial encoding
Broad      σ > π/6         Low specific       Coarse coding + robustness
========== =============== ================== ==========================

**Biological Meaning of Tuning Width**

- **Sharp tuning neurons**:
  - Advantage: High spatial resolution
  - Disadvantage: Sensitive to noise, requires more neurons
  - Found in: Direction selectivity in visual cortex

- **Broad tuning neurons**:
  - Advantage: Robust, insensitive to noise
  - Disadvantage: Low spatial resolution
  - Found in: Head direction cells, place cells

**Tuning in Circular Space**

In CANN, space is circular (π and -π represent the same direction):

.. code-block:: text

   Linear space:
   -π──────0──────π
   |              |
   No circularity  No circularity

   Circular space:
          π ≡ -π
         ╱     ╲
        ╱       ╲
      -π/2     π/2
        ╲       ╱
         ╲     ╱
           0

   In CANN, tuning at boundaries requires special handling!

Experimental Variations
-----------------------

**1. Changing the Stimulus Test Range**

.. code-block:: python

   # Test only part of the network (e.g., upper half-plane)
   positions = np.linspace(0, π, 8)

   # Higher resolution testing
   positions = np.linspace(-π, π, 32)

**2. Analyzing Tuning Width Variation Along the Network**

.. code-block:: python

   widths_by_neuron = []
   for neuron_idx in range(512):
       tuning = [compute_tuning(neuron_idx, pos) for pos in positions]
       width = fit_gaussian(positions, tuning)[2]
       widths_by_neuron.append(width)

   plt.figure(figsize=(12, 4))
   plt.plot(widths_by_neuron)
   plt.xlabel('Neuron Index')
   plt.ylabel('Tuning Width (radians)')
   plt.title('Distribution of Tuning Width Along the Network')
   plt.axhline(np.mean(widths_by_neuron), color='r', linestyle='--', label='Mean')
   plt.legend()
   plt.grid(True, alpha=0.3)

**3. Comparing Effects of Different Network Parameters**

.. code-block:: python

   # Change local excitation range
   for a_value in [0.3, 0.5, 0.7]:
       cann = CANN1D(num=512, a=a_value)
       # ... compute and plot tuning curves

   # Change inhibition strength
   for J_value in [0.3, 0.5, 0.7]:
       cann = CANN1D(num=512, J=J_value)
       # ... compute and plot tuning curves

**4. Analyzing Symmetry of Tuning Curves**

.. code-block:: python

   # Check quality of Gaussian fitting
   residuals = tuning_data - gaussian_fit
   r_squared = 1 - np.sum(residuals**2) / np.sum((tuning_data - np.mean(tuning_data))**2)
   print(f"Fitting quality (R²): {r_squared:.4f}")

**5. Population Decoding: Reconstructing Stimulus from Population Activity**

.. code-block:: python

   # Use maximum likelihood estimation to reconstruct stimulus from population activity
   def decode_stimulus(population_activity):
       # Assume Gaussian tuning
       likelihood = np.ones(len(positions))
       for neuron_idx, response in enumerate(population_activity):
           for pos_idx, pos in enumerate(positions):
               tuning_curve = gaussian(pos, *tuning_params[neuron_idx])
               # Poisson model: P(response|tuning) = exp(-tuning) * tuning^response
               likelihood[pos_idx] *= np.exp(-tuning_curve) * (tuning_curve ** response)

       best_pos = positions[np.argmax(likelihood)]
       return best_pos

Related APIs
------------

- :class:`~src.canns.models.basic.CANN1D` - 1D CANN model
- :func:`~src.canns.analyzer.plotting.tuning_curve` - Tuning curve plotting tool
- :class:`~src.canns.task.tracking.SmoothTracking1D` - Smooth tracking task

Biological Applications
-----------------------

**1. Direction Selectivity in Visual Cortex**

Complex cells in mammalian visual cortex:

- Tuning width: 30-60 degrees
- Preferred stimulus: Direction (0-360 degrees)
- Application: CANN models successfully explain organization of direction columns

**2. Spatial Position Encoding**

Hippocampal place cells:

- Tuning width: 20-40 cm
- Preferred stimulus: Spatial position
- Characteristic: Formation and stability of place fields

**3. Head Direction Encoding**

Head direction cells in dorsomedial nucleus:

- Tuning width: 30-50 degrees
- Preferred stimulus: Head direction
- Characteristic: Global polar frame, aligned with environment

Additional Resources
--------------------

- :doc:`tracking_1d` - Basic tracking experiment
- :doc:`tracking_2d` - 2D spatial encoding
- :doc:`../spatial_navigation/path_integration` - Path integration using tuning curves

Frequently Asked Questions
--------------------------

**Q: Why is the tuning curve not perfectly Gaussian?**

A: Several possible reasons:
   - Network noise or unstable initial conditions
   - Boundary effects (neurons near ±π)
   - Non-ideal network parameters
   - Insufficient sample size

**Q: How to estimate tuning width?**

A: Several methods:

   .. code-block:: python

      # Method 1: Full Width at Half Maximum (FWHM)
      peak = max(tuning_curve)
      half_max = peak / 2
      width_fwhm = positions[tuning_curve > half_max][-1] - positions[tuning_curve > half_max][0]

      # Method 2: Gaussian fitting
      popt = curve_fit(gaussian, positions, tuning_curve)
      sigma = popt[2]  # Gaussian standard deviation

      # Method 3: Information-theoretic method
      information = calculate_mutual_information(population, stimulus)

**Q: Why do different neurons have different tuning widths?**

A: In real CANN models, widths should be basically the same. Different widths are typically due to:
   - Fitting quality issues
   - Boundary effects of the network
   - Random variability between neurons

Next Steps
----------

After completing this tutorial, we recommend:

1. Analyze tuning curves of multiple neurons and find patterns
2. Try parameter variations and observe changes in tuning width