1D CANN Tracking
================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scene Description
-----------------

You want to understand how a CANN (Continuous Attractor Neural Network) responds to input in one-dimensional space, and observe how the "bump" activity pattern in the neural network tracks a moving stimulus. This is the best starting point for understanding the basic principles of CANN.

What You Will Learn
-------------------

- How to initialize a 1D CANN model (CANN1D)
- How to define a smooth tracking task
- How to use JAX's compiled loops for efficient simulation
- How to calculate and visualize tuning curves
- How to analyze neuronal response properties

Complete Example
----------------

.. literalinclude:: ../../../../examples/cann/cann1d_tuning_curve.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **Environment Setup and Model Initialization**

   .. code-block:: python

      import brainstate
      import numpy as np
      from canns.models.basic import CANN1D

      # Set environment parameters
      brainstate.environ.set(dt=0.1)  # Time step 0.1ms

      # Create CANN1D model
      cann = CANN1D(num=512, z_min=-np.pi, z_max=np.pi)
      cann.init_state()

   **Explanation**:
   - ``num=512``: The network has 512 neurons
   - ``z_min=-np.pi, z_max=np.pi``: Represents angular space, from -π to π
   - ``init_state()``: Initializes all state variables (membrane potential, activity, etc.)

2. **Define Tracking Task**

   .. code-block:: python

      from canns.task.tracking import SmoothTracking1D

      task = SmoothTracking1D(
          cann_instance=cann,
          Iext=(0., 0., np.pi, 2*np.pi),  # Stimulus position sequence
          duration=(2., 20., 20.),          # Duration of each phase
          time_step=brainstate.environ.get_dt(),
      )
      task.get_data()

   **Explanation**:
   - Stimuli appear at 4 locations: 0, 0, π, 2π
   - The 1st location lasts 2 seconds (warmup)
   - The 2nd, 3rd, 4th locations each last 20 seconds
   - Total simulation time = 2 + 20 + 20 + 20 = 62 seconds

3. **Define Simulation Steps**

   .. code-block:: python

      def run_step(t, inputs):
          """Single step simulation function"""
          cann(inputs)  # Neural network forward pass
          return cann.r.value, cann.inp.value

   **Explanation**:
   - Input receives the stimulus at the current time
   - ``cann.r.value``: Neuronal firing rate (output)
   - ``cann.inp.value``: Input current

4. **Compile and Run Simulation**

   .. code-block:: python

      import brainstate.compile

      rs, inps = brainstate.compile.for_loop(
          run_step,
          task.run_steps,           # Time step indices
          task.data,                # Input data
          pbar=brainstate.compile.ProgressBar(10)  # Progress bar
      )

   **Explanation**:
   - ``for_loop`` uses JAX JIT compilation for acceleration
   - 2-5 times faster than ordinary Python loops
   - ``pbar`` shows a progress bar (updates every 10 steps)

5. **Calculate and Plot Tuning Curves**

   .. code-block:: python

      from canns.analyzer.plotting import PlotConfigs, tuning_curve

      # Select neurons to analyze
      neuron_indices = [128, 256, 384]

      # Create plot configuration
      config = PlotConfigs.tuning_curve(
          num_bins=50,
          pref_stim=cann.x,
          title='1D CANN Tuning Curve',
          xlabel='Stimulus Position (radians)',
          ylabel='Average Firing Rate',
          show=False,
          save_path='tuning_curves_1d.png',
          kwargs={'linewidth': 2, 'marker': 'o', 'markersize': 4}
      )

      # Plot tuning curves
      tuning_curve(
          stimulus=task.Iext_sequence.squeeze(),
          firing_rates=rs,
          neuron_indices=neuron_indices,
          config=config
      )

   **Explanation**:
   - Tuning curves show neuronal response to different stimuli
   - Each neuron should have the strongest response at some position (preferred stimulus)
   - Responses gradually decrease at other positions

Running Results
---------------

Running this script generates:

1. **Tuning Curve Plot** (`tuning_curves_1d.png`)

   - X-axis: Stimulus position (radians)
   - Y-axis: Average firing rate
   - Each curve represents one neuron's response
   - Expected: Each curve has a peak (Gaussian shape)

2. **Expected Output Features**

   - Neuron 128: Peak at ~0 radians
   - Neuron 256: Peak at ~π radians
   - Neuron 384: Peak at ~2π radians (= -π)

   This reflects the topographic organization of CANN: adjacent neurons encode adjacent spatial positions.

3. **Performance Metrics**

   - Simulation time: ~2-5 seconds (including compilation)
   - Memory usage: ~200 MB
   - Number of time steps: ~6200 steps

Key Concepts
------------

**Bump Activity**

The CANN is characterized by forming a "bump"-like local activity pattern:

- The most active neuronal population corresponds to the stimulus location
- The bump moves with the stimulus (tracking)
- The width of the bump is determined by the inhibition range

**Tuning Curve**

A neuron's tuning curve reflects its spatial selectivity:

- **Sharp tuning**: Narrow curve, strong response only to specific positions
- **Broad tuning**: Wide curve, response to multiple positions
- **Gaussian shape**: The most common tuning curve shape

**Topographic Organization**

A key property of CANN:

- Adjacent neurons have highly similar responses to adjacent stimuli
- This topographic organization is the basis for achieving continuous tracking
- Similar to cortical maps in the brain

Experimental Variations
-----------------------

Try these modifications to deepen your understanding:

**1. Change Network Size**

.. code-block:: python

   # Larger network (finer representation)
   cann = CANN1D(num=1024)

   # Smaller network (coarser representation)
   cann = CANN1D(num=256)

**2. Change Input Stimulus**

.. code-block:: python

   # Fast-moving stimulus
   task = SmoothTracking1D(
       cann_instance=cann,
       Iext=(0., 2*np.pi, 0.),
       duration=(5., 10., 5.),  # Fast movement during middle 10 seconds
   )

**3. Change Neuronal Parameters**

.. code-block:: python

   cann = CANN1D(
       num=512,
       tau=0.1,           # Faster time constant
       a=0.5,             # Change local excitation range
       A=1.2,             # Change excitation strength
       J0=0.5,            # Change background input
   )

**4. Analyze Different Neurons**

.. code-block:: python

   # Analyze more neurons
   neuron_indices = [64, 128, 192, 256, 320, 384, 448]

   # Or analyze all neurons
   neuron_indices = list(range(0, 512, 32))

Related API
-----------

- :class:`~src.canns.models.basic.CANN1D` - 1D CANN model
- :class:`~src.canns.task.tracking.SmoothTracking1D` - Smooth tracking task
- :func:`~src.canns.analyzer.plotting.tuning_curve` - Tuning curve plotting

More Resources
--------------

- :doc:`tracking_2d` - Learn the 2D CANN extension
- :doc:`../spatial_navigation/index` - Learn spatial navigation based on CANN

Frequently Asked Questions
--------------------------

**Q: Why is the tuning curve not a perfect Gaussian?**

A: This is normal. The actual tuning curve is affected by multiple factors:
   - Limited network size (discretization)
   - Boundary effects (at the circular boundary)
   - Noise and initial conditions

**Q: How to speed up the simulation?**

A:
   - Increase ``dt`` (but will reduce accuracy)
   - Reduce ``num`` (number of neurons)
   - Use GPU: Set environment variable ``JAX_PLATFORM_NAME=gpu``

**Q: Why are the tuning curves of some neurons very flat?**

A: Possible reasons:
   - The network has not fully converged (extend ``duration``)
   - The preferred stimulus of that neuron is not in the test range
   - Network parameters need adjustment

Next Steps
----------

After completing this tutorial, it is recommended to:

1. Try the experimental variations above and observe the results
2. Read :doc:`tracking_2d` to learn the 2D version
3. Explore the navigation applications in :doc:`../spatial_navigation/index`