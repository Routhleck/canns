2D CANN Tracking
================

Scene Description
-----------------

You want to understand how CANN processes stimuli in 2D space and observe how the "bump" activity pattern in the neural network moves across a 2D plane. This is a crucial step in extending from 1D to more complex spatial representations.

What You Will Learn
--------------------

- How to initialize a 2D CANN model (CANN2D)
- Definition of 2D smooth tracking tasks
- How to visualize neural activity in 2D space
- Generation and interpretation of energy landscape animations
- Characteristics of 2D space encoding

Complete Example
----------------

.. literalinclude:: ../../../../examples/cann/cann2d_tracking.py
   :language: python
   :linenos:

Step-by-Step Explanation
-------------------------

1. **Initializing a 2D CANN Model**

   .. code-block:: python

      import brainstate as bst
      from canns.models.basic import CANN2D

      bst.environ.set(dt=0.1)  # Time step 0.1ms

      # Create CANN2D model
      cann = CANN2D(length=100)  # 100x100 neural grid
      cann.init_state()

   **Explanation**:
   - ``length=100``: Creates 100×100 = 10,000 neurons
   - Represents 2D space [0, length] × [0, length]
   - Each neuron corresponds to a 2D coordinate (x, y)

2. **Defining a 2D Tracking Task**

   .. code-block:: python

      from canns.task.tracking import SmoothTracking2D

      task_st = SmoothTracking2D(
          cann_instance=cann,
          Iext=([0., 0.], [1., 1.], [0.75, 0.75], [2., 2.], [1.75, 1.75], [3., 3.]),
          duration=(10., 10., 10., 10., 10.),
          time_step=brainstate.environ.get_dt(),
      )
      task_st.get_data()

   **Explanation**:
   - Stimulus appears at 6 locations: (0,0) → (1,1) → (0.75,0.75) → (2,2) → (1.75,1.75) → (3,3)
   - Each location lasts 10 seconds (5 locations = 50 seconds total time)
   - Each location represents a 2D coordinate (x, y)

3. **Defining the Simulation Step Function**

   .. code-block:: python

      def run_step(t, Iext):
          """Single-step simulation function handling 2D stimulus"""
          with bst.environ.context(t=t):
              cann(Iext)  # Pass 2D stimulus
              return cann.u.value, cann.r.value, cann.inp.value

   **Explanation**:
   - ``Iext`` is a 2D stimulus (x, y) coordinate
   - Returns three quantities: membrane potential, firing rate, input current
   - Uses ``context`` to manage the time environment

4. **Running the Compilation Loop**

   .. code-block:: python

      import brainstate.compile

      cann_us, cann_rs, inps = brainstate.compile.for_loop(
          run_step,
          task_st.run_steps,      # Time step indices
          task_st.data,           # 2D stimulus sequence
          pbar=brainstate.compile.ProgressBar(10)
      )

   **Explanation**:
   - ``cann_us``: Membrane potential at all time steps [time, x, y]
   - ``cann_rs``: Firing rate at all time steps [time, x, y]
   - ``inps``: Input at all time steps [time, x, y]

5. **Creating Energy Landscape Animation**

   .. code-block:: python

      from canns.analyzer.plotting import PlotConfigs, energy_landscape_2d_animation

      # Configure animation parameters
      config = PlotConfigs.energy_landscape_2d_animation(
          time_steps_per_second=100,
          fps=20,
          title='CANN2D Encoding',
          xlabel='Space X',
          ylabel='Space Y',
          clabel='Neural Activity',
          repeat=True,
          save_path='cann2d_tracking.gif',
          show=False
      )

      # Generate animation
      energy_landscape_2d_animation(
          zs_data=cann_us,
          config=config
      )

   **Explanation**:
   - ``time_steps_per_second=100``: 100 simulation steps = 1 second
   - ``fps=20``: Animation playback frame rate
   - Generates a GIF file showing the 2D bump movement

Running Results
---------------

Running this script generates:

1. **2D Energy Landscape Animation** (`cann2d_tracking.gif`)

   - X and Y axes: Spatial positions
   - Z axis (color): Neural activity intensity
   - White/bright areas: High activity bump
   - Dark areas: Low activity background

2. **Expected Animation Characteristics**

   - Bump moves smoothly across the 2D plane
   - Bump shape approximates a 2D Gaussian distribution
   - Movement trajectory: (0,0) → (1,1) → (0.75,0.75) → (2,2) → (1.75,1.75) → (3,3)
   - Stable activity maintained at each location for 10 seconds

3. **Performance Metrics**

   - Simulation time: ~5-10 seconds (including compilation)
   - Memory usage: ~500 MB (10,000 neurons)
   - Total time steps: ~5000 steps
   - GIF file size: ~2-5 MB

Key Concepts
------------

**2D Bump Activity**

Characteristics of bumps formed in 2D CANN:

- Bump is a **2D Gaussian-shaped** local activity region
- Center corresponds to stimulus position
- Width is determined by the neural connection range
- Surrounded by strong inhibition (WSC pattern)

.. code-block:: text

   Top-down view (looking down at activity):

         High activity
           ↑
           |    ★★★
           |   ★★★★★
           |    ★★★★★
           |     ★★★
           |
   Low activity └─────────────→

   Side view (through bump center):

            Activity
            ↑
            │      ╱╲
            │     ╱  ╲
            │    ╱    ╲
            │___╱      ╲___
            └──────────────→ Position

**Extension of Topological Organization in 2D**

- 2D CANN maintains 2D topological mapping
- Adjacent neurons have similar responses to adjacent spatial positions
- Forms a continuous, navigable representation space
- Similar to orientation columns and position encoding in the brain

**Comparison with 1D**

==================  ==============  ==============
Feature             1D CANN1D       2D CANN2D
==================  ==============  ==============
Number of neurons   512             10,000 (100²)
Representation      One line        2D plane
Bump shape          1D Gaussian     2D Gaussian
Memory requirement  ~100 MB         ~500 MB
Computational       O(n²)           O(n⁴)
complexity
==================  ==============  ==============

Experimental Variations
-----------------------

**1. Changing Network Size**

.. code-block:: python

   # Finer representation
   cann = CANN2D(length=150)

   # Coarser representation
   cann = CANN2D(length=64)

**2. Changing Tracking Trajectory**

.. code-block:: python

   # Moving within square boundaries
   task_st = SmoothTracking2D(
       cann_instance=cann,
       Iext=([0., 0.], [5., 0.], [5., 5.], [0., 5.], [0., 0.]),
       duration=(10., 10., 10., 10.),
   )

   # Circular trajectory
   import numpy as np
   n_points = 8
   angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
   radius = 2.0
   center = 3.0
   points = [[center + radius*np.cos(a), center + radius*np.sin(a)] for a in angles]
   task_st = SmoothTracking2D(
       cann_instance=cann,
       Iext=points,
       duration=(10.,) * (n_points - 1),
   )

**3. Changing Neural Parameters**

.. code-block:: python

   cann = CANN2D(
       length=100,
       tau=0.1,       # Membrane time constant
       a=0.5,         # Local excitation range
       A=1.2,         # Excitation strength
       J0=0.5,        # Background input
   )

**4. Analyzing Bump Movement Speed**

.. code-block:: python

   import numpy as np

   # Calculate bump center at each time step
   bump_centers = []
   for t in range(len(cann_us)):
       u = cann_us[t]
       # Find position of maximum activity
       max_idx = np.unravel_index(np.argmax(u), u.shape)
       bump_centers.append(max_idx)

   # Calculate bump movement speed
   velocities = np.diff(bump_centers, axis=0)
   speed = np.linalg.norm(velocities, axis=1)

Related API
-----------

- :class:`~src.canns.models.basic.CANN2D` - 2D CANN model
- :class:`~src.canns.task.tracking.SmoothTracking2D` - 2D smooth tracking task
- :func:`~src.canns.analyzer.plotting.energy_landscape_2d_animation` - 2D energy landscape animation

More Resources
---------------

- :doc:`tracking_1d` - Understanding 1D fundamentals
- :doc:`tuning_curves` - Analyzing neural tuning curves
- :doc:`../spatial_navigation/index` - Spatial navigation using 2D CANN

Frequently Asked Questions
---------------------------

**Q: Why does the bump become deformed?**

A: Several possible reasons:
   - Network has not fully converged
   - Boundary effects (near 0 or length)
   - Parameters ``a`` and ``A`` need adjustment

**Q: Animation is slow or fails to generate?**

A:
   - Reduce ``length`` to lower computational complexity
   - Increase ``fps`` to reduce frame count
   - Run on GPU: ``JAX_PLATFORM_NAME=gpu``

**Q: How to analyze bump width?**

A: You can analyze along the cross-section through the bump center and fit a Gaussian curve:

   .. code-block:: python

      from scipy.optimize import curve_fit

      # Cross-section at bump center position
      center = np.array(bump_centers[-1])  # Position at last time step
      x_profile = cann_us[-1, int(center[0]), :]

      # Fit Gaussian
      def gaussian(x, amp, center, width):
          return amp * np.exp(-(x - center)**2 / (2*width**2))

Next Steps
----------

After completing this tutorial, we recommend:

1. Try the experimental variations above to observe encoding in 2D space
2. Read :doc:`tuning_curves` to learn how to analyze 2D tuning curves
3. Explore :doc:`../spatial_navigation/index` for navigation applications
4. Study :doc:`oscillatory_tracking` to learn about dynamic tracking behavior