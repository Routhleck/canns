Oscillatory Tracking and Dynamic Stability
===========================================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scenario Description
--------------------

You want to understand how a CANN network stably tracks rapidly changing stimuli, and observe how the oscillation patterns of network activity change with stimuli. This reveals the dynamic stability and adaptive properties of CANN.

What You Will Learn
-------------------

- Tracking mechanisms under rapidly changing stimuli
- Oscillation and decay patterns of network activity
- Real-time dynamics of energy landscapes
- Relationship between tracking stability and parameters
- Visualization of attractor dynamics

Complete Example
----------------

.. literalinclude:: ../../../../examples/cann/cann1d_oscillatory_tracking.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **Initialize CANN Model**

   .. code-block:: python

      import brainstate
      from canns.models.basic import CANN1D

      brainstate.environ.set(dt=0.1)

      cann = CANN1D(num=512)
      cann.init_state()

   **Explanation**:
   - 512 neurons arranged in one dimension
   - dt=0.1ms: time step
   - Sufficient neuron density to support smooth bump activity

2. **Define Rapid Tracking Task**

   .. code-block:: python

      from canns.task.tracking import SmoothTracking1D

      task_st = SmoothTracking1D(
          cann_instance=cann,
          Iext=(1., 0.75, 2., 1.75, 3.),     # 5 stimulus positions (radians)
          duration=(10., 10., 10., 10.),      # 4 time periods, 10 seconds each
          time_step=brainstate.environ.get_dt(),
      )
      task_st.get_data()

   **Explanation**:
   - Stimulus sequence at 5 positions: 1 → 0.75 → 2 → 1.75 → 3
   - Adjacent position spacing approximately 0.25 radians (~14 degrees)
   - 10-second duration is sufficient to observe complete tracking dynamics

3. **Define Simulation Steps**

   .. code-block:: python

      def run_step(t, inputs):
          """Single-step simulation: receive stimulus input"""
          cann(inputs)
          return cann.u.value, cann.inp.value

   **Explanation**:
   - ``cann.u.value``: membrane potential (network state)
   - ``cann.inp.value``: external input current
   - Return these two quantities for dynamic analysis

4. **Run Compiled Loop**

   .. code-block:: python

      import brainstate.compile

      us, inps = brainstate.compile.for_loop(
          run_step,
          task_st.run_steps,
          task_st.data,
          pbar=brainstate.compile.ProgressBar(10)
      )

   **Explanation**:
   - ``us``: membrane potential at all time steps [time, neurons]
   - ``inps``: input at all time steps [time, neurons]
   - Total data points: 40 seconds × 10 Hz = 400 time steps

5. **Generate Animation Visualization**

   .. code-block:: python

      from canns.analyzer.plotting import PlotConfigs, energy_landscape_1d_animation

      config = PlotConfigs.energy_landscape_1d_animation(
          time_steps_per_second=100,
          fps=20,
          title='Smooth Tracking 1D',
          xlabel='Spatial Position',
          ylabel='Neural Activity',
          repeat=True,
          save_path='oscillatory_tracking.gif',
          show=False
      )

      energy_landscape_1d_animation(
          data_sets={'u': (cann.x, us), 'Iext': (cann.x, inps)},
          config=config
      )

   **Explanation**:
   - ``data_sets`` contains two datasets: membrane potential and input
   - Animation shows two layers of data over time progression
   - ``cann.x``: spatial position coordinates of neurons

Running Results
---------------

Running this script generates:

1. **Energy Landscape Time Series Animation** (`oscillatory_tracking.gif`)

   - **Upper layer**: input current (Iext) changes over time
   - **Lower layer**: bump formed by network membrane potential (u)
   - Color changes indicate intensity changes

2. **Expected Dynamic Features**

   .. code-block:: text

      t=0-10s: Tracking at position 1
      │ Iext:   [peak at x=1]
      │ u:      [bump forms rapidly, stabilizes at x=1]
      │
      t=10-20s: Tracking at position 0.75
      │ Iext:   [peak at x=0.75]
      │ u:      [bump smoothly moves from x=1 to x=0.75, may have oscillations]
      │
      t=20-30s: Tracking at position 2
      │ Iext:   [peak at x=2]
      │ u:      [bump jumps from x=0.75 to x=2, shows maximum acceleration]
      │
      t=30-40s: Tracking at positions 1.75→3
      │ Iext:   [peak at x=1.75, then moves to x=3]
      │ u:      [bump continues to track smoothly]

3. **Performance Metrics**

   - Simulation time: ~2-3 seconds (including compilation)
   - Memory usage: ~200 MB
   - Generated GIF size: ~3-5 MB

Key Concepts
------------

**Oscillation in Tracking**

When stimuli move rapidly, the network produces oscillations:

.. code-block:: text

   Bump response to stimulus step:

   Stimulus position: _______●-------●________

   Bump position:     _______●      ╱╱╱╱╱╱___●
                                  ╱oscillation╱
                                 ╱____╱

   - **Rising phase**: Bump moves rapidly toward new position
   - **Overshoot**: Bump overshoots the stimulus position
   - **Oscillation phase**: Oscillates around new position
   - **Stabilization**: Finally stabilizes at stimulus position

**Attractor Dynamics**

The bump activity of CANN is a manifestation of **attractors**:

1. **Stable Attractor**: bump remains stationary
   - Maintains activity even after stimulus disappears
   - Implements "memory" function

2. **Driven Attractor**: pulled by stimulus
   - Bump follows stimulus movement
   - Used for "tracking"

3. **Competing Attractors**: multiple competing states (in larger networks)
   - Represent different choices
   - Through inhibitory competition

**Stability Conditions**

Tracking stability depends on:

.. code-block:: python

   Stability = f(
       stimulus_speed,           # faster is harder to track
       network_time_constant,    # larger tau is harder to track
       local_excitation_range,   # larger a is easier to track
       excitation_strength,      # larger A is easier to track
   )

Experimental Variations
-----------------------

**1. Change Stimulus Speed**

.. code-block:: python

   # Slow movement (easy to track)
   task_st = SmoothTracking1D(
       cann_instance=cann,
       Iext=(0., np.pi/2, np.pi),
       duration=(20., 20.),  # longer time
   )

   # Fast movement (prone to oscillations)
   task_st = SmoothTracking1D(
       cann_instance=cann,
       Iext=(0., np.pi/2, np.pi),
       duration=(2., 2.),  # shorter time
   )

**2. Analyze Tracking Delay**

.. code-block:: python

   import numpy as np

   # Find bump center and stimulus position
   bump_peaks = []
   for t in range(len(us)):
       u = us[t]
       peak_idx = np.argmax(u)
       bump_peaks.append(peak_idx)

   # Calculate tracking delay
   stimulus_centers = task_st.Iext_sequence.squeeze()
   delays = bump_peaks - stimulus_centers

   print(f"Average tracking delay: {np.mean(np.abs(delays)):.2f} neurons")

**3. Change Network Parameters to Optimize Tracking**

.. code-block:: python

   # Increase local excitation (easier to track)
   cann = CANN1D(num=512, a=0.6)

   # Reduce time constant (faster response)
   cann = CANN1D(num=512, tau=0.05)

**4. Visualize Attractor Landscape**

.. code-block:: python

   import matplotlib.pyplot as plt

   # Spontaneous activity without external input
   cann.reset_state()
   cann.u[:] = 0.1  # small initial value
   cann.u[256] = 1.0  # activate one neuron at center

   fig, axes = plt.subplots(1, 2, figsize=(12, 4))

   # Bump stability without input
   for t in range(1000):
       cann(np.zeros(512))
       if t % 100 == 0:
           axes[0].plot(cann.u.value)

   axes[0].set_title("Spontaneous Bump Activity (No Input)")
   axes[0].set_xlabel("Neuron Index")
   axes[0].set_ylabel("Membrane Potential")

   # Driven bump tracking
   cann.reset_state()
   stimulus = np.zeros(512)
   stimulus[300] = 1.0
   for t in range(1000):
       cann(stimulus)
       if t % 100 == 0:
           axes[1].plot(cann.u.value)

   axes[1].set_title("Driven Bump Activity (With Input)")
   axes[1].set_xlabel("Neuron Index")

   plt.tight_layout()
   plt.savefig('attractor_dynamics.png')

Related API
-----------

- :class:`~src.canns.models.basic.CANN1D` - One-dimensional CANN model
- :class:`~src.canns.task.tracking.SmoothTracking1D` - Smooth tracking task
- :func:`~src.canns.analyzer.plotting.energy_landscape_1d_animation` - Energy landscape animation

Biological Applications
-----------------------

**Head Direction Cells**

Head direction cells in the dorsomedial nucleus (MEC) encode an animal's head direction. The CANN model successfully explains:

- Tuning curves of cells (bell-shaped curves)
- How bump tracks during head movement
- Multimodal input fusion (vision + vestibular sense)

**Spatial Navigation**

When rodents navigate:

1. Grid Cells provide localization information
2. Place Cells encode specific locations
3. CANN mechanism smoothly updates representations when location changes

**Motor Control**

Motor cortex also shows similar CANN dynamics:

- Bump in motor planning represents planned movement direction
- Bump smoothly moves during movement execution
- Reflects motor trajectory unfolding over time

More Resources
--------------

- :doc:`tracking_1d` - Understanding basic tracking
- :doc:`tracking_2d` - Tracking in 2D space
- :doc:`../spatial_navigation/index` - CANN applications in navigation

Frequently Asked Questions
--------------------------

**Q: Why does the bump oscillate?**

A: When stimuli move, CANN needs time to rebalance. Oscillations are the process of the network finding a new attractor state. Oscillation amplitude depends on:
   - Speed of stimulus movement (faster means more severe)
   - Network time constant (slower means more prone to oscillation)
   - Inhibition strength (stronger means easier to stabilize)

**Q: How to reduce tracking delay?**

A: Several methods:
   - Increase excitation strength (A): drive bump faster
   - Reduce time constant (tau): neurons respond faster
   - Increase local excitation range (a): bump forms more easily
   - But avoid over-adjustment causing instability

**Q: Can the bump "drift"?**

A: Drift can occur under certain conditions:
   - Asymmetric neuron count
   - Boundary effects (near network boundaries)
   - Noise accumulation
   Can be verified by running the network without input and checking if bump shifts over time

Next Steps
----------

After completing this tutorial, we recommend:

1. Perform the experimental variations above
2. Change network parameters and observe stability changes
3. Analyze bump movement speed and acceleration