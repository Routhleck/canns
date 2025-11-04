Theta Rhythm and Temporal Encoding
==================================

Scene Description
-----------------

You want to understand how Theta rhythm (~8Hz) in the brain interacts with spatial encoding, and how the "Theta Phase Precession" mechanism enables high-precision temporal and spatial encoding.

What You Will Learn
-------------------

- Properties and sources of Theta rhythm
- Mechanism of Theta Phase Precession
- Combination of temporal codes and location codes
- Role of Theta rhythm in path integration
- Oscillation and CANN interaction

Complete Example
----------------

Based on a navigation task with Theta rhythm:

.. code-block:: python

   import numpy as np
   import brainstate
   from canns.models.basic import CANN1D
   from canns.task.tracking import SmoothTracking1D

   # Set up Theta rhythm (~8 Hz)
   theta_frequency = 8.0  # Hz
   theta_period = 1.0 / theta_frequency  # seconds

   # Create Theta-modulated input
   def create_theta_modulated_input(position, time, theta_freq=8.0):
       """Create position input modulated by Theta rhythm"""
       theta_phase = np.sin(2 * np.pi * theta_freq * time)
       # Input intensity is modulated by Theta oscillation
       return position * (1.0 + 0.5 * theta_phase)

   # Run CANN tracking with Theta modulation
   cann = CANN1D(num=512)
   cann.init_state()

   firing_rates = []
   phases = []

   for t in np.linspace(0, 10, 1000):  # 10 seconds
       position = np.sin(t)  # sinusoidally varying position
       theta_input = create_theta_modulated_input(position, t)

       # Create input stimulus
       stimulus = np.zeros(512)
       stimulus[int(256 + 256 * position)] = theta_input

       cann(stimulus)
       firing_rates.append(cann.r.value.copy())
       phases.append(np.angle(np.exp(1j * 2 * np.pi * 8 * t)))

   firing_rates = np.array(firing_rates)
   phases = np.array(phases)

Step-by-Step Analysis
---------------------

1. **Theta Rhythm Basics**

   .. code-block:: python

      # Theta rhythm is a periodic oscillation
      theta_frequency = 8.0  # Hz (within 5-12Hz range)
      theta_period = 1.0 / theta_frequency  # ~125 ms

      # Standard Theta signal
      theta_signal = np.sin(2 * np.pi * theta_frequency * t)

   **Characteristics**:
   - Frequency: 5-12 Hz (depends on behavior and species)
   - Amplitude: 0.5-2 mV
   - Source: Medial septum
   - Function: Coordinate activity across multiple brain regions

2. **Phase Precession Mechanism**

   .. code-block:: python

      # Phase Precession: As position cells pass through their place field,
      # their firing progressively leads the phase of Theta rhythm

      # When the animal enters the place field:
      Position = 0.5, Theta phase = 0° → cell starts firing
      Position = 0.6, Theta phase = 30° → stronger firing
      Position = 0.7, Theta phase = 60° → stronger
      Position = 0.8, Theta phase = 90° → maximum
      Position = 0.9, Theta phase = 120° → starts to weaken
      Position = 1.0, Theta phase = 180° → leaves place field

   **Results**:
   - Firing time leads relative to position
   - Multiple positions can be encoded within one cycle
   - Time × velocity = position information

3. **Implementing Phase Precession in CANN**

   .. code-block:: python

      def analyze_phase_precession(firing_rates, phases, position_trajectory):
          """Analyze Phase Precession"""
          precession_data = []

          for neuron_idx in range(firing_rates.shape[1]):
              neuron_activity = firing_rates[:, neuron_idx]

              # Find peak activity of this neuron
              peak_times = np.where(neuron_activity > np.max(neuron_activity) * 0.5)[0]

              if len(peak_times) > 1:
                  # Calculate relationship between firing time and Theta phase
                  for peak_time in peak_times:
                      precession_data.append({
                          'neuron': neuron_idx,
                          'position': position_trajectory[peak_time],
                          'theta_phase': phases[peak_time],
                          'firing_rate': neuron_activity[peak_time]
                      })

          return precession_data

   **Explanation**:
   - Plot: x-axis = position, y-axis = phase
   - You should see a monotonically decreasing relationship (phase advances with position)

Key Concepts
------------

**Combination of Theta and Spatial Encoding**

.. code-block:: text

   Benefits provided by Theta rhythm:

   1. Time window segmentation
      ├─ Theta cycle 1: time 0-125ms
      ├─ Theta cycle 2: time 125-250ms
      └─ Different information encoded within each cycle

   2. Computational advantages of Phase Precession
      Position = v · t + θ · τ_phase
      where τ_phase is phase offset, using ~30Hz "beta oscillation"

   3. Sequence generation
      - When rats run, place cell sequences synchronize with Theta cycles
      - Can be used for sequence learning and replay

**Theta Skipping and Theta Cycling**

Two different encoding mechanisms:

.. code-block:: text

   Theta Skipping:
   - One place field per Theta cycle
   - Occurs during low-speed movement
   - Temporal resolution: number of positions = Theta frequency / velocity

   Theta Cycling:
   - Multiple Theta cycles per place field
   - Occurs during high-speed movement
   - Alternately encodes movement trajectory and other information

Experimental Variations
-----------------------

**1. Changing Theta Frequency**

.. code-block:: python

   for theta_freq in [4, 6, 8, 10, 12]:  # 4-12 Hz range
       theta_input = create_theta_modulated_input(position, t, theta_freq)
       # Observe how Phase Precession changes

**2. Analyzing Effects of Theta Modulation Strength**

.. code-block:: python

   for modulation_strength in [0.1, 0.3, 0.5, 0.7, 0.9]:
       theta_input = position * (1.0 + modulation_strength * theta_phase)

**3. Measuring Temporal Encoding Ability**

.. code-block:: python

   # How many different positions can be encoded within two Theta cycles?
   def measure_temporal_resolution(firing_rates, phases):
       """Measure temporal resolution"""
       # Cluster firing events by Theta phase within one cycle
       clusters = cluster_by_theta_phase(phases)
       positions_per_cycle = len(clusters)
       return positions_per_cycle

Related APIs
------------

- :class:`~src.canns.models.basic.CANN1D` - CANN1D supporting Theta modulation
- :class:`~src.canns.task.tracking.SmoothTracking1D` - Can add Theta rhythm

Biological Applications
-----------------------

**Evidence from Hippocampal Recording**

- O'Keefe and Recce (1993) discovered Phase Precession in place cells
- Within each Theta cycle, different place cells fire in sequence
- Effect is like "compressed replay" of future path

**Theta and Learning**

- Synaptic plasticity occurs within Theta oscillation periods
- STDP is correlated with Theta phase
- Supports efficient reinforcement learning

More Resources
--------------

- :doc:`path_integration` - Understanding basic path integration
- :doc:`grid_place_cells` - Integration of grid and place cells

FAQ
---

**Q: What are the computational advantages of Phase Precession?**

A: It allows encoding multiple positions within one Theta cycle. If Theta frequency is 8Hz, velocity is 1m/s, and place field width is 40cm, then one Theta cycle can encode 8 different positions!

**Q: Why is Theta rhythm necessary?**

A:
- Coordinate different brain regions
- Select time windows for learning
- Support prospective encoding
- Possibly related to consciousness

**Q: How is Phase Precession implemented in CANN?**

A: By modulating input intensity at Theta frequency. Input is strong at Theta peaks and weak at Theta troughs, causing input intensity to vary at different Theta phases, which changes firing time.

Next Steps
----------

1. Analyze the strength and duration of Phase Precession
2. Compare encoding at different velocities
3. Study interaction between Theta and Beta oscillations
4. Read :doc:`grid_place_cells` to learn more