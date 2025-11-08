Hopfield Network Basics: Associative Memory
============================================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scenario Description
--------------------

You want to understand how to use neural networks to store and retrieve memories. The Hopfield network is the simplest and most elegant model for implementing associative memory: given an incomplete or corrupted pattern as input, the network can recover the complete stored pattern.

What You Will Learn
-------------------

- How Hopfield networks work
- Energy functions and attractor dynamics
- Mechanisms of pattern storage and retrieval
- Capacity limitations and interference
- Possibilities for biological implementation

Complete Example
----------------

Pattern storage and retrieval based on Hopfield networks:

.. code-block:: python

   import numpy as np
   from canns.models.brain_inspired import HopfieldNetwork

   # Create Hopfield network
   network = HopfieldNetwork(num_neurons=100)
   network.init_state()

   # Store patterns
   patterns = [
       np.random.randn(100) > 0,  # Pattern 1: random binary
       np.random.randn(100) > 0,  # Pattern 2
       np.random.randn(100) > 0,  # Pattern 3
   ]

   # Store using Hebbian rule
   for pattern in patterns:
       network.store(pattern)

   # Test: partial input → complete output
   corrupted = patterns[0].copy()
   corrupted[:10] = np.random.rand(10) > 0.5  # Corrupt first 10 bits

   retrieved = network.retrieve(corrupted, steps=100)

   # Check retrieval accuracy
   accuracy = np.mean(retrieved == patterns[0])
   print(f"Retrieval accuracy: {accuracy:.1%}")

Step-by-Step Analysis
---------------------

1. **Hopfield Network Structure**

   .. code-block:: python

      # Fully connected symmetric network
      # Total neurons: N
      # Connection weights: W[i,j] = W[j,i]  (symmetric)
      # No self-connections: W[i,i] = 0

      import numpy as np

      N = 100  # Number of neurons
      W = np.random.randn(N, N)
      W = (W + W.T) / 2  # Symmetrize
      np.fill_diagonal(W, 0)  # Remove self-connections

   **Network Topology**:

   .. code-block:: text

      ○─────○─────○
      │╲   │╱  │
      │ ╲ ╱ ╲ │
      │  ×   ╲│
      │ ╱ ╲  ╱
      │╱   ╲│
      ○─────○─────○

      - All neurons connected to all other neurons
      - Weight matrix: N×N symmetric matrix
      - Is a recurrent network

2. **Hebbian Learning Rule**

   .. code-block:: python

      # Pattern storage: set weights using Hebbian rule
      # W = (1/N) * Σ ξ ξ^T  (outer product)

      def store_patterns_hebbian(patterns):
          N = len(patterns[0])
          W = np.zeros((N, N))

          for pattern in patterns:
              # Outer product: ξ ξ^T
              W += np.outer(pattern, pattern)

          # Normalize and remove self-connections
          W = W / N
          np.fill_diagonal(W, 0)

          return W

   **Weight Significance**:
   - W[i,j] > 0: if neurons i and j are usually activated together
   - W[i,j] < 0: if neurons i and j are usually activated oppositely
   - Weight magnitude reflects the frequency of co-activation

3. **Energy Function and Attractors**

   .. code-block:: python

      # Energy function of Hopfield network
      def compute_energy(state, W, b=None):
          """Compute the energy of the network"""
          # E = -0.5 * s^T W s - b^T s
          energy = -0.5 * np.dot(state, np.dot(W, state))
          if b is not None:
              energy -= np.dot(b, state)
          return energy

      # Key property: energy monotonically decreases with synchronous updates
      # Network eventually converges to a local minimum (attractor)

   **Energy Landscape**:

   .. code-block:: text

      Energy
      ↑
      │     ╱╲      ╱╲
      │    ╱  ╲    ╱  ╲      Attractors (energy valleys)
      │___╱____╱╲__╱____╲____
      │         ╲  ╱
      │          ╲╱
      └─────────────────────→ Neuron state space

4. **Pattern Retrieval Process**

   .. code-block:: python

      def retrieve_pattern(initial_state, W, max_steps=100):
          """Recover complete pattern from partial input"""
          state = initial_state.copy()

          for step in range(max_steps):
              old_state = state.copy()

              # Synchronous update: all neurons update simultaneously
              activation = np.dot(W, state)
              state = (activation > 0).astype(float)

              # Check convergence
              if np.array_equal(state, old_state):
                  print(f"Converged at step {step}")
                  break

          return state

   **Convergence Guarantee**:
   - Hopfield networks always converge
   - Energy function monotonically decreases
   - Final state is an attractor (stored pattern or spurious attractor)

Running Results
---------------

A successful retrieval process:

.. code-block:: text

   Input (80% corrupted):        ███░░░░░░░░░░░░░░░░
   After 1st iteration:          ████░░░░░░░░░░░░░░░░
   After 5th iteration:          ████████░░░░░░░░░░░░
   After 10th iteration:         ████████████░░░░░░░░
   After 20th iteration:         ████████████████████ ✓ Complete recovery

Key Concepts
------------

**Attractor Dynamics**

The core of Hopfield networks is attractors:

.. code-block:: text

   Attractor basin:
   ┌──────────────────────┐
   │   Attractor basin 1  │
   │  ╱───────╲          │
   │ ╱  Stored  ╲        │
   │╱  Pattern1 ╲       │
   │            ╲       │
   │  Partial    → Complete
   │  pattern     pattern 1
   │
   └──────────────────────┘

   ┌──────────────────────┐
   │   Attractor basin 2  │
   │  ╱───────╲          │
   │ ╱  Stored  ╲       │
   │╱  Pattern2 ╲      │
   └──────────────────────┘

**Capacity and Interference**

How many patterns can a Hopfield network store?

.. code-block:: python

   # Theoretical capacity: α = C/N ≈ 0.138
   # Where C is the number of stored patterns, N is the number of neurons

   N = 100
   max_patterns = int(0.138 * N)  # ~14 patterns

   # Exceeding capacity:
   # - Retrieval errors increase
   # - Spurious attractors appear
   # - Stored patterns may not be recoverable

**Spurious Attractors**

The network may converge to non-stored attractors:

.. code-block:: text

   Spurious attractor example (storing [1,0] and [0,1]):
   - [1,1]: possible spurious attractor (mixture of two stored patterns)
   - [0,0]: possible spurious attractor
   - Only [1,0] and [0,1] are desired attractors

Experimental Variations
------------------------

**1. Vary the Number of Stored Patterns**

.. code-block:: python

   for num_patterns in [2, 5, 10, 15, 20]:
       patterns = [np.random.randn(100) > 0 for _ in range(num_patterns)]
       network = HopfieldNetwork(100)
       for pattern in patterns:
           network.store(pattern)

       # Test retrieval success rate
       success_rate = test_retrieval(network, patterns)
       print(f"{num_patterns} patterns: success rate {success_rate:.1%}")

**2. Analyze the Effect of Corruption Level**

.. code-block:: python

   for corruption_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
       corrupted = patterns[0].copy()
       mask = np.random.rand(len(corrupted)) < corruption_level
       corrupted[mask] = 1 - corrupted[mask]

       retrieved = network.retrieve(corrupted)
       accuracy = np.mean(retrieved == patterns[0])
       print(f"Corruption {corruption_level:.0%}: retrieval accuracy {accuracy:.1%}")

**3. Visualize Energy Landscape**

.. code-block:: python

   # Visualize energy on a 2D projection
   import matplotlib.pyplot as plt

   x = np.linspace(-1, 1, 100)
   y = np.linspace(-1, 1, 100)
   X, Y = np.meshgrid(x, y)
   Z = np.zeros_like(X)

   for i in range(100):
       for j in range(100):
           state = np.zeros(N)
           state[0] = X[i, j]
           state[1] = Y[i, j]
           Z[i, j] = compute_energy(state, W)

   plt.contourf(X, Y, Z, levels=20, cmap='viridis')
   plt.colorbar(label='Energy')
   plt.title('Energy Landscape of Hopfield Network')

Related API
-----------

- :class:`~src.canns.models.brain_inspired.HopfieldNetwork` - Hopfield network
- :class:`~src.canns.trainer.HebbianTrainer` - Hebbian learner
- :func:`~src.canns.analyzer.spatial.compute_energy` - Energy computation

Biological Applications
-----------------------

**Associative Memory in the Brain**

- **Olfactory Cortex**: odor recognition
- **Hippocampal CA3**: contextual association and pattern completion
- **Prefrontal Cortex**: working memory maintenance

**Advantages of Hopfield Networks**

- Simple learning rule (Hebbian)
- Automatic error correction (partial → complete)
- Biologically plausible
- Content-addressable storage

More Resources
--------------

- :doc:`pattern_storage_1d` - One-dimensional pattern storage
- :doc:`mnist_memory` - Memory for MNIST digits
- :doc:`energy_diagnostics` - Energy analysis tools
- :doc:`hebbian_vs_antihebbian` - Comparison of different learning rules

FAQ
---

**Q: Why does the Hopfield network always converge?**

A: Because the energy function monotonically decreases (or remains constant) with each update. The final state must be a local minimum, i.e., an attractor. This guarantees network stability.

**Q: What about spurious attractors?**

A: Spurious attractors are a fundamental limitation of Hopfield networks. They can be reduced by:
   - Reducing the number of stored patterns
   - Using better learning rules (e.g., projection rule)
   - Using orthogonalization of patterns

**Q: What practical applications can Hopfield networks be used for?**

A:
   - Error-correcting codes
   - Image denoising
   - Combinatorial optimization (convertible to patterns)
   - Keyword matching
   - Medical diagnostic systems

Next Steps
----------

1. Try storing more patterns and observe retrieval performance
2. Analyze the properties of spurious attractors
3. Compare synchronous and asynchronous updates
4. Read :doc:`hebbian_vs_antihebbian` to learn about different learning rules