Energy Diagnostics and Network Analysis
========================================

Scenario Description
--------------------

You want to understand the internal workings of Hopfield networks through energy functions to diagnose network behavior, identify spurious attractors, and optimize network parameters.

What You Will Learn
--------------------

- Computing and visualizing energy functions
- Analysis of attractor basins
- Identification of spurious attractors
- Network debugging methods

Complete Example
----------------

.. literalinclude:: ../../../../examples/brain_inspired/hopfield_energy_diagnostics.py
   :language: python
   :linenos:

Step-by-Step Explanation
-------------------------

1. **Energy Function Computation**

   .. code-block:: python

      def compute_energy(state, weights, bias=None):
          """E = -0.5 * s^T W s - b^T s"""
          energy = -0.5 * np.dot(state, np.dot(weights, state))
          if bias is not None:
              energy -= np.dot(bias, state)
          return energy

2. **Tracking Energy Changes Across Iterations**

   .. code-block:: python

      state = initial_state.copy()
      energies = [compute_energy(state, W)]

      for step in range(max_steps):
          state = update(state, W)
          energy = compute_energy(state, W)
          energies.append(energy)

      plt.plot(energies)
      plt.xlabel('Iteration Steps')
      plt.ylabel('Energy')

3. **Identifying Attractors**

   .. code-block:: python

      def find_attractors(W, num_trials=100):
          """Find all attractors from random initializations"""
          attractors = []

          for trial in range(num_trials):
              init_state = np.random.rand(len(W)) > 0.5
              final_state = retrieve_pattern(init_state, W)
              attractors.append(tuple(final_state))

          unique_attractors = list(set(attractors))
          return unique_attractors

Execution Results
-----------------

Energy Descent Curve:

.. code-block:: text

   Energy
   ↑  ╲
      │  ╲___
      │      ╲___
      │          ╲_____
      │                ╲ ✓ Attractor (minimum)
      └─────────────────→ Iterations

Key Concepts
------------

**Energy Minimization**

Hopfield networks are equivalent to solving:

.. math::

   \\min_s E(s) = -0.5 s^T W s - b^T s

**Attractor Basins**

Different initializations converge to different attractors:

.. code-block:: text

   State Space

   Basin 1 → Attractor 1 (Target Pattern A)
   Basin 2 → Attractor 2 (Spurious Attractor)
   Basin 3 → Attractor 3 (Target Pattern B)

Experimental Variations
-----------------------

**1. Capacity vs. Spurious Attractors**

.. code-block:: python

   for num_patterns in range(1, 20):
       attractors = find_attractors(W)
       spurious = len(attractors) - num_patterns
       print(f"{num_patterns} stored → {spurious} spurious")

**2. Energy Landscape Visualization**

.. code-block:: python

   # Visualization after dimensionality reduction
   from sklearn.decomposition import PCA

   pca = PCA(n_components=2)
   projected_states = pca.fit_transform(all_states)

Related API
-----------

- :func:`~src.canns.analyzer.energy_analysis`

Next Steps
----------

- :doc:`hopfield_basics` - Theoretical Foundations
- :doc:`hebbian_vs_antihebbian` - Learning Rules Comparison
