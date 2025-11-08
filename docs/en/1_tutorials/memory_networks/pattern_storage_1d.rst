1D Pattern Storage and Retrieval
=================================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scenario Description
--------------------

You want to understand the mechanism of pattern storage in one-dimensional space, which is the easiest configuration to implement and measure in practical neuroscience experiments.

What You Will Learn
-------------------

- Hebbian storage of 1D patterns
- Capacity limits and information-theoretic bounds
- Retrieval performance versus network size
- Experimental validation methods

Complete Example
----------------

.. literalinclude:: ../../../../examples/brain_inspired/hopfield_train_1d.py
   :language: python
   :linenos:

Step-by-Step Explanation
-------------------------

Key steps for 1D pattern storage:

1. **Generate 1D Patterns**

   .. code-block:: python

      N = 100  # Network size
      patterns = [np.random.randn(N) > 0 for _ in range(5)]  # 5 random patterns

2. **Hebbian Weight Calculation**

   .. code-block:: python

      W = np.zeros((N, N))
      for pattern in patterns:
          W += np.outer(pattern, pattern)
      W = W / N
      np.fill_diagonal(W, 0)

3. **Test Retrieval**

   .. code-block:: python

      for pattern_idx, pattern in enumerate(patterns):
          # Create corrupted version (30% corruption)
          corrupted = pattern.copy()
          corrupted[:30] = 1 - corrupted[:30]

          retrieved = retrieve(corrupted, W)
          accuracy = np.mean(retrieved == pattern)
          print(f"Pattern {pattern_idx}: accuracy {accuracy:.1%}")

Results
-------

Performance curve for 1D pattern storage:

.. code-block:: text

   Retrieval Accuracy
   ↑
   100%│     ╱──────
       │    ╱
    75%│   ╱
       │  ╱
    50%│ ╱
       │╱
     0%└────────────────→ Number of Stored Patterns
       0   5   10   15

Key Concepts
------------

**Information Capacity**

Capacity of 1D Hopfield network:

.. math::

   C \\approx 0.138 \\cdot N

For example: 100 neurons can store ~14 patterns

**Retrieval Complexity**

Expected number of steps for asynchronous updates:

.. math::

   E[steps] \\propto \\log(N)

Experimental Variations
-----------------------

**1. Vary Network Size**

.. code-block:: python

   for N in [50, 100, 200, 500]:
       patterns = generate_patterns(N, num_patterns=int(0.1*N))
       network = HopfieldNetwork(N)
       success_rate = test_retrieval(network, patterns)

**2. Vary Corruption Level**

.. code-block:: python

   for corruption in [0.1, 0.3, 0.5, 0.7]:
       accuracy = test_with_corruption(corruption)

Related API
-----------

- :class:`~src.canns.models.brain_inspired.HopfieldNetwork`
- :class:`~src.canns.trainer.HebbianTrainer`

FAQ
---

**Q: Why is capacity limited?**

A: Because the weight matrix is finite (N² weights), storing too many patterns causes interference.

Next Steps
----------

- :doc:`mnist_memory` - Learn with real data
- :doc:`energy_diagnostics` - Analyze energy