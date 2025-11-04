Hebbian and Anti-Hebbian Learning
==================================

Scenario Description
--------------------

You want to compare two opposing learning rules — Hebbian ("connections between simultaneously active neurons are strengthened") and Anti-Hebbian ("connections between simultaneously active neurons are weakened") — to understand how they produce completely different network behaviors and functionalities.

What You Will Learn
--------------------

- Mathematical formulations of Hebbian and Anti-Hebbian rules
- Different network behaviors they produce
- Biological evidence and application scenarios
- Relationship between learning rules and network functionality

Complete Example
----------------

.. literalinclude:: ../../../../examples/brain_inspired/hopfield_hebbian_vs_antihebbian.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **Hebbian Learning**

   .. code-block:: python

      # Rule: simultaneous activity → strengthen connections
      # ΔW = η · y · x^T

      def hebbian_learning(activity, input_signal, learning_rate=0.01):
          """Hebbian learning rule"""
          delta_w = learning_rate * np.outer(activity, input_signal)
          return delta_w

      # Uses:
      # - Associative memory (association)
      # - Pattern storage
      # - Output-driven learning

2. **Anti-Hebbian Learning**

   .. code-block:: python

      # Rule: simultaneous activity → weaken connections
      # ΔW = -η · y · x^T

      def antihebbian_learning(activity, input_signal, learning_rate=0.01):
          """Anti-Hebbian learning rule"""
          delta_w = -learning_rate * np.outer(activity, input_signal)
          return delta_w

      # Uses:
      # - Sparse code learning
      # - Competitive networks
      # - Decomposition / Independent Component Analysis

3. **Comparison of Learning Objectives**

   .. code-block:: python

      # Hebbian: extracts direction of maximum variance (PCA)
      # Anti-Hebbian: extracts mutually independent directions (ICA)

      # Concretized in Oja's rule
      oja_rule = "ΔW = η · (y · x^T - y² · W)"
      #                 ↑            ↑
      #            Hebbian    normalization (essentially Anti-Hebbian)

Key Concepts
------------

**Function of Hebbian Learning**

.. code-block:: text

   Associative learning:
   Stimulus X (red apple) ↔ Output Y ("red")
   Multiple pairings → strengthen X→Y connection

   Future: seeing red → automatically "think of" red concept

**Function of Anti-Hebbian Learning**

.. code-block:: text

   Competitive learning:
   Input 1 (high activity) → inhibit Input 2
   Input 1 and Input 2 separated → each neuron encodes specific feature

   Result: sparse, distributed representation

**Biological Evidence**

Hebbian:
- Hippocampal synaptic strengthening
- Long-Term Potentiation (LTP)
- Activity-dependent dendritic spine formation

Anti-Hebbian:
- Certain cerebellar synapses
- Long-Term Depression (LTD)
- Competitive synaptic pruning

Experimental Variations
-----------------------

**1. Comparing Learning Speed**

.. code-block:: python

   hebbian_network = train_with_hebbian(data, epochs=100)
   antihebbian_network = train_with_antihebbian(data, epochs=100)

   plt.plot(hebbian_network.losses, label='Hebbian')
   plt.plot(antihebbian_network.losses, label='Anti-Hebbian')

**2. Comparing Representation Quality**

.. code-block:: python

   # Hebbian: reconstruction error (PCA)
   reconstruction_error_h = np.mean((data - reconstructed_h)**2)

   # Anti-Hebbian: independence measure (ICA)
   independence_ica = compute_independence_measure(unmixed)

**3. Network Capacity Comparison**

.. code-block:: python

   for num_patterns in [5, 10, 15, 20]:
       h_capacity = test_capacity(hebbian_network, num_patterns)
       ah_capacity = test_capacity(antihebbian_network, num_patterns)

Key Concepts
------------

**Properties of Weight Matrices**

Hebbian:
- W is symmetric (W^T = W)
- All eigenvalues are real
- Energy function well-defined

Anti-Hebbian:
- W is generally non-symmetric
- May have complex eigenvalues
- Oscillations may occur

**Learning Dynamics**

.. code-block:: text

   Hebbian:
   ┌─────────────┐
   │  Positive   │  → unbounded weight growth
   │  Reinforces │
   │  strong     │
   │  signals    │
   └─────────────┘
   Requires normalization: W ← W / ||W||

   Anti-Hebbian:
   ┌──────────────┐
   │  Negative    │  → automatically stable
   │  Inhibits    │
   │  strong      │
   │  signals     │
   └──────────────┘
   Naturally bounded

Related APIs
------------

- :class:`~src.canns.trainer.HebbianTrainer`
- :class:`~src.canns.trainer.AntiHebbianTrainer`
- :class:`~src.canns.trainer.OjaTrainer` (Hebbian + normalization)

Biological Applications
-----------------------

**Hebbian in Learning**

- Experience-dependent synaptic strengthening
- Habit formation
- Skill learning

**Anti-Hebbian in Competition**

- Neurons compete for different features
- Orientation selectivity formation
- Feature map development

More Resources
--------------

- :doc:`hopfield_basics` - Hebbian applications
- :doc:`../unsupervised_learning/oja_pca` - Hebbian + normalization

Frequently Asked Questions
---------------------------

**Q: Which learning rule is better?**

A: Depends on the task:
   - Associative memory → Hebbian
   - Feature separation → Anti-Hebbian
   - Dimensionality reduction → Hebbian + normalization

**Q: Does the brain use both?**

A: Yes! Different brain regions use different rules, and even different synapses on the same neuron may use different rules.

**Q: How do I choose the learning rate?**

A: Hebbian requires small learning rate + normalization. Anti-Hebbian is naturally stable and can use larger learning rates.

Next Steps
----------

1. Compare the two learners on the same data
2. Analyze the learned weight matrices
3. Test generalization ability on new data
4. Explore hybrid rules
