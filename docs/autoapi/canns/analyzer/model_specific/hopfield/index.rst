canns.analyzer.model_specific.hopfield
======================================

.. py:module:: canns.analyzer.model_specific.hopfield

.. autoapi-nested-parse::

   Hopfield network analysis tools.



Classes
-------

.. autoapisummary::

   canns.analyzer.model_specific.hopfield.HopfieldAnalyzer


Module Contents
---------------

.. py:class:: HopfieldAnalyzer(model, stored_patterns = None)

   Analyzer for Hopfield associative memory networks.

   Provides diagnostics such as pattern overlap, energy, and recall quality.

   .. rubric:: Examples

   >>> import jax.numpy as jnp
   >>> from canns.models.brain_inspired import AmariHopfieldNetwork
   >>> from canns.analyzer.model_specific.hopfield import HopfieldAnalyzer
   >>>
   >>> # Dummy inputs (patterns) based on analyzer tests
   >>> patterns = [
   ...     jnp.array([1.0, -1.0, 1.0]),
   ...     jnp.array([-1.0, 1.0, -1.0]),
   ... ]
   >>> model = AmariHopfieldNetwork(num_neurons=3)
   >>> analyzer = HopfieldAnalyzer(model, stored_patterns=patterns)
   >>> diagnostics = analyzer.analyze_recall(patterns[0], patterns[0])
   >>> print(sorted(diagnostics.keys()))
   ['best_match_idx', 'best_match_overlap', 'input_output_overlap', 'output_energy']

   Initialize Hopfield analyzer.

   :param model: The Hopfield network model to analyze
   :param stored_patterns: List of patterns stored in the network (optional)


   .. py:method:: analyze_recall(input_pattern, output_pattern)

      Analyze pattern recall quality.

      :param input_pattern: Input (noisy) pattern
      :param output_pattern: Recalled pattern

      :returns:     - best_match_idx: Index of best matching stored pattern
                    - best_match_overlap: Overlap with best matching pattern
                    - input_output_overlap: Overlap between input and output
                    - output_energy: Energy of the recalled pattern
      :rtype: Dictionary with diagnostic metrics



   .. py:method:: compute_energy(pattern)

      Compute energy of a given pattern.

      :param pattern: Pattern to compute energy for

      :returns: Energy value E = -0.5 * s^T W s



   .. py:method:: compute_overlap(pattern1, pattern2)

      Compute normalized overlap between two patterns.

      :param pattern1: First pattern
      :param pattern2: Second pattern

      :returns: Overlap value between -1 and 1



   .. py:method:: compute_pattern_energies()

      Compute energy for each stored pattern.



   .. py:method:: compute_weight_symmetry_error()

      Compute the symmetry error of the weight matrix.

      Hopfield networks require symmetric weights (W_ij = W_ji).
      This metric quantifies how much the weight matrix deviates from symmetry.

      :returns: Symmetry error as ||W - W^T||_F / ||W||_F



   .. py:method:: estimate_capacity()

      Estimate theoretical storage capacity of the network.

      Uses the rule of thumb: capacity ≈ N / (4 * ln(N))
      where N is the number of neurons.

      :returns: Estimated number of patterns that can be reliably stored



   .. py:method:: get_statistics()

      Get comprehensive statistics about stored patterns.

      :returns:     - num_patterns: Number of stored patterns
                    - capacity_estimate: Theoretical capacity estimate
                    - capacity_usage: Fraction of capacity used
                    - mean_pattern_energy: Mean energy of stored patterns
                    - std_pattern_energy: Standard deviation of energies
                    - min_pattern_energy: Minimum energy
                    - max_pattern_energy: Maximum energy
      :rtype: Dictionary with pattern statistics



   .. py:method:: set_patterns(patterns)

      Set the stored patterns and compute their energies.

      :param patterns: List of patterns stored in the network



   .. py:attribute:: model


   .. py:property:: pattern_energies
      :type: list[float]


      Get energies of stored patterns.


   .. py:attribute:: stored_patterns


