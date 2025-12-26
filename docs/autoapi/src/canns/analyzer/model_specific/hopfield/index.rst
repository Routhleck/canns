src.canns.analyzer.model_specific.hopfield
==========================================

.. py:module:: src.canns.analyzer.model_specific.hopfield

.. autoapi-nested-parse::

   Hopfield network analysis tools.



Classes
-------

.. autoapisummary::

   src.canns.analyzer.model_specific.hopfield.HopfieldAnalyzer


Module Contents
---------------

.. py:class:: HopfieldAnalyzer(model, stored_patterns = None)

   Analyzer for Hopfield associative memory networks.

   Provides diagnostic and analysis tools for Hopfield networks including:
   - Pattern storage capacity estimation
   - Energy landscape analysis
   - Overlap metrics for pattern retrieval
   - Recall quality diagnostics

   The Hopfield network stores patterns as attractors in an energy landscape.
   Energy function:
       E = -0.5 * s^T W s

   Reference:
       Hopfield, J. J. (1982). Neural networks and physical systems with
       emergent collective computational abilities. PNAS, 79(8), 2554-2558.

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


