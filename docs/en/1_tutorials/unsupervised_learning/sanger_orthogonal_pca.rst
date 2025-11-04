Sanger Rule: Orthogonal PCA Multi-Component Extraction
=======================================================

Scenario Description
--------------------

You want to extract multiple orthogonal principal components from high-dimensional data, going beyond the limitation of Oja's rule which can extract only one principal component. Sanger's rule (Generalized Hebbian Algorithm) implements simultaneous extraction of multiple principal components through a simple local learning rule.

What You Will Learn
-------------------

- Mathematical derivation of Sanger's rule
- Relationship with Oja's rule
- Extraction of multiple orthogonal principal components
- Convergence with standard PCA
- Applications in online learning

Complete Example
----------------

.. literalinclude:: ../../../../examples/brain_inspired/oja_vs_sanger_comparison.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **Mathematics of Sanger's Rule**

   .. code-block:: python

      # Sanger's rule (Generalized Hebbian Algorithm)
      # ΔW = η · (y · x^T - y · (W^T · y) · 1^T)
      #
      # Where:
      # - y = W^T · x  (output)
      # - W^T · y     (self-feedback)
      # - 1^T         ("Gram-Schmidt" term)

      def sanger_learning(X, W, learning_rate=0.001, epochs=20):
          """Implementation of Sanger's rule"""
          N_features, N_components = W.shape

          for epoch in range(epochs):
              for x in X:
                  y = np.dot(W.T, x)  # Output

                  # Key aspect of Sanger's rule:
                  lower_triangular = np.tril(np.dot(y.reshape(-1, 1), y.reshape(1, -1)))
                  delta_w = learning_rate * np.dot(x.reshape(-1, 1), y.reshape(1, -1))
                  delta_w -= learning_rate * np.dot(W, lower_triangular)

                  W += delta_w

          return W

2. **Comparison with Oja's Rule**

   .. code-block:: python

      # Oja's rule (single principal component):
      oja_rule = "ΔW = η · (y · x^T - y² · W)"

      # Sanger's rule (multiple principal components):
      sanger_rule = "ΔW[i,:] = η · (y[i] · x^T - y[i] · Σ_j<i y[j] · W[j,:])"

      # Meaning:
      # - Learning of the first principal component is identical to Oja's
      # - Subsequent principal components are orthogonal to earlier ones

3. **Testing Orthogonality**

   .. code-block:: python

      # Check if learned principal components are orthogonal
      W_learned = train_with_sanger(data)

      for i in range(n_components):
          for j in range(i+1, n_components):
              dot_product = np.dot(W_learned[i], W_learned[j])
              print(f"Inner product of components {i} and {j}: {dot_product:.6f}")

   **Expected**: Inner product should be close to 0 (completely orthogonal)

4. **Convergence with PCA**

   .. code-block:: python

      from sklearn.decomposition import PCA

      # Standard PCA
      pca = PCA(n_components=3)
      pca_components = pca.fit_transform(data)

      # Learning with Sanger's rule
      sanger_components = train_with_sanger(data, n_components=3)

      # Comparison: should learn the same principal component directions
      for i in range(3):
          similarity = np.abs(np.dot(sanger[i], pca.components_[i]))
          print(f"PC{i} similarity: {similarity:.4f}")  # Should be > 0.95

Running Results
---------------

Learning curves for Sanger's rule:

.. code-block:: text

   Explained Variance Ratio
   ↑
   │  PC1: ════════════════ 52%
   │  PC2: ════════ 18%
   │  PC3: ═══ 7%
   │
   └──────────────────────→ Components

   Convergence time: 10-20 epochs

Key Concepts
------------

**Gram-Schmidt Orthogonalization**

Sanger's rule implements the Gram-Schmidt process online:

.. math::

   w_i = x_i - \\sum_{j<i} (w_j^T x_i) w_j

Corresponding to:

.. code-block:: text

   New component = Original direction - Projections of earlier components

**Information-Theoretic Interpretation**

Sanger's rule maximizes:

.. math::

   I(y; x) = \\sum_{i=1}^{n} I(y_i; x)

(Mutual information between output and input)

Subject to the constraint:

.. math::

   y_i \\perp y_j, \\quad i \\neq j

Experimental Variations
-----------------------

**1. Varying the Number of Principal Components**

.. code-block:: python

   for n_pc in [1, 2, 3, 5, 10]:
       W = train_sanger(data, n_components=n_pc)
       variance_exp = compute_variance_explained(W, data)
       print(f"Number of PCs {n_pc}: variance {variance_exp:.1%}")

**2. Effect of Learning Rate**

.. code-block:: python

   for lr in [0.0001, 0.001, 0.01, 0.1]:
       W = train_sanger(data, learning_rate=lr)
       convergence_time = measure_convergence(W, target=pca.components_)

**3. Online Learning Capability**

.. code-block:: python

   # Online Sanger on streaming data
   W = np.random.randn(n_features, n_pc) * 0.1

   for batch in data_stream:
       for x in batch:
           y = np.dot(W.T, x)
           W += learning_rate * (np.dot(x.reshape(-1,1), y.reshape(1,-1)) -
                                 np.dot(W, np.tril(np.outer(y, y))))

Related API
-----------

- :class:`~src.canns.trainer.SangerTrainer` - Sanger rule trainer
- :class:`~src.canns.trainer.OjaTrainer` - Oja rule (for comparison)

Biological Applications
-----------------------

**Multi-Channel Sensory Processing**

- Vision: Simultaneous extraction of multiple features (color, orientation, spatial frequency)
- Audition: Separation of multiple frequency components
- Olfaction: Recognition of multiple molecular features

**Diversity in Neural Coding**

Cortical neurons appear to be organized as multiple independent feature extractors, which may be implemented through Sanger's rule or similar mechanisms.

Additional Resources
--------------------

- :doc:`oja_pca` - Single principal component extraction (basics)
- :doc:`algorithm_comparison` - Comparison with other methods

Frequently Asked Questions
--------------------------

**Q: When is Sanger superior to standard PCA?**

A: In online scenarios and when computation is constrained. Sanger requires only memory proportional to the number of samples, while PCA needs to store the entire covariance matrix (N² memory).

**Q: Why is the Gram-Schmidt term necessary?**

A: To ensure that extracted principal components are orthogonal. Without it, subsequent components would repeatedly extract in the direction of the first principal component.

**Q: How long does it take to converge to true PCA?**

A: Typically 20-50 epochs, depending on data dimensionality and learning rate. In online learning, multiple passes through the data are needed.

Next Steps
----------

1. Compare Oja and Sanger performance on the same data
2. Test online learning capability
3. Apply to real data (images, audio, etc.)
4. Explore nonlinear extensions (kernel PCA)

References
----------

- Sanger, T. D. (1989). Optimal unsupervised learning in a single-layer linear feedforward neural network. Neural Networks
- Oja, E., & Karhunen, J. (1985). On stochastic approximation of the eigenvectors and eigenvalues of the expectation of a random matrix. Journal of Mathematical Analysis and Applications