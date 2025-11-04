Unsupervised Learning
=====================

Scenario Description
--------------------

Unsupervised learning is the core of self-organized learning in neural networks. This series of tutorials focuses on classical bio-inspired learning algorithms, particularly the neural implementation of Principal Component Analysis (PCA):

- **Oja's Rule**: Extracting the first principal component
- **Sanger's Rule**: Extracting multiple orthogonal principal components
- Comparison with traditional PCA algorithms

What You Will Learn
-------------------

1. Mathematical principles and implementation of Oja's rule
2. How to extract principal components from high-dimensional data
3. Orthogonalization mechanism of Sanger's rule
4. Convergence analysis of learning algorithms
5. Comparative verification with sklearn PCA

Tutorial List
-------------

.. toctree::
   :maxdepth: 1

   oja_pca
   sanger_orthogonal_pca
   algorithm_comparison

Target Audience
---------------

- Researchers interested in bio-inspired learning algorithms
- Students who need to understand neural implementation of PCA
- Developers researching unsupervised learning

Prerequisites
-------------

- Linear algebra (eigenvalues, eigenvectors)
- Basic statistics (variance, covariance)
- Python and NumPy

Core Algorithms
---------------

**Oja's Rule**:

.. math::

   \Delta W = \eta \cdot (y \cdot x^T - y^2 \cdot W)

- First term: Hebbian enhancement
- Second term: Weight normalization

**Sanger's Rule (GHA)**:

.. math::

   \Delta W_{ij} = \eta \cdot y_i \cdot (x_j - \sum_{k=1}^{i} W_{kj} y_k)

- Extract multiple principal components through Gram-Schmidt orthogonalization

Practical Applications
----------------------

- **Dimensionality Reduction**: Low-dimensional representation of high-dimensional data
- **Feature Extraction**: Identifying principal directions of data variation
- **Data Compression**: Preserving the most important information
- **Denoising**: Filtering noise while retaining signals

Theoretical Significance
------------------------

Oja's and Sanger's rules demonstrate that complex statistical operations (PCA) can emerge from simple local learning rules, which is of great importance for understanding the learning mechanisms of biological neural networks.

Getting Started
---------------

Start with :doc:`oja_pca` and learn how neural networks automatically discover data structure!
