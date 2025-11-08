Associative Memory for MNIST Digits
====================================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scene Description
-----------------

You want to see how Hopfield networks perform on real data — storing handwritten digit images and recovering complete digits from partial images. This demonstrates the capabilities of biologically-inspired models on practical tasks.

What You Will Learn
--------------------

- Preprocessing of image data
- Storage of high-dimensional patterns
- Image completion tasks
- Practical limitations of network capacity
- Methods for visualizing results

Complete Example
----------------

.. literalinclude:: ../../../../examples/brain_inspired/hopfield_train_mnist.py
   :language: python
   :linenos:

Step-by-Step Analysis
---------------------

1. **MNIST Data Preparation**

   .. code-block:: python

      from torchvision import datasets
      import numpy as np

      mnist = datasets.MNIST(root='./data', download=True, train=True)
      images = mnist.data.numpy() / 255  # Normalization
      images = images.reshape(len(images), -1)  # Flatten to vectors
      # Binarization
      binary_images = (images > 0.5).astype(float)

2. **Storage of MNIST Patterns**

   .. code-block:: python

      N = 28 * 28  # 784 dimensions
      num_images = 10  # Store 10 different digits

      selected_images = binary_images[:num_images]
      W = compute_hebbian_weights(selected_images)

3. **Recovery from Partial Images**

   .. code-block:: python

      # Mask upper half
      corrupted = selected_images[0].copy()
      corrupted[:392] = 0.5 * np.ones(392)  # Masking

      recovered = retrieve_pattern(corrupted, W)

      # Visualization
      plot_three_images(selected_images[0], corrupted, recovered)

Execution Results
------------------

- Successfully recovers complete digits (from 50% corrupted input)
- But capacity is limited (typically can only store 3-5 different digits)

Key Concepts
-------------

**Curse of Dimensionality**

Problems in high-dimensional spaces:

.. code-block:: python

   # MNIST: 784 dimensions
   # Number of storable patterns: ~108 (0.138 * 784)
   # But in practice, capacity is lower due to high dimensionality

**Spurious Attractors**

Spurious attractors increase in high dimensions:

.. code-block:: text

   2 dimensions: Few spurious attractors
   100 dimensions: Spurious attractors noticeably increase
   784 dimensions: Many spurious attractors

Experimental Variations
------------------------

**1. Changing the Number of Stored Digits**

.. code-block:: python

   for num_digits in [1, 2, 3, 5, 10]:
       success_rates[num_digits] = test_retrieval(num_digits)

**2. Different Corruption Patterns**

.. code-block:: python

   # Upper half corrupted
   # Left half corrupted
   # Random pixel corruption (salt-and-pepper noise)

Related API
-----------

- :class:`~src.canns.models.brain_inspired.HopfieldNetwork`

Next Steps
----------

- :doc:`energy_diagnostics` - Analyze why high dimensions fail
- :doc:`hopfield_basics` - Theoretical background
