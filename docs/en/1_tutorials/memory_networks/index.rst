Memory Networks Training
========================

Scenario Description
--------------------

Hopfield networks are the most classical associative memory models, capable of storing and retrieving patterns. This series of tutorials will teach you how to:

- Train Hopfield networks using Hebbian learning
- Store and recover multiple patterns (including images)
- Analyze the network's energy landscape and storage capacity
- Compare the effectiveness of Hebbian and Anti-Hebbian learning

What You Will Learn
-------------------

1. Fundamentals and training of Hopfield networks
2. Storage and retrieval of 1D binary patterns
3. Memory tasks on MNIST handwritten digits
4. Energy function analysis and diagnostic tools
5. Comparison between Hebbian and Anti-Hebbian learning

Tutorial List
-------------

.. toctree::
   :maxdepth: 1

   hopfield_basics
   pattern_storage_1d
   mnist_memory
   energy_diagnostics
   hebbian_vs_antihebbian

Target Audience
---------------

- Researchers interested in associative memory
- Students learning neural network fundamentals
- Developers who need to implement content-addressable memory

Prerequisites
-------------

- Basic neural network concepts
- Linear algebra fundamentals
- Python and NumPy

Practical Applications
----------------------

- **Pattern Recognition**: Store and recognize visual patterns
- **Error Correction**: Recover corrupted data through associative memory
- **Content-Addressable Storage**: Retrieve complete content from partial information
- **Cognitive Modeling**: Understand the neural basis of human memory

Key Concepts
------------

- **Energy Function**: E = -∑ᵢⱼ Wᵢⱼ sᵢ sⱼ
- **Attractors**: Patterns as stable points in network dynamics
- **Storage Capacity**: ~0.14N (N is the number of neurons)

Get Started
-----------

Start with :doc:`hopfield_basics` to train your first memory network!
