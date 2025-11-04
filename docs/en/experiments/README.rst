Experiments
===========

This directory is used to store experimental extensions and user-defined content.

About This Directory
--------------------

The ``experiments/`` directory provides a dedicated space for users to:

- Test new ideas and algorithms
- Develop custom analysis tools
- Store ongoing research projects
- Experiment with model variants

This directory is not included in the CANNs core code, but will be referenced and explained in the documentation.

Usage Recommendations
---------------------

1. **Organization Structure**

   It is recommended to create subdirectories by project or topic:

   .. code-block:: text

      experiments/
      ├── custom_learning_rules/
      │   ├── my_stdp_variant.py
      │   └── README.md
      ├── network_architectures/
      │   ├── modular_cann.py
      │   └── experiments.ipynb
      └── analysis_tools/
          └── custom_analyzer.py

2. **Documentation**

   Add a README to each experimental project with:

   - Experimental objectives
   - Methods used
   - Expected results
   - Current status

3. **Version Control**

   You can add this directory to ``.gitignore`` to keep it private, or selectively commit it to the repository.

Example: Custom Learning Rules
-------------------------------

.. code-block:: python

   # experiments/custom_learning_rules/my_rule.py
   from canns.trainer import Trainer
   import jax.numpy as jnp

   class MyCustomTrainer(Trainer):
       """My custom learning rule"""

       def __init__(self, model, learning_rate=0.01):
           super().__init__(model=model)
           self.learning_rate = learning_rate

       def train(self, train_data):
           # Implement your learning rule
           for pattern in train_data:
               # ... custom logic
               pass

Share Your Experiments
----------------------

If your experiments yield interesting results, you are welcome to:

1. Share them in GitHub Issues
2. Submit a Pull Request to integrate them into the core library
3. Exchange ideas in community discussions

Related Resources
-----------------

- :doc:`../1_tutorials/index` - Learn basic usage of CANNs
- `GitHub Issues <https://github.com/your-org/canns/issues>`_ - Discuss new ideas

Getting Started with Experiments
---------------------------------

Create your first experimental project:

.. code-block:: bash

   cd docs/zh/experiments/
   mkdir my_experiment
   cd my_experiment
   touch experiment.py README.md

Then start coding in ``experiment.py``!