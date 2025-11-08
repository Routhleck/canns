Building Processing Pipelines
=============================

.. warning::

   ⚠️ **Important Notice**: Some content in this documentation is still under development and validation, and may be incomplete. It is recommended for reference only. Please confirm with the development team about the completeness of relevant features before using them in important projects.



Scenario Description
--------------------

You want to combine multiple modules (data loading, preprocessing, model training, analysis) into an automated, repeatable processing pipeline for large-scale data or multiple experiments.

What You Will Learn
-------------------

- Pipeline design patterns
- Modularity and composition
- Error handling and logging
- Configuration management
- Parallelization and scaling

Complete Example
----------------

.. code-block:: python

   from canns.pipeline import Pipeline, Task
   import yaml

   # Define configuration
   config = {
       'data': {
           'path': '/data/experiments',
           'batch_size': 32
       },
       'model': {
           'type': 'CANN1D',
           'num_neurons': 512
       },
       'training': {
           'epochs': 50,
           'learning_rate': 0.01
       }
   }

   # Define tasks
   pipeline = Pipeline()

   # Task 1: Data loading
   @pipeline.task('load_data')
   def load_data(config):
       data = load_experimental_data(config['data']['path'])
       return {'data': data}

   # Task 2: Preprocessing
   @pipeline.task('preprocess', depends_on=['load_data'])
   def preprocess(inputs, config):
       data = inputs['data']
       preprocessor = DataPreprocessor()
       clean_data = preprocessor.process_pipeline(data)
       return {'clean_data': clean_data}

   # Task 3: Model training
   @pipeline.task('train', depends_on=['preprocess'])
   def train(inputs, config):
       model = CANN1D(num_neurons=config['model']['num_neurons'])
       trainer = STDPTrainer(model, learning_rate=config['training']['learning_rate'])

       for epoch in range(config['training']['epochs']):
           loss = trainer.train(inputs['clean_data'])

       return {'model': model, 'loss': loss}

   # Task 4: Analysis
   @pipeline.task('analyze', depends_on=['train'])
   def analyze(inputs, config):
       model = inputs['model']
       analysis_results = analyze_network(model)
       return {'results': analysis_results}

   # Run pipeline
   results = pipeline.run(config)

   # Save results
   pipeline.save_results(results, output_dir='/results')

Key Concepts
------------

**Pipeline DAG (Directed Acyclic Graph)**

.. code-block:: text

   load_data
      ↓
   preprocess
      ↓
   train
      ↓
   analyze
      ↓
   visualize
      ↓
   save

**Task Dependencies**

- Sequential execution: A→B (B waits for A to complete)
- Parallel execution: A||B (A and B can run simultaneously)
- Conditional execution: Choose branches based on results

Experimental Variations
-----------------------

**1. Multiple Pipelines**

.. code-block:: python

   # Run multiple pipelines in parallel
   configs = load_all_experiments()
   results = parallel_run_pipelines(configs)

**2. Distributed Processing**

.. code-block:: python

   # Run on a cluster
   pipeline.set_backend('dask')  # or 'spark'
   pipeline.run(config)

**3. Fault Tolerance**

.. code-block:: python

   # Automatically retry failed tasks
   pipeline.set_retry_policy(max_retries=3, backoff=2.0)

Related API
-----------

- :class:`~src.canns.pipeline.Pipeline`
- :class:`~src.canns.pipeline.Task`
- :func:`~src.canns.pipeline.run_parallel`

Next Steps
----------

- :doc:`external_trajectories` - Integrating external data
- :doc:`parameter_customization` - Parameter tuning