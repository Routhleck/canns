构建处理流程（Pipeline）
==========================

场景描述
--------

你想要将多个模块（数据加载、预处理、模型训练、分析）组合成一个自动化的、可重复的处理流程，用于大规模数据或多个实验。

你将学到
--------

- Pipeline设计模式
- 模块化和组合
- 错误处理和日志
- 配置管理
- 并行化和扩展

完整示例
--------

.. code-block:: python

   from canns.pipeline import Pipeline, Task
   import yaml

   # 定义配置
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

   # 定义任务
   pipeline = Pipeline()

   # 任务1：数据加载
   @pipeline.task('load_data')
   def load_data(config):
       data = load_experimental_data(config['data']['path'])
       return {'data': data}

   # 任务2：预处理
   @pipeline.task('preprocess', depends_on=['load_data'])
   def preprocess(inputs, config):
       data = inputs['data']
       preprocessor = DataPreprocessor()
       clean_data = preprocessor.process_pipeline(data)
       return {'clean_data': clean_data}

   # 任务3：模型训练
   @pipeline.task('train', depends_on=['preprocess'])
   def train(inputs, config):
       model = CANN1D(num_neurons=config['model']['num_neurons'])
       trainer = STDPTrainer(model, learning_rate=config['training']['learning_rate'])

       for epoch in range(config['training']['epochs']):
           loss = trainer.train(inputs['clean_data'])

       return {'model': model, 'loss': loss}

   # 任务4：分析
   @pipeline.task('analyze', depends_on=['train'])
   def analyze(inputs, config):
       model = inputs['model']
       analysis_results = analyze_network(model)
       return {'results': analysis_results}

   # 运行流程
   results = pipeline.run(config)

   # 保存结果
   pipeline.save_results(results, output_dir='/results')

关键概念
--------

**流程DAG（有向无环图）**

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

**任务依赖**

- 顺序执行：A→B（B等待A完成）
- 并行执行：A||B（A和B可同时运行）
- 条件执行：根据结果选择分支

实验变化
--------

**1. 多重流程**

.. code-block:: python

   # 并行运行多个pipeline
   configs = load_all_experiments()
   results = parallel_run_pipelines(configs)

**2. 分布式处理**

.. code-block:: python

   # 在集群上运行
   pipeline.set_backend('dask')  # 或 'spark'
   pipeline.run(config)

**3. 容错机制**

.. code-block:: python

   # 自动重试失败的任务
   pipeline.set_retry_policy(max_retries=3, backoff=2.0)

相关API
-------

- :class:`~src.canns.pipeline.Pipeline`
- :class:`~src.canns.pipeline.Task`
- :func:`~src.canns.pipeline.run_parallel`

下一步
------

- :doc:`external_trajectories` - 集成外部数据
- :doc:`parameter_customization` - 参数调优
