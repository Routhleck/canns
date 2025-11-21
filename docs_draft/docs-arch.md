
- Why CANNs?
- Basic Intro
  - How to build CANN model?
  - How to generate task data?
  - How to analyze CANN model?
  - How to analyze experimental data?
  - How to train brain-inspired model?
- Core Concepts
  - Overview (Design Philosophy)
  - Model Collections
    - Basic CANN Models
    - Hybrid CANN Models (TODO)
    - Brain-Inspired Models
  - Task Generators
  - Analysis methods
    - Model Analyzer
    - Data Analyzer
      - Experimental Data Analysis
        - CANN1d
        - CANN2d
      - RNN Dynamics Analysis
        - Slow and Fixed Points
  - Brain-Inspired Training
- Full Details Tutorials (FOR ALL PROVIDED CLASSES & APIS)
  - Model Collections
    - Basic CANN Models
      - CANN1D
      - CANN2D
      - Hierarchical Path Integration Model
      - Theta Sweep Models
        - Direction Cell Network
        - Grid Cell Network
        - Place Cell Network
    - Hybrid CANN Models (TODO)
    - Brain-Inspired Models
      - AmariHopfield Model
      - Linear Feedforward Model
      - Spike (LIF) Model
  - Task Generators
    - Tracking
    - Closed-Loop Navigation
    - Open-Loop Navigation
  - Analysis methods
    - Model Analyzer
      - Plot Config
      - Basic
        - Energy Landscape
        - Spike Plot
        - Firing Field
        - Tuning Curve
      - Theta Sweep Analysis
    - Data Analyzer
      - Experimental Data Analysis
      - RNN Dynamics Analysis



Dir Structure

- 0_why_canns.ipynb

- 1_quick_starts

  - 1_00_installation.ipynb
  - 1_01_models.ipynb
  - 1_02_tasks.ipynb
  - 1_03_analyze_model.ipynb
  - 1_04_analyze_data.ipynb
  - 1_05_train.ipynb

- 2_core_concepts

  - 2_00_design_philosophy.ipynb
  - 2_01_model_collections
    - 2_01_01_basic.ipynb
    - 2_01_02_hybrid.ipynb
    - 2_01_03_brain_inspired.ipynb
  - 2_02_task_generators.ipynb
  - 2_03_analysis_methods
    - 2_03_01_model_analyzer.ipynb
    - 2_03_02_experimental_data_analyzer.ipynb
    - 2_03_03_rnn_dynamics_analyzer.ipynb
  - 2_04_brain_inspired_training.ipynb

- 3_full_detail_tutorials

  - Scenario 1: CANN modeling and simulation
  
    - 实现CANN模型与调用内置CANN模型

      > CANNs库中的model都是基于brainstate(JAX)来实现的，所以还是需要简单介绍下brainstate的模型构建逻辑，然后如果想要深入brainstate，可以去看brainstate的readthedocs。这里以CANN1D举例，然后最后要说下内置有丰富CANN模型

    - Task数据生成方法并与CANN模型结合进行仿真模拟
  
      > 这里讲解下如何使用tracking的task生成（这里随便挑一个最简单的Population Coding，然后说明不同方法基本上就是初始化参数不太一样，使用方法基本一致），并且讲解具体如何与一个CANN1D的模型进行结合，使用brainstate的for_loop方法进行仿真模拟
  
    - 使用分析方法来对CANN模型仿真模拟结果进行可视化分析
  
      > 这里介绍analyzer模块中的各种分析可视化方法（尽量涵盖所有的1d model analysis的方法），这里主要用oscillation tracking task来去进行各类方法的讲解。然里用不同的tracking的task来去生成不同的energy landscape。最后用CANN2D的例子来展示下2d对应的plot方法。
  
    - 不同CANN参数的影响
  
      > 在这里尝试讲解下CANN1D的各种参数对模型的影响，就以CANN1D，oscillation tracking任务，energy landscape可视化方法，进行展示。
  
    - Hierarchical Path Integration Network的实现与Navigation Task结合仿真模拟
  
      > 这里大致介绍下Hierarchical Path Integration Network是如何实现的，以及如何使用不同于tracking task的navigation task，来去进行仿真模拟，并且最后用对应的分析可视化方法展示各个的firing heatmap
  
    - Theta Sweep System Model (Head-direction Cell Network与Grid Cell Network)
  
      > 同上，也是介绍该network的大致实现，然后配合navigation task来去进行仿真模拟，最终生成分析的动图和一些静态图
  
    - Theta Sweep Place Cell Network
  
      > 同上
  
  - Scenario 2: Data Analysis
  
    - 实验数据Attractor Structure分析
  
      > 这里仅放下placeholder，todo
  
    - RNN Slowpoint Analysis
  
      > 这里也是placeholder
  
  - Scenario 3: Brain-Inspired Learning
  
    - 实现Brain-Inspired模型与调用内置Brain-Inspired模型
    - 在图像记忆存储任务上训练Brain-Inspired模型
  
  - Scenario 4: End-to-End Pipeline
  
    - Using external trajectory data for theta sweep analysis
