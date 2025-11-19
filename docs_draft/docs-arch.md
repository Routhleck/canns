
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

  - 3_01_model_collections
    - 3_01_01_basic
      - CANN1D
      - CANN2D
      - Hierarchical Path Integration Network
      - Theta Sweep Head Direction Cell Network
      - Theta Sweep Grid Cell Network
      - Theta Sweep Place Cell Network
    - 3_01_02_hybrid
      - TODO
    - 3_01_03_brain_inspired
      - AmariHopfield Network
      - LIF
    
  - 3_02_task_generators
    - 3_02_01_tracking
    - 3_02_02_navigation
  - 3_03_analysis_methods
    - 3_03_01_model_analyzer
    - 3_03_02_experimental_data_analyzer
    - 3_03_03_rnn_dynamics_analyzer
  
  - 3_04_brain_inspired_training
