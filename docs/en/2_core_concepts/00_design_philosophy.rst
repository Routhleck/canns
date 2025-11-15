Design Philosophy
=================

Design Philosophy Handbook for the Continuous Attractor Neural Networks (CANNs) Python
library. Quickly get started with this project by understanding the functionality of each module.

This library provides a unified high-level API around CANNs, enabling users to easily load, analyze, and train
state-of-the-art CANN architectures, thereby helping researchers and developers rapidly conduct experiments
and deploy brain-inspired solutions.

Module Overview
---------------

- ``model`` The built-in model module of this library.

  - ``basic`` Basic CANNs models and their various variants.
  - ``brain_inspired`` Various brain-inspired models.
  - ``hybrid`` Hybrid models combining CANN with ANN or others.

- ``task``
  Task module for CANNs, including task generation, saving, loading, importing, and visualization.
- ``analyzer`` Analysis module, mainly for visualization and plotting.

  - ``model analyzer``
    Focuses on analyzing CANN models, including energy landscapes, firing rates, tuning curves, etc.
  - ``data analyzer`` Focuses on CANN analysis of experimental data or dynamics analysis of virtual RNN models.

- ``trainer`` Training module, providing unified learning and prediction workflows.
- ``pipeline``
  Forms an end-to-end workflow by combining the above modules, minimizing function calls for certain requirements
  and providing user-friendly interfaces.

Detailed Module Explanations
----------------------------

``models``
~~~~~~~~~~

Overview
^^^^^^^^

The model module implements basic CANN models and their variants across different dimensions, brain-inspired models,
and hybrid CANN models. This module is the foundation of the library and can interact with other modules to enable
various application scenarios.

Models are categorized by type:

- Basic Models (:mod:`~src.canns.models.basic`) Basic CANNs models and their variants.
- Brain-Inspired Models (:mod:`~src.canns.models.brain_inspired`) Brain-inspired models.
- Hybrid Models (:mod:`~src.canns.models.hybrid`) Hybrid models combining CANN with ANN or others.

The implementation primarily relies on\ `brainstate <https://brainstate.readthedocs.io>`__\ from the\ `Brain simulation
ecosystem <https://brainmodeling.readthedocs.io/index.html>`__\ .\ ``brainstate``
is the core framework for dynamical systems in the Brain Simulation Ecosystem, built on
JAX/BrainUnit at its foundation. It provides the ``brainstate.nn.Dynamics``
abstraction,\ ``State``/``HiddenState``/``ParamState`` state containers, and
``brainstate.environ`` for unified time-step management, working together with
tools like ``brainstate.transform.for_loop``\ and ``brainstate.random``
to allow writing neural network dynamics that can be both JIT-compiled and support automatic differentiation.
With these interfaces, CANN models only need to describe variables and update equations, while ``brainstate`` 
handles time advancement, parallelization, and random number management, significantly reducing implementation costs.

Usage Examples
^^^^^^^^^^^^^^

The following examples outline the complete workflow for using models in the library, which can be referenced from
``examples/cann/cann1d_oscillatory_tracking.py``\ , ``examples/cann/cann2d_tracking.py``
and ``examples/brain_inspired/hopfield_train.py``\ respectively. Primarily using :class:`~src.canns.models.basic.CANN1D` and :class:`~src.canns.models.basic.CANN2D`
models, :class:`~src.canns.task.tracking.SmoothTracking1D` and :class:`~src.canns.task.tracking.SmoothTracking2D` tasks,
and :class:`~src.canns.analyzer.plotting.PlotConfigs` configuration tools:

.. code:: ipython3

    import brainstate as bst
    from canns.models.basic import CANN1D, CANN2D
    from canns.task.tracking import SmoothTracking1D, SmoothTracking2D
    from canns.analyzer.plotting import (
        PlotConfigs,
        energy_landscape_1d_animation,
        energy_landscape_2d_animation,
    )
    
    bst.environ.set(dt=0.1)
    
    # Create a 1D CANN instance and initialize its state
    cann = CANN1D(num=512)  # 512 neurons
    cann.init_state()       # Initialize neural network state
    
    # Use SmoothTracking1D task here; details will be introduced in later sections
    task_1d = SmoothTracking1D(
        cann_instance=cann,
        Iext=(1.0, 0.75, 2.0, 1.75, 3.0),
        duration=(10.0,) * 4,
        time_step=bst.environ.get_dt(),
    )
    task_1d.get_data()  # Generate task data
    
    # Write a step function that takes stimulus and runs the CANN1D instance
    def step_1d(_, stimulus):
        cann(stimulus)                          # Update CANN state with the provided stimulus
        return cann.u.value, cann.inp.value     # Return neural membrane potential and input
    
    us, inputs = bst.compile.for_loop(step_1d, task_1d.run_steps, task_1d.data) # Compile step function using brainstate's for_loop


.. parsed-literal::

    <SmoothTracking1D> Generating Task data: 400it [00:00, 2409.20it/s]


For brain-inspired models, refer to the following Hopfield example (see
``examples/brain_inspired/hopfield_train.py``\ ), which completes pattern recovery on noisy images. Using :class:`~src.canns.models.brain_inspired.AmariHopfieldNetwork`
model and :class:`~src.canns.trainer.HebbianTrainer` trainer:

.. code:: ipython3

    from canns.models.brain_inspired import AmariHopfieldNetwork
    from canns.trainer import HebbianTrainer
    
    # Create an Amari Hopfield network instance and initialize its state
    model = AmariHopfieldNetwork(num_neurons=128 * 128, asyn=False, activation='sign')
    model.init_state()  # Initialize neural network state
    
    trainer = HebbianTrainer(model) # Create a HebbianTrainer instance; details will be introduced in later sections
    trainer.train(train_patterns)  # train_patterns: List[np.ndarray] of shape (N,), perform training
    denoised = trainer.predict_batch(noisy_patterns, show_sample_progress=True)

Extension Development Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since the base implementation is entirely dependent on
``brainstate``\ , developers extending models are advised to simultaneously consult the official documentation:
https://brainstate.readthedocs.io , with emphasis on the state registration methods of ``nn.Dynamics``\ ,
time management of ``environ.set/get_dt``\ , batch execution paradigm of ``compile.for_loop``\ , and
usage conventions of ``ParamState``/``HiddenState``\ .
These concepts help in writing numerical structures and APIs compatible with existing models.

For basic models
''''''''''''''''

Each model inherits from :class:`~src.canns.models.basic.BasicModel` or :class:`~src.canns.models.basic.BasicModelGroup` and implements the following main methods:

Main work to be completed in basic models:

- Inherit from :class:`~src.canns.models.basic.BasicModel` or :class:`~src.canns.models.basic.BasicModelGroup`\ , call the parent class constructor in
  ``__init__`` (e.g.,
  ``super().__init__(math.prod(shape), **kwargs)``\ ) and save
  ``shape``\ , ``varshape`` and other dimension information;
- Implement ``make_conn()`` to generate connection matrices and assign them to
  ``self.conn_mat`` in the constructor (refer to the Gaussian kernel implementation in ``src/canns/models/basic/cann.py``\ );
- Implement
  ``get_stimulus_by_pos(pos)``\ , which returns external stimuli based on feature space positions for use by the task module;
- Register ``brainstate.HiddenState``/``State`` in ``init_state()`` (common ones include
  ``self.u``\ , ``self.r``\ , ``self.inp``\ ), ensuring the update function can read and write directly;
- Write single-step dynamics in ``update(inputs)``\ , remember to multiply by
  ``brainstate.environ.get_dt()`` to maintain numerical stability;
- When exposing diagnostic quantities or axis information, return them through properties/methods (such as
  ``self.x``\ , ``self.rho``\ ) for reuse by tasks, analyzers, and pipelines.

For brain-inspired models
'''''''''''''''''''''''''

Each model inherits from :class:`~src.canns.models.brain_inspired.BrainInspiredModel` or :class:`~src.canns.models.brain_inspired.BrainInspiredModelGroup` and implements

To extend brain-inspired models (inheriting from :class:`~src.canns.models.brain_inspired.BrainInspiredModel` or
:class:`~src.canns.models.brain_inspired.BrainInspiredModelGroup`\ ), ensure that:

- Register at least the state vector (default ``self.s``\ ) and connection weights
  ``self.W`` in ``init_state()``\ , where ``self.W`` is recommended to use ``brainstate.ParamState`` for
  direct Hebbian learning writes;
- If the weight attribute name is not ``W``\ , override ``weight_attr`` so that
  ``HebbianTrainer`` can locate it;
- Implement ``update(...)`` and ``energy``
  properties to ensure the trainer can run universal prediction loops and determine convergence;
- Implement ``apply_hebbian_learning(patterns)`` when customizing Hebbian rules,
  otherwise rely completely on the trainer's universal implementation;
- If the model supports dynamic resizing, override
  ``resize(num_neurons, preserve_submatrix=True)``\ , referring to the approach in
  ``src/canns/models/brain_inspired/hopfield.py``\ .

For hybrid models
'''''''''''''''''

To be implemented in the future, pending.

``task``
~~~~~~~~

Overview
^^^^^^^^

The task module is primarily used for generating, saving, loading, importing, and visualizing various CANN tasks.
This module provides multiple predefined task types and allows users to create custom tasks to meet specific requirements.

Usage Examples
^^^^^^^^^^^^^^

Taking the one-dimensional tracking task as an example (see
``examples/cann/cann1d_oscillatory_tracking.py``\ ), using :class:`~src.canns.task.tracking.SmoothTracking1D` task:

.. code:: ipython3

    from canns.task.tracking import SmoothTracking1D
    from canns.models.basic import CANN1D
    from canns.analyzer.plotting import energy_landscape_1d_animation, PlotConfigs
    
    # Create a SmoothTracking1D task
    task_st = SmoothTracking1D(
        cann_instance=cann,
        Iext=(1., 0.75, 2., 1.75, 3.),   # External input strengths; for SmoothTracking1D tasks, this represents the initial and final input strengths of different phases, corresponding to the duration below
        duration=(10., 10., 10., 10.),   # Duration of each phase; here indicates the task is divided into 4 phases, each lasting 10.0 time units
        time_step=bst.environ.get_dt(),
    )
    task_st.get_data()  # Generate task data
    
    task_st.data  # Task data, including time series and corresponding external inputs


.. parsed-literal::

    <SmoothTracking1D> Generating Task data: 400it [00:00, 9206.62it/s]




.. parsed-literal::

    array([[0.10189284, 0.09665093, 0.09165075, ..., 0.11314222, 0.10738649,
            0.10189275],
           [0.10079604, 0.09560461, 0.09065294, ..., 0.11193825, 0.10623717,
            0.10079593],
           [0.09970973, 0.0945684 , 0.08966482, ..., 0.11074577, 0.10509886,
            0.09970973],
           ...,
           [9.72546482, 9.68417931, 9.64015198, ..., 9.79967213, 9.76397419,
            9.72546482],
           [9.76497078, 9.72653675, 9.68532467, ..., 9.83337116, 9.80059338,
            9.76497078],
           [9.80151176, 9.76596642, 9.72760582, ..., 9.86403942, 9.8342123 ,
            9.80151081]], shape=(400, 512))



``SmoothTracking1D``/``SmoothTracking2D``
automatically generates smooth trajectories from keypoints. ``task.data`` and ``task.Iext_sequence``
can be directly fed to models or analyzers.

All tasks inherit the base class's ``save_data``/``load_data`` methods for convenient experiment repetition:

.. code:: ipython3

    task.save_data("outputs/tracking_task.npz")
    # ... later or on another machine
    restored = SmoothTracking1D(
        cann_instance=cann_model,
        Iext=(1.0, 0.8, 2.2, 1.5),
        duration=(8.0,) * 3,
        time_step=bst.environ.get_dt(),
    )
    restored.load_data("outputs/tracking_task.npz")


.. parsed-literal::

    Data successfully saved to: outputs/tracking_task.npz
    Data successfully loaded from: outputs/tracking_task.npz


When ``self.data`` is a dataclass (such as
``OpenLoopNavigationData``\ ), the base class automatically splits fields for saving and reconstructs structured
objects when reading.

:class:`~src.canns.task.open_loop_navigation.OpenLoopNavigationTask`
can both self-generate trajectories and support importing experimental data. For specific usage, refer to
``examples/cann/theta_sweep_grid_cell_network.py`` . For importing experimental trajectory data, see
``examples/cann/import_external_trajectory.py``\ :

.. code:: ipython3

    import numpy as np
    import os
    from canns.task.open_loop_navigation import OpenLoopNavigationTask
    
    # Load external position data using numpy
    data = np.load(os.path.join(os.getcwd(), "..", "..", "en", "notebooks", "external_trajectory.npz"))
    positions = data["positions"]  # shape (time_steps, 2)
    times = data["times"]          # shape (time_steps,)
    simulate_time = times[-1] - times[0]
    env_size = 1.8
    dt = 0.1
    
    task = OpenLoopNavigationTask(duration=simulate_time, width=env_size, height=env_size, dt=dt)
    task.import_data(position_data=positions, times=times)  # Import external position data
    task.calculate_theta_sweep_data()   # Calculate theta sweep data
    task.show_trajectory_analysis(save_path="trajectory.png", show=True, smooth_window=50) # Visualize trajectory analysis


.. parsed-literal::

    Successfully imported trajectory data with 800 time steps
    Spatial dimensions: 2D
    Time range: 0.000 to 1.598 s
    Mean speed: 1.395 units/s
    Trajectory analysis saved to: trajectory.png


.. image:: ../../_static/00_design_philosophy_1.png


Extension Development Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can create custom tasks by inheriting from :class:`~src.canns.task.Task` class. The following main methods need to be implemented:

When creating custom tasks, follow these steps:

- Inherit from :class:`~src.canns.task.Task`\ , parse configuration in the constructor and (optionally) specify
  ``data_class``\ ;
- Implement ``get_data()`` to generate or load data and write the result to
  ``self.data`` (can be ``numpy.ndarray`` or a dataclass);
- Provide helper methods like ``import_data(...)`` when importing external data, maintaining
  ``self.data`` structure consistent with ``get_data()`` output;
- Implement ``show_data(show=True, save_path=None)``\ , providing the most important visualization;
- For persistence, directly reuse base class ``save_data``/``load_data`` to avoid reinventing the wheel.

``analyzer``
~~~~~~~~~~~~

Overview
^^^^^^^^

The analysis module provides rich tools for in-depth analysis and visualization of CANN models and experimental data.
This module is divided into two main categories: model analysis and data analysis.

Usage
^^^^^

Model Analysis
''''''''''''''

After matching models with tasks, the analyzer can be used to generate visualizations. For example, using :class:`~src.canns.models.basic.CANN1D` and :class:`~src.canns.analyzer.plotting.PlotConfigs`
to generate 1D tracking visualizations:

.. code:: ipython3

    import brainstate
    from canns.task.tracking import SmoothTracking1D
    from canns.models.basic import CANN1D
    from canns.analyzer.plotting import energy_landscape_1d_animation, PlotConfigs
    
    brainstate.environ.set(dt=0.1)
    
    
    
    # Create a SmoothTracking1D task
    task_st = SmoothTracking1D(
        cann_instance=cann,
        Iext=(1., 0.75, 2., 1.75, 3.),   # External input strengths; for SmoothTracking1D tasks, this represents the initial and final input strengths of different phases, corresponding to the duration below
        duration=(10., 10., 10., 10.),   # Duration of each phase; here indicates the task is divided into 4 phases, each lasting 10.0 time units
        time_step=brainstate.environ.get_dt(),
    )
    task_st.get_data()  # Generate task data
    
    
    # Write a step function that takes inputs and runs the CANN1D instance
    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.inp.value
    
    # Compile step function using brainstate.transform.for_loop
    us, inps = brainstate.transform.for_loop(
        run_step,
        task_st.run_steps,  # Total time steps needed for the task
        task_st.data,       # Task data, here the stimulus generated by SmoothTracking1D
        pbar=brainstate.transform.ProgressBar(10) # Update progress bar every 10 steps
    )
    
    # Configure and generate energy landscape animation (using PlotConfigs configuration, see :class:`~src.canns.analyzer.plotting.PlotConfigs`)
    config = PlotConfigs.energy_landscape_1d_animation(
        time_steps_per_second=100,
        fps=20,
        title='Smooth Tracking 1D',
        xlabel='State',
        ylabel='Activity',
        repeat=True,
        save_path='smooth_tracking_1d.gif',
        show=False
    )

    # Generate energy landscape animation (see :func:`~src.canns.analyzer.plotting.energy_landscape_1d_animation`)
    energy_landscape_1d_animation(
        data_sets={'u': (cann.x, us), 'Iext': (cann.x, inps)},
        config=config
    )

.. figure:: ../../_static/smooth_tracking_1d.gif

For the two-dimensional case, call ``energy_landscape_2d_animation(zs_data=...)``
to output a two-dimensional activity heatmap.

.. figure:: ../../_static/CANN2D_encoding.gif

Data Analysis
''''''''''''''

Experimental data analysis workflows can be directly referenced from two scripts in the repository:

- ``examples/experimental_cann1d_analysis.py``\ : ``load_roi_data()``
  reads sample ROI data, then uses
  ``bump_fits``\ , ``create_1d_bump_animation`` to fit and generate 1D bump
  animations;
- ``examples/experimental_cann2d_analysis.py``\ : after generating embedding results in
  ``embed_spike_trains``\ , combine UMAP with
  ``plot_projection`` for dimensionality reduction visualization, then call
  ``tda_vis``\ , ``decode_circular_coordinates`` and
  ``plot_3d_bump_on_torus`` to complete topological analysis and torus animations.

.. figure:: ../../_static/bump_analysis_demo.gif

.. figure:: ../../_static/torus_bump.gif

Extension Development Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model Analysis
''''''''''''''

Although Analyzer has no unified base class, it is recommended to follow the configuration paradigm in
``src/canns/analyzer/plotting/config.py`` : use :class:`~src.canns.analyzer.plotting.PlotConfig`/:class:`~src.canns.analyzer.plotting.PlotConfigs`
to uniformly manage titles, axes, animation frame rates and other parameters, and receive ``config``
objects in plotting functions. This approach keeps visualization interfaces consistent and makes it convenient
for users to customize default styles.

Data Analysis
''''''''''''''

Similarly, data analysis tools also have no unified base class. Users can create their own data analysis tools
based on specific requirements.

``trainer``
~~~~~~~~~~~

Overview
^^^^^^^^

The training module provides a unified interface for training and evaluating brain-inspired models. Currently,
only Hebbian learning training methods are provided; more brain-inspired training methods will be added in the future.

Usage
^^^^^

Using :class:`~src.canns.trainer.HebbianTrainer` as an example, refer to
``examples/brain_inspired/hopfield_train.py``\ :

.. code:: ipython3

    import numpy as np
    import skimage.data
    from matplotlib import pyplot as plt
    from skimage.color import rgb2gray
    from skimage.filters import threshold_mean
    from skimage.transform import resize
    
    from canns.models.brain_inspired import AmariHopfieldNetwork
    from canns.trainer import HebbianTrainer
    
    np.random.seed(42)
    
    def preprocess_image(img, w=128, h=128) -> np.ndarray:
        """Resize, grayscale (if needed), threshold to binary, then map to {-1,+1}."""
        if img.ndim == 3:
            img = rgb2gray(img)
        img = resize(img, (w, h), anti_aliasing=True)
        img = img.astype(np.float32, copy=False)
        thresh = threshold_mean(img)
        binary = img > thresh
        shift = np.where(binary, 1.0, -1.0).astype(np.float32)
        return shift.reshape(w * h)
    
    # Load training data from skimage
    camera = preprocess_image(skimage.data.camera())
    astronaut = preprocess_image(skimage.data.astronaut())
    horse = preprocess_image(skimage.data.horse().astype(np.float32))
    coffee = preprocess_image(skimage.data.coffee())
    
    data_list = [camera, astronaut, horse, coffee]
    
    # Create an Amari Hopfield network instance and initialize its state (see :class:`~src.canns.models.brain_inspired.AmariHopfieldNetwork`)
    model = AmariHopfieldNetwork(num_neurons=data_list[0].shape[0], asyn=False, activation="sign")
    model.init_state()

    # Create HebbianTrainer and train (see :class:`~src.canns.trainer.HebbianTrainer`)
    trainer = HebbianTrainer(model)
    trainer.train(data_list)
    
    # Generate test data (with noise added)
    def get_corrupted_input(input, corruption_level):
        corrupted = np.copy(input)
        inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
        for i, v in enumerate(input):
            if inv[i]:
                corrupted[i] = -1 * v
        return corrupted
    
    tests = [get_corrupted_input(d, 0.3) for d in data_list]
    
    # Predict corrupted images
    predicted = trainer.predict_batch(tests, show_sample_progress=True)
    
    # Display prediction results
    def plot(data, test, predicted, figsize=(5, 6)):
        def reshape(data):
            dim = int(np.sqrt(len(data)))
            data = np.reshape(data, (dim, dim))
            return data
    
        data = [reshape(d) for d in data]
        test = [reshape(d) for d in test]
        predicted = [reshape(d) for d in predicted]
    
        fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
        for i in range(len(data)):
            if i==0:
                axarr[i, 0].set_title('Train data')
                axarr[i, 1].set_title("Input data")
                axarr[i, 2].set_title('Output data')
    
            axarr[i, 0].imshow(data[i], cmap='gray')
            axarr[i, 0].axis('off')
            axarr[i, 1].imshow(test[i], cmap='gray')
            axarr[i, 1].axis('off')
            axarr[i, 2].imshow(predicted[i], cmap='gray')
            axarr[i, 2].axis('off')
    
        plt.tight_layout()
        plt.savefig("discrete_hopfield_train.png")
        plt.show()
    
    
    plot(data_list, tests, predicted, figsize=(5, 6))


.. parsed-literal::

    Processing samples: 100%|█████████████| 4/4 [00:04<00:00,  1.05s/it, sample=4/4]



.. image:: ../../_static/00_design_philosophy_2.png


Extension Development Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can create custom trainers by inheriting from :class:`~src.canns.trainer.Trainer` class. The following main methods need to be implemented:

To implement a new trainer, inherit from :class:`~src.canns.trainer.Trainer` and:

- Save the target model and progress display configuration in the constructor;
- Implement ``train(self, train_data)``\ , defining the parameter update strategy;
- Implement
  ``predict(self, pattern, *args, **kwargs)``\ , providing single-sample inference logic, using
  ``predict_batch`` to wrap batch inference when necessary;
- Follow the default ``configure_progress``
  convention, allowing users to enable/disable progress bars or compilation modes;
- When trainers need to cooperate with specific models, agree on common attribute names (such as weights, state vectors)
  to ensure interoperability.

Pipeline
~~~~~~~~

Overview
^^^^^^^^

The pipeline module forms an end-to-end workflow by combining models, tasks, analysis, and training modules together,
minimizing function calls for certain requirements and providing user-friendly interfaces.

Usage
^^^^^

End-to-end workflows can be implemented using :class:`~src.canns.pipeline.ThetaSweepPipeline`\ (see
``examples/pipeline/theta_sweep_from_external_data.py``\ ):

.. code:: ipython3

    from canns.pipeline import ThetaSweepPipeline
    
    pipeline = ThetaSweepPipeline(
        trajectory_data=positions,
        times=times,
        env_size=env_size,
    )
    results = pipeline.run(output_dir="theta_sweep_results")


.. parsed-literal::

    🚀 Starting Theta Sweep Pipeline...
    📊 Setting up spatial navigation task...
    Successfully imported trajectory data with 800 time steps
    Spatial dimensions: 2D
    Time range: 0.000 to 1.598 s
    Mean speed: 1.395 units/s
    🧠 Setting up neural networks...
    ⚡ Running theta sweep simulation...


.. parsed-literal::

    /Users/sichaohe/Documents/GitHub/canns/.venv/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    Running for 800 iterations: 100%|██████████| 800/800 [00:10<00:00, 75.01it/s]


.. parsed-literal::

    📈 Generating trajectory analysis...
    Trajectory analysis saved to: theta_sweep_results/trajectory_analysis.png
    📊 Generating population activity plot...
    Plot saved to: theta_sweep_results/population_activity.png
    🎬 Creating theta sweep animation...
    [theta_sweep] Using imageio backend for theta sweep animation (auto-detected).
    [theta_sweep] Detected JAX; using 'spawn' start method to avoid fork-related deadlocks.


.. parsed-literal::

    <theta_sweep> Rendering frames: 100%|██████████| 80/80 [03:42<00:00,  2.78s/it]


.. parsed-literal::

    ✅ Pipeline completed successfully!
    📁 Results saved to: /Users/sichaohe/Documents/GitHub/canns/docs/zh/notebooks/theta_sweep_results



.. image:: ../../_static/00_design_philosophy_3.png


.. figure:: ../../_static/theta_sweep_animation.gif

``results``
returns a dictionary containing animations, trajectory analysis, and raw simulation data, which can be passed
on to custom analysis.

Extension Development Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can create custom pipelines by inheriting from :class:`~src.canns.pipeline.Pipeline` class. The following main methods need to be implemented:

When creating custom pipelines:

- Inherit from :class:`~src.canns.pipeline.Pipeline` and implement
  ``run(...)``\ , returning a dictionary containing main products;
- Call ``prepare_output_dir()`` as needed to manage output directories and use
  ``set_results()`` to cache results for subsequent ``get_results()``\ ;
- Combine calls to models, tasks, and analyzers within ``run()``\ ,
  maintaining clear input/output formats;
- If there are multiple reuse scenarios, call ``reset()`` before execution
  to clean up cached states from the previous run.