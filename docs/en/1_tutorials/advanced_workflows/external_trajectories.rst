Integrating External Trajectory Data
=====================================

Scenario Description
--------------------

You have real movement trajectories of animals (from video tracking, position chips, etc.) and want to use them to drive neural network simulations to study how the network encodes animal locations in real behavior.

What You Will Learn
--------------------

- Loading and processing trajectory data
- Synchronizing trajectories with models
- Real-time and offline simulations
- Multimodal data fusion
- Behavioral state recognition

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from canns.task.open_loop_navigation import OpenLoopNavigationTask

   class ExternalTrajectoryTask(OpenLoopNavigationTask):
       def __init__(self, trajectory_file, dt=0.05):
           """Load external trajectories from file"""
           self.data = self.load_trajectory(trajectory_file)
           self.dt = dt

       def load_trajectory(self, file):
           """Support multiple formats"""
           if file.endswith('.csv'):
               return np.loadtxt(file, delimiter=',', skiprows=1)
           elif file.endswith('.npz'):
               return np.load(file)['position']
           elif file.endswith('.mat'):
               import scipy.io as sio
               return sio.loadmat(file)['position']

       def get_input_at_time(self, t):
           """Get trajectory at specific time"""
           idx = int(t / self.dt)
           if idx < len(self.data):
               return self.data[idx]
           return self.data[-1]

   # Usage
   task = ExternalTrajectoryTask('animal_trajectory.csv')
   network = HierarchicalNetwork(num_module=5, num_place=30)

   for t in np.arange(0, task.duration, task.dt):
       position = task.get_input_at_time(t)
       velocity = np.gradient(task.data[int(t/task.dt)])

       network(velocity=velocity, loc=position)

Key Concepts
------------

**Trajectory Data Formats**

- CSV: time, x, y columns
- NPZ: numpy compressed format
- H5: HDF5 format
- MAT: MATLAB format

**Synchronization Issues**

- Sampling rate matching
- Time alignment
- Coordinate system transformation

Experimental Variations
-----------------------

**1. Trajectories of Multiple Animals**

.. code-block:: python

   for animal_id in range(num_animals):
       trajectory = load_animal_trajectory(animal_id)
       # Analyze each animal separately

**2. Multimodal Data Fusion**

.. code-block:: python

   # Position + brain imaging
   # Position + behavioral video
   # Position + neural recordings

Related API
-----------

- :class:`~src.canns.task.open_loop_navigation.ExternalTrajectoryTask`

Next Steps
----------

- :doc:`parameter_customization` - Parameter optimization
- :doc:`../spatial_navigation/complex_environments` - Handling complex environments
