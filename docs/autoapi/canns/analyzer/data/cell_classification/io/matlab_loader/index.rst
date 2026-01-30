canns.analyzer.data.cell_classification.io.matlab_loader
========================================================

.. py:module:: canns.analyzer.data.cell_classification.io.matlab_loader

.. autoapi-nested-parse::

   MATLAB Data Loader

   Functions and classes for loading neuroscience data from MATLAB .mat files.



Attributes
----------

.. autoapisummary::

   canns.analyzer.data.cell_classification.io.matlab_loader.loader


Classes
-------

.. autoapisummary::

   canns.analyzer.data.cell_classification.io.matlab_loader.MATFileLoader
   canns.analyzer.data.cell_classification.io.matlab_loader.TuningCurve
   canns.analyzer.data.cell_classification.io.matlab_loader.Unit


Module Contents
---------------

.. py:class:: MATFileLoader

   Loader for MATLAB .mat files containing neuroscience data.

   Handles both MATLAB v5/v7 files (via scipy.io) and v7.3+ files (via h5py).


   .. py:method:: load(filepath)
      :staticmethod:


      Load a .mat file, automatically detecting the version.

      :param filepath: Path to .mat file
      :type filepath: str

      :returns: **data** -- Dictionary containing the loaded data
      :rtype: dict

      .. rubric:: Examples

      >>> loader = MATFileLoader()
      >>> data = loader.load("example.mat")
      >>> print(data.keys())



   .. py:method:: load_example_cells(filepath)
      :staticmethod:


      Load example cell data from exampleIdCells.mat format.

      Expected structure:
      - res: struct array with fields:
        - recName, id
        - hdTuning, posTuning
        - tempAcorr (temporal autocorrelation)

      :param filepath: Path to example cells .mat file
      :type filepath: str

      :returns: **units** -- List of Unit objects
      :rtype: list of Unit

      .. rubric:: Examples

      >>> loader = MATFileLoader()
      >>> cells = loader.load_example_cells("../results/exampleIdCells.mat")
      >>> print(f"Loaded {len(cells)} example cells")



   .. py:method:: load_unit_data(filepath)
      :staticmethod:


      Load unit data from a .mat file.

      Expected structure (from unit_data_25953.mat):
      - units: struct array with fields:
        - id or spikeInds or spikeTimes
        - rmf.hd, rmf.pos, rmf.theta (tuning structures)
        - isGrid (boolean)

      :param filepath: Path to unit data .mat file
      :type filepath: str

      :returns: **units** -- List of Unit objects
      :rtype: list of Unit

      .. rubric:: Examples

      >>> loader = MATFileLoader()
      >>> units = loader.load_unit_data("../results/unit_data_25953.mat")
      >>> print(f"Loaded {len(units)} units")
      >>> print(f"Grid cells: {sum(u.is_grid for u in units if u.is_grid)}")



.. py:class:: TuningCurve

   Represents a tuning curve (e.g., head direction or spatial tuning).

   .. attribute:: bins

      Bin centers (e.g., angles for HD, positions for spatial)

      :type: np.ndarray

   .. attribute:: rates

      Firing rates in each bin (Hz)

      :type: np.ndarray

   .. attribute:: mvl

      Mean Vector Length (for directional tuning)

      :type: float, optional

   .. attribute:: center_of_mass

      Preferred direction/position

      :type: float, optional

   .. attribute:: peak_rate

      Maximum firing rate

      :type: float, optional


   .. py:method:: __post_init__()

      Compute derived properties.



   .. py:attribute:: bins
      :type:  numpy.ndarray


   .. py:attribute:: center_of_mass
      :type:  float | None
      :value: None



   .. py:attribute:: mvl
      :type:  float | None
      :value: None



   .. py:attribute:: peak_rate
      :type:  float | None
      :value: None



   .. py:attribute:: rates
      :type:  numpy.ndarray


.. py:class:: Unit

   Represents a single neural unit (neuron).

   .. attribute:: unit_id

      Unique identifier for this unit

      :type: int or str

   .. attribute:: spike_times

      Spike times in seconds

      :type: np.ndarray

   .. attribute:: spike_indices

      Indices into session time array

      :type: np.ndarray, optional

   .. attribute:: hd_tuning

      Head direction tuning curve

      :type: TuningCurve, optional

   .. attribute:: pos_tuning

      Spatial position tuning (2D rate map)

      :type: TuningCurve, optional

   .. attribute:: theta_tuning

      Theta phase tuning

      :type: TuningCurve, optional

   .. attribute:: is_grid

      Whether this is a grid cell

      :type: bool, optional

   .. attribute:: is_hd

      Whether this is a head direction cell

      :type: bool, optional

   .. attribute:: gridness_score

      Grid cell score

      :type: float, optional

   .. attribute:: metadata

      Additional metadata

      :type: dict


   .. py:attribute:: gridness_score
      :type:  float | None
      :value: None



   .. py:attribute:: hd_tuning
      :type:  TuningCurve | None
      :value: None



   .. py:attribute:: is_grid
      :type:  bool | None
      :value: None



   .. py:attribute:: is_hd
      :type:  bool | None
      :value: None



   .. py:attribute:: metadata
      :type:  dict[str, Any]


   .. py:attribute:: pos_tuning
      :type:  TuningCurve | None
      :value: None



   .. py:attribute:: spike_indices
      :type:  numpy.ndarray | None
      :value: None



   .. py:attribute:: spike_times
      :type:  numpy.ndarray


   .. py:attribute:: theta_tuning
      :type:  TuningCurve | None
      :value: None



   .. py:attribute:: unit_id
      :type:  Any


.. py:data:: loader

