src.canns.data
==============

.. py:module:: src.canns.data

.. autoapi-nested-parse::

   Data utilities for CANNs.

   This module provides dataset management, loading, and downloading utilities.
   It consolidates data-related functionality previously scattered across the codebase.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/src/canns/data/datasets/index
   /autoapi/src/canns/data/loaders/index


Attributes
----------

.. autoapisummary::

   src.canns.data.DATASETS
   src.canns.data.DEFAULT_DATA_DIR
   src.canns.data.HUGGINGFACE_REPO


Functions
---------

.. autoapisummary::

   src.canns.data.download_dataset
   src.canns.data.get_data_dir
   src.canns.data.get_dataset_path
   src.canns.data.get_huggingface_upload_guide
   src.canns.data.list_datasets
   src.canns.data.load
   src.canns.data.load_grid_data
   src.canns.data.load_roi_data
   src.canns.data.quick_setup


Package Contents
----------------

.. py:function:: download_dataset(dataset_key, force = False)

   Download a specific dataset.

   :param dataset_key: Key of the dataset to download (e.g., 'grid_1', 'roi_data').
   :type dataset_key: str
   :param force: Whether to force re-download if file already exists.
   :type force: bool

   :returns: Path to downloaded file if successful, None otherwise.
   :rtype: Path or None


.. py:function:: get_data_dir()

   Get the data directory, creating it if necessary.


.. py:function:: get_dataset_path(dataset_key, auto_setup = True)

   Get path to a dataset, downloading/setting up if necessary.

   :param dataset_key: Key of the dataset.
   :type dataset_key: str
   :param auto_setup: Whether to automatically attempt setup if dataset not found.
   :type auto_setup: bool

   :returns: Path to dataset file if available, None otherwise.
   :rtype: Path or None


.. py:function:: get_huggingface_upload_guide()

   Get guide for uploading datasets to Hugging Face.

   :returns: Upload guide text.
   :rtype: str


.. py:function:: list_datasets()

   List available datasets with descriptions.


.. py:function:: load(url, cache_dir = None, force_download = False, file_type = None)

   Universal data loading function that downloads and reads data from URLs.

   :param url: URL to download data from.
   :type url: str
   :param cache_dir: Directory to cache downloaded files. If None, uses temporary directory.
   :type cache_dir: str or Path, optional
   :param force_download: Force re-download even if file exists in cache.
   :type force_download: bool
   :param file_type: Force specific file type ('text', 'numpy', 'json', 'pickle', 'hdf5').
                     If None, auto-detect from file extension.
   :type file_type: str, optional

   :returns: Loaded data.
   :rtype: Any

   .. rubric:: Examples

   >>> # Load numpy data
   >>> data = load('https://example.com/data.npz')
   >>>
   >>> # Load text data with custom cache
   >>> data = load('https://example.com/data.txt', cache_dir='./cache')
   >>>
   >>> # Force specific file type
   >>> data = load('https://example.com/data.bin', file_type='numpy')


.. py:function:: load_grid_data(source = None, dataset_key = 'grid_1')

   Load grid cell data for 2D CANN analysis.

   :param source: Data source. Can be:
                  - URL string: downloads and loads from URL
                  - Path: loads from local file
                  - None: uses default CANNs dataset
   :type source: str, Path, or None
   :param dataset_key: Which default dataset to use ('grid_1' or 'grid_2') when source is None.
   :type dataset_key: str

   :returns: Dictionary containing spike data and metadata if successful, None otherwise.
             Expected keys: 'spike', 't', and optionally 'x', 'y' for position data.
   :rtype: dict or None

   .. rubric:: Examples

   >>> # Load default dataset
   >>> grid_data = load_grid_data()
   >>>
   >>> # Load from URL
   >>> grid_data = load_grid_data('https://example.com/grid_data.npz')
   >>>
   >>> # Load specific default dataset
   >>> grid_data = load_grid_data(dataset_key='grid_2')


.. py:function:: load_roi_data(source = None)

   Load ROI data for 1D CANN analysis.

   :param source: Data source. Can be:
                  - URL string: downloads and loads from URL
                  - Path: loads from local file
                  - None: uses default CANNs dataset
   :type source: str, Path, or None

   :returns: ROI data array if successful, None otherwise.
   :rtype: ndarray or None

   .. rubric:: Examples

   >>> # Load default dataset
   >>> roi_data = load_roi_data()
   >>>
   >>> # Load from URL
   >>> roi_data = load_roi_data('https://example.com/roi_data.txt')
   >>>
   >>> # Load from local file
   >>> roi_data = load_roi_data('./my_roi_data.txt')


.. py:function:: quick_setup()

   Quick setup function to get datasets ready.

   :returns: True if successful, False otherwise.
   :rtype: bool


.. py:data:: DATASETS

.. py:data:: DEFAULT_DATA_DIR

.. py:data:: HUGGINGFACE_REPO
   :value: 'canns-team/data-analysis-datasets'


