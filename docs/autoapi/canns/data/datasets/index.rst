canns.data.datasets
===================

.. py:module:: canns.data.datasets

.. autoapi-nested-parse::

   Universal data loading utilities for CANNs.

   This module provides generic functions to download and load data from URLs,
   with specialized support for CANNs example datasets.



Attributes
----------

.. autoapisummary::

   canns.data.datasets.BASE_URL
   canns.data.datasets.DATASETS
   canns.data.datasets.DEFAULT_DATA_DIR
   canns.data.datasets.HAS_DOWNLOAD_DEPS
   canns.data.datasets.HAS_NUMPY
   canns.data.datasets.HUGGINGFACE_REPO
   canns.data.datasets.LEFT_RIGHT_DATASET_DIR


Functions
---------

.. autoapisummary::

   canns.data.datasets.compute_file_hash
   canns.data.datasets.detect_file_type
   canns.data.datasets.download_dataset
   canns.data.datasets.download_file_with_progress
   canns.data.datasets.get_data_dir
   canns.data.datasets.get_dataset_path
   canns.data.datasets.get_huggingface_upload_guide
   canns.data.datasets.get_left_right_data_session
   canns.data.datasets.get_left_right_npz
   canns.data.datasets.list_datasets
   canns.data.datasets.load
   canns.data.datasets.load_file
   canns.data.datasets.quick_setup


Module Contents
---------------

.. py:function:: compute_file_hash(filepath)

   Compute SHA256 hash of a file.


.. py:function:: detect_file_type(filepath)

   Detect file type based on extension.


.. py:function:: download_dataset(dataset_key, force = False)

   Download a specific dataset.

   :param dataset_key: Key of the dataset to download (e.g., 'grid_1', 'roi_data').
   :type dataset_key: str
   :param force: Whether to force re-download if file already exists.
   :type force: bool

   :returns: Path to downloaded file if successful, None otherwise.
   :rtype: Path or None


.. py:function:: download_file_with_progress(url, filepath, chunk_size = 8192)

   Download a file with progress bar.


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


.. py:function:: get_left_right_data_session(session_id, auto_download = True, force = False)

   Download and return files for a Left_Right_data_of session.

   :param session_id: Session folder name, e.g. "24365_2".
   :type session_id: str
   :param auto_download: Whether to download missing files automatically.
   :type auto_download: bool
   :param force: Whether to force re-download of existing files.
   :type force: bool

   :returns: Mapping with keys: "manifest", "full_file", "module_files".
   :rtype: dict or None


.. py:function:: get_left_right_npz(session_id, filename, auto_download = True, force = False)

   Download and return a specific Left_Right_data_of NPZ file.

   :param session_id: Session folder name, e.g. "26034_3".
   :type session_id: str
   :param filename: File name inside the session folder, e.g.
                    "26034_3_ASA_mec_gridModule02_n104_cm.npz".
   :type filename: str
   :param auto_download: Whether to download the file if missing.
   :type auto_download: bool
   :param force: Whether to force re-download of existing files.
   :type force: bool

   :returns: Path to the requested file if available, None otherwise.
   :rtype: Path or None


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


.. py:function:: load_file(filepath, file_type = None)

   Load data from file based on file type.

   :param filepath: Path to the data file.
   :type filepath: Path
   :param file_type: Force specific file type. If None, auto-detect from extension.
   :type file_type: str, optional

   :returns: Loaded data.
   :rtype: Any


.. py:function:: quick_setup()

   Quick setup function to get datasets ready.

   :returns: True if successful, False otherwise.
   :rtype: bool


.. py:data:: BASE_URL
   :value: 'https://huggingface.co/datasets/canns-team/data-analysis-datasets/resolve/main/'


.. py:data:: DATASETS

.. py:data:: DEFAULT_DATA_DIR

.. py:data:: HAS_DOWNLOAD_DEPS
   :value: True


.. py:data:: HAS_NUMPY
   :value: True


.. py:data:: HUGGINGFACE_REPO
   :value: 'canns-team/data-analysis-datasets'


.. py:data:: LEFT_RIGHT_DATASET_DIR
   :value: 'Left_Right_data_of'


