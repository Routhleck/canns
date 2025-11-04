src.canns.data.loaders
======================

.. py:module:: src.canns.data.loaders

.. autoapi-nested-parse::

   Experimental data processing utilities for CANNs.

   This module provides specialized functions for processing experimental data
   typically used in CANN analyses, including ROI data, grid cell data, and
   other neurophysiological _datasets.



Functions
---------

.. autoapisummary::

   src.canns.data.loaders.get_data_summary
   src.canns.data.loaders.load_grid_data
   src.canns.data.loaders.load_roi_data
   src.canns.data.loaders.preprocess_spike_data
   src.canns.data.loaders.validate_grid_data
   src.canns.data.loaders.validate_roi_data


Module Contents
---------------

.. py:function:: get_data_summary(data)

   Get summary statistics for experimental data.

   :param data: ROI data (ndarray) or grid data (dict).
   :type data: ndarray or dict

   :returns: Summary statistics.
   :rtype: dict


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


.. py:function:: preprocess_spike_data(spike_data, time_window = None, min_spike_count = 10)

   Preprocess spike data for analysis.

   :param spike_data: Raw spike data.
   :type spike_data: list or ndarray
   :param time_window: (start, end) time window to filter spikes.
   :type time_window: tuple, optional
   :param min_spike_count: Minimum number of spikes required per neuron.
   :type min_spike_count: int

   :returns: Processed spike data, or None if processing fails.
   :rtype: ndarray or None


.. py:function:: validate_grid_data(data)

   Validate grid data format for 2D CANN analysis.

   :param data: Grid data dictionary.
   :type data: dict

   :returns: True if data is valid, False otherwise.
   :rtype: bool


.. py:function:: validate_roi_data(data)

   Validate ROI data format for 1D CANN analysis.

   :param data: ROI data array.
   :type data: ndarray

   :returns: True if data is valid, False otherwise.
   :rtype: bool


