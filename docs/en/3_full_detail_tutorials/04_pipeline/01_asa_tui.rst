Tutorial 2: End-to-End ASA TUI Analysis Tutorial (Legacy)
========================================================

This tutorial introduces how the ASA TUI (Attractor Structure Analyzer) integrates data preprocessing, TDA, decoding, and visualization from the ASA pipeline into an interactive terminal workflow. You will learn to complete analyses through the interface and manage results in your working directory.

.. note::
   The ASA TUI is a legacy interface provided for transitional use. It is recommended to prioritize the
   :doc:`ASA GUI Tutorial <03_asa_gui>`.

Tutorial Objectives
-------------------

- Complete an end-to-end analysis using ASA TUI
- Understand input data structure, parameter meanings, and output directories
- Master common analysis modes and their dependencies

Target Audience
---------------

- Researchers needing rapid analysis of experimental neural recordings
- Users who wish to execute the ASA pipeline through a visual interface
- Experimental groups requiring reusable TDA/decoding/visualization workflows

Prerequisites
-------------

- CANNs installed (recommended via ``pip install -e .`` or ``pip install canns``)
- Terminal width of at least 120 columns (minimum 100 columns)
- Prepared ASA or Neuron + Trajectory data

Launching ASA TUI
-----------------

Execute in your project environment:

.. code-block:: bash

   python -m canns.pipeline.asa
   # or
   canns-tui

.. note::
   ``canns-tui`` is now the unified entry point. Upon launch, you first select **ASA** or **Model Gallery**.
   To enter ASA directly, use ``python -m canns.pipeline.asa``.

If a size warning appears after launch, enlarge your terminal window or reduce the font size.

Interface Overview
------------------

The interface is divided into three columns:

- Left: Working directory, run buttons, progress bar, and status information
- Center: Parameter panel + file tree of the working directory
- Right: Result preview and log output

.. figure:: /_static/asa_tui_overview.png
   :alt: ASA TUI Interface Overview
   :width: 90%

   Main interface overview of ASA TUI

Interface Element Inventory
---------------------------

Left Action Panel
^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Workdir Tab**
     - Displays the current working directory path.
   * - **Change Workdir**
     - Opens the working directory selection dialog (equivalent to ``Ctrl-W``).
   * - **Page Indicator**
     - Shows the current page: ``Preprocess`` or ``Analysis``.
   * - **Continue →**
     - Triggers preprocessing and switches to the analysis page (visible only on the preprocessing page).
   * - **← Back**
     - Returns to the preprocessing page (visible only on the analysis page).
   * - **Run Analysis**
     - Runs the current analysis mode (visible only on the analysis page).
   * - **Stop**
     - Requests termination of the current running task (available on the analysis page).
   * - **Progress Bar**
     - Displays preprocessing/analysis progress.
   * - **Status**
     - Shows run status (Idle / Running / Success / Error).

Center Parameter Panel
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Parameters**
     - Parameter control area, switching between preprocessing and analysis pages.
   * - **Input Mode**
     - Select ``ASA File`` or ``Neuron + Traj``.
   * - **Preset**
     - Preset templates: ``Grid`` / ``HD`` / ``None`` (refreshes TDA/GridScore defaults).
   * - **Method**
     - Preprocessing method: ``None`` / ``Embed Spike Trains``.
   * - **Files in Workdir**
     - File tree of the working directory; selecting an image enables preview, and selecting a ``.npz`` file sets it as input.

Right Result Panel
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - **Setup Tab**
     - Displays quick operation hints.
   * - **Results Tab**
     - Result preview and hint messages; automatically switches after analysis completion.
   * - **Log Viewer**
     - Displays real-time logs; also writes to a log file.

.. note::
   Image previews in ASA TUI are rendered as **terminal character graphics** (based on ``climage``).
   If your system has a graphical interface and an image viewer, you can click **Open** to view the actual image externally.

Keyboard Shortcuts
------------------

.. list-table::
   :widths: 30 70

   * - Shortcut
     - Function
   * - ``Ctrl-W``
     - Select working directory
   * - ``Ctrl-R``
     - Run current page action (Preprocess page = Continue, Analysis page = Run Analysis)
   * - ``F5``
     - Refresh preview
   * - ``?``
     - Open help
   * - ``Esc``
     - Exit application
   * - ``Tab``
     - Switch focus between panels

Workflow Overview
-----------------

1. Select a working directory (containing data files)
2. Choose input mode and data files
3. Configure preprocessing parameters
4. Navigate to the analysis page and select an analysis mode
5. Run the analysis and view Results output

Step 1: Select Working Directory
--------------------------------

Click the **Change Workdir** button on the left or press ``Ctrl-W`` to select a directory containing your data.
The file tree will refresh synchronously, facilitating input data selection and result inspection.

.. figure:: /_static/asa_tui_workdir.png
   :alt: Select Working Directory
   :width: 85%

   Working directory selection dialog

Working Directory Notes
^^^^^^^^^^^^^^^^^^^^^^^

- The working directory determines the root location for the **file tree** and **result output directory**.
- Switching the working directory resets the current input state; you must reselect data files.
- Press ``F5`` to refresh the file tree and current path display.

Step 2: Select Input Mode and Files
----------------------------------

Input mode is located in the **Input Mode** dropdown:

- **ASA File**: A single ``.npz`` file containing ``spike`` and ``t`` (recommended to include ``x``/``y``).
- **Neuron + Traj**: Separate neuron and trajectory files.

ASA File Format
^^^^^^^^^^^^^^^

Must contain at least ``spike`` and ``t``:

- ``spike``: A dense ``T x N`` matrix, or a spike data structure that can be embedded
- ``t``: Time series (synchronized with ``spike``)
- Optional: ``x`` / ``y`` for trajectory-related analyses

.. code-block:: python

   import numpy as np
   np.savez(
       "session_asa.npz",
       spike=spike,
       t=t,
       x=x,
       y=y,
   )

After selecting a ``.npz`` file in the file tree, the log area will indicate that an ASA file has been chosen.

Neuron + Traj Mode
^^^^^^^^^^^^^^^^^^

Requires two types of ``.npz`` files:

- **Neuron**: Contains ``spike`` (or the entire file is a spike array)
- **Traj**: Contains ``x`` / ``y`` / ``t``

.. code-block:: python

   np.savez("neuron.npz", spike=spike)
   np.savez("traj.npz", x=x, y=y, t=t)

.. note::
   The current TUI version's file tree only supports direct selection of ASA files. For Neuron + Traj mode,
   it is recommended to first merge them into an ASA ``.npz`` file via script, or extend the selection logic as needed.

File Tree and Preview
^^^^^^^^^^^^^^^^^^^^^

- When **ASA File** mode is selected and a ``.npz`` file is chosen, it sets the ASA input file.
- Selecting ``.png/.jpg/.jpeg/.gif/.bmp`` files enables image preview on the right.
- After results are generated, the file tree automatically refreshes for easy browsing of the output directory.

Step 3: Preprocessing Configuration
-----------------------------------

**Method** offers two options:

- ``None``: Assumes input is already a dense ``T x N`` matrix
- ``Embed Spike Trains``: Embeds spike times into a dense matrix (recommended)

Preset Explanation
^^^^^^^^^^^^^^^^^^

- ``Grid`` / ``HD`` restores TDA and GridScore parameters to their respective defaults.
- ``None`` makes no changes, preserving current parameter inputs.
- Manually adjusted parameters may be overwritten when switching presets.

Key Parameter Descriptions:

.. list-table::
   :widths: 25 75

   * - Parameter
     - Description
   * - ``res``
     - Embedding resolution / sampling length
   * - ``dt``
     - Sampling time step
   * - ``sigma``
     - Smoothing kernel scale
   * - ``smooth``
     - Whether to apply smoothing
   * - ``speed_filter``
     - Whether to filter by speed
   * - ``min_speed``
     - Minimum speed threshold

.. note::
   The above parameters are editable only when ``Embed Spike Trains`` is selected; they are disabled when ``None`` is chosen.

After configuration, click **Continue →** to proceed to the analysis page.

.. figure:: /_static/asa_tui_preprocess.png
   :alt: Preprocessing Parameter Configuration
   :width: 90%

   Preprocessing parameter section

Preprocessing Page Button Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Continue →**: Triggers the preprocessing task; automatically switches to the analysis page upon completion.
- Buttons are locked during preprocessing to prevent duplicate submissions.
- If required fields are missing from input data, an error dialog will appear.

Step 4: Select Analysis Mode
----------------------------

The analysis page supports the following modes:

- **TDA**: Persistent homology analysis, generating barcodes
- **CohoMap**: Phase decoding and cohomap based on TDA
- **PathCompare**: Comparison of real vs. decoded trajectories
- **CohoSpace**: Phase manifold projection and trajectory visualization
- **FR**: Population firing rate heatmap
- **FRM**: Single-neuron firing rate map
- **GridScore**: Grid cell metric analysis

After selecting an analysis mode, the parameter panel automatically switches to the corresponding parameter group; **Decode / CohoMap** parameters are shared among CohoMap, PathCompare, and CohoSpace modes.

Dependencies:

- CohoMap requires prior completion of TDA
- PathCompare / CohoSpace depend on CohoMap

Analysis Parameter Details
^^^^^^^^^^^^^^^^^^^^^^^^^^

TDA Parameters
~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``dim``
     - Embedding dimension
   * - ``num_times``
     - Number of samples
   * - ``active_times``
     - Active time duration
   * - ``k``
     - Number of neighbors
   * - ``n_points``
     - Number of sample points
   * - ``metric``
     - Distance metric (cosine / euclidean / correlation)
   * - ``nbs``
     - Neighborhood sampling count
   * - ``maxdim``
     - Maximum homology dimension
   * - ``coeff``
     - Homology coefficient
   * - ``do_shuffle``
     - Whether to shuffle randomly
   * - ``num_shuffles``
     - Number of shuffles (editable only when ``do_shuffle`` is enabled)
   * - ``standardize``
     - Whether to use StandardScaler for normalization

Decode / CohoMap Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``decode_version``
     - Decoding version: ``v2`` (multi) or ``v0`` (legacy)
   * - ``num_circ``
     - Expected number of circles
   * - ``cohomap_subsample``
     - Cohomap trajectory subsampling
   * - ``real_ground`` / ``real_of``
     - Enabled only in ``v0`` mode

.. note::
   When ``decode_version`` is ``v2``, ``real_ground`` and ``real_of`` are disabled.

PathCompare Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``use_box``
     - Use ``coordsbox`` / ``times_box`` cropping
   * - ``interp_to_full``
     - Available only when ``use_box`` is enabled
   * - ``dim_mode``
     - ``1d`` or ``2d``
   * - ``dim`` / ``dim1`` / ``dim2``
     - Dimension selection (enabled based on ``dim_mode``)
   * - ``coords_key`` / ``times_box_key``
     - Optional field name override
   * - ``slice_mode``
     - ``time`` or ``index``
   * - ``tmin`` / ``tmax``
     - Time slicing (``-1`` for auto)
   * - ``imin`` / ``imax``
     - Index slicing (``-1`` for auto)
   * - ``stride``
     - Sampling stride
   * - ``theta_scale``
     - Angle unit (rad / deg / unit / auto)

.. note::
   When ``dim_mode`` is ``1d``, only ``dim`` is enabled; when ``2d``, ``dim1``/``dim2`` are enabled.
   When ``slice_mode`` is ``time``, ``tmin``/``tmax`` are enabled; when ``index``, ``imin``/``imax`` are enabled.
   When ``use_box`` is disabled, ``interp_to_full`` is disabled.

CohoSpace Parameters
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``dim_mode``
     - ``1d`` or ``2d``
   * - ``dim`` / ``dim1`` / ``dim2``
     - Dimension selection (enabled based on ``dim_mode``)
   * - ``mode``
     - Use ``fr`` or ``spike`` signals
   * - ``top_percent``
     - Percentage for selecting highly active neurons
   * - ``view``
     - ``both`` / ``single`` / ``population``
   * - ``neuron_id``
     - Specifiable in single-neuron view
   * - ``subsample``
     - Trajectory subsampling ratio
   * - ``unfold``
     - ``square`` or ``skew`` unfolding
   * - ``skew_show_grid`` / ``skew_tiles``
     - Enabled only when ``unfold=skew``

.. note::
   When ``dim_mode`` is ``1d``, only ``dim`` is enabled; when ``2d``, ``dim1``/``dim2`` are enabled.
   When ``unfold`` is ``skew``, ``skew_show_grid`` and ``skew_tiles`` are enabled.

FR Parameters
~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``neuron_start`` / ``neuron_end``
     - Neuron range (blank for all)
   * - ``time_start`` / ``time_end``
     - Time range (blank for all)
   * - ``mode``
     - ``fr`` or ``spike``
   * - ``normalize``
     - ``zscore_per_neuron`` / ``minmax_per_neuron`` / ``none``

.. note::
   Leaving ``neuron_start`` / ``neuron_end`` and ``time_start`` / ``time_end`` blank implies full range.

FRM Parameters
~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``neuron_id``
     - Neuron ID to plot
   * - ``bins``
     - Number of spatial bins
   * - ``min_occupancy``
     - Minimum occupancy threshold
   * - ``smoothing``
     - Whether to smooth
   * - ``smooth_sigma``
     - Smoothing scale
   * - ``mode``
     - ``fr`` or ``spike``

GridScore Parameters
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70

   * - ``annulus_inner`` / ``annulus_outer``
     - Inner and outer radii of annular region
   * - ``bin_size``
     - Spatial bin size
   * - ``smooth_sigma``
     - Smoothing scale

Step 5: Run and View Results
----------------------------

Click **Run Analysis** or press ``Ctrl-R`` to start the analysis. Progress and logs are displayed in real time.
Final visualization results are previewed in the image preview widget.

.. figure:: /_static/asa_tui_results.png
   :alt: Result Preview
   :width: 90%

   Analysis results and log area

Image Preview Widget
^^^^^^^^^^^^^^^^^^^^

In the Image Preview area of the **Results** tab, you can directly operate:

- **Load**: Reads the path from the input box and previews it within TUI; supports paths relative to the workdir.
- **Open**: Invokes the system image viewer to open the original image (requires a graphical OS interface).
- **Zoom + / Zoom -**: Zooms in or out of the terminal character graphic preview.
- **Fit**: Resets zoom level and fits the current preview area.
- **↑/↓/←/→**: Pans within the preview area (affects view only, not the image file).

.. note::
   If ``climage`` is not installed, the preview area displays a filename hint; installation is recommended for character graphic previews.

Final results are saved by default in:

``<workdir>/Results/<dataset_id>/``

where ``dataset_id`` is composed of the input filename and a hash prefix.
Common output examples:

- ``TDA/``: ``barcode.png``, ``persistence.npz``
- ``CohoMap/``: ``cohomap.png``, ``decoding.npz``
- ``PathCompare/``: ``path_compare.png``
- ``CohoSpace/``: ``cohospace_trajectory.png``, (optional) ``cohospace_neuron_*.png``
- ``FR/``: ``fr_heatmap.png``
- ``FRM/``: ``frm_neuron_<id>.png``
- ``GridScore/``: ``gridscore_distribution.png``, ``gridscore.npz``

Log files are written by default to:

``<workdir>/Results/<dataset_id>/asa_tui.log``

You can also preview image files by selecting them in the file tree.

Button and Status Execution Logic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Run Analysis** reads current page parameters and starts the analysis task.
- During execution, **Stop** can request task cancellation; the log will indicate cancellation status.
- After preprocessing and analysis complete, the file tree is automatically refreshed and the results page is activated.
- Result previews prioritize displaying artifacts like ``barcode``, ``cohomap``, ``path_compare``, ``cohospace_trajectory``,
  ``fr_heatmap``, ``frm``, and ``distribution``.

Dialogs and Help
----------------

- A warning dialog appears if the terminal size is insufficient, recommending a larger size.
- An error dialog appears for parameter or input errors, showing the cause.
- Press ``?`` to open the help screen, listing shortcuts and workflow instructions.