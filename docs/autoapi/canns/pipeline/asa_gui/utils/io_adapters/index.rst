[ANONYMOUS_PROJECT].pipeline.asa_gui.utils.io_adapters
========================================

.. py:module:: [ANONYMOUS_PROJECT].pipeline.asa_gui.utils.io_adapters

.. autoapi-nested-parse::

   I/O adapters for loading ASA GUI inputs.



Functions
---------

.. autoapisummary::

   [ANONYMOUS_PROJECT].pipeline.asa_gui.utils.io_adapters.load_asa_npz
   [ANONYMOUS_PROJECT].pipeline.asa_gui.utils.io_adapters.pack_neuron_traj_to_asa


Module Contents
---------------

.. py:function:: load_asa_npz(path)

.. py:function:: pack_neuron_traj_to_asa(neuron_path, traj_path, out_path)

   Pack neuron and trajectory arrays into ASA-style .npz.

   Minimal format support: neuron_path is .npy (T,N), traj_path is .npy (T,2).


