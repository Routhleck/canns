"""
MATLAB Data Loader

Functions and classes for loading neuroscience data from MATLAB .mat files.
"""

import numpy as np
import scipy.io
import h5py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import warnings


@dataclass
class TuningCurve:
    """
    Represents a tuning curve (e.g., head direction or spatial tuning).

    Attributes
    ----------
    bins : np.ndarray
        Bin centers (e.g., angles for HD, positions for spatial)
    rates : np.ndarray
        Firing rates in each bin (Hz)
    mvl : float, optional
        Mean Vector Length (for directional tuning)
    center_of_mass : float, optional
        Preferred direction/position
    peak_rate : float, optional
        Maximum firing rate
    """
    bins: np.ndarray
    rates: np.ndarray
    mvl: Optional[float] = None
    center_of_mass: Optional[float] = None
    peak_rate: Optional[float] = None

    def __post_init__(self):
        """Compute derived properties."""
        if self.peak_rate is None:
            self.peak_rate = np.max(self.rates) if len(self.rates) > 0 else 0.0


@dataclass
class Unit:
    """
    Represents a single neural unit (neuron).

    Attributes
    ----------
    unit_id : int or str
        Unique identifier for this unit
    spike_times : np.ndarray
        Spike times in seconds
    spike_indices : np.ndarray, optional
        Indices into session time array
    hd_tuning : TuningCurve, optional
        Head direction tuning curve
    pos_tuning : TuningCurve, optional
        Spatial position tuning (2D rate map)
    theta_tuning : TuningCurve, optional
        Theta phase tuning
    is_grid : bool, optional
        Whether this is a grid cell
    is_hd : bool, optional
        Whether this is a head direction cell
    gridness_score : float, optional
        Grid cell score
    metadata : dict
        Additional metadata
    """
    unit_id: Any
    spike_times: np.ndarray
    spike_indices: Optional[np.ndarray] = None
    hd_tuning: Optional[TuningCurve] = None
    pos_tuning: Optional[TuningCurve] = None
    theta_tuning: Optional[TuningCurve] = None
    is_grid: Optional[bool] = None
    is_hd: Optional[bool] = None
    gridness_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MATFileLoader:
    """
    Loader for MATLAB .mat files containing neuroscience data.

    Handles both MATLAB v5/v7 files (via scipy.io) and v7.3+ files (via h5py).
    """

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """
        Load a .mat file, automatically detecting the version.

        Parameters
        ----------
        filepath : str
            Path to .mat file

        Returns
        -------
        data : dict
            Dictionary containing the loaded data

        Examples
        --------
        >>> loader = MATFileLoader()
        >>> data = loader.load("example.mat")
        >>> print(data.keys())
        """
        try:
            # Try scipy.io first (works for v5/v7)
            data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
            # Remove MATLAB metadata
            data = {k: v for k, v in data.items() if not k.startswith('__')}
            return data
        except NotImplementedError:
            # Fall back to h5py for v7.3+
            return MATFileLoader._load_h5py(filepath)

    @staticmethod
    def _load_h5py(filepath: str) -> Dict[str, Any]:
        """Load MATLAB v7.3+ file using h5py."""
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                if not key.startswith('#'):
                    data[key] = MATFileLoader._h5py_to_numpy(f[key])
        return data

    @staticmethod
    def _h5py_to_numpy(h5_obj):
        """Recursively convert h5py objects to numpy arrays or dicts."""
        if isinstance(h5_obj, h5py.Dataset):
            return np.array(h5_obj)
        elif isinstance(h5_obj, h5py.Group):
            return {k: MATFileLoader._h5py_to_numpy(v) for k, v in h5_obj.items()}
        else:
            return h5_obj

    @staticmethod
    def load_unit_data(filepath: str) -> List[Unit]:
        """
        Load unit data from a .mat file.

        Expected structure (from unit_data_25953.mat):
        - units: struct array with fields:
          - id or spikeInds or spikeTimes
          - rmf.hd, rmf.pos, rmf.theta (tuning structures)
          - isGrid (boolean)

        Parameters
        ----------
        filepath : str
            Path to unit data .mat file

        Returns
        -------
        units : list of Unit
            List of Unit objects

        Examples
        --------
        >>> loader = MATFileLoader()
        >>> units = loader.load_unit_data("../results/unit_data_25953.mat")
        >>> print(f"Loaded {len(units)} units")
        >>> print(f"Grid cells: {sum(u.is_grid for u in units if u.is_grid)}")
        """
        data = MATFileLoader.load(filepath)

        # Find the units structure
        if 'units' in data:
            units_struct = data['units']
        else:
            raise ValueError("Could not find 'units' in .mat file")

        # Handle both array of structs and single struct
        if not isinstance(units_struct, np.ndarray):
            units_struct = [units_struct]

        units = []
        for i, unit_data in enumerate(units_struct):
            try:
                unit = MATFileLoader._parse_unit_struct(unit_data, unit_id=i)
                units.append(unit)
            except Exception as e:
                warnings.warn(f"Failed to parse unit {i}: {e}")
                continue

        return units

    @staticmethod
    def _parse_unit_struct(unit_struct, unit_id: Any = None) -> Unit:
        """Parse a MATLAB unit structure into a Unit object."""
        # Extract spike times
        spike_times = None
        spike_indices = None

        if hasattr(unit_struct, 'spikeTimes'):
            spike_times = np.asarray(unit_struct.spikeTimes).ravel()
        elif hasattr(unit_struct, 'spikeInds'):
            spike_indices = np.asarray(unit_struct.spikeInds).ravel()

        # Extract unit ID
        if unit_id is None:
            if hasattr(unit_struct, 'id'):
                unit_id = unit_struct.id
            else:
                unit_id = 0

        # Extract tuning curves from rmf (rate map field) structure
        hd_tuning = None
        pos_tuning = None
        theta_tuning = None

        if hasattr(unit_struct, 'rmf'):
            rmf = unit_struct.rmf

            # Head direction tuning
            if hasattr(rmf, 'hd'):
                hd_tuning = MATFileLoader._parse_tuning_curve(rmf.hd)

            # Position tuning
            if hasattr(rmf, 'pos'):
                pos_tuning = MATFileLoader._parse_tuning_curve(rmf.pos)

            # Theta phase tuning
            if hasattr(rmf, 'theta'):
                theta_tuning = MATFileLoader._parse_tuning_curve(rmf.theta)

        # Extract classification flags
        is_grid = None
        if hasattr(unit_struct, 'isGrid'):
            is_grid = bool(unit_struct.isGrid)

        # Create Unit object
        unit = Unit(
            unit_id=unit_id,
            spike_times=spike_times if spike_times is not None else np.array([]),
            spike_indices=spike_indices,
            hd_tuning=hd_tuning,
            pos_tuning=pos_tuning,
            theta_tuning=theta_tuning,
            is_grid=is_grid
        )

        return unit

    @staticmethod
    def _parse_tuning_curve(tuning_struct) -> Optional[TuningCurve]:
        """Parse a MATLAB tuning structure into a TuningCurve object."""
        if tuning_struct is None:
            return None

        try:
            # Extract bins and rates
            bins = None
            rates = None
            mvl = None
            center_of_mass = None

            if hasattr(tuning_struct, 'z'):
                # 'z' typically contains the tuning curve values
                rates = np.asarray(tuning_struct.z)

                # For 1D tuning curves, create default bins
                if rates.ndim == 1:
                    bins = np.linspace(-np.pi, np.pi, len(rates))
                else:
                    # For 2D rate maps, bins might not be meaningful
                    bins = np.arange(rates.shape[0])

            if hasattr(tuning_struct, 'mvl'):
                mvl = float(tuning_struct.mvl)

            if hasattr(tuning_struct, 'centerOfMass'):
                center_of_mass = float(tuning_struct.centerOfMass)

            if rates is not None:
                return TuningCurve(
                    bins=bins,
                    rates=rates,
                    mvl=mvl,
                    center_of_mass=center_of_mass
                )

        except Exception as e:
            warnings.warn(f"Failed to parse tuning curve: {e}")

        return None

    @staticmethod
    def load_example_cells(filepath: str) -> List[Unit]:
        """
        Load example cell data from exampleIdCells.mat format.

        Expected structure:
        - res: struct array with fields:
          - recName, id
          - hdTuning, posTuning
          - tempAcorr (temporal autocorrelation)

        Parameters
        ----------
        filepath : str
            Path to example cells .mat file

        Returns
        -------
        units : list of Unit
            List of Unit objects

        Examples
        --------
        >>> loader = MATFileLoader()
        >>> cells = loader.load_example_cells("../results/exampleIdCells.mat")
        >>> print(f"Loaded {len(cells)} example cells")
        """
        data = MATFileLoader.load(filepath)

        # Find the result structure
        if 'res' in data:
            res_struct = data['res']
        else:
            raise ValueError("Could not find 'res' in .mat file")

        if not isinstance(res_struct, np.ndarray):
            res_struct = [res_struct]

        units = []
        for i, cell_data in enumerate(res_struct):
            try:
                # Extract basic info
                unit_id = cell_data.id if hasattr(cell_data, 'id') else i
                rec_name = cell_data.recName if hasattr(cell_data, 'recName') else ""

                # Extract HD tuning
                hd_tuning = None
                if hasattr(cell_data, 'hdTuning'):
                    hd_struct = cell_data.hdTuning
                    if hasattr(hd_struct, 'z') and hasattr(hd_struct, 'mvl'):
                        rates = np.asarray(hd_struct.z).ravel()
                        bins = np.linspace(-np.pi, np.pi, len(rates))
                        mvl = float(hd_struct.mvl)
                        hd_tuning = TuningCurve(bins=bins, rates=rates, mvl=mvl)

                # Extract position tuning
                pos_tuning = None
                if hasattr(cell_data, 'posTuning'):
                    pos_struct = cell_data.posTuning
                    if hasattr(pos_struct, 'z'):
                        rates = np.asarray(pos_struct.z)
                        # For 2D rate maps, bins are spatial coordinates
                        bins = np.arange(rates.shape[0])
                        pos_tuning = TuningCurve(bins=bins, rates=rates)

                # Create Unit
                unit = Unit(
                    unit_id=unit_id,
                    spike_times=np.array([]),  # Not provided in example format
                    hd_tuning=hd_tuning,
                    pos_tuning=pos_tuning,
                    metadata={'recording': rec_name}
                )
                units.append(unit)

            except Exception as e:
                warnings.warn(f"Failed to parse example cell {i}: {e}")
                continue

        return units


if __name__ == "__main__":
    # Simple tests (will only work if data files exist)
    print("Testing MATLAB data loader...")

    loader = MATFileLoader()

    # Test 1: Try to load example cells
    try:
        print("\nTest 1 - Loading example cells:")
        example_path = "../results/exampleIdCells.mat"
        cells = loader.load_example_cells(example_path)
        print(f"  Loaded {len(cells)} example cells")

        if len(cells) > 0:
            cell = cells[0]
            print(f"  First cell ID: {cell.unit_id}")
            if cell.hd_tuning:
                print(f"  HD tuning MVL: {cell.hd_tuning.mvl:.3f}")
                print(f"  HD tuning curve shape: {cell.hd_tuning.rates.shape}")

    except FileNotFoundError:
        print("  File not found (expected if running outside results directory)")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 2: Try to load unit data
    try:
        print("\nTest 2 - Loading unit data:")
        unit_path = "../results/unit_data_25953.mat"
        units = loader.load_unit_data(unit_path)
        print(f"  Loaded {len(units)} units")

        if len(units) > 0:
            grid_cells = [u for u in units if u.is_grid]
            print(f"  Grid cells: {len(grid_cells)}")
            if len(grid_cells) > 0:
                print(f"  First grid cell ID: {grid_cells[0].unit_id}")

    except FileNotFoundError:
        print("  File not found (expected if running outside results directory)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nData loader tests completed!")
