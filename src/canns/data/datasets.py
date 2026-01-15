"""
Universal data loading utilities for CANNs.

This module provides generic functions to download and load data from URLs,
with specialized support for CANNs example datasets.
"""

import hashlib
import tempfile
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import requests
    from tqdm import tqdm

    HAS_DOWNLOAD_DEPS = True
except ImportError:
    HAS_DOWNLOAD_DEPS = False
    warnings.warn(
        "Download dependencies not available. Install with: pip install requests tqdm",
        ImportWarning,
        stacklevel=2,
    )

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".canns" / "data"

# URLs for datasets on Hugging Face
HUGGINGFACE_REPO = "canns-team/data-analysis-datasets"
BASE_URL = f"https://huggingface.co/datasets/{HUGGINGFACE_REPO}/resolve/main/"

# Dataset registry with metadata
DATASETS = {
    "roi_data": {
        "filename": "ROI_data.txt",
        "description": "1D CANN ROI data for bump analysis",
        "size_mb": 0.7,
        "format": "txt",
        "usage": "1D CANN analysis, MCMC bump fitting",
        "sha256": None,
        "url": f"{BASE_URL}ROI_data.txt",
    },
    "grid_1": {
        "filename": "grid_1.npz",
        "description": "Grid cell spike data with position information",
        "size_mb": 8.7,
        "format": "npz",
        "usage": "2D CANN analysis, topological data analysis, circular coordinate decoding",
        "sha256": None,
        "url": f"{BASE_URL}grid_1.npz",
    },
    "grid_2": {
        "filename": "grid_2.npz",
        "description": "Second grid cell dataset",
        "size_mb": 4.5,
        "format": "npz",
        "usage": "2D CANN analysis, comparison studies",
        "sha256": None,
        "url": f"{BASE_URL}grid_2.npz",
    },
}


def get_data_dir() -> Path:
    """Get the data directory, creating it if necessary."""
    data_dir = DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file_with_progress(url: str, filepath: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    if not HAS_DOWNLOAD_DEPS:
        raise ImportError(
            "requests and tqdm are required for downloading. "
            "Install with: pip install requests tqdm"
        )

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                desc=filepath.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                pbar.update(size)

        return True

    except Exception as e:
        print(f"Download failed: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def list_datasets() -> None:
    """List available datasets with descriptions."""
    print("Available CANNs Datasets:")
    print("=" * 60)

    for key, info in DATASETS.items():
        status = "Available" if info["url"] else "Setup required"
        print(f"\nDataset: {key}")
        print(f"  File: {info['filename']}")
        print(f"  Size: {info['size_mb']} MB")
        print(f"  Description: {info['description']}")
        print(f"  Usage: {info['usage']}")
        print(f"  Status: {status}")


def download_dataset(dataset_key: str, force: bool = False) -> Path | None:
    """
    Download a specific dataset.

    Parameters
    ----------
    dataset_key : str
        Key of the dataset to download (e.g., 'grid_1', 'roi_data').
    force : bool
        Whether to force re-download if file already exists.

    Returns
    -------
    Path or None
        Path to downloaded file if successful, None otherwise.
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return None

    info = DATASETS[dataset_key]

    if not info["url"]:
        print(f"{dataset_key} not yet available for download")
        print("Please use setup_local_datasets() to copy from local repository")
        return None

    data_dir = get_data_dir()
    filepath = data_dir / info["filename"]

    # Check if file already exists
    if filepath.exists() and not force:
        if info["sha256"]:
            # Verify hash if available
            current_hash = compute_file_hash(filepath)
            if current_hash == info["sha256"]:
                print(f"{dataset_key} already exists and is valid")
                return filepath
            else:
                print(f"{dataset_key} exists but hash mismatch, re-downloading...")
        else:
            print(f"{dataset_key} already exists")
            return filepath

    print(f"Downloading {dataset_key} ({info['size_mb']} MB)...")

    url = info["url"]
    if download_file_with_progress(url, filepath):
        print(f"Download completed: {filepath}")
        return filepath
    else:
        return None


def get_dataset_path(dataset_key: str, auto_setup: bool = True) -> Path | None:
    """
    Get path to a dataset, downloading/setting up if necessary.

    Parameters
    ----------
    dataset_key : str
        Key of the dataset.
    auto_setup : bool
        Whether to automatically attempt setup if dataset not found.

    Returns
    -------
    Path or None
        Path to dataset file if available, None otherwise.
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        return None

    data_dir = get_data_dir()
    filepath = data_dir / DATASETS[dataset_key]["filename"]

    if filepath.exists():
        return filepath

    if auto_setup:
        print(f"Dataset {dataset_key} not found, attempting setup...")

        if filepath.exists():
            return filepath

        # Then try download (if URL available)
        downloaded_path = download_dataset(dataset_key)
        if downloaded_path:
            return downloaded_path

    print(f"Dataset {dataset_key} not available")
    print("Try running setup_local_datasets() or download_dataset() manually")
    return None


def detect_file_type(filepath: Path) -> str:
    """Detect file type based on extension."""
    suffix = filepath.suffix.lower()
    if suffix in [".txt", ".dat", ".csv"]:
        return "text"
    elif suffix in [".npz", ".npy"]:
        return "numpy"
    elif suffix in [".pkl", ".pickle"]:
        return "pickle"
    elif suffix in [".json"]:
        return "json"
    elif suffix in [".h5", ".hdf5"]:
        return "hdf5"
    else:
        return "unknown"


def load_file(filepath: Path, file_type: str | None = None) -> Any:
    """
    Load data from file based on file type.

    Parameters
    ----------
    filepath : Path
        Path to the data file.
    file_type : str, optional
        Force specific file type. If None, auto-detect from extension.

    Returns
    -------
    Any
        Loaded data.
    """
    if file_type is None:
        file_type = detect_file_type(filepath)

    if file_type == "text":
        if not HAS_NUMPY:
            raise ImportError("numpy is required to load text data")
        try:
            return np.loadtxt(filepath)
        except Exception:
            # Fallback to reading as plain text
            with open(filepath) as f:
                return f.read()

    elif file_type == "numpy":
        if not HAS_NUMPY:
            raise ImportError("numpy is required to load numpy data")

        if filepath.suffix.lower() == ".npz":
            return dict(np.load(filepath, allow_pickle=True))
        else:
            return np.load(filepath, allow_pickle=True)

    elif file_type == "json":
        import json

        with open(filepath) as f:
            return json.load(f)

    elif file_type == "pickle":
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)

    elif file_type == "hdf5":
        try:
            import h5py

            return h5py.File(filepath, "r")
        except ImportError as err:
            raise ImportError("h5py is required to load HDF5 data") from err

    else:
        # Try to read as text
        with open(filepath) as f:
            return f.read()


def load(
    url: str,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    file_type: str | None = None,
) -> Any:
    """
    Universal data loading function that downloads and reads data from URLs.

    Parameters
    ----------
    url : str
        URL to download data from.
    cache_dir : str or Path, optional
        Directory to cache downloaded files. If None, uses temporary directory.
    force_download : bool
        Force re-download even if file exists in cache.
    file_type : str, optional
        Force specific file type ('text', 'numpy', 'json', 'pickle', 'hdf5').
        If None, auto-detect from file extension.

    Returns
    -------
    Any
        Loaded data.

    Examples
    --------
    >>> # Load numpy data
    >>> data = load('https://example.com/data.npz')
    >>>
    >>> # Load text data with custom cache
    >>> data = load('https://example.com/data.txt', cache_dir='./cache')
    >>>
    >>> # Force specific file type
    >>> data = load('https://example.com/data.bin', file_type='numpy')
    """
    if not HAS_DOWNLOAD_DEPS:
        raise ImportError(
            "requests and tqdm are required for downloading. "
            "Install with: pip install requests tqdm"
        )

    # Parse URL to get filename
    parsed_url = urlparse(url)
    filename = Path(parsed_url.path).name
    if not filename:
        filename = "downloaded_data"

    # Set up cache directory
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "canns_cache"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    filepath = cache_dir / filename

    # Download if needed
    if not filepath.exists() or force_download:
        print(f"Downloading from {url}...")
        if not download_file_with_progress(url, filepath):
            raise RuntimeError(f"Failed to download {url}")
    else:
        print(f"Using cached file: {filepath}")

    # Load and return data
    return load_file(filepath, file_type)


def get_huggingface_upload_guide() -> str:
    """
    Get guide for uploading datasets to Hugging Face.

    Returns
    -------
    str
        Upload guide text.
    """
    guide = """
Hugging Face Dataset Upload Guide

1. Create a Hugging Face account at https://huggingface.co

2. Install huggingface_hub:
   pip install huggingface_hub

3. Create a new dataset repository:
   - Go to https://huggingface.co/new-dataset  
   - Name: canns-datasets (or similar)
   - Make it public for easy access

4. Upload the data files using Python:

   from huggingface_hub import HfApi, login
   
   # Login (one time setup)
   login()
   
   # Upload files
   api = HfApi()
   
   for filename in ["ROI_data.txt", "grid_1.npz", "grid_2.npz"]:
       api.upload_file(
           path_or_fileobj=f"CANN-data-analysis/data/{filename}",
           path_in_repo=filename,
           repo_id="your-username/canns-datasets", 
           repo_type="dataset"
       )

5. Update this module:
   - Edit HUGGINGFACE_REPO variable
   - Set the 'url' field for each dataset in DATASETS dict

6. Create a README.md for the dataset repository with:
   - Dataset descriptions
   - Usage examples
   - Citation information
   - License information

Once uploaded, users can easily access the datasets through the CANNs package.
"""
    return guide


def quick_setup() -> bool:
    """
    Quick setup function to get datasets ready.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    print("CANNs Dataset Quick Setup")
    print("=" * 40)

    # First try downloading from Hugging Face
    print("Attempting to download datasets from Hugging Face...")
    download_success = True

    for dataset_key in DATASETS.keys():
        try:
            result = download_dataset(dataset_key)
            if result is None:
                download_success = False
                break
        except Exception as e:
            print(f"Download failed for {dataset_key}: {e}")
            download_success = False
            break

    if download_success:
        print("All datasets downloaded successfully from Hugging Face!")
        return True

    # If that fails, show instructions
    print("\nManual Setup Required:")
    print("1. Install download dependencies: pip install requests tqdm")
    print("2. Or clone the CANN-data-analysis repository:")
    print("   git clone https://github.com/Airs702/CANN-data-analysis.git")
    print("3. Run: setup_local_datasets('path/to/CANN-data-analysis/data')")

    return False
