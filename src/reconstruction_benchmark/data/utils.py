"""Common utilities for data preparation."""

import warnings
from importlib import import_module
from pathlib import Path
from typing import Optional

import anndata as ad
from anndata import AnnData

from ..config import DataConfig
from ..masking import mask_data


def format_mask_percentage(mask_percentage: float) -> str:
    """Format mask percentage as string (e.g., 0.15 -> '15')."""
    return f"{mask_percentage * 100:g}"


def discover_raw_data_file(raw_dir: Path, dataset_name: str) -> Optional[Path]:
    """
    Auto-discover the raw data file in the raw directory.

    Returns the most recently modified .h5ad file in the directory.
    Shows a warning if multiple files are found.

    Parameters
    ----------
    raw_dir
        Directory containing raw data files
    dataset_name
        Name of the dataset (e.g., "pbmc3k") - used for warning messages

    Returns
    -------
    Path to raw data file if found, None otherwise
    """
    if not raw_dir.exists():
        return None

    # Find all .h5ad files in the raw directory
    h5ad_files = [f for f in raw_dir.glob("*.h5ad") if f.is_file()]

    if not h5ad_files:
        return None

    # Find the most recently modified file
    latest_file = max(h5ad_files, key=lambda p: p.stat().st_mtime)

    if len(h5ad_files) > 1:
        file_list = ", ".join(f.name for f in h5ad_files)
        warnings.warn(
            f"Multiple files found in raw directory for dataset '{dataset_name}': {file_list}. "
            f"Using the most recently modified file: {latest_file.name}",
            UserWarning,
            stacklevel=2,
        )

    return latest_file


def create_output_directories(output_dir: Path) -> tuple[Path, Path]:
    """
    Create raw and masked output directories.

    Parameters
    ----------
    output_dir
        Base output directory

    Returns
    -------
    Tuple of (raw_dir, masked_dir)
    """
    raw_dir = output_dir / "raw"
    masked_dir = output_dir / "masked"
    raw_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, masked_dir


def save_masked_data(
    adata: AnnData, masked_dir: Path, mask_percentage: float, seed: int
) -> Path:
    """
    Save masked data with mask_<pct>_seed_<seed>.h5ad filename.

    Parameters
    ----------
    adata
        Masked AnnData object to save
    masked_dir
        Directory to save masked data
    mask_percentage
        Percentage of values masked
    seed
        Random seed used for masking

    Returns
    -------
    Path to saved file
    """
    mask_pct_str = format_mask_percentage(mask_percentage)
    masked_path = masked_dir / f"mask_{mask_pct_str}_seed_{seed}.h5ad"
    adata.write(masked_path)
    print(f"Saved masked data to {masked_path}")
    return masked_path


def _get_download_function(dataset_name: str):
    """
    Dynamically import and return the download_dataset function for a dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset (e.g., "pbmc3k")

    Returns
    -------
    The download_dataset function from the dataset module

    Raises
    ------
    ImportError
        If the dataset module cannot be imported
    AttributeError
        If the download_dataset function is not found in the module
    """
    try:
        module_name = f".datasets.{dataset_name}"
        module = import_module(module_name, package="reconstruction_benchmark.data")
        download_dataset = getattr(module, "download_dataset")
        return download_dataset
    except ImportError as e:
        raise ImportError(
            f"Could not import dataset module for '{dataset_name}': {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Dataset module '{dataset_name}' does not have a 'download_dataset' function: {e}"
        ) from e


def _discover_datasets() -> list[str]:
    """
    Discover all available dataset modules in the datasets folder.

    Returns
    -------
    List of dataset names (file stems without .py extension)
    """
    datasets_dir = Path(__file__).parent / "datasets"
    if not datasets_dir.exists():
        return []
    dataset_files = [
        f.stem for f in datasets_dir.glob("*.py") if f.name != "__init__.py"
    ]
    return sorted(dataset_files)


def _prepare_single_dataset(
    dataset_name: str,
    output_dir: Path,
    mask_percentage: float,
    seed: int,
    data_dir: Path,
) -> None:
    """
    Prepare a single dataset by downloading (if needed) and masking it.

    Parameters
    ----------
    dataset_name
        Name of the dataset (e.g., "pbmc3k")
    output_dir
        Base output directory for this dataset (e.g., data/pbmc3k)
    mask_percentage
        Percentage of values to mask
    seed
        Random seed for masking
    data_dir
        Base data directory (for downloading datasets)
    """
    # Create output directories
    raw_dir, masked_dir = create_output_directories(output_dir)

    # Auto-discover existing raw data file
    raw_path = discover_raw_data_file(raw_dir, dataset_name)
    if raw_path is not None:
        print(f"Raw data already exists at {raw_path}, skipping download...")
        adata = ad.read_h5ad(raw_path)
    else:
        # Dynamically import dataset-specific download function
        download_dataset = _get_download_function(dataset_name)

        print(f"Downloading {dataset_name} dataset to {data_dir}...")
        adata_downloaded, downloaded_path = download_dataset(data_dir=data_dir)

        # Discover the actual downloaded file
        dataset_raw_dir = data_dir / dataset_name / "raw"
        raw_path = discover_raw_data_file(dataset_raw_dir, dataset_name)
        if raw_path is not None:
            adata = ad.read_h5ad(raw_path)
            print(f"Loaded {dataset_name} dataset from {raw_path}")
        else:
            # Fallback to the downloaded dataset
            adata = adata_downloaded

    # Create masked version
    mask_pct_str = format_mask_percentage(mask_percentage)
    print(f"Masking {mask_pct_str}% of values for {dataset_name}...")
    masked_adata = mask_data(
        adata,
        mask_percentage=mask_percentage,
        seed=seed,
        inplace=False,
    )

    # Save masked data
    save_masked_data(masked_adata, masked_dir, mask_percentage, seed)


def prepare_dataset(config: DataConfig) -> None:
    """
    Prepare a dataset by downloading and masking it.

    This function dispatches to dataset-specific download functions
    and uses common utilities for directory creation, masking, and saving.

    Raw data is only downloaded if it doesn't already exist (shared across experiments).
    The original filename is preserved (not renamed).
    Masked data is always created for the specified mask percentage and seed.

    If dataset_name is None, all available datasets will be prepared.

    Parameters
    ----------
    config
        Data configuration. If dataset_name is None, all datasets will be prepared.
    """
    if config.dataset_name is None:
        # Discover all available datasets
        dataset_names = _discover_datasets()
        if not dataset_names:
            raise ValueError(
                "No datasets found in datasets folder. "
                "Please specify a dataset name or add dataset modules to data/datasets/"
            )
        print(f"Preparing all datasets: {', '.join(dataset_names)}")
        for dataset_name in dataset_names:
            print(f"\n{'=' * 60}")
            print(f"Preparing dataset: {dataset_name}")
            print(f"{'=' * 60}")
            # Create config for this specific dataset
            dataset_config = DataConfig(
                dataset_name=dataset_name,
                mask_percentage=config.mask_percentage,
                seed=config.seed,
                data_dir=config.data_dir,
            )
            # dataset_config.output_dir is guaranteed to be set since dataset_name is not None
            assert dataset_config.output_dir is not None
            _prepare_single_dataset(
                dataset_name=dataset_name,
                output_dir=dataset_config.output_dir,
                mask_percentage=dataset_config.mask_percentage,
                seed=dataset_config.seed,
                data_dir=dataset_config.data_dir,
            )
    else:
        # Prepare single dataset
        _prepare_single_dataset(
            dataset_name=config.dataset_name,
            output_dir=config.output_dir,
            mask_percentage=config.mask_percentage,
            seed=config.seed,
            data_dir=config.data_dir,
        )
