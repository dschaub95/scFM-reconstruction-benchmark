"""Common utilities for data preparation."""

from pathlib import Path
from typing import Optional

from anndata import AnnData

from ..config import DataConfig
from ..masking import mask_data


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


def save_raw_data(adata: AnnData, raw_dir: Path, existing_file: Optional[Path] = None) -> Path:
    """
    Save raw data to the raw directory.
    
    If an existing file is provided, it will be renamed to data.h5ad.
    Otherwise, the AnnData object will be saved as data.h5ad.
    
    Parameters
    ----------
    adata
        AnnData object (used if existing_file is None)
    raw_dir
        Directory to save raw data
    existing_file
        Optional path to existing file that should be renamed
        
    Returns
    -------
    Path to saved file
    """
    raw_path = raw_dir / "data.h5ad"
    
    if existing_file is not None and existing_file.exists():
        # Rename existing file instead of saving again
        existing_file.rename(raw_path)
        print(f"Renamed {existing_file.name} to {raw_path.name}")
    else:
        # Save the AnnData object
        adata.write(raw_path)
        print(f"Saved raw data to {raw_path}")
    
    return raw_path


def save_masked_data(adata: AnnData, masked_dir: Path, mask_percentage: float, seed: int) -> Path:
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
    mask_pct_value = mask_percentage * 100
    mask_pct_str = f"{mask_pct_value:g}"
    masked_path = masked_dir / f"mask_{mask_pct_str}_seed_{seed}.h5ad"
    adata.write(masked_path)
    print(f"Saved masked data to {masked_path}")
    return masked_path


def prepare_dataset(config: DataConfig) -> None:
    """
    Prepare a dataset by downloading and masking it.
    
    This function dispatches to dataset-specific download functions
    and uses common utilities for directory creation, masking, and saving.
    
    Raw data is only downloaded if it doesn't already exist (shared across experiments).
    Masked data is always created for the specified mask percentage and seed.
    
    Parameters
    ----------
    config
        Data configuration
    """
    # Create output directories
    raw_dir, masked_dir = create_output_directories(config.output_dir)
    
    # Check if raw data already exists
    raw_path = raw_dir / "data.h5ad"
    if raw_path.exists():
        print(f"Raw data already exists at {raw_path}, skipping download...")
        import anndata as ad
        adata = ad.read_h5ad(raw_path)
    else:
        # Import dataset-specific download function
        dataset_name_lower = config.dataset_name.lower()
        
        if dataset_name_lower == "pbmc":
            from .pbmc import download_pbmc_dataset
            
            # Download dataset directly to raw folder
            print(f"Downloading {config.dataset_name} dataset directly to {raw_dir}...")
            adata, downloaded_path = download_pbmc_dataset(cache_dir=raw_dir)
        else:
            raise ValueError(f"Unknown dataset: {config.dataset_name}")
        
        # Rename downloaded file to data.h5ad (or save if no existing file)
        save_raw_data(adata, raw_dir, existing_file=downloaded_path)
    
    # Create masked version
    print(f"Masking {config.mask_percentage * 100:.1f}% of values...")
    masked_adata = mask_data(
        adata,
        mask_percentage=config.mask_percentage,
        seed=config.seed,
        inplace=False,
    )
    
    # Save masked data
    save_masked_data(masked_adata, masked_dir, config.mask_percentage, config.seed)

