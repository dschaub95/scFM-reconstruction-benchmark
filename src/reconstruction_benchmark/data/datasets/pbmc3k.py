"""PBMC3k dataset download utilities."""

from pathlib import Path

import scanpy as sc
from anndata import AnnData


def download_dataset(data_dir: Path) -> tuple[AnnData, Path]:
    """
    Download the PBMC3k dataset from Scanpy to the specified directory.

    Downloads:
    - pbmc3k to data/pbmc3k/raw/pbmc3k_raw.h5ad

    Parameters
    ----------
    data_dir
        Base data directory (e.g., Path("data"))

    Returns
    -------
    Tuple of (pbmc3k AnnData object, path to pbmc3k downloaded file)
    """
    # Create directory for pbmc3k dataset
    pbmc3k_raw_dir = data_dir / "pbmc3k" / "raw"
    pbmc3k_raw_dir.mkdir(parents=True, exist_ok=True)

    # Save original dataset directory
    original_dataset_dir = sc.settings.datasetdir

    # Download pbmc3k dataset
    sc.settings.datasetdir = pbmc3k_raw_dir
    try:
        adata_3k = sc.datasets.pbmc3k()
        # Scanpy saves the file as pbmc3k_raw.h5ad
        downloaded_path_3k = pbmc3k_raw_dir / "pbmc3k_raw.h5ad"
        print(f"Downloaded pbmc3k dataset to {downloaded_path_3k}")
    finally:
        sc.settings.datasetdir = original_dataset_dir

    return adata_3k, downloaded_path_3k

