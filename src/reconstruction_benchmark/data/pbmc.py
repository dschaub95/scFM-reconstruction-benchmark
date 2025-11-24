"""PBMC dataset download utilities."""

from pathlib import Path

import scanpy as sc
from anndata import AnnData


def download_pbmc_dataset(cache_dir: Path) -> tuple[AnnData, Path]:
    """
    Download the PBMC dataset from Scanpy directly to the specified cache directory.

    Parameters
    ----------
    cache_dir
        Directory where scanpy should download the data

    Returns
    -------
    Tuple of (AnnData object, path to downloaded file)
    """
    # Set scanpy's dataset directory to download directly to raw folder
    original_dataset_dir = sc.settings.datasetdir
    sc.settings.datasetdir = cache_dir

    try:
        adata = sc.datasets.pbmc3k()
        # Scanpy saves the file as pbmc3k_raw.h5ad
        downloaded_path = cache_dir / "pbmc3k_raw.h5ad"
    finally:
        # Restore original dataset directory
        sc.settings.datasetdir = original_dataset_dir

    return adata, downloaded_path
